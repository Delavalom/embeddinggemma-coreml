import Accelerate

/// Antialiased bicubic resize of 8-bit images, ported from the path the reference takes: torchvision
/// `resize(uint8, BICUBIC, antialias=True)` -> torch `_upsample_bicubic2d_aa` on uint8 (CPU, no AVX2), i.e.
/// `separable_upsample_generic_Nd_kernel_impl` with the uint8 specialization:
///
/// - per output index: `center = scale * (i + 0.5)`, `support = 2 * max(scale, 1)`, taps
///   `[int(center - support + 0.5), int(center + support + 0.5))` clipped to the input, weights
///   `cubic((j + xmin - center + 0.5) / max(scale, 1))` with a = -0.5, normalized to sum 1 (in Double);
/// - weights rounded to int16 fixed point with the largest precision that keeps `max weight * 2^(p+1) < 2^15`;
/// - horizontal pass first, then vertical, each rounding with `1 << (p - 1)`, shifting and clamping to 0...255
///   (so the intermediate image is uint8 too).
///
/// It matches s0's pixels bit for bit when given the same decoded RGB (checked in numpy against s0 before this was
/// written). vImage's high-quality scaling was tried first and missed the 1/255 p99 gate by 11x (see
/// stages/s5_swift/README.md).
enum Resampler {
  struct Taps {
    let start: [Int]
    let count: [Int]
    let weights: [Int32]  // out x maxTaps, row-major
    let maxTaps: Int
    let precision: Int
  }

  static func cubic(_ x: Double, a: Double = -0.5) -> Double {
    let x = abs(x)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
    return 0
  }

  static func taps(inSize: Int, outSize: Int) -> Taps {
    let scale = Double(inSize) / Double(outSize)
    let support = 2.0 * (scale >= 1 ? scale : 1)
    let invscale = scale >= 1 ? 1 / scale : 1
    let maxTaps = Int(support.rounded(.up)) * 2 + 1
    var start = [Int](repeating: 0, count: outSize), count = start
    var w = [Double](repeating: 0, count: outSize * maxTaps)
    var wmax = 0.0
    for i in 0..<outSize {
      let center = scale * (Double(i) + 0.5)
      let xmin = max(Int(center - support + 0.5), 0)  // Int() truncates toward zero, as static_cast<int64_t>
      let xsize = min(max(min(Int(center + support + 0.5), inSize) - xmin, 0), maxTaps)
      var total = 0.0
      for j in 0..<xsize {
        let v = cubic((Double(j + xmin) - center + 0.5) * invscale)
        w[i * maxTaps + j] = v
        total += v
      }
      if total != 0 {
        for j in 0..<xsize {
          w[i * maxTaps + j] /= total
          wmax = max(wmax, w[i * maxTaps + j])
        }
      }
      (start[i], count[i]) = (xmin, xsize)
    }
    var precision = 0
    while precision < 22, Int(0.5 + wmax * Double(1 << (precision + 1))) < (1 << 15) { precision += 1 }
    let fixed = w.map { v -> Int32 in
      let s = v * Double(1 << precision)
      return Int32((s < 0 ? s - 0.5 : s + 0.5).rounded(.towardZero))
    }
    return Taps(start: start, count: count, weights: fixed, maxTaps: maxTaps, precision: precision)
  }

  /// Exact fixed-point rounding of accumulated sums: `clamp(floor(x / 2^p), 0, 255)`, as `(acc >> p)` clamped.
  /// Every value is an integer far below 2^53, so Double arithmetic here is exact and matches int32 bit for bit.
  static func finish(_ x: UnsafeMutablePointer<Double>, count: Int, precision: Int) {
    var scale = 1.0 / Double(1 << precision), lo = 0.0, hi = 255.0
    var n = Int32(count)
    vDSP_vsmulD(x, 1, &scale, x, 1, vDSP_Length(count))
    vvfloor(x, x, &n)
    vDSP_vclipD(x, 1, &lo, &hi, x, 1, vDSP_Length(count))
  }

  /// `dst` rows (`taps.start.count` of them, `length` long) = round + sum_j w[i][j] * src row (start[i] + j).
  static func combineRows(_ src: UnsafePointer<Double>, _ dst: UnsafeMutablePointer<Double>, length: Int,
                          taps t: Taps, weights: UnsafePointer<Double>) {
    var round = t.precision > 0 ? Double(1 << (t.precision - 1)) : 0
    let n = vDSP_Length(length)
    for i in 0..<t.start.count {
      let acc = dst + i * length
      vDSP_vfillD(&round, acc, 1, n)
      for j in 0..<t.count[i] {
        var w = weights[i * t.maxTaps + j]
        vDSP_vsmaD(src + (t.start[i] + j) * length, 1, &w, acc, 1, acc, 1, n)
      }
    }
  }

  /// RGBX (4 bytes per pixel, X ignored) `width x height` -> planar RGB `[3, toHeight, toWidth]`.
  /// Horizontal pass first (on transposed row strips, so both passes combine whole contiguous rows), then vertical.
  static func resizeRGBX(_ src: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
    let h = taps(inSize: width, outSize: toWidth), v = taps(inSize: height, outSize: toHeight)
    let hw = h.weights.map(Double.init), vw = v.weights.map(Double.init)
    let stripRows = max(1, min(height, (1 << 20) / width))
    var strip = [Double](repeating: 0, count: stripRows * width), stripT = strip
    var accT = [Double](repeating: 0, count: stripRows * toWidth)
    var mid = [Double](repeating: 0, count: height * toWidth)  // one channel after the horizontal pass
    var outD = [Double](repeating: 0, count: toHeight * toWidth)
    var out = [UInt8](repeating: 0, count: 3 * toHeight * toWidth)
    src.withUnsafeBufferPointer { s in
      for c in 0..<3 {
        for r0 in stride(from: 0, to: height, by: stripRows) {
          let n = min(stripRows, height - r0)
          vDSP_vfltu8D(s.baseAddress! + r0 * width * 4 + c, 4, &strip, 1, vDSP_Length(n * width))
          if width == toWidth {  // torch skips a pass whose size does not change
            mid.withUnsafeMutableBufferPointer { m in (m.baseAddress! + r0 * width).update(from: strip, count: n * width) }
            continue
          }
          vDSP_mtransD(strip, 1, &stripT, 1, vDSP_Length(width), vDSP_Length(n))  // width x n
          combineRows(stripT, &accT, length: n, taps: h, weights: hw)  // toWidth x n
          finish(&accT, count: toWidth * n, precision: h.precision)
          mid.withUnsafeMutableBufferPointer { m in
            vDSP_mtransD(accT, 1, m.baseAddress! + r0 * toWidth, 1, vDSP_Length(n), vDSP_Length(toWidth))
          }
        }
        if height == toHeight {
          outD.withUnsafeMutableBufferPointer { o in o.baseAddress!.update(from: mid, count: mid.count) }
        } else {
          combineRows(mid, &outD, length: toWidth, taps: v, weights: vw)
          finish(&outD, count: outD.count, precision: v.precision)
        }
        out.withUnsafeMutableBufferPointer { o in
          vDSP_vfixu8D(outD, 1, o.baseAddress! + c * outD.count, 1, vDSP_Length(outD.count))
        }
      }
    }
    return out
  }
}
