import Accelerate
import Foundation

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

  /// torch's `aa_filter` for bicubic: `cubic_convolution1` / `cubic_convolution2` with A = -0.5, written exactly
  /// as torch writes them (same operations in the same order, so the Double weights are identical).
  static func cubic(_ x: Double) -> Double {
    let a = -0.5, x = abs(x)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
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

  /// For output rows `rows`: `dst[i - rows.lowerBound]` = round + sum_j w[i][j] * src row (start[i] + j - srcBase),
  /// every row `length` long.
  static func combineRows(_ src: UnsafePointer<Double>, srcBase: Int, _ dst: UnsafeMutablePointer<Double>,
                          length: Int, rows: Range<Int>, taps t: Taps, weights: UnsafePointer<Double>) {
    var round = t.precision > 0 ? Double(1 << (t.precision - 1)) : 0
    let n = vDSP_Length(length)
    for i in rows {
      let acc = dst + (i - rows.lowerBound) * length
      vDSP_vfillD(&round, acc, 1, n)
      for j in 0..<t.count[i] {
        var w = weights[i * t.maxTaps + j]
        vDSP_vsmaD(src + (t.start[i] + j - srcBase) * length, 1, &w, acc, 1, acc, 1, n)
      }
    }
  }

  /// Rows per horizontal strip: about 64k pixels, so each worker's buffers stay near 1 MB at any source size.
  static func stripRows(width: Int, height: Int) -> Int { max(1, min(height, (1 << 16) / max(width, 1))) }

  /// RGBX `width x height` (from `rows`) -> planar RGB `[3, toHeight, toWidth]`. The horizontal pass runs over
  /// row strips in parallel and keeps only the uint8 intermediate (`3 x height x toWidth`); `rows` is released
  /// right after it. The vertical pass runs over (channel, block of output rows) in parallel.
  static func resize(width: Int, height: Int, toWidth: Int, toHeight: Int, rows source: RowProvider) -> [UInt8] {
    let h = taps(inSize: width, outSize: toWidth), v = taps(inSize: height, outSize: toHeight)
    let hw = h.weights.map(Double.init), vw = v.weights.map(Double.init)
    var mid = [UInt8](repeating: 0, count: 3 * height * toWidth)  // planar, after the horizontal pass
    var out = [UInt8](repeating: 0, count: 3 * toHeight * toWidth)
    let rowsPerStrip = stripRows(width: width, height: height)
    let strips = (height + rowsPerStrip - 1) / rowsPerStrip
    mid.withUnsafeMutableBufferPointer { midBuf in
      let midBox = Shared(midBuf.baseAddress!)
      hw.withUnsafeBufferPointer { hwBuf in
        let hwBox = Shared(hwBuf.baseAddress!)
        DispatchQueue.concurrentPerform(iterations: strips) { s in
          let midP = midBox.pointer, hwP = hwBox.pointer
          let r0 = s * rowsPerStrip, n = min(rowsPerStrip, height - r0)
          let rgbx = UnsafeMutablePointer<UInt8>.allocate(capacity: n * width * 4)
          let strip = UnsafeMutablePointer<Double>.allocate(capacity: n * max(width, toWidth))
          let stripT = UnsafeMutablePointer<Double>.allocate(capacity: n * max(width, toWidth))
          defer { rgbx.deallocate(); strip.deallocate(); stripT.deallocate() }
          source.fill(y0: r0, rows: n, into: rgbx)
          for c in 0..<3 {
            let plane = midP + c * height * toWidth + r0 * toWidth
            if width == toWidth {  // torch skips a pass whose size does not change
              for p in 0..<(n * width) { plane[p] = rgbx[p * 4 + c] }
              continue
            }
            vDSP_vfltu8D(rgbx + c, 4, strip, 1, vDSP_Length(n * width))
            vDSP_mtransD(strip, 1, stripT, 1, vDSP_Length(width), vDSP_Length(n))  // width x n
            combineRows(stripT, srcBase: 0, strip, length: n, rows: 0..<toWidth, taps: h, weights: hwP)
            finish(strip, count: toWidth * n, precision: h.precision)  // toWidth x n
            vDSP_mtransD(strip, 1, stripT, 1, vDSP_Length(n), vDSP_Length(toWidth))  // n x toWidth
            vDSP_vfixu8D(stripT, 1, plane, 1, vDSP_Length(n * toWidth))
          }
        }
      }
    }
    source.release()
    if height == toHeight { return mid }
    let blockRows = 32, blocks = (toHeight + blockRows - 1) / blockRows
    mid.withUnsafeBufferPointer { midBuf in
      out.withUnsafeMutableBufferPointer { outBuf in
        let midBox = Shared(midBuf.baseAddress!), outBox = Shared(outBuf.baseAddress!)
        vw.withUnsafeBufferPointer { vwBuf in
          let vwBox = Shared(vwBuf.baseAddress!)
          DispatchQueue.concurrentPerform(iterations: 3 * blocks) { job in
            let midP = midBox.pointer, outP = outBox.pointer, vwP = vwBox.pointer
            let c = job / blocks, rows = (job % blocks) * blockRows..<min(toHeight, (job % blocks + 1) * blockRows)
            let first = v.start[rows.lowerBound]
            let last = rows.map { v.start[$0] + v.count[$0] }.max()!
            let src = UnsafeMutablePointer<Double>.allocate(capacity: (last - first) * toWidth)
            let dst = UnsafeMutablePointer<Double>.allocate(capacity: rows.count * toWidth)
            defer { src.deallocate(); dst.deallocate() }
            vDSP_vfltu8D(midP + c * height * toWidth + first * toWidth, 1, src, 1, vDSP_Length((last - first) * toWidth))
            combineRows(src, srcBase: first, dst, length: toWidth, rows: rows, taps: v, weights: vwP)
            finish(dst, count: rows.count * toWidth, precision: v.precision)
            vDSP_vfixu8D(dst, 1, outP + c * toHeight * toWidth + rows.lowerBound * toWidth, 1,
                         vDSP_Length(rows.count * toWidth))
          }
        }
      }
    }
    return out
  }

}

/// A pointer shared by the `concurrentPerform` workers of one resize. Safe because the buffer outlives the call and
/// every worker reads shared inputs and writes only its own disjoint rows.
struct Shared<P>: @unchecked Sendable {
  let pointer: P
  init(_ pointer: P) { self.pointer = pointer }
}
