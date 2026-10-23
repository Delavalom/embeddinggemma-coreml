import CoreGraphics
import CoreImage
import CryptoKit
import Foundation
import ImageIO
import Testing
@testable import EmbeddingGemma

func loadCGImage(_ url: URL) throws -> CGImage {
  let src = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
  return try #require(CGImageSourceCreateImageAtIndex(src, 0, nil))
}

/// A width x height RGB image whose pixel (x, y) is (x % 256, y % 256, 7), drawn in sRGB.
func gradient(width: Int, height: Int) throws -> CGImage {
  var bytes = [UInt8](repeating: 255, count: width * height * 4)
  for y in 0..<height { for x in 0..<width { bytes[(y * width + x) * 4] = UInt8(x % 256); bytes[(y * width + x) * 4 + 1] = UInt8(y % 256); bytes[(y * width + x) * 4 + 2] = 7 } }
  let ctx = try #require(CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                   space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                   bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
  return try #require(ctx.makeImage())
}

func percentile(_ xs: [Float], _ q: Double) -> Float {
  let s = xs.sorted()
  return s.isEmpty ? 0 : s[min(s.count - 1, Int((Double(s.count - 1) * q).rounded()))]
}

@Suite struct ImagePreprocessorTests {
  static let gate: Float = 0.003922  // [s5_swift].max_pixel_abs_diff in gates.toml (1/255 rounded up)

  /// |swift - reference| in 0-1 units over the committed samples, and whether the full image is bit-identical.
  static func compare(_ out: PreprocessedImage, _ ref: PixelBudget, samples: Int) throws -> (diffs: [Float], exact: Bool) {
    #expect(out.bucket.name.rawValue == ref.bucket && out.bucket.height == ref.height && out.bucket.width == ref.width)
    let values = [UInt8](try #require(Data(base64Encoded: ref.sample_b64)))
    let idx = TestSupport.sampleIndices(out.planar.count, samples)
    let diffs = zip(idx, values).map { abs(Float(out.planar[$0.0]) - Float($0.1)) / 255 }
    let sha = SHA256.hash(data: out.interleavedRGB()).map { String(format: "%02x", $0) }.joined()
    return (diffs, sha == ref.sha256)
  }

  /// The pixel gate, on decode-identical inputs (ADR-0004): synthetic lossless images resized by torch's uint8
  /// antialiased bicubic, at every bucket and both budgets. The p99 is gated at 1/255; the port is in fact
  /// bit-exact, which is asserted too.
  @Test(arguments: 0..<10)
  func resizeMatchesTorchOnLosslessInputs(index: Int) throws {
    let fx = try TestSupport.load(PixelFixtures.self, "pixels.json")
    let s = fx.synthetic[index]
    let img = try TestSupport.synthetic(height: s.height, width: s.width)
    for (budget, ref) in s.budgets.sorted(by: { $0.key < $1.key }) {
      let (diffs, exact) = try Self.compare(try ImagePreprocessor(budget: Int(budget)!).preprocess(img), ref,
                                            samples: fx.samples)
      #expect(percentile(diffs, 0.99) <= Self.gate, "\(s.height)x\(s.width)@\(budget)")
      #expect(exact, "\(s.height)x\(s.width)@\(budget) is not bit-exact (max diff \(diffs.max() ?? 0))")
    }
  }

  /// 9x to 12x downscales (12 MP at budget 70, extreme aspect ratios), bit-exact against torch (SHA-256 of the
  /// whole output). 48 MP is checked by `perfLargeImages` in release builds.
  @Test(arguments: [(3024, 4032, 70), (3024, 4032, 280), (5000, 1200, 70), (1080, 7000, 70)])
  func largeDownscalesAreBitExact(h: Int, w: Int, budget: Int) throws {
    let fx = try TestSupport.load(PixelFixtures.self, "pixels.json")
    let ref = try #require(fx.large_synthetic.first { $0.height == h && $0.width == w && $0.budget == budget })
    let out = try ImagePreprocessor(budget: budget).preprocess(try TestSupport.synthetic(height: h, width: w))
    #expect(out.bucket.name.rawValue == ref.bucket)
    #expect(SHA256.hash(data: out.interleavedRGB()).map { String(format: "%02x", $0) }.joined() == ref.sha256)
  }

  /// Non-premultiplied sRGB RGBA is read directly: alpha dropped, stored colours kept at every alpha, exactly as
  /// PIL's convert("RGB"). Forced through the drawing path (premultiplied), colours survive within 8-bit rounding
  /// and alpha 0 becomes black (ADR-0005).
  @Test func alphaIsDroppedKeepingColours() throws {
    func rgba(alpha: UInt8) throws -> CGImage {
      var px = [UInt8]()
      for _ in 0..<(96 * 96) { px += [200, 100, 50, alpha] }
      let provider = try #require(CGDataProvider(data: Data(px) as CFData))
      return try #require(CGImage(width: 96, height: 96, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 96 * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
                                  decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }
    let p = try ImagePreprocessor(budget: 70)
    for (alpha, drawnTolerance) in [(UInt8(255), 0), (128, 2), (16, 16), (0, 255)] {
      let direct = try p.preprocess(try rgba(alpha: alpha))
      let drawn = try p.preprocess(try rgba(alpha: alpha), orientation: .up, read: .never)
      let n = direct.bucket.height * direct.bucket.width
      for (c, want) in [200, 100, 50].enumerated() {
        #expect(Int(direct.planar[c * n + n / 2]) == want, "direct, alpha \(alpha) channel \(c)")
        #expect(abs(Int(drawn.planar[c * n + n / 2]) - want) <= drawnTolerance, "drawn, alpha \(alpha) channel \(c)")
      }
    }
    #expect(try p.preprocess(try rgba(alpha: 0), orientation: .up, read: .never).planar.allSatisfy { $0 == 0 })
  }

  /// For bitmaps in memory, the direct read and the drawing path give identical pixels in every layout the direct
  /// path accepts (RGBX, XRGB, BGRX little-endian, packed RGB). Production only reads non-premultiplied alpha
  /// directly (see ImagePreprocessor on ImageIO's two JPEG decodes).
  @Test func directReadEqualsDrawing() throws {
    let base = try TestSupport.synthetic(height: 300, width: 410)
    let ctxs: [(CGImageAlphaInfo, CGBitmapInfo, Int)] = [
      (.noneSkipLast, [], 32), (.noneSkipFirst, [], 32), (.noneSkipFirst, .byteOrder32Little, 32),
      (.noneSkipLast, .byteOrder32Little, 32),
    ]
    let p = try ImagePreprocessor(budget: 70)
    let want = try p.preprocess(base, orientation: .up, read: .never).planar
    for (alpha, order, _) in ctxs {
      let ctx = try #require(CGContext(data: nil, width: 410, height: 300, bitsPerComponent: 8, bytesPerRow: 0,
                                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                       bitmapInfo: alpha.rawValue | order.rawValue))
      ctx.draw(base, in: CGRect(x: 0, y: 0, width: 410, height: 300))
      let img = try #require(ctx.makeImage())
      #expect(RowProvider.directLayout(img) != nil, "\(alpha.rawValue) \(order.rawValue)")
      #expect(try p.preprocess(img, orientation: .up, read: .whenPossible).planar == want, "\(alpha.rawValue) \(order.rawValue)")
    }
    // Packed 24-bit RGB.
    let rgbx = try ImagePreprocessor.drawUpright(base, orientation: .up).rgbx
    let rgb = Data(rgbx.enumerated().filter { $0.offset % 4 != 3 }.map(\.element))
    let packed = try #require(CGImage(width: 410, height: 300, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: 410 * 3,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: [],
                                      provider: CGDataProvider(data: rgb as CFData)!, decode: nil,
                                      shouldInterpolate: false, intent: .defaultIntent))
    #expect(RowProvider.directLayout(packed) != nil)
    #expect(try p.preprocess(packed, orientation: .up, read: .whenPossible).planar == want)
  }

  /// Display P3 pixels are converted to sRGB (ADR-0005), not passed through as if they were sRGB.
  @Test func displayP3IsConvertedToSRGB() throws {
    var px = [UInt8]()
    for _ in 0..<(96 * 96) { px += [200, 100, 50, 255] }
    let ctx = try #require(CGContext(data: &px, width: 96, height: 96, bitsPerComponent: 8, bytesPerRow: 96 * 4,
                                     space: CGColorSpace(name: CGColorSpace.displayP3)!,
                                     bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    let out = try ImagePreprocessor(budget: 70).preprocess(try #require(ctx.makeImage()))
    let n = out.bucket.height * out.bucket.width
    let rgb = (0..<3).map { Int(out.planar[$0 * n]) }
    #expect(rgb[0] > 205 && rgb[1] < 100)  // P3 (200, 100, 50) is about sRGB (215, 95, 40)
  }

  /// Release only (`swift test -c release --filter perfLargeImages`): 12 MP and 48 MP, bit-exact, with timings.
  @Test(.enabled(if: !_isDebugAssertConfiguration(), "release builds only"), .timeLimit(.minutes(5)))
  func perfLargeImages() throws {
    let fx = try TestSupport.load(PixelFixtures.self, "pixels.json")
    for (h, w) in [(3024, 4032), (6048, 8064)] {
      let img = try TestSupport.synthetic(height: h, width: w)
      for budget in [70, 280] {
        // EGCM_PERF_CASE=HxW@B runs one case, so each peak is measured in a fresh process.
        if let only = ProcessInfo.processInfo.environment["EGCM_PERF_CASE"], only != "\(h)x\(w)@\(budget)" { continue }
        let ref = try #require(fx.large_synthetic.first { $0.height == h && $0.width == w && $0.budget == budget })
        var times: [Double] = []
        var sha = ""
        returnFreedMemoryToOS()
        let base = physFootprint(), sampler = PeakSampler()
        sampler.start()
        for _ in 0..<5 {
          let t0 = ContinuousClock.now
          let out = try ImagePreprocessor(budget: budget).preprocess(img)
          let d = ContinuousClock.now - t0
          times.append(Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
          sha = SHA256.hash(data: out.interleavedRGB()).map { String(format: "%02x", $0) }.joined()
        }
        let extraMB = Double(sampler.stop() - base) / 1_048_576
        #expect(sha == ref.sha256, "\(h)x\(w)@\(budget)")
        print("perfLargeImages \(h)x\(w)@\(budget): min \(Int(times.min()!)) ms, runs \(times.map { Int($0) }), "
              + "peak extra \(Int(extraMB)) MB")
      }
    }
  }

  /// The fixture photos against s0. PNG decodes identically everywhere, so those are gated and must be bit-exact.
  /// JPEG decoding differs between ImageIO and PIL/libjpeg-turbo (chroma upsampling, IDCT), by up to ~48/255 per
  /// pixel at full size; after the resize that leaves p99 2/255 on these fixtures. That is recorded, not gated (the
  /// end-to-end cosine gate covers it); the 3/255 bound only catches regressions.
  @Test func fixturePhotosAgainstS0() throws {
    let fx = try TestSupport.load(PixelFixtures.self, "pixels.json")
    #expect(fx.hf_revision == TestSupport.hfRevision && fx.images.count == 20 && fx.synthetic.count == 10)
    var jpeg: [Float] = [], png: [Float] = [], exact = 0, pngCount = 0
    for img in fx.images {
      let cg = try loadCGImage(TestSupport.fixturesDir.appending(path: img.file))
      #expect(cg.width == img.width && cg.height == img.height)
      let isPNG = img.file.hasSuffix(".png")
      for (budget, ref) in img.budgets.sorted(by: { $0.key < $1.key }) {
        let (diffs, same) = try Self.compare(try ImagePreprocessor(budget: Int(budget)!).preprocess(cg), ref,
                                             samples: fx.samples)
        if isPNG { png += diffs; pngCount += 1; #expect(same, "\(img.id)@\(budget) is not bit-exact") } else { jpeg += diffs }
        if same { exact += 1 }
      }
    }
    print("pixel parity vs s0: PNG p99 \(percentile(png, 0.99)) (\(pngCount) cases); JPEG (decode-inclusive) p99 "
          + "\(percentile(jpeg, 0.99)), p99.9 \(percentile(jpeg, 0.999)), max \(jpeg.max() ?? 0); "
          + "bit-exact \(exact)/\(fx.images.count * 2)")
    #expect(pngCount >= 4 && percentile(png, 0.99) <= Self.gate)
    #expect(percentile(jpeg, 0.99) <= 3.0 / 255)
  }

  @Test func multiArrayIsChannelsFirstFloat32InUnitRange() throws {
    let out = try ImagePreprocessor(budget: 70).preprocess(try gradient(width: 200, height: 200))
    #expect(out.bucket.name == .square)
    let m = try out.multiArray()
    #expect(m.shape == [1, 3, 384, 384] && m.dataType == .float32)
    let p = m.dataPointer.assumingMemoryBound(to: Float.self)
    for i in [0, 1000, 384 * 384 + 5, 2 * 384 * 384 + 77] {
      #expect(p[i] == Float(out.planar[i]) * Float(1.0 / 255.0))  // u8.astype(f32) * f32(1/255), as the reference
    }
    #expect(p[2 * 384 * 384 + 77] == Float(7) * Float(1.0 / 255.0))  // blue plane is constant 7
  }

  @Test func orientationIsAppliedBeforeChoosingTheBucket() throws {
    let wide = try gradient(width: 240, height: 100)  // r = 2.4: wide when upright
    #expect(try ImagePreprocessor(budget: 70).preprocess(wide).bucket.name == .wide)
    for o in [CGImagePropertyOrientation.right, .left, .rightMirrored, .leftMirrored] {
      #expect(try ImagePreprocessor(budget: 70).preprocess(wide, orientation: o).bucket.name == .tall)
    }
    // .down rotates by 180 degrees: the top-left pixel comes from the bottom-right one.
    let up = try ImagePreprocessor(budget: 70).preprocess(try gradient(width: 240, height: 100))
    let down = try ImagePreprocessor(budget: 70).preprocess(try gradient(width: 240, height: 100), orientation: .down)
    let (h, w) = (up.bucket.height, up.bucket.width)
    #expect(down.planar[0] == up.planar[h * w - 1])
  }

  /// Our orientation transforms agree with Core Image's `oriented(_:)` for all eight EXIF orientations.
  @Test(arguments: [CGImagePropertyOrientation.up, .upMirrored, .down, .downMirrored, .left, .leftMirrored, .right,
                    .rightMirrored])
  func orientationMatchesCoreImage(o: CGImagePropertyOrientation) throws {
    let img = try gradient(width: 5, height: 3)
    let ours = try ImagePreprocessor.drawUpright(img, orientation: o)
    let ci = CIImage(cgImage: img).oriented(o)
    let ctx = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    var theirs = [UInt8](repeating: 0, count: ours.width * ours.height * 4)
    ctx.render(ci, toBitmap: &theirs, rowBytes: ours.width * 4, bounds: ci.extent, format: .RGBA8, colorSpace: nil)
    #expect((Int(ci.extent.width), Int(ci.extent.height)) == (ours.width, ours.height))
    let rgb = { (b: [UInt8]) in b.enumerated().filter { $0.offset % 4 != 3 }.map(\.element) }
    #expect(rgb(ours.rgbx) == rgb(theirs))
  }

  @Test func grayImagesBecomeEqualRGB() throws {
    var bytes = [UInt8](repeating: 0, count: 64 * 64)
    for i in bytes.indices { bytes[i] = UInt8(i % 251) }
    let ctx = try #require(CGContext(data: &bytes, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 64,
                                     space: CGColorSpace(name: CGColorSpace.linearGray)!,
                                     bitmapInfo: CGImageAlphaInfo.none.rawValue))
    let out = try ImagePreprocessor(budget: 70).preprocess(try #require(ctx.makeImage()))
    let n = out.bucket.height * out.bucket.width
    #expect(out.planar[0..<n] == out.planar[n..<2 * n] && out.planar[n..<2 * n] == out.planar[2 * n..<3 * n])
  }

  @Test func unsupportedBudgetThrows() {
    #expect(throws: BucketError.unsupportedBudget(100)) { _ = try ImagePreprocessor(budget: 100) }
  }
}
