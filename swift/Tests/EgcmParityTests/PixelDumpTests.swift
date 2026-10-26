import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import egcm_parity

/// The `--pixels` dump (ADR-0008): s5 feeds these PNGs to Python Core ML and the FP reference, so they must hold the
/// bucket image's exact bytes. s5 also checks the sha256 of what it reads against parity.json.
@Suite struct PixelDumpTests {
  /// A deterministic channels-first image with varied byte values.
  static func planar(height h: Int, width w: Int) -> [UInt8] {
    (0..<(3 * h * w)).map { UInt8(truncatingIfNeeded: $0 &* 2_654_435_761 >> 7) }
  }

  @Test func interleavesChannelsFirstPlanes() {
    // 1x2 image: R plane [1, 2], G plane [3, 4], B plane [5, 6]
    #expect(interleaved(planar: [1, 2, 3, 4, 5, 6], height: 1, width: 2) == [1, 3, 5, 2, 4, 6])
  }

  @Test func pngRoundTripIsLossless() throws {
    let (h, w) = (37, 53)
    let planes = Self.planar(height: h, width: w)
    let url = FileManager.default.temporaryDirectory.appending(path: "egcm-parity-\(UUID().uuidString).png")
    defer { try? FileManager.default.removeItem(at: url) }
    try writePNG(planar: planes, height: h, width: w, to: url)

    let src = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
    let img = try #require(CGImageSourceCreateImageAtIndex(src, 0, nil))
    #expect((img.width, img.height, img.bitsPerComponent) == (w, h, 8))
    var rgbx = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = try #require(CGContext(data: &rgbx, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                     space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                     bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    ctx.setBlendMode(.copy)
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    let rgb = (0..<(w * h)).flatMap { rgbx[$0 * 4..<$0 * 4 + 3] }
    #expect(rgb == interleaved(planar: planes, height: h, width: w))
    #expect(interleavedSHA256(planar: planes, height: h, width: w).count == 64)
  }

  @Test func pixelsOptionIsParsed() throws {
    let o = try Options.parse(["--model", "/m", "--tokenizer", "/t", "--fixtures", "/f", "--out", "/o.json",
                               "--pixels", "/p"])
    #expect(o.pixels?.path == "/p")
    #expect(try Options.parse(["--model", "/m", "--tokenizer", "/t", "--fixtures", "/f", "--out", "/o"]).pixels == nil)
  }
}
