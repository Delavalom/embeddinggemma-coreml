import Accelerate
import CoreGraphics
import CoreML
import ImageIO

/// An image resized to its bucket, as the image functions take it.
public struct PreprocessedImage: Sendable {
  public let bucket: Bucket
  /// uint8 RGB planes, channels first: `[3, H, W]` flattened.
  public let planar: [UInt8]

  /// `pixels` input `[1, 3, H, W]` in 0-1: `Float(u8) * Float(1/255)`, exactly as the reference processor.
  public func multiArray(dataType: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
    let m = try MLMultiArray(shape: [1, 3, bucket.height, bucket.width].map { NSNumber(value: $0) }, dataType: dataType)
    let scale = Float(1.0 / 255.0)
    switch dataType {
    case .float16:
      let p = m.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in planar.indices { p[i] = Float16(Float(planar[i]) * scale) }
    default:
      let p = m.dataPointer.assumingMemoryBound(to: Float.self)
      for i in planar.indices { p[i] = Float(planar[i]) * scale }
    }
    return m
  }

  /// uint8 HxWx3 (the layout s0 stores), for checksums.
  func interleavedRGB() -> Data {
    let n = bucket.height * bucket.width
    var out = Data(count: n * 3)
    out.withUnsafeMutableBytes { (o: UnsafeMutableRawBufferPointer) in
      for i in 0..<n { for c in 0..<3 { o[i * 3 + c] = planar[c * n + i] } }
    }
    return out
  }
}

/// CGImage -> bucket-sized RGB pixels, matching the reference processor (RGB, bicubic + antialias, rescale by 1/255,
/// no normalization).
///
/// The image is first drawn at its own size into an 8-bit sRGB context, which applies the orientation and converts
/// any colour space to sRGB (the fixtures and the reference are sRGB; the reference ignores ICC profiles). The
/// bucket is chosen from the oriented size.
public struct ImagePreprocessor: Sendable {
  public let budget: Int

  public init(budget: Int) throws {
    guard Bucket.budgets.contains(budget) else { throw BucketError.unsupportedBudget(budget) }
    self.budget = budget
  }

  public func preprocess(_ image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> PreprocessedImage {
    let (rgbx, width, height) = try Self.drawUpright(image, orientation: orientation)
    let bucket = try Bucket.choose(height: height, width: width, budget: budget)
    let planar = Resampler.resizeRGBX(rgbx, width: width, height: height, toWidth: bucket.width, toHeight: bucket.height)
    return PreprocessedImage(bucket: bucket, planar: planar)
  }

  /// RGBX bytes of the upright image in sRGB, and its upright size.
  static func drawUpright(_ image: CGImage, orientation: CGImagePropertyOrientation) throws
    -> (rgbx: [UInt8], width: Int, height: Int) {
    let (w0, h0) = (image.width, image.height)
    let swaps = [.left, .right, .leftMirrored, .rightMirrored].contains(orientation)
    let (w, h) = swaps ? (h0, w0) : (w0, h0)
    guard w > 0, h > 0, let srgb = CGColorSpace(name: CGColorSpace.sRGB) else { throw EmbeddingGemmaError.invalidImage }
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    let ok = bytes.withUnsafeMutableBytes { buf -> Bool in
      guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: srgb, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
      ctx.interpolationQuality = .none
      ctx.concatenate(Self.transform(orientation, width: CGFloat(w0), height: CGFloat(h0)))
      ctx.draw(image, in: CGRect(x: 0, y: 0, width: w0, height: h0))
      return true
    }
    guard ok else { throw EmbeddingGemmaError.invalidImage }
    return (bytes, w, h)
  }

  /// Maps the stored image (w x h, Core Graphics' bottom-left origin) onto the upright canvas.
  static func transform(_ o: CGImagePropertyOrientation, width w: CGFloat, height h: CGFloat) -> CGAffineTransform {
    switch o {
    case .up: .identity
    case .upMirrored: CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)
    case .down: CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)
    case .downMirrored: CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)
    case .left: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: h, ty: 0)
    case .leftMirrored: CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: h, ty: w)
    case .right: CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: w)
    case .rightMirrored: CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
    @unknown default: .identity
    }
  }
}
