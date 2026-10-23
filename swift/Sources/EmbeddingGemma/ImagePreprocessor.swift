import Accelerate
import CoreGraphics
import CoreML
import ImageIO

/// An image resized to its bucket, as the image functions take it.
package struct PreprocessedImage: Sendable {
  package let bucket: Bucket
  /// uint8 RGB planes, channels first: `[3, H, W]` flattened.
  package let planar: [UInt8]

  /// `pixels` input `[1, 3, H, W]` in 0-1: `Float(u8) * Float(1/255)`, exactly as the reference processor.
  /// FP32 or FP16 (converted with vImage); anything else throws.
  package func multiArray(dataType: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
    guard dataType == .float32 || dataType == .float16 else {
      throw EmbeddingGemmaError.contractViolation("unsupported type \(dataType.rawValue) for `pixels`")
    }
    let m = try MLMultiArray(shape: [1, 3, bucket.height, bucket.width].map { NSNumber(value: $0) }, dataType: dataType)
    var values = [Float](repeating: 0, count: planar.count)
    vDSP.convertElements(of: planar, to: &values)
    vDSP.multiply(Float(1.0 / 255.0), values, result: &values)  // one rounding, as u8.astype(f32) * f32(1/255)
    if dataType == .float32 {
      m.dataPointer.assumingMemoryBound(to: Float.self).update(from: values, count: values.count)
    } else {
      values.withUnsafeBufferPointer {
        CoreMLIO.toFloat16($0.baseAddress!, m.dataPointer.assumingMemoryBound(to: UInt16.self), count: values.count)
      }
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
/// - **Orientation and colour:** the image is drawn once into an 8-bit sRGB context, which applies the EXIF
///   orientation and converts the colour space (Display P3, gray, ...) to sRGB. The reference ignores ICC profiles; converting is the semantically right input for the model (ADR-0005).
///   The bucket is chosen from the upright size.
/// - **Why draw, not read the bytes:** ImageIO decodes a JPEG differently depending on whether it is first drawn
///   or first read through its data provider (up to 73/255 apart), and caches whichever came first. The drawn
///   decode is the one closer to PIL (p99 2/255 vs 4/255 on the fixtures), so opaque images are always drawn; for
///   bitmaps already in memory drawing sRGB into sRGB is an exact copy.
/// - **Alpha:** PIL's `convert("RGB")` drops alpha and keeps the stored colours. Upright 8-bit sRGB images with
///   non-premultiplied alpha (PNGs; a lossless decode, so no decoder question) are read directly, which does exactly
///   that. Others are drawn premultiplied and unpremultiplied: exact where alpha is 255, within 8-bit rounding
///   where it is partial, black where it is 0 (ADR-0005).
/// - **Memory:** the drawn copy (or the direct bytes) is released as soon as the horizontal pass is done.
package struct ImagePreprocessor: Sendable {
  package let budget: Int

  package init(budget: Int) throws {
    guard Bucket.budgets.contains(budget) else { throw BucketError.unsupportedBudget(budget) }
    self.budget = budget
  }

  package func preprocess(_ image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> PreprocessedImage {
    try preprocess(image, orientation: orientation, read: .alphaOnly)
  }

  /// `read` chooses when the bytes are read directly instead of drawn (tests force each path).
  func preprocess(_ image: CGImage, orientation: CGImagePropertyOrientation, read: RowProvider.DirectRead) throws
    -> PreprocessedImage {
    let rows = try RowProvider.make(image, orientation: orientation, read: read)
    let bucket = try Bucket.choose(height: rows.height, width: rows.width, budget: budget)
    let planar = Resampler.resize(width: rows.width, height: rows.height, toWidth: bucket.width,
                                  toHeight: bucket.height, rows: rows)
    return PreprocessedImage(bucket: bucket, planar: planar)
  }

  static func uprightSize(_ image: CGImage, _ o: CGImagePropertyOrientation) -> (Int, Int) {
    [.left, .right, .leftMirrored, .rightMirrored].contains(o) ? (image.height, image.width) : (image.width, image.height)
  }

  /// RGBX bytes of the whole upright image in sRGB (drawing path), and its upright size.
  static func drawUpright(_ image: CGImage, orientation: CGImagePropertyOrientation) throws
    -> (rgbx: [UInt8], width: Int, height: Int) {
    let rows = try RowProvider.make(image, orientation: orientation, read: .never)
    var bytes = [UInt8](repeating: 0, count: rows.width * rows.height * 4)
    bytes.withUnsafeMutableBufferPointer { rows.fill(y0: 0, rows: rows.height, into: $0.baseAddress!) }
    return (bytes, rows.width, rows.height)
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

/// `CGImage` is immutable and safe to read from several threads; the box just says so to the compiler.
struct SendableImage: @unchecked Sendable { let image: CGImage }

/// Upright sRGB rows as RGBX, for the resampler's horizontal pass (called concurrently for disjoint rows), then
/// released. Either the image's own bytes (8-bit sRGB, upright, alpha not premultiplied) or one drawn copy.
final class RowProvider: @unchecked Sendable {
  /// `.alphaOnly` (production): direct read only for non-premultiplied alpha. `.whenPossible`: any direct layout.
  enum DirectRead { case never, alphaOnly, whenPossible }

  let width: Int, height: Int
  private var data: CFData?  // the direct bytes or the drawn copy; nil after release()
  private let rowBytes: Int, pixelBytes: Int
  private let offsets: (r: Int, g: Int, b: Int)

  private init(width: Int, height: Int, data: CFData, rowBytes: Int, pixelBytes: Int, offsets: (Int, Int, Int)) {
    (self.width, self.height, self.data, self.rowBytes, self.pixelBytes, self.offsets) =
      (width, height, data, rowBytes, pixelBytes, offsets)
  }

  static func make(_ image: CGImage, orientation: CGImagePropertyOrientation, read: DirectRead) throws -> RowProvider {
    let (w, h) = ImagePreprocessor.uprightSize(image, orientation)
    guard w > 0, h > 0 else { throw EmbeddingGemmaError.invalidImage }
    let allowed = switch read {
    case .never: false
    case .alphaOnly: [.last, .first].contains(image.alphaInfo)
    case .whenPossible: true
    }
    if allowed, orientation == .up, let layout = directLayout(image), let data = image.dataProvider?.data,
       CFDataGetLength(data) >= image.bytesPerRow * (h - 1) + w * layout.pixelBytes {
      return RowProvider(width: w, height: h, data: data, rowBytes: image.bytesPerRow, pixelBytes: layout.pixelBytes,
                         offsets: layout.offsets)
    }
    return RowProvider(width: w, height: h, data: try draw(image, orientation: orientation, width: w, height: h),
                       rowBytes: w * 4, pixelBytes: 4, offsets: (0, 1, 2))
  }

  /// Byte offsets of R, G, B for 8-bit sRGB layouts whose stored colours are what PIL would read.
  static func directLayout(_ image: CGImage) -> (pixelBytes: Int, offsets: (Int, Int, Int))? {
    guard image.bitsPerComponent == 8, image.colorSpace?.name == CGColorSpace.sRGB,
          image.bitmapInfo.intersection(.floatComponents).isEmpty else { return nil }
    let alpha = image.alphaInfo, order = image.bitmapInfo.intersection(.byteOrderMask)
    if image.bitsPerPixel == 24, alpha == .none, order == [] { return (3, (0, 1, 2)) }
    guard image.bitsPerPixel == 32 else { return nil }
    let alphaFirst: Bool
    switch alpha {
    case .noneSkipLast, .last: alphaFirst = false
    case .noneSkipFirst, .first: alphaFirst = true
    default: return nil  // premultiplied or no alpha slot: draw
    }
    let logical = alphaFirst ? (1, 2, 3) : (0, 1, 2)  // in big-endian (memory) order
    switch order {
    case [], .byteOrder32Big: return (4, logical)
    case .byteOrder32Little: return (4, (3 - logical.0, 3 - logical.1, 3 - logical.2))
    default: return nil
    }
  }

  private static func draw(_ image: CGImage, orientation: CGImagePropertyOrientation, width w: Int, height h: Int) throws -> CFData {
    let hasAlpha = ![.none, .noneSkipFirst, .noneSkipLast].contains(image.alphaInfo)
    guard let out = CFDataCreateMutable(nil, w * h * 4), let srgb = CGColorSpace(name: CGColorSpace.sRGB) else {
      throw EmbeddingGemmaError.invalidImage
    }
    CFDataSetLength(out, w * h * 4)
    let p = CFDataGetMutableBytePtr(out)!
    let info = hasAlpha ? CGImageAlphaInfo.premultipliedLast : .noneSkipLast
    guard let ctx = CGContext(data: p, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: srgb,
                              bitmapInfo: info.rawValue) else { throw EmbeddingGemmaError.invalidImage }
    ctx.interpolationQuality = .none
    ctx.setBlendMode(.copy)
    ctx.concatenate(ImagePreprocessor.transform(orientation, width: CGFloat(image.width), height: CGFloat(image.height)))
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    if hasAlpha {
      var buf = vImage_Buffer(data: p, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
      vImageUnpremultiplyData_RGBA8888(&buf, &buf, vImage_Flags(kvImageNoFlags))
    }
    return out
  }

  /// Rows `y0..<y0+rows` as RGBX (`width * 4` bytes per row).
  func fill(y0: Int, rows: Int, into dst: UnsafeMutablePointer<UInt8>) {
    guard let data, let base = CFDataGetBytePtr(data) else { return }
    if pixelBytes == 4, offsets == (0, 1, 2) {
      for y in 0..<rows { (dst + y * width * 4).update(from: base + (y0 + y) * rowBytes, count: width * 4) }
      return
    }
    for y in 0..<rows {
      let src = base + (y0 + y) * rowBytes, d = dst + y * width * 4
      for x in 0..<width {
        let s = src + x * pixelBytes
        (d[x * 4], d[x * 4 + 1], d[x * 4 + 2]) = (s[offsets.r], s[offsets.g], s[offsets.b])
      }
    }
  }

  /// Drops the pixels (the image's bytes or the drawn copy) once the horizontal pass no longer needs them.
  func release() { data = nil }
}
