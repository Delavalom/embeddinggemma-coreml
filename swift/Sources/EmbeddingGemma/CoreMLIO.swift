import Accelerate
import CoreML

/// Building inputs and reading the `embedding` output of the Core ML functions (spec section 7), whatever
/// element types the converted model ended up with. FP16 goes through vImage on `UInt16` storage rather than
/// Swift's `Float16`, which is unavailable on Intel Macs.
enum CoreMLIO {
  static let embeddingName = "embedding"
  static let embeddingSize = 768

  /// The declared element type of an input, defaulting to `fallback` if the model does not say.
  static func dataType(of input: String, in model: MLModel, fallback: MLMultiArrayDataType) throws -> MLMultiArrayDataType {
    guard let d = model.modelDescription.inputDescriptionsByName[input] else {
      throw EmbeddingGemmaError.contractViolation("no input `\(input)`")
    }
    return d.multiArrayConstraint?.dataType ?? fallback
  }

  /// `[1, values.count]` token ids or mask. FP16 is refused: it cannot hold token ids above 2048 exactly.
  static func integers(_ values: [Int32], as type: MLMultiArrayDataType, input: String) throws -> MLMultiArray {
    let m = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: type)
    switch type {
    case .int32: m.dataPointer.assumingMemoryBound(to: Int32.self).update(from: values, count: values.count)
    case .float32:
      var view = UnsafeMutableBufferPointer(start: m.dataPointer.assumingMemoryBound(to: Float.self), count: values.count)
      vDSP.convertElements(of: values, to: &view)
    case .double:
      let p = m.dataPointer.assumingMemoryBound(to: Double.self)
      for (i, v) in values.enumerated() { p[i] = Double(v) }
    default: throw EmbeddingGemmaError.contractViolation("unsupported type \(type.rawValue) for `\(input)`")
    }
    return m
  }

  /// Float32 -> FP16 bits.
  static func toFloat16(_ src: UnsafePointer<Float>, _ dst: UnsafeMutablePointer<UInt16>, count: Int) {
    var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count),
                          rowBytes: count * 4)
    var d = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
    vImageConvert_PlanarFtoPlanar16F(&s, &d, vImage_Flags(kvImageNoFlags))
  }

  /// FP16 bits -> Float32.
  static func fromFloat16(_ src: [UInt16]) -> [Float] {
    let n = src.count
    var src = src, out = [Float](repeating: 0, count: n)
    src.withUnsafeMutableBytes { s in
      out.withUnsafeMutableBytes { o in
        var sb = vImage_Buffer(data: s.baseAddress, height: 1, width: vImagePixelCount(n), rowBytes: s.count)
        var ob = vImage_Buffer(data: o.baseAddress, height: 1, width: vImagePixelCount(n), rowBytes: o.count)
        vImageConvert_Planar16FtoPlanarF(&sb, &ob, vImage_Flags(kvImageNoFlags))
      }
    }
    return out
  }

  /// The 768 values of `embedding`: any shape with one axis of 768 and all others 1 (`[1, 768]`, `[1, 768, 1, 1]`),
  /// read with that axis's stride.
  static func embedding(from out: any MLFeatureProvider) throws -> [Float] {
    guard let m = out.featureValue(for: embeddingName)?.multiArrayValue else {
      throw EmbeddingGemmaError.contractViolation("no `\(embeddingName)` output")
    }
    return try embedding(from: m)
  }

  static func embedding(from m: MLMultiArray) throws -> [Float] {
    let shape = m.shape.map(\.intValue)
    guard let axis = shape.firstIndex(of: embeddingSize), shape.enumerated().allSatisfy({ $0.offset == axis || $0.element == 1 })
    else { throw EmbeddingGemmaError.contractViolation("`\(embeddingName)` has shape \(shape), expected [1, 768]") }
    let stride = m.strides[axis].intValue, n = embeddingSize
    return try m.withUnsafeBytes { raw -> [Float] in
      switch m.dataType {
      case .float32: let p = raw.bindMemory(to: Float.self); return (0..<n).map { p[$0 * stride] }
      case .double: let p = raw.bindMemory(to: Double.self); return (0..<n).map { Float(p[$0 * stride]) }
      case .float16: let p = raw.bindMemory(to: UInt16.self); return fromFloat16((0..<n).map { p[$0 * stride] })
      default: throw EmbeddingGemmaError.contractViolation("unsupported output type \(m.dataType.rawValue)")
      }
    }
  }

  /// The first `dimensions` values, L2-normalized (Matryoshka truncation). Throws on NaN or infinity.
  static func truncateAndNormalize(_ v: [Float], dimensions: Int, function: String) throws -> [Float] {
    guard v.allSatisfy(\.isFinite) else { throw EmbeddingGemmaError.nonFiniteOutput(function: function) }
    var head = Array(v.prefix(dimensions))
    var sumsq: Float = 0
    vDSP_svesq(head, 1, &sumsq, vDSP_Length(head.count))
    guard sumsq > 0, sumsq.isFinite else { return head }
    var inv = 1 / sumsq.squareRoot()
    vDSP_vsmul(head, 1, &inv, &head, 1, vDSP_Length(head.count))
    return head
  }
}
