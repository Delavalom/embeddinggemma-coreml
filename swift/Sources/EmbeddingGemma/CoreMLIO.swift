import Accelerate
import CoreML

/// Building inputs and reading the `embedding` output of the Core ML functions (spec section 7), whatever
/// element types the converted model ended up with.
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

  /// `[1, values.count]` array of integers in the given element type.
  static func integers(_ values: [Int32], as type: MLMultiArrayDataType) throws -> MLMultiArray {
    let m = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: type)
    switch type {
    case .int32: m.dataPointer.assumingMemoryBound(to: Int32.self).update(from: values, count: values.count)
    case .float32: for (i, v) in values.enumerated() { m.dataPointer.assumingMemoryBound(to: Float.self)[i] = Float(v) }
    case .float16: for (i, v) in values.enumerated() { m.dataPointer.assumingMemoryBound(to: Float16.self)[i] = Float16(v) }
    case .double: for (i, v) in values.enumerated() { m.dataPointer.assumingMemoryBound(to: Double.self)[i] = Double(v) }
    default: throw EmbeddingGemmaError.contractViolation("unsupported input type \(type.rawValue)")
    }
    return m
  }

  /// The 768 values of `embedding` (`[1, 768]` or `[768]`), honouring the array's strides.
  static func embedding(from out: any MLFeatureProvider) throws -> [Float] {
    guard let m = out.featureValue(for: embeddingName)?.multiArrayValue else {
      throw EmbeddingGemmaError.contractViolation("no `\(embeddingName)` output")
    }
    guard m.count == embeddingSize, let last = m.strides.last?.intValue else {
      throw EmbeddingGemmaError.contractViolation("`\(embeddingName)` has shape \(m.shape), expected [1, 768]")
    }
    return try m.withUnsafeBytes { raw -> [Float] in
      switch m.dataType {
      case .float32: let p = raw.bindMemory(to: Float.self); return (0..<embeddingSize).map { p[$0 * last] }
      case .float16: let p = raw.bindMemory(to: Float16.self); return (0..<embeddingSize).map { Float(p[$0 * last]) }
      case .double: let p = raw.bindMemory(to: Double.self); return (0..<embeddingSize).map { Float(p[$0 * last]) }
      default: throw EmbeddingGemmaError.contractViolation("unsupported output type \(m.dataType.rawValue)")
      }
    }
  }

  /// The first `dimensions` values, L2-normalized (Matryoshka truncation).
  static func truncateAndNormalize(_ v: [Float], dimensions: Int) -> [Float] {
    var head = Array(v.prefix(dimensions))
    var sumsq: Float = 0
    vDSP_svesq(head, 1, &sumsq, vDSP_Length(head.count))
    guard sumsq > 0, sumsq.isFinite else { return head }
    var inv = 1 / sumsq.squareRoot()
    vDSP_vsmul(head, 1, &inv, &head, 1, vDSP_Length(head.count))
    return head
  }
}
