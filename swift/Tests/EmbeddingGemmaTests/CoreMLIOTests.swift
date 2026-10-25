import CoreML
import Testing
@testable import EmbeddingGemma

@Suite struct CoreMLIOTests {
  /// [1, 768] with the 768 axis at stride 2 (as padded FP16 buffers can be): read with that stride.
  @Test func stridedOutputUsesTheStrideOfThe768Axis() throws {
    let storage = UnsafeMutablePointer<Float>.allocate(capacity: 768 * 2)
    for i in 0..<(768 * 2) { storage[i] = i % 2 == 0 ? Float(i / 2) : -1 }
    let m = try MLMultiArray(dataPointer: storage, shape: [1, 768], dataType: .float32, strides: [1536, 2]) { $0.deallocate() }
    let v = try CoreMLIO.embedding(from: m)
    #expect(v == (0..<768).map(Float.init))
  }

  @Test(arguments: [[1, 768, 1, 1], [768], [1, 1, 768]])
  func singletonAxesAreAccepted(shape: [Int]) throws {
    let m = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    for i in 0..<768 { m[i] = NSNumber(value: Float(i)) }
    #expect(try CoreMLIO.embedding(from: m) == (0..<768).map(Float.init))
  }

  @Test(arguments: [[1, 767], [2, 768], [1, 768, 2]])
  func otherShapesAreRejected(shape: [Int]) throws {
    let m = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    #expect(throws: EmbeddingGemmaError.self) { try CoreMLIO.embedding(from: m) }
  }

  @Test func fp16RoundTripsThroughVImage() throws {
    let values: [Float] = [0, 1, -2.5, 0.000_061_035_156, 65504, 1.0 / 3]
    var bits = [UInt16](repeating: 0, count: values.count)
    values.withUnsafeBufferPointer { v in bits.withUnsafeMutableBufferPointer { CoreMLIO.toFloat16(v.baseAddress!, $0.baseAddress!, count: v.count) } }
    #expect(bits[1] == 0x3C00 && bits[4] == 0x7BFF)
    let back = CoreMLIO.fromFloat16(bits)
    #expect(back[0...4] == values[0...4] && abs(back[5] - 1.0 / 3) < 1e-3)
  }

  @Test func idsRefuseFP16() {
    #expect(throws: EmbeddingGemmaError.self) { try CoreMLIO.integers([2, 300_000], as: .float16, input: "input_ids") }
    #expect(throws: Never.self) { try CoreMLIO.integers([2, 300_000], as: .int32, input: "input_ids") }
  }

  @Test func nonFiniteValuesThrow() {
    for bad in [Float.nan, .infinity, -.infinity] {
      var v = [Float](repeating: 0.1, count: 768)
      v[700] = bad  // beyond the truncation: still rejected, the whole output is suspect
      #expect(throws: EmbeddingGemmaError.self) { try CoreMLIO.truncateAndNormalize(v, dimensions: 256, function: "f") }
    }
  }

  @Test func multiArrayRejectsUnsupportedTypes() throws {
    let pre = PreprocessedImage(bucket: Bucket.all[2], planar: [UInt8](repeating: 255, count: 3 * 384 * 384))
    #expect(throws: EmbeddingGemmaError.self) { try pre.multiArray(dataType: .int32) }
    let h = try pre.multiArray(dataType: .float16)
    #expect(h.dataPointer.assumingMemoryBound(to: UInt16.self)[0] == 0x3C00)  // 1.0
    let f = try pre.multiArray(dataType: .float32)
    #expect(f.dataPointer.assumingMemoryBound(to: Float.self)[5] == Float(255) * Float(1.0 / 255.0))
  }
}
