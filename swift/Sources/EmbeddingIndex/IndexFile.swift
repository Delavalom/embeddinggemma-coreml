import Foundation

/// On-disk layout (all integers little-endian):
///
///     "EGIX" | version u32 | dims u32 | count u32 | revisionLength u32 | revision utf8
///     count * dims FP16 values | ids JSON (array of `count` strings, to end of file)
enum IndexFile {
  static let magic = Data("EGIX".utf8)
  static let version: UInt32 = 1

  struct Contents {
    var dimensions: Int
    var revision: String
    var vectors: [Float16]   // count * dimensions
    var ids: [String]
  }

  static func encode(_ c: Contents) throws -> Data {
    precondition(c.vectors.count == c.ids.count * c.dimensions, "vector block does not match ids")
    let revision = Data(c.revision.utf8)
    var out = Data()
    out.reserveCapacity(20 + revision.count + c.vectors.count * 2 + c.ids.count * 16)
    out.append(magic)
    for v in [version, UInt32(c.dimensions), UInt32(c.ids.count), UInt32(revision.count)] {
      withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) }
    }
    out.append(revision)
    c.vectors.withUnsafeBytes { out.append(contentsOf: $0) }
    out.append(try JSONEncoder().encode(c.ids))
    return out
  }

  static func decode(_ data: Data) throws -> Contents {
    var offset = 0
    func take(_ n: Int) throws -> Range<Int> {
      guard n >= 0, n <= data.count - offset else { throw VectorIndexError.corrupt }
      defer { offset += n }
      return offset..<(offset + n)
    }
    func u32() throws -> Int {
      let r = try take(4)
      return Int(data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: r.lowerBound, as: UInt32.self)) })
    }
    let base = data.startIndex
    guard data.count >= 4, data[base..<(base + 4)] == magic else { throw VectorIndexError.corrupt }
    offset = 4
    guard try u32() == Int(version) else { throw VectorIndexError.corrupt }
    let dims = try u32()
    let count = try u32()
    let revLen = try u32()
    let revRange = try take(revLen)
    guard dims > 0, let revision = String(data: data[(base + revRange.lowerBound)..<(base + revRange.upperBound)], encoding: .utf8) else {
      throw VectorIndexError.corrupt
    }
    let (n, overflow) = count.multipliedReportingOverflow(by: dims)
    let (bytes, overflow2) = n.multipliedReportingOverflow(by: 2)
    guard !overflow, !overflow2 else { throw VectorIndexError.corrupt }
    let block = try take(bytes)
    var vectors = [Float16](repeating: 0, count: n)
    vectors.withUnsafeMutableBytes { dst in
      data.withUnsafeBytes { src in
        dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[block]))
      }
    }
    guard let ids = try? JSONDecoder().decode([String].self, from: data[(base + offset)...]),
          ids.count == count, Set(ids).count == count
    else { throw VectorIndexError.corrupt }
    return Contents(dimensions: dims, revision: revision, vectors: vectors, ids: ids)
  }

  static func read(_ url: URL) throws -> Contents {
    try decode(Data(contentsOf: url, options: .alwaysMapped))
  }

  static func write(_ c: Contents, to url: URL) throws {
    try encode(c).write(to: url, options: .atomic)
  }
}
