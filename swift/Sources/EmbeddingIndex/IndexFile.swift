import Accelerate
import Foundation

/// On-disk layout, version 2. All integers are little-endian; vectors are
/// little-endian IEEE 754 half precision (FP16).
///
///     "EGIX" | version u32 | dims u32 | count u32 | revisionLength u32 | revision utf8
///     zero padding up to the next multiple of 16 bytes
///     count * dims FP16 values        <- starts 16-byte aligned (page-aligned mapping + aligned offset)
///     ids JSON (array of `count` unique strings, to end of file)
enum IndexFile {
  static let magic = Data("EGIX".utf8)
  static let version: UInt32 = 2
  static let maxDimensions = 65_535
  private static let fixedHeaderSize = 20

  /// Offset of the FP16 block for a given revision length.
  static func blockOffset(revisionLength: Int) -> Int {
    (fixedHeaderSize + revisionLength + 15) & ~15
  }

  struct Loaded {
    var vectors: [Float]   // count * dimensions, widened from FP16
    var ids: [String]
  }

  // MARK: Reading

  /// Reads the file, checking revision and dimensions *before* converting any vectors.
  static func read(_ url: URL, expectedDimensions: Int, expectedRevision: String) throws -> Loaded {
    let data = try Data(contentsOf: url, options: .alwaysMapped)
    return try data.withUnsafeBytes { raw in
      guard raw.count >= 8, raw.baseAddress != nil, Data(raw[0..<4]) == magic else { throw VectorIndexError.corrupt }
      let v = u32(raw, at: 4)
      guard v == version else { throw VectorIndexError.unsupportedVersion(v) }
      guard raw.count >= fixedHeaderSize else { throw VectorIndexError.corrupt }
      let dims = Int(u32(raw, at: 8))
      let count = Int(u32(raw, at: 12))
      let revLen = Int(u32(raw, at: 16))
      guard (1...maxDimensions).contains(dims), revLen <= raw.count - fixedHeaderSize else { throw VectorIndexError.corrupt }
      let offset = blockOffset(revisionLength: revLen)
      guard offset <= raw.count,
            raw[(fixedHeaderSize + revLen)..<offset].allSatisfy({ $0 == 0 }),
            let revision = String(data: Data(raw[fixedHeaderSize..<(fixedHeaderSize + revLen)]), encoding: .utf8)
      else { throw VectorIndexError.corrupt }

      guard revision == expectedRevision else {
        throw VectorIndexError.revisionMismatch(expected: expectedRevision, found: revision)
      }
      guard dims == expectedDimensions else {
        throw VectorIndexError.dimensionMismatch(expected: expectedDimensions, found: dims)
      }

      let (n, o1) = count.multipliedReportingOverflow(by: dims)
      let (bytes, o2) = n.multipliedReportingOverflow(by: 2)
      guard !o1, !o2, bytes <= raw.count - offset else { throw VectorIndexError.corrupt }

      let vectors = try [Float](unsafeUninitializedCapacity: n) { buf, initialized in
        initialized = 0
        guard n > 0 else { return }
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress! + offset),
                                height: vImagePixelCount(count), width: vImagePixelCount(dims), rowBytes: dims * 2)
        var dst = vImage_Buffer(data: buf.baseAddress!, height: vImagePixelCount(count),
                                width: vImagePixelCount(dims), rowBytes: dims * 4)
        guard vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
          throw VectorIndexError.corrupt
        }
        initialized = n
      }
      let json = Data(raw[(offset + bytes)...])
      guard let ids = try? JSONDecoder().decode([String].self, from: json),
            ids.count == count, Set(ids).count == count
      else { throw VectorIndexError.corrupt }
      return Loaded(vectors: vectors, ids: ids)
    }
  }

  private static func u32(_ raw: UnsafeRawBufferPointer, at offset: Int) -> UInt32 {
    UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
  }

  // MARK: Writing

  /// Streams the live slots to a temp file next to `url` and renames it into place,
  /// so readers only ever see the old file or the complete new one.
  static func write(to url: URL, dimensions: Int, revision: String, vectors: [Float], idBySlot: [String?]) throws {
    let ids = idBySlot.compactMap { $0 }
    assert(vectors.count == idBySlot.count * dimensions, "vector buffer does not match slot table")

    let tmp = url.deletingLastPathComponent()
      .appending(path: ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
    guard FileManager.default.createFile(atPath: tmp.path, contents: nil) else {
      throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: tmp.path])
    }
    var renamed = false
    defer { if !renamed { try? FileManager.default.removeItem(at: tmp) } }

    let handle = try FileHandle(forWritingTo: tmp)
    do {
      try handle.write(contentsOf: header(dimensions: dimensions, count: ids.count, revision: revision))

      // Convert runs of consecutive live slots in bounded chunks.
      let chunkRows = max(1, min(1024, (1 << 20) / dimensions))
      var chunk = [UInt16](repeating: 0, count: chunkRows * dimensions)  // FP16 bits
      let slotCount = idBySlot.count
      try vectors.withUnsafeBufferPointer { src in
        var s = 0
        while s < slotCount {
          guard idBySlot[s] != nil else { s += 1; continue }
          var end = s
          while end < slotCount, idBySlot[end] != nil { end += 1 }
          var r = s
          while r < end {
            let rows = min(chunkRows, end - r)
            try chunk.withUnsafeMutableBytes { out in
              var sb = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress! + r * dimensions),
                                     height: vImagePixelCount(rows), width: vImagePixelCount(dimensions), rowBytes: dimensions * 4)
              var db = vImage_Buffer(data: out.baseAddress!, height: vImagePixelCount(rows),
                                     width: vImagePixelCount(dimensions), rowBytes: dimensions * 2)
              guard vImageConvert_PlanarFtoPlanar16F(&sb, &db, vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
                throw CocoaError(.fileWriteUnknown)
              }
              try handle.write(contentsOf: Data(bytesNoCopy: out.baseAddress!, count: rows * dimensions * 2, deallocator: .none))
            }
            r += rows
          }
          s = end
        }
      }
      try handle.write(contentsOf: JSONEncoder().encode(ids))
      try handle.synchronize()
      try handle.close()
    } catch {
      try? handle.close()
      throw error
    }

    guard rename(tmp.path, url.path) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
    renamed = true
  }

  private static func header(dimensions: Int, count: Int, revision: String) -> Data {
    let rev = Data(revision.utf8)
    var out = Data()
    out.append(magic)
    for v in [version, UInt32(dimensions), UInt32(count), UInt32(rev.count)] {
      withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) }
    }
    out.append(rev)
    out.append(Data(count: blockOffset(revisionLength: rev.count) - out.count))
    return out
  }
}
