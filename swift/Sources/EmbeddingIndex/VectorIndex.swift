import Accelerate
import Foundation

public enum VectorIndexError: Error, Equatable {
  case revisionMismatch(expected: String, found: String)
  case dimensionMismatch(Int, Int)
  case corrupt
}

/// A flat, exact nearest-neighbour index over unit vectors (score = dot product).
///
/// Vectors are stored as FP16 (the on-disk format) with a Float32 shadow that
/// is kept in sync on every mutation, so `search` is one matrix-vector product
/// plus a size-k heap. A value type: copies are independent (copy-on-write).
public struct VectorIndex: Sendable {
  public let dimensions: Int
  public let modelRevision: String

  private let url: URL
  private var half: [Float16] = []        // slotCount * dimensions
  private var shadow: [Float] = []        // same values as `half`, widened
  private var slotIDs: [String?] = []     // nil = free slot
  private var slot: [String: Int] = [:]
  private var freeSlots: [Int] = []

  public var count: Int { slot.count }

  /// Opens the index at `url`, or starts an empty one if no file exists yet.
  /// Throws `.revisionMismatch` if the file was written for another model revision.
  public init(url: URL, dimensions: Int, modelRevision: String) throws {
    precondition(dimensions > 0, "dimensions must be positive")
    self.url = url
    self.dimensions = dimensions
    self.modelRevision = modelRevision
    guard FileManager.default.fileExists(atPath: url.path) else { return }

    let file = try IndexFile.read(url)
    guard file.revision == modelRevision else {
      throw VectorIndexError.revisionMismatch(expected: modelRevision, found: file.revision)
    }
    guard file.dimensions == dimensions else {
      throw VectorIndexError.dimensionMismatch(dimensions, file.dimensions)
    }
    half = file.vectors
    shadow = Self.widen(file.vectors)
    slotIDs = file.ids
    for (i, id) in file.ids.enumerated() { slot[id] = i }
  }

  public mutating func upsert(id: String, vector: [Float]) throws {
    guard vector.count == dimensions else { throw VectorIndexError.dimensionMismatch(dimensions, vector.count) }
    let s: Int
    if let existing = slot[id] {
      s = existing
    } else if let free = freeSlots.popLast() {
      s = free
    } else {
      s = slotIDs.count
      slotIDs.append(nil)
      half.append(contentsOf: repeatElement(0, count: dimensions))
      shadow.append(contentsOf: repeatElement(0, count: dimensions))
    }
    let base = s * dimensions
    for i in 0..<dimensions {
      let h = Float16(vector[i])
      half[base + i] = h
      shadow[base + i] = Float(h)
    }
    slotIDs[s] = id
    slot[id] = s
  }

  /// Removes `id`; removing an unknown id is a no-op.
  public mutating func remove(id: String) throws {
    guard let s = slot.removeValue(forKey: id) else { return }
    let base = s * dimensions
    for i in 0..<dimensions { half[base + i] = 0; shadow[base + i] = 0 }
    slotIDs[s] = nil
    freeSlots.append(s)
  }

  public func vector(for id: String) -> [Float]? {
    guard let s = slot[id] else { return nil }
    return Array(shadow[(s * dimensions)..<((s + 1) * dimensions)])
  }

  /// Top-`k` by dot product, best first. Returns `[]` if `query` has the wrong length.
  public func search(_ query: [Float], k: Int, filter: ((String) -> Bool)? = nil) -> [(id: String, score: Float)] {
    let slots = slotIDs.count
    guard k > 0, slots > 0, query.count == dimensions else { return [] }

    var scores = [Float](repeating: 0, count: slots)
    shadow.withUnsafeBufferPointer { m in
      query.withUnsafeBufferPointer { q in
        scores.withUnsafeMutableBufferPointer { out in
          // (slots x dims) * (dims x 1) -> (slots x 1)
          vDSP_mmul(m.baseAddress!, 1, q.baseAddress!, 1, out.baseAddress!, 1,
                    vDSP_Length(slots), 1, vDSP_Length(dimensions))
        }
      }
    }

    var heap = TopK(capacity: k)
    for s in 0..<slots {
      guard let id = slotIDs[s] else { continue }
      let score = scores[s]
      guard heap.wouldAccept(score: score, slot: s) else { continue }
      if let filter, !filter(id) { continue }
      heap.insert(score: score, slot: s)
    }
    return heap.sortedBestFirst().map { (id: slotIDs[$0.slot]!, score: $0.score) }
  }

  /// Writes the index atomically. Free slots are compacted away.
  public func save() throws {
    var vectors: [Float16] = []
    var ids: [String] = []
    vectors.reserveCapacity(count * dimensions)
    ids.reserveCapacity(count)
    for (s, id) in slotIDs.enumerated() {
      guard let id else { continue }
      ids.append(id)
      vectors.append(contentsOf: half[(s * dimensions)..<((s + 1) * dimensions)])
    }
    try IndexFile.write(.init(dimensions: dimensions, revision: modelRevision, vectors: vectors, ids: ids), to: url)
  }

  private static func widen(_ h: [Float16]) -> [Float] {
    guard !h.isEmpty else { return [] }
    var src = h
    var out = [Float](repeating: 0, count: h.count)
    src.withUnsafeMutableBytes { s in
      out.withUnsafeMutableBytes { d in
        var sb = vImage_Buffer(data: s.baseAddress, height: 1, width: vImagePixelCount(h.count), rowBytes: h.count * 2)
        var db = vImage_Buffer(data: d.baseAddress, height: 1, width: vImagePixelCount(h.count), rowBytes: h.count * 4)
        vImageConvert_Planar16FtoPlanarF(&sb, &db, vImage_Flags(kvImageNoFlags))
      }
    }
    return out
  }
}

/// Fixed-capacity min-heap keeping the k best (score, slot) pairs.
/// Ties on score prefer the lower slot, so results are deterministic.
private struct TopK {
  struct Entry { var score: Float; var slot: Int }
  let capacity: Int
  var items: [Entry] = []

  init(capacity: Int) { self.capacity = capacity; items.reserveCapacity(capacity) }

  /// true if `a` ranks strictly below `b`.
  private static func worse(_ a: Entry, _ b: Entry) -> Bool {
    a.score < b.score || (a.score == b.score && a.slot > b.slot)
  }

  func wouldAccept(score: Float, slot: Int) -> Bool {
    items.count < capacity || Self.worse(items[0], Entry(score: score, slot: slot))
  }

  mutating func insert(score: Float, slot: Int) {
    let e = Entry(score: score, slot: slot)
    if items.count < capacity {
      items.append(e)
      var i = items.count - 1
      while i > 0 {
        let p = (i - 1) / 2
        guard Self.worse(items[i], items[p]) else { break }
        items.swapAt(i, p); i = p
      }
    } else {
      items[0] = e
      var i = 0
      while true {
        let l = 2 * i + 1, r = l + 1
        var m = i
        if l < items.count, Self.worse(items[l], items[m]) { m = l }
        if r < items.count, Self.worse(items[r], items[m]) { m = r }
        if m == i { break }
        items.swapAt(i, m); i = m
      }
    }
  }

  func sortedBestFirst() -> [Entry] {
    items.sorted { Self.worse($1, $0) }
  }
}
