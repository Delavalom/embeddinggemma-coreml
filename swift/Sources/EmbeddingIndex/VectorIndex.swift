import Accelerate
import Foundation

public enum VectorIndexError: Error, Equatable {
  /// The file on disk was written for a different model revision.
  case revisionMismatch(expected: String, found: String)
  case dimensionMismatch(expected: Int, found: Int)
  /// `dimensions` must be in 1...65_535.
  case invalidDimensions(Int)
  case unsupportedVersion(UInt32)
  case corrupt
}

/// A flat, exact nearest-neighbour index (score = dot product).
///
/// **Ownership.** This is a reference type that is deliberately *not* `Sendable`: the
/// buffer is 50+ MB at 50k x 256, and a value type would copy it silently on the first
/// mutation of any shared copy. Own it from a single actor (the app wraps it in one)
/// and call it only from there.
///
/// **Storage.** One contiguous `[Float]` buffer. Values are rounded to FP16 precision
/// on `upsert` (`Float(Float16(x))`, done with vImage so it also builds for Intel Macs), so what `search` and `vector(for:)` see is exactly
/// what a save and reload gives back. Magnitudes above 65,504 overflow FP16: callers
/// pass unit vectors (scores are only cosine similarities if they do).
public final class VectorIndex {
  public let dimensions: Int
  public let modelRevision: String

  private let url: URL
  private var buffer: [Float] = []         // slotCount * dimensions
  private var idBySlot: [String?] = []     // nil = free slot
  private var slotByID: [String: Int] = [:]
  private var freeSlots: [Int] = []

  private var slotCount: Int { idBySlot.count }

  /// Number of live vectors.
  public var count: Int { slotByID.count }

  /// Opens the index at `url`, or starts an empty one if no file exists yet.
  /// Throws `.revisionMismatch` if the file was written for another model revision.
  public init(url: URL, dimensions: Int, modelRevision: String) throws {
    guard (1...IndexFile.maxDimensions).contains(dimensions) else {
      throw VectorIndexError.invalidDimensions(dimensions)
    }
    self.url = url
    self.dimensions = dimensions
    self.modelRevision = modelRevision
    guard FileManager.default.fileExists(atPath: url.path) else { return }

    let file = try IndexFile.read(url, expectedDimensions: dimensions, expectedRevision: modelRevision)
    buffer = file.vectors
    idBySlot = file.ids
    slotByID.reserveCapacity(file.ids.count)
    for (i, id) in file.ids.enumerated() { slotByID[id] = i }
  }

  /// Pre-sizes storage for `n` vectors, avoiding repeated growth while bulk loading.
  public func reserveCapacity(_ n: Int) {
    guard n > 0 else { return }
    buffer.reserveCapacity(n * dimensions)
    idBySlot.reserveCapacity(n)
    slotByID.reserveCapacity(n)
  }

  public func upsert(id: String, vector: [Float]) throws {
    guard vector.count == dimensions else {
      throw VectorIndexError.dimensionMismatch(expected: dimensions, found: vector.count)
    }
    let s: Int
    if let existing = slotByID[id] {
      s = existing
    } else if let free = freeSlots.popLast() {
      s = free
    } else {
      s = slotCount
      idBySlot.append(nil)
      buffer.append(contentsOf: repeatElement(0, count: dimensions))
    }
    let base = s * dimensions
    buffer.withUnsafeMutableBufferPointer { b in
      vector.withUnsafeBufferPointer { FP16.round($0.baseAddress!, into: b.baseAddress! + base, count: dimensions) }
    }
    idBySlot[s] = id
    slotByID[id] = s
  }

  /// Removes `id`; removing an unknown id is a no-op.
  public func remove(id: String) throws {
    guard let s = slotByID.removeValue(forKey: id) else { return }
    let base = s * dimensions
    for i in 0..<dimensions { buffer[base + i] = 0 }
    idBySlot[s] = nil
    freeSlots.append(s)
  }

  public func vector(for id: String) -> [Float]? {
    guard let s = slotByID[id] else { return nil }
    return Array(buffer[(s * dimensions)..<((s + 1) * dimensions)])
  }

  /// Top-`k` by dot product, best first.
  ///
  /// - Returns `[]` when the index is empty, `k <= 0`, or `query` is unusable (wrong length
  ///   or containing NaN/infinity). The last two are programmer errors and `assert` in debug builds.
  /// - `k` larger than `count` (even `Int.max`) is fine.
  /// - `filter` must be pure and cheap: it is called once per candidate that would
  ///   enter the top `k`, in slot order.
  /// - Equal scores are ordered by internal slot, which is stable within a session but is
  ///   **not** preserved across `save()` and reload (slots are compacted).
  public func search(_ query: [Float], k: Int, filter: ((String) -> Bool)? = nil) -> [(id: String, score: Float)] {
    assert(query.count == dimensions, "query has \(query.count) dimensions, index has \(dimensions)")
    assert(query.allSatisfy(\.isFinite), "query contains NaN or infinity")
    guard k > 0, count > 0, query.count == dimensions, query.allSatisfy(\.isFinite) else { return [] }

    let slots = slotCount
    var scores = [Float](repeating: 0, count: slots)
    buffer.withUnsafeBufferPointer { m in
      query.withUnsafeBufferPointer { q in
        scores.withUnsafeMutableBufferPointer { out in
          // (slots x dims) * (dims x 1) -> (slots x 1)
          vDSP_mmul(m.baseAddress!, 1, q.baseAddress!, 1, out.baseAddress!, 1,
                    vDSP_Length(slots), 1, vDSP_Length(dimensions))
        }
      }
    }

    var heap = TopK(capacity: min(k, count))
    for s in 0..<slots {
      guard let id = idBySlot[s] else { continue }
      let score = scores[s]
      guard heap.wouldAccept(score: score, slot: s) else { continue }
      if let filter, !filter(id) { continue }
      heap.insert(score: score, slot: s)
    }
    return heap.sortedBestFirst().map { (id: idBySlot[$0.slot]!, score: $0.score) }
  }

  /// Every live vector scored against every query, unsorted: one matrix product over the whole buffer.
  ///
  /// `scores` is row-major `queries.count x ids.count`: row q holds the dot products of query q with the vectors
  /// in `ids` order (internal slot order, live slots only; stable within a session, not across `save()`). Returns
  /// no scores (and asserts in debug builds) if any query has the wrong length or contains NaN/infinity.
  public func dotAll(_ queries: [[Float]]) -> BulkScores {
    let live = (0..<slotCount).filter { idBySlot[$0] != nil }
    let ids = live.map { idBySlot[$0]! }
    let ok = queries.allSatisfy { $0.count == dimensions && $0.allSatisfy(\.isFinite) }
    assert(ok, "every query must have \(dimensions) finite values")
    guard ok, !queries.isEmpty, !ids.isEmpty else {
      return BulkScores(ids: ids, queryCount: ok ? queries.count : 0, scores: [])
    }
    let p = queries.count, slots = slotCount
    var qT = [Float](repeating: 0, count: dimensions * p)  // dims x p
    for (q, v) in queries.enumerated() { for i in 0..<dimensions { qT[i * p + q] = v[i] } }
    var bySlot = [Float](repeating: 0, count: slots * p)  // (slots x dims) * (dims x p) -> slots x p
    buffer.withUnsafeBufferPointer { m in
      vDSP_mmul(m.baseAddress!, 1, qT, 1, &bySlot, 1, vDSP_Length(slots), vDSP_Length(p), vDSP_Length(dimensions))
    }
    var scores = [Float](repeating: 0, count: p * ids.count)
    if live.count == slots {
      vDSP_mtrans(bySlot, 1, &scores, 1, vDSP_Length(p), vDSP_Length(slots))  // -> p x slots
    } else {
      for (j, s) in live.enumerated() { for q in 0..<p { scores[q * ids.count + j] = bySlot[s * p + q] } }
    }
    return BulkScores(ids: ids, queryCount: p, scores: scores)
  }

  /// Streams the index to a temp file in the same directory, then renames it over `url`.
  /// Free slots are compacted away.
  public func save() throws {
    try IndexFile.write(to: url, dimensions: dimensions, revision: modelRevision,
                        vectors: buffer, idBySlot: idBySlot)
  }
}

/// Fixed-capacity min-heap keeping the k best (score, slot) pairs.
/// Ties on score prefer the lower slot, so results are deterministic within a session.
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

/// Scores from `VectorIndex.dotAll`: `queryCount x ids.count`, row-major.
public struct BulkScores: Sendable {
  public let ids: [String]
  public let queryCount: Int
  public let scores: [Float]

  /// Query `q`'s scores, aligned with `ids`.
  public func scores(forQuery q: Int) -> ArraySlice<Float> {
    scores[(q * ids.count)..<((q + 1) * ids.count)]
  }
}

/// FP16 rounding with vImage (Swift's `Float16` does not exist on Intel Macs).
enum FP16 {
  /// `dst[i] = Float(Float16(src[i]))`.
  static func round(_ src: UnsafePointer<Float>, into dst: UnsafeMutablePointer<Float>, count: Int) {
    var half = [UInt16](repeating: 0, count: count)
    half.withUnsafeMutableBytes { h in
      var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count),
                            rowBytes: count * 4)
      var m = vImage_Buffer(data: h.baseAddress, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
      var d = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
      vImageConvert_PlanarFtoPlanar16F(&s, &m, vImage_Flags(kvImageNoFlags))
      vImageConvert_Planar16FtoPlanarF(&m, &d, vImage_Flags(kvImageNoFlags))
    }
  }
}
