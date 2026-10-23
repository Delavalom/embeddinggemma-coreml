import Testing
import Foundation
@testable import EmbeddingIndex

/// Deterministic unit vectors from a seeded LCG.
func rand(_ n: Int, _ d: Int, seed: UInt64) -> [[Float]] {
  var s = seed
  return (0..<n).map { _ in
    let v = (0..<d).map { _ -> Float in
      s = s &* 6364136223846793005 &+ 1
      return Float(Int64(bitPattern: s >> 11) % 2000) / 1000 - 1
    }
    let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
    return v.map { $0 / n }
  }
}

func tempURL() -> URL {
  FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
}

@Test func topKMatchesBruteForce() throws {
  let url = tempURL()
  var idx = try VectorIndex(url: url, dimensions: 64, modelRevision: "r1")
  let vs = rand(500, 64, seed: 7)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "a\(i)", vector: v) }
  let q = vs[42]
  let hits = idx.search(q, k: 5)
  #expect(hits.first?.id == "a42")
  let brute = vs.enumerated().map { ($0.offset, zip($0.element, q).reduce(Float(0)) { $0 + $1.0 * $1.1 }) }
    .sorted { $0.1 > $1.1 }.prefix(5).map { "a\($0.0)" }
  #expect(hits.map(\.id) == Array(brute))
}

@Test func persistsAndRejectsOtherRevision() throws {
  let url = tempURL()
  var idx = try VectorIndex(url: url, dimensions: 8, modelRevision: "r1")
  try idx.upsert(id: "x", vector: [1,0,0,0,0,0,0,0]); try idx.save()
  let again = try VectorIndex(url: url, dimensions: 8, modelRevision: "r1")
  #expect(again.vector(for: "x")?.first == 1)
  #expect(throws: VectorIndexError.revisionMismatch(expected: "r2", found: "r1")) {
    _ = try VectorIndex(url: url, dimensions: 8, modelRevision: "r2")
  }
}

@Test func removeAndFilter() throws {
  let url = tempURL()
  var idx = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.upsert(id: "b", vector: [0.9,0.1,0,0])
  try idx.remove(id: "a")
  #expect(idx.search([1,0,0,0], k: 2).map(\.id) == ["b"])
  #expect(idx.search([1,0,0,0], k: 2, filter: { $0 != "b" }).isEmpty)
}

@Test func upsertReplacesAndRemovedSlotsAreReused() throws {
  var idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0])
  try idx.upsert(id: "a", vector: [0,1,0,0])
  #expect(idx.count == 1)
  #expect(idx.vector(for: "a") == [0,1,0,0])
  try idx.remove(id: "a")
  try idx.upsert(id: "b", vector: [0,0,1,0])
  #expect(idx.count == 1)
  #expect(idx.vector(for: "a") == nil)
  #expect(idx.search([0,0,1,0], k: 5).map(\.id) == ["b"])
}

@Test func rejectsWrongDimensions() throws {
  var idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  #expect(throws: VectorIndexError.dimensionMismatch(4, 3)) { try idx.upsert(id: "a", vector: [1,0,0]) }
  #expect(idx.search([1,0,0], k: 3).isEmpty)
}

@Test func saveCompactsRemovedSlotsAndRoundTrips() throws {
  let url = tempURL()
  var idx = try VectorIndex(url: url, dimensions: 16, modelRevision: "r")
  let vs = rand(20, 16, seed: 3)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "k\(i)", vector: v) }
  for i in stride(from: 0, to: 20, by: 2) { try idx.remove(id: "k\(i)") }
  try idx.save()
  let again = try VectorIndex(url: url, dimensions: 16, modelRevision: "r")
  #expect(again.count == 10)
  #expect(again.search(vs[5], k: 3).map(\.id) == idx.search(vs[5], k: 3).map(\.id))
  #expect(again.vector(for: "k4") == nil)
}

@Test func rejectsDimensionMismatchAndCorruptFiles() throws {
  let url = tempURL()
  var idx = try VectorIndex(url: url, dimensions: 8, modelRevision: "r")
  try idx.upsert(id: "x", vector: [1,0,0,0,0,0,0,0]); try idx.save()
  #expect(throws: VectorIndexError.dimensionMismatch(4, 8)) {
    _ = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  }
  let data = try Data(contentsOf: url)
  try data.prefix(data.count - 3).write(to: url)
  #expect(throws: VectorIndexError.corrupt) {
    _ = try VectorIndex(url: url, dimensions: 8, modelRevision: "r")
  }
  try Data("garbage".utf8).write(to: url)
  #expect(throws: VectorIndexError.corrupt) {
    _ = try VectorIndex(url: url, dimensions: 8, modelRevision: "r")
  }
}

/// 50k x 256, k = 50. The 20 ms budget is only enforced in release builds
/// (`swift test -c release --filter perf`); debug builds just run the search.
@Test(.timeLimit(.minutes(5))) func perf50kSearch() throws {
  var idx = try VectorIndex(url: tempURL(), dimensions: 256, modelRevision: "perf")
  let vs = rand(50_000, 256, seed: 99)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "v\(i)", vector: v) }
  let q = vs[1234]
  _ = idx.search(q, k: 50) // warm up
  var times: [Double] = []
  var hits: [(id: String, score: Float)] = []
  for _ in 0..<20 {
    let t0 = ContinuousClock.now
    hits = idx.search(q, k: 50)
    let d = ContinuousClock.now - t0
    times.append(Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
  }
  times.sort()
  let median = times[times.count / 2]
  print("perf50kSearch: median \(median) ms, min \(times[0]) ms, max \(times[times.count - 1]) ms")
  #expect(hits.count == 50)
  #expect(hits.first?.id == "v1234")
  #if !DEBUG
  #expect(median < 20)
  #endif
}
