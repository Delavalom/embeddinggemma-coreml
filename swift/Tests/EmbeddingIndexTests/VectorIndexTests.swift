import Testing
import Foundation
@testable import EmbeddingIndex

/// Deterministic unit vectors from a seeded LCG.
struct LCG {
  var s: UInt64
  mutating func unitVector(_ d: Int) -> [Float] {
    let v = (0..<d).map { _ -> Float in
      s = s &* 6364136223846793005 &+ 1
      return Float(Int64(bitPattern: s >> 11) % 2000) / 1000 - 1
    }
    let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
    return v.map { $0 / n }
  }
}

func rand(_ n: Int, _ d: Int, seed: UInt64) -> [[Float]] {
  var g = LCG(s: seed)
  return (0..<n).map { _ in g.unitVector(d) }
}

func tempURL() -> URL {
  FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
}

func tempDir() throws -> URL {
  let d = tempURL()
  try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
  return d
}

@Test func topKMatchesBruteForce() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 64, modelRevision: "r1")
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
  let idx = try VectorIndex(url: url, dimensions: 8, modelRevision: "r1")
  try idx.upsert(id: "x", vector: [1,0,0,0,0,0,0,0]); try idx.save()
  let again = try VectorIndex(url: url, dimensions: 8, modelRevision: "r1")
  #expect(again.vector(for: "x")?.first == 1)
  #expect(throws: VectorIndexError.revisionMismatch(expected: "r2", found: "r1")) {
    _ = try VectorIndex(url: url, dimensions: 8, modelRevision: "r2")
  }
}

@Test func removeAndFilter() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.upsert(id: "b", vector: [0.9,0.1,0,0])
  try idx.remove(id: "a")
  #expect(idx.search([1,0,0,0], k: 2).map(\.id) == ["b"])
  #expect(idx.search([1,0,0,0], k: 2, filter: { $0 != "b" }).isEmpty)
}

@Test func upsertReplacesAndRemovedSlotsAreReused() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0])
  try idx.upsert(id: "a", vector: [0,1,0,0])
  #expect(idx.count == 1)
  #expect(idx.vector(for: "a") == [0,1,0,0])
  try idx.remove(id: "a")
  try idx.upsert(id: "b", vector: [0,0,1,0])
  #expect(idx.count == 1)
  #expect(idx.vector(for: "a") == nil)
  #expect(idx.search([0,0,1,0], k: 5).map(\.id) == ["b"])
  try idx.remove(id: "never-existed") // no-op
  #expect(idx.count == 1)
}

@Test func upsertRejectsWrongDimensions() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  #expect(throws: VectorIndexError.dimensionMismatch(expected: 4, found: 3)) { try idx.upsert(id: "a", vector: [1,0,0]) }
}

@Test(arguments: [0, -1, 65_536, Int.max])
func invalidDimensionsThrowInsteadOfTrapping(_ d: Int) {
  #expect(throws: VectorIndexError.invalidDimensions(d)) {
    _ = try VectorIndex(url: tempURL(), dimensions: d, modelRevision: "r")
  }
}

@Test func maxDimensionsIsAccepted() throws {
  _ = try VectorIndex(url: tempURL(), dimensions: 65_535, modelRevision: "r")
}

@Test func hugeAndDegenerateK() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 16, modelRevision: "r")
  #expect(idx.search(rand(1, 16, seed: 1)[0], k: 10).isEmpty) // empty index
  let vs = rand(30, 16, seed: 5)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "k\(i)", vector: v) }
  let all = idx.search(vs[0], k: 1_000)
  #expect(all.count == 30)
  #expect(all.map(\.score) == all.map(\.score).sorted(by: >))
  #expect(idx.search(vs[0], k: Int.max).map(\.id) == all.map(\.id))
  #expect(idx.search(vs[0], k: 0).isEmpty)
  #expect(idx.search(vs[0], k: -5).isEmpty)
  #expect(idx.search(vs[0], k: Int.min).isEmpty)
}

@Test func reserveCapacityIsHarmless() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  idx.reserveCapacity(0); idx.reserveCapacity(-3); idx.reserveCapacity(1_000)
  try idx.upsert(id: "a", vector: [1,0,0,0])
  #expect(idx.count == 1)
}

// Debug builds trap on these inputs by design (assert); release builds return [].
#if !DEBUG
@Test func unusableQueriesReturnEmpty() throws {
  let idx = try VectorIndex(url: tempURL(), dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0])
  #expect(idx.search([1,0,0], k: 3).isEmpty)
  #expect(idx.search([.nan,0,0,0], k: 3).isEmpty)
  #expect(idx.search([.infinity,0,0,0], k: 3).isEmpty)
}
#endif

@Test func valuesAreFP16RoundedAndSurviveSaveAndReload() throws {
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 32, modelRevision: "r")
  let vs = rand(50, 32, seed: 11)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "k\(i)", vector: v) }
  #expect(idx.vector(for: "k0") == vs[0].map { Float(Float16($0)) })
  try idx.save()
  let again = try VectorIndex(url: url, dimensions: 32, modelRevision: "r")
  for i in 0..<50 { #expect(again.vector(for: "k\(i)") == idx.vector(for: "k\(i)")) }
  #expect(again.search(vs[3], k: 10).map(\.id) == idx.search(vs[3], k: 10).map(\.id))
}

@Test func saveCompactsRemovedSlotsAndRoundTrips() throws {
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 16, modelRevision: "r")
  let vs = rand(20, 16, seed: 3)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "k\(i)", vector: v) }
  for i in stride(from: 0, to: 20, by: 2) { try idx.remove(id: "k\(i)") }
  try idx.save()
  let again = try VectorIndex(url: url, dimensions: 16, modelRevision: "r")
  #expect(again.count == 10)
  #expect(again.search(vs[5], k: 3).map(\.id) == idx.search(vs[5], k: 3).map(\.id))
  #expect(again.vector(for: "k4") == nil)
}

@Test func saveLeavesNoTempFilesAndReplacesExistingFile() throws {
  let dir = try tempDir()
  let url = dir.appending(path: "index.egix")
  let idx = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.save()
  try idx.upsert(id: "b", vector: [0,1,0,0]); try idx.save()
  #expect(try FileManager.default.contentsOfDirectory(atPath: dir.path) == ["index.egix"])
  #expect(try VectorIndex(url: url, dimensions: 4, modelRevision: "r").count == 2)
}

@Test func saveAcrossManyChunksWithGaps() throws {
  // dims 1024 -> 1024 rows per chunk. Removing k100...110 and k2000 leaves live runs of 100, 1889 (two chunks), 499 and 1 slots.
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 1024, modelRevision: "r")
  idx.reserveCapacity(2_500)
  var g = LCG(s: 21)
  for i in 0..<2_500 { try idx.upsert(id: "k\(i)", vector: g.unitVector(1024)) }
  for i in (100...110) + [2000] { try idx.remove(id: "k\(i)") }
  try idx.save()
  let again = try VectorIndex(url: url, dimensions: 1024, modelRevision: "r")
  #expect(again.count == idx.count && again.count == 2_488)
  #expect(again.vector(for: "k105") == nil)
  for i in [0, 99, 111, 1023, 1024, 1025, 1134, 1999, 2001, 2_499] { #expect(again.vector(for: "k\(i)") == idx.vector(for: "k\(i)")) }
}

// MARK: File format

@Test func fp16BlockStartsSixteenByteAligned() throws {
  for rev in ["", "r", "abcdefghijkl", "a-much-longer-revision-name-0123456789"] {
    let off = IndexFile.blockOffset(revisionLength: rev.utf8.count)
    #expect(off % 16 == 0 && off >= 20 + rev.utf8.count && off < 20 + rev.utf8.count + 16)

    let url = tempURL()
    let idx = try VectorIndex(url: url, dimensions: 4, modelRevision: rev)
    try idx.upsert(id: "one", vector: [1, 0, 0, 0]); try idx.save()
    let bytes = [UInt8](try Data(contentsOf: url))
    #expect(Array(bytes[off..<(off + 2)]) == [0x00, 0x3C]) // 1.0 as little-endian FP16
    #expect(bytes[4] == 2) // version 2
  }
}

@Test func unsupportedVersionIsReported() throws {
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.save()
  var bytes = [UInt8](try Data(contentsOf: url))
  bytes[4] = 1
  try Data(bytes).write(to: url)
  #expect(throws: VectorIndexError.unsupportedVersion(1)) {
    _ = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  }
}

@Test func dimensionMismatchOnOpen() throws {
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 8, modelRevision: "r")
  try idx.upsert(id: "x", vector: [1,0,0,0,0,0,0,0]); try idx.save()
  #expect(throws: VectorIndexError.dimensionMismatch(expected: 4, found: 8)) {
    _ = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  }
}

@Test func garbageFilesAreCorrupt() throws {
  let url = tempURL()
  for junk in ["", "EG", "garbage", "garbage-but-long-enough-to-hold-a-header"] {
    try Data(junk.utf8).write(to: url)
    #expect(throws: VectorIndexError.corrupt) { _ = try VectorIndex(url: url, dimensions: 8, modelRevision: "r") }
  }
}

@Test func everyTruncatedPrefixIsRejected() throws {
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 4, modelRevision: "rev")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.upsert(id: "b", vector: [0,1,0,0]); try idx.save()
  let full = try Data(contentsOf: url)
  for n in 0..<full.count {
    try full.prefix(n).write(to: url)
    #expect(throws: VectorIndexError.self, "prefix of \(n)/\(full.count) bytes") {
      _ = try VectorIndex(url: url, dimensions: 4, modelRevision: "rev")
    }
  }
  try full.write(to: url) // the untruncated file still opens
  #expect(try VectorIndex(url: url, dimensions: 4, modelRevision: "rev").count == 2)
}

@Test func everyHeaderByteFlipIsRejected() throws {
  let url = tempURL()
  let idx = try VectorIndex(url: url, dimensions: 4, modelRevision: "rev")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.save()
  let full = [UInt8](try Data(contentsOf: url))
  let blockStart = IndexFile.blockOffset(revisionLength: 3)
  for i in 0..<blockStart {
    var bytes = full
    bytes[i] ^= 0xFF
    try Data(bytes).write(to: url)
    #expect(throws: VectorIndexError.self, "flip of header byte \(i)") {
      _ = try VectorIndex(url: url, dimensions: 4, modelRevision: "rev")
    }
  }
}

// MARK: Scale (serialized so their memory and timing do not disturb each other)

@Suite(.serialized) struct ScaleTests {
  /// 50k x 256, k = 50. The 20 ms budget is only enforced in release builds
  /// (`swift test -c release --filter perf`); debug builds just run the search.
  @Test(.timeLimit(.minutes(5))) func perf50kSearch() throws {
    let idx = try VectorIndex(url: tempURL(), dimensions: 256, modelRevision: "perf")
    idx.reserveCapacity(50_000)
    var g = LCG(s: 99)
    var q: [Float] = []
    for i in 0..<50_000 {
      let v = g.unitVector(256)
      if i == 1234 { q = v }
      try idx.upsert(id: "v\(i)", vector: v)
    }
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

  #if !DEBUG
  /// Build, save and reload 50k x 256 (51 MB of floats): peak footprint growth must stay under 80 MB,
  /// i.e. no second copy of the vectors (no intermediate [Float16], no copy-on-write duplicate).
  /// Run it alone for a clean baseline: `swift test -c release --filter memoryFootprint50k`.
  /// In a full run, memory freed by earlier tests can mask growth (the bound still holds).
  @Test(.timeLimit(.minutes(5))) func memoryFootprint50k() throws {
    let mb = { (b: UInt64) in Double(b) / 1_048_576 }
    let url = tempURL()
    returnFreedMemoryToOS() // earlier tests' freed buffers would otherwise hide our growth
    let base = physFootprint()
    let sampler = PeakSampler(); sampler.start()

    var idx: VectorIndex? = try VectorIndex(url: url, dimensions: 256, modelRevision: "mem")
    idx!.reserveCapacity(50_000)
    var g = LCG(s: 5)
    var q: [Float] = []
    for i in 0..<50_000 {
      let v = g.unitVector(256)
      if i == 777 { q = v }
      try idx!.upsert(id: "v\(i)", vector: v)
    }
    let built = physFootprint()
    try idx!.save()
    let saved = physFootprint()
    idx = nil
    returnFreedMemoryToOS()
    let released = physFootprint()
    let loaded = try VectorIndex(url: url, dimensions: 256, modelRevision: "mem")
    let reloaded = physFootprint()
    let peak = sampler.stop()

    print(String(format: "memoryFootprint50k: baseline %.1f MB | built +%.1f | saved +%.1f | released +%.1f | reloaded +%.1f | PEAK growth %.1f MB",
                 mb(base), mb(built) - mb(base), mb(saved) - mb(base), mb(released) - mb(base),
                 mb(reloaded) - mb(base), mb(peak) - mb(base)))
    #expect(loaded.count == 50_000)
    #expect(loaded.search(q, k: 1).first?.id == "v777")
    #expect(mb(peak) - mb(base) < 80)
  }
  #endif
}
