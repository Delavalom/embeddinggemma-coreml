import Testing
@testable import EmbeddingGemma

/// The same cases as tests/test_buckets.py, so Swift and Python pick identical buckets.
@Suite struct BucketsTests {
  @Test func tenBuckets() {
    #expect(Bucket.all.count == 10)
    #expect(Set(Bucket.all.map(\.name)) == Set(Bucket.Shape.allCases))
    #expect(Set(Bucket.all.map(\.functionName)).count == 10)
  }

  @Test func tableMatchesPython() {
    let table = Bucket.all.map { "\($0.budget) \($0.name.rawValue) \($0.height)x\($0.width)" }
    #expect(table == [
      "70 tall 576x240", "70 portrait 480x336", "70 square 384x384", "70 landscape 336x480", "70 wide 240x576",
      "280 tall 1152x528", "280 portrait 960x672", "280 square 768x768", "280 landscape 672x960",
      "280 wide 528x1152",
    ])
    #expect(Bucket.all[1].functionName == "embed_image_70_portrait")
    #expect(Bucket.all[1].tokens == 70 && Bucket.all[7].tokens == 256)
  }

  @Test func bucketsAreFixedPointsOfTheResize() throws {
    for b in Bucket.all {
      let size = try Bucket.aspectPreservingSize(height: b.height, width: b.width, budget: b.budget)
      #expect(size.height == b.height && size.width == b.width)
      #expect((b.height / 48) * (b.width / 48) == b.tokens && b.tokens <= b.budget)
    }
  }

  @Test func eachBucketIsChosenByItsOwnShape() throws {
    for b in Bucket.all { #expect(try Bucket.choose(height: b.height, width: b.width, budget: b.budget) == b) }
  }

  @Test func matchesUpstreamFormula() throws {
    let s = try Bucket.aspectPreservingSize(height: 3024, width: 4032, budget: 70)
    #expect(s.height == 336 && s.width == 432)
  }

  /// Expected values from egcm.buckets.aspect_preserving_size (which tests/test_buckets.py checks against upstream).
  @Test(arguments: [
    (10, 100000, 70, 48, 3360), (10, 100000, 280, 48, 13440), (100000, 10, 70, 3360, 48),
    (1, 1000, 280, 48, 13440), (1000, 1, 70, 3360, 48), (7, 5000, 70, 48, 3360),
    (1064, 1600, 70, 288, 480), (1064, 1600, 280, 624, 960), (4032, 3024, 280, 912, 672),
  ])
  func resizeMatchesPython(h: Int, w: Int, budget: Int, eh: Int, ew: Int) throws {
    let s = try Bucket.aspectPreservingSize(height: h, width: w, budget: budget)
    #expect(s.height == eh && s.width == ew)
  }

  @Test func choose() throws {  // (height, width, budget)
    func name(_ h: Int, _ w: Int, _ b: Int) throws -> Bucket.Shape { try Bucket.choose(height: h, width: w, budget: b).name }
    #expect(try name(3024, 4032, 70) == .landscape)
    #expect(try name(4032, 3024, 70) == .portrait)
    #expect(try name(1000, 1000, 280) == .square)
    #expect(try name(2532, 1170, 70) == .tall)
    #expect(try name(1170, 2532, 280) == .wide)
    #expect(try name(1080, 1920, 70) == .wide)
    #expect(try name(2000, 3000, 70) == .landscape)
    #expect(try name(5000, 20000, 280) == .wide)
  }

  @Test func boundaries() throws {
    #expect(Bucket.aspectEdges == [0.6, 0.83, 1.2, 1.67])
    func name(_ r: Double) throws -> Bucket.Shape {  // height 10000 keeps r exact
      try Bucket.choose(height: 10000, width: Int((r * 10000).rounded()), budget: 70).name
    }
    #expect(try name(0.5999) == .tall)
    #expect(try name(0.6) == .portrait)
    #expect(try name(0.8299) == .portrait)
    #expect(try name(0.83) == .square)
    #expect(try name(1.2) == .square)
    #expect(try name(1.2001) == .landscape)
    #expect(try name(1.67) == .landscape)
    #expect(try name(1.6701) == .wide)
  }

  @Test(arguments: [(0, 100), (100, 0), (0, 0), (-5, 10)])
  func zeroOrNegativeDimensionsThrow(h: Int, w: Int) {
    #expect(throws: BucketError.nonPositiveSize(height: h, width: w)) { try Bucket.choose(height: h, width: w, budget: 70) }
    #expect(throws: BucketError.nonPositiveSize(height: h, width: w)) {
      try Bucket.aspectPreservingSize(height: h, width: w, budget: 70)
    }
  }

  @Test func unsupportedBudgetThrows() {
    #expect(throws: BucketError.unsupportedBudget(100)) { try Bucket.choose(height: 100, width: 100, budget: 100) }
    #expect(throws: BucketError.unsupportedBudget(100)) {
      try Bucket.aspectPreservingSize(height: 100, width: 100, budget: 100)
    }
  }
}
