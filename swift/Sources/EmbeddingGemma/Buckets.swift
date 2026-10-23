import Foundation

package enum BucketError: Error, Equatable {
  case nonPositiveSize(height: Int, width: Int)
  case unsupportedBudget(Int)
}

/// One fixed image shape of the Core ML image functions. Mirrors egcm/buckets.py exactly: the same ten-bucket
/// table (sizes are height x width) and the same aspect-ratio edges, inclusive and exclusive as there.
package struct Bucket: Sendable, Hashable {
  package enum Shape: String, Sendable, CaseIterable { case tall, portrait, square, landscape, wide }

  package static let patch = 16, pool = 3, side = 48
  package static let budgets = [70, 280]
  /// r = width / height: r < 0.6 tall | r < 0.83 portrait | r <= 1.2 square | r <= 1.67 landscape | else wide.
  package static let aspectEdges = [0.6, 0.83, 1.2, 1.67]

  package let budget: Int
  package let name: Shape
  package let height: Int
  package let width: Int

  /// Soft tokens: one per 3x3 block of 16 px patches.
  package var tokens: Int { (height / Self.side) * (width / Self.side) }
  package var functionName: String { "embed_image_\(budget)_\(name.rawValue)" }

  package static let all: [Bucket] = [
    Bucket(budget: 70, name: .tall, height: 576, width: 240),
    Bucket(budget: 70, name: .portrait, height: 480, width: 336),
    Bucket(budget: 70, name: .square, height: 384, width: 384),
    Bucket(budget: 70, name: .landscape, height: 336, width: 480),
    Bucket(budget: 70, name: .wide, height: 240, width: 576),
    Bucket(budget: 280, name: .tall, height: 1152, width: 528),
    Bucket(budget: 280, name: .portrait, height: 960, width: 672),
    Bucket(budget: 280, name: .square, height: 768, width: 768),
    Bucket(budget: 280, name: .landscape, height: 672, width: 960),
    Bucket(budget: 280, name: .wide, height: 528, width: 1152),
  ]

  private static func check(height: Int, width: Int, budget: Int) throws {
    guard height > 0, width > 0 else { throw BucketError.nonPositiveSize(height: height, width: width) }
    guard budgets.contains(budget) else { throw BucketError.unsupportedBudget(budget) }
  }

  /// The bucket for an image of the given size (height first, as in Python), from r = width / height.
  package static func choose(height: Int, width: Int, budget: Int) throws -> Bucket {
    try check(height: height, width: width, budget: budget)
    let r = Double(width) / Double(height)
    let name: Shape =
      if r < aspectEdges[0] { .tall }
      else if r < aspectEdges[1] { .portrait }
      else if r <= aspectEdges[2] { .square }
      else if r <= aspectEdges[3] { .landscape }
      else { .wide }
    return all.first { $0.budget == budget && $0.name == name }!
  }

  /// The size the reference processor resizes to (port of `aspect_preserving_size`). The buckets are fixed points
  /// of it; the runtime only needs `choose`, this keeps the table honest.
  package static func aspectPreservingSize(height: Int, width: Int, budget: Int) throws -> (height: Int, width: Int) {
    try check(height: height, width: width, budget: budget)
    let maxPatches = budget * pool * pool
    let targetPx = Double(maxPatches * patch * patch)
    let f = (targetPx / (Double(height) * Double(width))).squareRoot()
    var th = Int((f * Double(height) / Double(side)).rounded(.down)) * side
    var tw = Int((f * Double(width) / Double(side)).rounded(.down)) * side
    let maxSide = (maxPatches / (pool * pool)) * side
    if th == 0 {
      th = side
      tw = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
    } else if tw == 0 {
      tw = side
      th = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
    }
    return (th, tw)
  }
}
