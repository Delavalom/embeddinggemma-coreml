import CoreGraphics
import CoreML
import CryptoKit
import EmbeddingGemma
import Foundation
import ImageIO

/// Command-line options. Every path is required except `synthetic`.
struct Options {
  var model: URL
  var tokenizer: URL
  var fixtures: URL
  var out: URL
  /// A folder of lossless PNGs (written by stages/s5_swift/run.py): pixels only, no embeddings.
  var synthetic: URL?
  var samples = 8192
  var computeUnits = MLComputeUnits.cpuAndNeuralEngine
  /// Only the first N texts / images (0 = all), for quick runs.
  var limit = 0

  static let usage = """
    usage: egcm-parity --model <EmbeddingGemma2.mlpackage> --tokenizer <dir> --fixtures <repo>/fixtures --out <json>
                       [--synthetic <dir of PNGs>] [--samples 8192] [--compute-units cpuAndNeuralEngine|cpuOnly|all]
                       [--limit N]
    """

  static func parse(_ args: [String]) throws -> Options {
    var values: [String: String] = [:]
    var i = 0
    while i < args.count {
      let key = args[i]
      guard key.hasPrefix("--"), i + 1 < args.count else { throw ParityError.usage("bad argument \(key)") }
      values[String(key.dropFirst(2))] = args[i + 1]
      i += 2
    }
    func path(_ k: String) throws -> URL {
      guard let v = values.removeValue(forKey: k) else { throw ParityError.usage("--\(k) is required") }
      return URL(filePath: v).standardizedFileURL
    }
    var o = Options(model: try path("model"), tokenizer: try path("tokenizer"), fixtures: try path("fixtures"),
                    out: try path("out"))
    if values["synthetic"] != nil { o.synthetic = try path("synthetic") }
    if let s = values.removeValue(forKey: "samples") {
      guard let n = Int(s), n > 0 else { throw ParityError.usage("--samples must be a positive integer") }
      o.samples = n
    }
    if let l = values.removeValue(forKey: "limit") {
      guard let n = Int(l), n >= 0 else { throw ParityError.usage("--limit must be >= 0") }
      o.limit = n
    }
    if let cu = values.removeValue(forKey: "compute-units") {
      switch cu {
      case "cpuAndNeuralEngine": o.computeUnits = .cpuAndNeuralEngine
      case "cpuOnly": o.computeUnits = .cpuOnly
      case "all": o.computeUnits = .all
      default: throw ParityError.usage("unknown --compute-units \(cu)")
      }
    }
    if let extra = values.keys.sorted().first { throw ParityError.usage("unknown option --\(extra)") }
    return o
  }
}

enum ParityError: Error, CustomStringConvertible {
  case usage(String)
  case badFixture(String)

  var description: String {
    switch self {
    case .usage(let m): "\(m)\n\(Options.usage)"
    case .badFixture(let m): "bad fixture: \(m)"
    }
  }
}

struct TextRow: Decodable { let id: String; let text: String; let kind: String }
struct ImageRow: Decodable { let id: String; let file: String }

enum Fixtures {
  static func jsonl<T: Decodable>(_ url: URL, as _: T.Type) throws -> [T] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return try text.split(separator: "\n").filter { !$0.allSatisfy(\.isWhitespace) }.map {
      try JSONDecoder().decode(T.self, from: Data($0.utf8))
    }
  }

  /// The prompt s0 used for a fixture kind (stages/s0_reference/reference.py PROMPT_FOR_KIND).
  static func task(_ kind: String) throws -> EmbeddingTask {
    switch kind {
    case "query": .query
    case "document": .document
    default: throw ParityError.badFixture("unknown text kind \(kind)")
    }
  }

  /// The decoded image and its EXIF orientation (fixtures are stored upright, so normally `.up`).
  static func image(_ url: URL) throws -> (CGImage, CGImagePropertyOrientation) {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
      throw ParityError.badFixture("cannot decode \(url.path)")
    }
    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    let raw = (props?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
    return (img, CGImagePropertyOrientation(rawValue: raw) ?? .up)
  }
}

/// The sampling rule of stages/s5_swift/export_fixtures.py: index k -> (k * n) // samples, or all n if n <= samples.
func sampleIndices(_ n: Int, _ samples: Int) -> [Int] {
  n <= samples ? Array(0..<n) : (0..<samples).map { $0 * n / samples }
}

/// SHA-256 of the uint8 HxWx3 layout s0 stores, from channels-first planes.
func interleavedSHA256(planar: [UInt8], height: Int, width: Int) -> String {
  let n = height * width
  var out = [UInt8](repeating: 0, count: n * 3)
  for i in 0..<n { for c in 0..<3 { out[i * 3 + c] = planar[c * n + i] } }
  return SHA256.hash(data: out).map { String(format: "%02x", $0) }.joined()
}

/// What egcm-parity writes. `run.py` reads it (stages/s5_swift/compare.py).
struct TextResult: Encodable { let ids: [Int32]; let emb: [Float] }
struct ImageResult: Encodable {
  let id: String
  let budget: Int
  let bucket: String
  let height: Int
  let width: Int
  let sha256: String
  /// uint8 values of the `[3, H, W]` tensor at `sampleIndices(3*H*W, samples)`.
  let pixel_sample: [UInt8]
  /// L2-normalized 768-d vector (nil for synthetic images).
  let emb: [Float]?
}
struct ParityOutput: Encodable {
  let model: String
  let model_revision: String
  let compute_units: String
  let samples: Int
  let texts: [String: TextResult]
  /// Keyed `<id>_<budget>`.
  let images: [String: ImageResult]
  let synthetic: [String: ImageResult]
}
