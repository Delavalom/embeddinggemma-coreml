import CoreGraphics
import CoreML
import CryptoKit
import EmbeddingGemma
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The packages `--model` names (ADR-0010).
enum ModelPackages: Equatable {
  /// `--model <pkg>`: one 14-function package that serves every budget.
  case all(URL)
  /// `--model 70=<pkg> --model 280=<pkg>`: one package per image budget.
  case perBudget([Int: URL])

  static let budgets = [70, 280]

  /// The package used for `budget`.
  func package(budget: Int) -> URL? {
    switch self {
    case .all(let url): url
    case .perBudget(let urls): urls[budget]
    }
  }

  /// What `--model` was given, for parity.json: the path, or `70=<pkg> 280=<pkg>`.
  var argument: String {
    switch self {
    case .all(let url): url.path
    case .perBudget(let urls): urls.keys.sorted().map { "\($0)=\(urls[$0]!.path)" }.joined(separator: " ")
    }
  }

  /// Every `--model` value in order: one bare path, or `<budget>=<path>` once for each of 70 and 280.
  static func parse(_ values: [String]) throws -> ModelPackages {
    guard !values.isEmpty else { throw ParityError.usage("--model is required") }
    var perBudget: [Int: URL] = [:]
    var bare: [String] = []
    for v in values {
      if let m = v.wholeMatch(of: /([0-9]+)=(.+)/) {
        guard let b = Int(m.1), budgets.contains(b) else {
          throw ParityError.usage("--model \(v): the budget must be 70 or 280")
        }
        guard perBudget[b] == nil else { throw ParityError.usage("--model \(b)=... is given twice") }
        perBudget[b] = URL(filePath: String(m.2)).standardizedFileURL
      } else {
        bare.append(v)
      }
    }
    if bare.count == 1 && perBudget.isEmpty { return .all(URL(filePath: bare[0]).standardizedFileURL) }
    guard bare.isEmpty else {
      throw ParityError.usage("--model takes one package path, or one 70=<pkg> and one 280=<pkg>")
    }
    if let missing = budgets.first(where: { perBudget[$0] == nil }) {
      throw ParityError.usage("--model \(missing)=<pkg> is missing (one package per budget needs both)")
    }
    return .perBudget(perBudget)
  }
}

/// Command-line options. Every path is required except `synthetic` and `pixels`.
struct Options {
  var model: ModelPackages
  var tokenizer: URL
  var fixtures: URL
  var out: URL
  /// A folder of lossless PNGs (written by stages/s5_swift/run.py): pixels only, no embeddings.
  var synthetic: URL?
  /// Where to write each fixture's uint8 bucket image as `<id>_<budget>.png` (lossless), so s5 can feed Swift's exact
  /// pixels to Python Core ML and to the FP reference (ADR-0008).
  var pixels: URL?
  var samples = 8192
  var computeUnits = MLComputeUnits.cpuAndNeuralEngine
  /// Only the first N texts / images (0 = all), for quick runs.
  var limit = 0

  static let usage = """
    usage: egcm-parity --model <EmbeddingGemma2.mlpackage> --tokenizer <dir> --fixtures <repo>/fixtures --out <json>
                       [--synthetic <dir of PNGs>] [--pixels <dir>] [--samples 8192]
                       [--compute-units cpuAndNeuralEngine|cpuOnly|all] [--limit N]
           --model <pkg>                          one 14-function package for both budgets
           --model 70=<pkg> --model 280=<pkg>     one package per image budget (ADR-0010)
    """

  static func parse(_ args: [String]) throws -> Options {
    var values: [String: String] = [:]
    var models: [String] = []
    var i = 0
    while i < args.count {
      let key = args[i]
      guard key.hasPrefix("--"), i + 1 < args.count else { throw ParityError.usage("bad argument \(key)") }
      if key == "--model" { models.append(args[i + 1]) } else { values[String(key.dropFirst(2))] = args[i + 1] }
      i += 2
    }
    func path(_ k: String) throws -> URL {
      guard let v = values.removeValue(forKey: k) else { throw ParityError.usage("--\(k) is required") }
      return URL(filePath: v).standardizedFileURL
    }
    var o = Options(model: try ModelPackages.parse(models), tokenizer: try path("tokenizer"),
                    fixtures: try path("fixtures"), out: try path("out"))
    if values["synthetic"] != nil { o.synthetic = try path("synthetic") }
    if values["pixels"] != nil { o.pixels = try path("pixels") }
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

/// The uint8 HxWx3 layout s0 stores, from channels-first planes.
func interleaved(planar: [UInt8], height: Int, width: Int) -> [UInt8] {
  let n = height * width
  var out = [UInt8](repeating: 0, count: n * 3)
  for i in 0..<n { for c in 0..<3 { out[i * 3 + c] = planar[c * n + i] } }
  return out
}

/// SHA-256 of the uint8 HxWx3 layout s0 stores, from channels-first planes.
func interleavedSHA256(planar: [UInt8], height: Int, width: Int) -> String {
  SHA256.hash(data: interleaved(planar: planar, height: height, width: width))
    .map { String(format: "%02x", $0) }.joined()
}

/// Writes channels-first uint8 planes as an 8-bit RGB PNG, which stores the values losslessly (no colour
/// conversion: the image is tagged sRGB, the bytes are written as they are). s5 checks the sha256 on reading.
func writePNG(planar: [UInt8], height: Int, width: Int, to url: URL) throws {
  let rgb = Data(interleaved(planar: planar, height: height, width: width))
  guard let provider = CGDataProvider(data: rgb as CFData), let srgb = CGColorSpace(name: CGColorSpace.sRGB),
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
                            space: srgb, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    throw ParityError.badFixture("cannot encode \(url.path)")
  }
  CGImageDestinationAddImage(dest, image, nil)
  guard CGImageDestinationFinalize(dest) else { throw ParityError.badFixture("cannot write \(url.path)") }
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
  /// `--model` as given: the package path, or `70=<pkg> 280=<pkg>`.
  let model: String
  /// The package used per budget, keyed `"70"`, `"280"`.
  let models: [String: String]
  /// `modelRevision` of the budget 70 model (texts are embedded with it).
  let model_revision: String
  /// `modelRevision` per budget, keyed `"70"`, `"280"` (equal for one 14-function package).
  let model_revisions: [String: String]
  let compute_units: String
  let samples: Int
  let texts: [String: TextResult]
  /// Keyed `<id>_<budget>`.
  let images: [String: ImageResult]
  let synthetic: [String: ImageResult]
}
