import EmbeddingGemma
import SwiftUI

@main
struct PhotoSearchDemoApp: App {
  private let launch = LaunchArguments(CommandLine.arguments)

  var body: some Scene {
    WindowGroup {
      if launch.compile {
        CompileView(runID: launch.runID)
      } else if launch.benchmark {
        BenchmarkView(runID: launch.runID)
      } else {
        SearchView()
      }
    }
  }
}

/// `-compile` runs `CompileRunner` and `-benchmark` runs `BenchmarkRunner` instead of the search UI; `-runID <id>`
/// is echoed into the result file so stages/s6_device/run.py can tell this run's result from an older one. A literal
/// `--` is ignored.
struct LaunchArguments {
  let compile: Bool
  let benchmark: Bool
  let runID: String

  init(_ args: [String]) {
    let args = args.filter { $0 != "--" }
    compile = args.contains("-compile")
    benchmark = args.contains("-benchmark")
    if let i = args.firstIndex(of: "-runID"), i + 1 < args.count { runID = args[i + 1] } else { runID = "manual" }
  }
}

/// Where the demo gets its model.
///
/// The model is not published yet, so `just stage 6` assembles the published folder layout (`files.json`,
/// `EmbeddingGemma2.mlpackage/`, `tokenizer/`) from the s4 package into `Examples/PhotoSearchDemo/Model/`, and the
/// app bundles it as the `Model` resource. `EmbeddingGemma.load(.file(...))` then verifies every file against
/// `files.json`, copies it into Application Support and compiles it once (later launches reuse that). Without a
/// bundled folder the app uses the published Hugging Face default (`EmbeddingGemma.defaultRevision`).
enum DemoModel {
  /// Vector length stored in the index.
  static let dimensions = 256

  static var bundledFolder: URL? {
    guard let url = Bundle.main.url(forResource: "Model", withExtension: nil),
          FileManager.default.fileExists(atPath: url.appending(path: "files.json").path) else { return nil }
    return url
  }

  static var source: EmbeddingGemma.Source {
    if let folder = bundledFolder { return .file(folder) }
    return .huggingFace(revision: EmbeddingGemma.defaultRevision)
  }

  static func load(budget: EmbeddingGemma.ImageBudget = .auto) async throws -> EmbeddingGemma {
    try await EmbeddingGemma.load(source, budget: budget, dimensions: dimensions)
  }

  /// A hint for the UI when loading fails.
  static var setupHint: String {
    bundledFolder == nil
      ? "No model is bundled and none is published yet. Run `just stage 6` to bundle the s4 model."
      : "The bundled model failed to load."
  }
}
