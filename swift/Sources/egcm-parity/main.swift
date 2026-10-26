// egcm-parity: the s5 CLI. Runs the Swift package end to end on every fixture and writes what it computed, so
// stages/s5_swift/run.py can compare it with s0. It goes through the same path an app does: a published folder
// (package + tokenizer + files.json) -> ModelStore (verify, compile) -> EmbeddingGemma.load -> embed.
//
//   swift run -c release egcm-parity --model artifacts/s4_compress/EmbeddingGemma2.mlpackage \
//     --tokenizer <dir with tokenizer.json> --fixtures fixtures --out artifacts/s5_swift/parity.json
//
// For each fixture text: the token ids (prompt, <bos>, <eos>) and the 768-d vector. For each fixture image at
// budgets 70 and 280: the bucket, the sha256 of the uint8 bucket image, a pixel sample and the 768-d vector; with
// --pixels <dir>, also the whole bucket image as <dir>/<id>_<budget>.png (lossless), for s5's references (ADR-0008).
// Vectors are L2-normalized (dimensions 768, so nothing is truncated); cosine does not depend on the norm.

import CoreML
import EmbeddingGemma
import Foundation

func log(_ s: String) { FileHandle.standardError.write(Data("[egcm-parity] \(s)\n".utf8)) }

/// A folder in the published layout, built from the local package and tokenizer (clones on APFS, so no extra space).
func publish(model: URL, tokenizer: URL, into root: URL) throws -> URL {
  let fm = FileManager.default
  let tok = root.appending(path: ModelStore.tokenizerDir)
  try fm.createDirectory(at: tok, withIntermediateDirectories: true)
  try fm.copyItem(at: model.resolvingSymlinksInPath(), to: root.appending(path: ModelStore.packageName))
  for name in ["tokenizer.json", "tokenizer_config.json"] {
    // The HF cache stores these as symlinks into blobs/; copy the file itself.
    try fm.copyItem(at: tokenizer.appending(path: name).resolvingSymlinksInPath(), to: tok.appending(path: name))
  }
  _ = try FilesManifest.write(root: root)
  return root
}

func imageResult(id: String, url: URL, budget: Int, samples: Int, model: EmbeddingGemma?,
                 pixels: URL? = nil) async throws -> ImageResult {
  let (img, orientation) = try Fixtures.image(url)
  let pre = try ImagePreprocessor(budget: budget).preprocess(img, orientation: orientation)
  let (h, w) = (pre.bucket.height, pre.bucket.width)
  if let pixels {
    try writePNG(planar: pre.planar, height: h, width: w, to: pixels.appending(path: "\(id)_\(budget).png"))
  }
  let emb = try await model?.embed(image: img, orientation: orientation)
  return ImageResult(id: id, budget: budget, bucket: pre.bucket.name.rawValue, height: h, width: w,
                     sha256: interleavedSHA256(planar: pre.planar, height: h, width: w),
                     pixel_sample: sampleIndices(pre.planar.count, samples).map { pre.planar[$0] }, emb: emb)
}

func run(_ o: Options) async throws {
  let work = FileManager.default.temporaryDirectory.appending(path: "egcm-parity-\(UUID().uuidString)")
  defer { try? FileManager.default.removeItem(at: work) }
  let started = Date()
  let folder = try publish(model: o.model, tokenizer: o.tokenizer, into: work.appending(path: "published"))
  let store = ModelStore(root: work.appending(path: "store"))
  var models: [Int: EmbeddingGemma] = [:]
  for (budget, b) in [(70, EmbeddingGemma.ImageBudget.tokens70), (280, .tokens280)] {
    models[budget] = try await EmbeddingGemma.load(.file(folder), budget: b, dimensions: 768, store: store,
                                                   computeUnits: o.computeUnits)
  }
  log("model installed and compiled (\(Int(Date().timeIntervalSince(started))) s)")
  let encoder = try await TextEncoder(tokenizerFolder: folder.appending(path: ModelStore.tokenizerDir))

  var texts = try Fixtures.jsonl(o.fixtures.appending(path: "texts.jsonl"), as: TextRow.self)
  if o.limit > 0 { texts = Array(texts.prefix(o.limit)) }
  var textOut: [String: TextResult] = [:]
  for t in texts {
    let task = try Fixtures.task(t.kind)
    textOut[t.id] = TextResult(ids: try encoder.encode(text: t.text, prompt: task.prompt),
                               emb: try await models[70]!.embed(text: t.text, task: task))
  }
  log("\(textOut.count) texts")

  var images = try Fixtures.jsonl(o.fixtures.appending(path: "images.jsonl"), as: ImageRow.self)
  if o.limit > 0 { images = Array(images.prefix(o.limit)) }
  var imageOut: [String: ImageResult] = [:]
  if let dir = o.pixels { try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true) }
  for (n, row) in images.enumerated() {
    for budget in [70, 280] {
      imageOut["\(row.id)_\(budget)"] = try await imageResult(id: row.id, url: o.fixtures.appending(path: row.file),
                                                              budget: budget, samples: o.samples, model: models[budget],
                                                              pixels: o.pixels)
    }
    if (n + 1) % 50 == 0 { log("\(n + 1)/\(images.count) images") }
  }

  var synthOut: [String: ImageResult] = [:]
  if let dir = o.synthetic {
    let pngs = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "png" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    for url in pngs {
      let id = url.deletingPathExtension().lastPathComponent
      for budget in [70, 280] {
        synthOut["\(id)_\(budget)"] = try await imageResult(id: id, url: url, budget: budget, samples: o.samples,
                                                            model: nil)
      }
    }
    log("\(pngs.count) synthetic images")
  }

  let units = switch o.computeUnits {
  case .cpuOnly: "cpuOnly"
  case .all: "all"
  default: "cpuAndNeuralEngine"
  }
  let result = ParityOutput(model: o.model.path, model_revision: models[70]!.modelRevision, compute_units: units,
                            samples: o.samples, texts: textOut, images: imageOut, synthetic: synthOut)
  let enc = JSONEncoder()
  enc.outputFormatting = [.sortedKeys]
  try FileManager.default.createDirectory(at: o.out.deletingLastPathComponent(), withIntermediateDirectories: true)
  try enc.encode(result).write(to: o.out, options: .atomic)
  log("wrote \(o.out.path) in \(Int(Date().timeIntervalSince(started))) s")
}

do {
  try await run(try Options.parse(Array(CommandLine.arguments.dropFirst())))
} catch let e as ParityError {
  log("error: \(e)")
  exit(2)
} catch {
  log("error: \(error)")
  exit(1)
}
