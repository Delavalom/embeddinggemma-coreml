import EmbeddingGemma
import Foundation

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

/// The folder handed to `.file(...)` (ADR-0010): the published layout at `root` for one package, or one layout per
/// budget under `root/b70/` and `root/b280/` (each with its own files.json), which `.file` picks by budget.
func publish(_ models: ModelPackages, tokenizer: URL, into root: URL) throws -> URL {
  switch models {
  case .all(let model):
    return try publish(model: model, tokenizer: tokenizer, into: root)
  case .perBudget(let packages):
    for (budget, model) in packages.sorted(by: { $0.key < $1.key }) {
      _ = try publish(model: model, tokenizer: tokenizer, into: root.appending(path: ModelStore.budgetFolder(budget)))
    }
    return root
  }
}
