import Foundation
import Tokenizers

/// Token ids padded to one of the text functions' fixed lengths.
public struct PaddedTokens: Sendable, Equatable {
  public let ids: [Int32]
  /// 1 for real tokens, 0 for padding.
  public let mask: [Int32]
  public var length: Int { ids.count }
  public var tokenCount: Int { mask.reduce(0) { $0 + Int($1) } }
  /// The Core ML function for this length: `embed_text_<length>`.
  public var functionName: String { "embed_text_\(length)" }
}

/// The Gemma tokenizer (swift-transformers) with the sequence layout s0 recorded:
/// `[<bos>] + tokens(prompt + text) + [<eos>]`, the prompt concatenated as plain text.
public struct TextEncoder: Sendable {
  /// Sequence lengths of the text functions `embed_text_{16,32,64,128}`.
  public static let lengths = [16, 32, 64, 128]
  public static let maxTokens = 128
  static let padID: Int32 = 0  // <pad>

  let tokenizer: any Tokenizer

  /// Loads `tokenizer.json` and `tokenizer_config.json` from `tokenizerFolder`.
  public init(tokenizerFolder: URL) async throws {
    tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
  }

  /// Ids for `prompt + text`, bos and eos included. Longer than `maxTokens` keeps bos and the first
  /// `maxTokens - 2` tokens, then eos (as Hugging Face truncation does with special tokens).
  public func encode(text: String, prompt: String, maxTokens: Int = TextEncoder.maxTokens) -> [Int32] {
    let ids = tokenizer.encode(text: prompt + text, addSpecialTokens: true).map { Int32($0) }
    guard ids.count > maxTokens, let eos = ids.last else { return ids }
    return Array(ids.prefix(maxTokens - 1)) + [eos]
  }

  /// The smallest function length that holds `n` tokens (n <= 128).
  public static func paddedLength(_ n: Int) -> Int { lengths.first { $0 >= n } ?? lengths.last! }

  /// Pads `ids` (at most 128) with `<pad>` to `paddedLength(ids.count)`.
  public static func pad(_ ids: [Int32]) -> PaddedTokens {
    precondition(ids.count <= maxTokens, "encode truncates at \(maxTokens) tokens")
    let n = paddedLength(ids.count), extra = n - ids.count
    return PaddedTokens(ids: ids + Array(repeating: padID, count: extra),
                        mask: Array(repeating: 1, count: ids.count) + Array(repeating: 0, count: extra))
  }
}
