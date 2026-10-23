import Foundation
import Tokenizers

/// Token ids padded to one of the text functions' fixed lengths.
package struct PaddedTokens: Sendable, Equatable {
  package let ids: [Int32]
  /// 1 for real tokens, 0 for padding.
  package let mask: [Int32]
  package var length: Int { ids.count }
  package var tokenCount: Int { mask.reduce(0) { $0 + Int($1) } }
  /// The Core ML function for this length: `embed_text_<length>`.
  package var functionName: String { "embed_text_\(length)" }
}

/// The Gemma tokenizer (swift-transformers) with the sequence layout s0 recorded:
/// `[<bos>] + tokens(prompt + text) + [<eos>]`, the prompt concatenated as plain text.
///
/// **Truncation:** at most 128 tokens: `<bos>`, the first 126 tokens of prompt + text, `<eos>`, the same ids the
/// reference tokenizer gives with `truncation=True, max_length=128` (checked against it in the tests). Before
/// tokenizing, input is cut to `maxCharacters` (16,384) characters so a huge string costs nothing; that only changes
/// the result if the first 126 tokens average more than 128 characters each.
package struct TextEncoder: Sendable {
  /// Sequence lengths of the text functions `embed_text_{16,32,64,128}`.
  package static let lengths = [16, 32, 64, 128]
  package static let maxTokens = 128
  /// 4 x maxTokens x 32 characters per token (a generous bound; Gemma averages about 4).
  package static let maxCharacters = 4 * maxTokens * 32
  static let padID: Int32 = 0  // <pad>

  /// s0's ids for fixture t000 ("screenshots of tweets" with the query prompt); `load` checks them.
  static let referenceText = "task: search result | query: screenshots of tweets"
  static let referenceIDs: [Int32] = [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 75490, 529, 46331, 1]

  let tokenizer: any Tokenizer

  /// Loads `tokenizer.json` and `tokenizer_config.json` from `tokenizerFolder` and checks that it reproduces the
  /// reference ids (`.tokenizerMismatch` otherwise).
  package init(tokenizerFolder: URL) async throws {
    tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
    let found = try encode(text: Self.referenceText, prompt: "")
    guard found == Self.referenceIDs else {
      throw EmbeddingGemmaError.tokenizerMismatch(expected: Self.referenceIDs, found: found)
    }
  }

  /// Ids for `prompt + text`, bos and eos included, truncated to `maxTokens` keeping bos and eos.
  package func encode(text: String, prompt: String, maxTokens: Int = TextEncoder.maxTokens) throws -> [Int32] {
    guard maxTokens >= 2 else { throw EmbeddingGemmaError.contractViolation("maxTokens must be at least 2") }
    let input = prompt + text
    let capped = maxTokens == .max ? input : String(input.prefix(Self.maxCharacters))
    let ids = tokenizer.encode(text: capped, addSpecialTokens: true)
    guard let eos = ids.last, ids.allSatisfy({ (0...Int(Int32.max)).contains($0) }) else {
      throw EmbeddingGemmaError.contractViolation("the tokenizer returned no or out-of-range ids")
    }
    let out = ids.map { Int32($0) }
    guard out.count > maxTokens else { return out }
    return Array(out.prefix(maxTokens - 1)) + [Int32(eos)]
  }

  /// The smallest function length that holds `n` tokens.
  package static func paddedLength(_ n: Int) -> Int? { lengths.first { $0 >= n } }

  /// Pads `ids` (at most 128) with `<pad>` to `paddedLength(ids.count)`.
  package static func pad(_ ids: [Int32]) throws -> PaddedTokens {
    guard let n = paddedLength(ids.count) else {
      throw EmbeddingGemmaError.contractViolation("\(ids.count) tokens is more than \(maxTokens)")
    }
    let extra = n - ids.count
    return PaddedTokens(ids: ids + Array(repeating: padID, count: extra),
                        mask: Array(repeating: 1, count: ids.count) + Array(repeating: 0, count: extra))
  }
}
