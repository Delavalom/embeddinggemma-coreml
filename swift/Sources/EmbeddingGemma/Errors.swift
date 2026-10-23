import Foundation

public enum EmbeddingGemmaError: Error {
  /// A file could not be downloaded (network, HTTP status, or a malformed `files.json`).
  case downloadFailed(underlying: any Error)
  /// A downloaded file's size or SHA-256 differs from `files.json` (or `files.json` from the pinned hash).
  case checksumMismatch(file: String)
  /// The model could not be compiled or loaded.
  case modelLoadFailed(underlying: any Error)
  /// The image has no pixels or could not be drawn.
  case invalidImage
  /// `dimensions` must be in 1...768 (the 768-d vector is truncated, then renormalized).
  case invalidDimensions(Int)
  /// A model function lacks an input or output of spec section 7, or has an unsupported element type.
  case contractViolation(String)
  /// The repo is not `owner/name`, or the revision is not a 40-character commit SHA.
  case invalidSource(String)
  /// Reading or writing the model folder failed (permissions, disk full, ...).
  case storageFailed(underlying: any Error)
  /// The volume has less free space than the install needs (3x the download).
  case insufficientStorage(required: Int64, available: Int64)
  /// The model produced NaN or infinity (for example an FP16 overflow on the Neural Engine).
  case nonFiniteOutput(function: String)
  /// The installed tokenizer does not reproduce the reference token ids.
  case tokenizerMismatch(expected: [Int32], found: [Int32])
}

extension EmbeddingGemmaError: LocalizedError {
  public var errorDescription: String? {
    switch self {
    case .downloadFailed(let e): "Downloading the model failed: \(e.localizedDescription)"
    case .checksumMismatch(let f): "The downloaded file \(f) does not match its checksum."
    case .modelLoadFailed(let e): "Loading the model failed: \(e.localizedDescription)"
    case .invalidImage: "The image could not be read."
    case .invalidDimensions(let n): "dimensions must be between 1 and 768, got \(n)."
    case .contractViolation(let s): "The model does not have the expected interface: \(s)."
    case .invalidSource(let s): "Invalid model source: \(s)."
    case .storageFailed(let e): "Storing the model failed: \(e.localizedDescription)"
    case .insufficientStorage(let r, let a):
      "Not enough free space: the model needs \(ByteCountFormatter.string(fromByteCount: r, countStyle: .file)), "
        + "\(ByteCountFormatter.string(fromByteCount: a, countStyle: .file)) is available."
    case .nonFiniteOutput(let f): "The model function \(f) produced a non-finite value."
    case .tokenizerMismatch: "The tokenizer does not match the model's reference tokenizer."
    }
  }
}

extension EmbeddingGemmaError {
  /// Wraps a file-system error: disk full and the like become `.storageFailed`; our own errors and cancellation
  /// pass through unchanged.
  static func storage(_ error: any Error) -> any Error {
    if error is EmbeddingGemmaError || error is CancellationError { return error }
    return EmbeddingGemmaError.storageFailed(underlying: error)
  }

  /// True for "no space left" errors from Foundation or POSIX.
  static func isOutOfSpace(_ error: any Error) -> Bool {
    let ns = error as NSError
    if ns.domain == NSCocoaErrorDomain && ns.code == NSFileWriteOutOfSpaceError { return true }
    if ns.domain == NSPOSIXErrorDomain && ns.code == Int(ENOSPC) { return true }
    if let inner = ns.userInfo[NSUnderlyingErrorKey] as? NSError { return isOutOfSpace(inner) }
    return false
  }
}
