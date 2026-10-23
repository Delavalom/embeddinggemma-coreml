import Foundation
import Synchronization

/// Downloads one URL to a local file. Injectable so tests can count or fail fetches.
public protocol FileFetcher: Sendable {
  /// Downloads `url` to `destination` (replacing it). On failure or cancellation, resume data (if the server
  /// supports it) is stored in `resumeFile` and used by the next call.
  func fetch(_ url: URL, to destination: URL, resumeFile: URL) async throws
}

public struct HTTPStatusError: Error, Equatable, CustomStringConvertible {
  public let url: URL
  public let status: Int
  public var description: String { "HTTP \(status) for \(url.absoluteString)" }
}

/// `URLSession` download tasks, resumable through `resumeData` (stored on error and on task cancellation).
/// Works for https:// and file:// URLs.
public struct URLSessionFetcher: FileFetcher {
  let session: URLSession

  public init(session: URLSession = .shared) { self.session = session }

  public func fetch(_ url: URL, to destination: URL, resumeFile: URL) async throws {
    if let data = try? Data(contentsOf: resumeFile) {
      do { return try await attempt(url, destination, resumeFile, resumeData: data) } catch let e as URLError
        where e.code != .cancelled && e.downloadTaskResumeData == nil {
        // Stale or unusable resume data: start over once.
        try? FileManager.default.removeItem(at: resumeFile)
      }
    }
    try await attempt(url, destination, resumeFile, resumeData: nil)
  }

  private func attempt(_ url: URL, _ destination: URL, _ resumeFile: URL, resumeData: Data?) async throws {
    let current = Mutex<URLSessionDownloadTask?>(nil)
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
        let done: @Sendable (URL?, URLResponse?, (any Error)?) -> Void = { tmp, response, error in
          if let error {
            if let data = (error as? URLError)?.downloadTaskResumeData { try? data.write(to: resumeFile) }
            return cont.resume(throwing: error)
          }
          if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            return cont.resume(throwing: HTTPStatusError(url: url, status: http.statusCode))
          }
          guard let tmp else { return cont.resume(throwing: URLError(.cannotCreateFile)) }
          do {  // the temporary file is deleted when this handler returns, so move it now
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: tmp, to: destination)
            try? FileManager.default.removeItem(at: resumeFile)
            cont.resume()
          } catch { cont.resume(throwing: error) }
        }
        let task = resumeData.map { session.downloadTask(withResumeData: $0, completionHandler: done) }
          ?? session.downloadTask(with: url, completionHandler: done)
        current.withLock { $0 = task }
        task.resume()
      }
    } onCancel: {
      current.withLock { $0 }?.cancel { data in if let data { try? data.write(to: resumeFile) } }
    }
  }
}

/// Fetch-and-verify for one manifest entry.
enum VerifiedDownload {
  /// `dir/<entry.path>` with the entry's size and SHA-256. A file that is already there and verifies is kept;
  /// otherwise it is downloaded from `base/<entry.path>` next to its destination, verified, then moved into place.
  /// A mismatch deletes the download and throws `.checksumMismatch(file: entry.path)`.
  static func fetch(_ entry: ManifestEntry, base: URL, into dir: URL, fetcher: any FileFetcher) async throws {
    let dest = dir.appending(path: entry.path)
    if verifies(dest, entry) { return }
    let part = dest.appendingPathExtension("download")
    do {
      try await fetcher.fetch(base.appending(path: entry.path), to: part,
                              resumeFile: dest.appendingPathExtension("resume"))
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw EmbeddingGemmaError.downloadFailed(underlying: error)
    }
    guard verifies(part, entry) else {
      try? FileManager.default.removeItem(at: part)
      throw EmbeddingGemmaError.checksumMismatch(file: entry.path)
    }
    try? FileManager.default.removeItem(at: dest)
    try FileManager.default.moveItem(at: part, to: dest)
  }

  static func verifies(_ file: URL, _ entry: ManifestEntry) -> Bool {
    guard let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize, Int64(size) == entry.size
    else { return false }
    return (try? Checksums.sha256(file: file)) == entry.sha256
  }
}
