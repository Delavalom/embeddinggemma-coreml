import Foundation
import Synchronization

/// Downloads one URL to a local file. Injectable so tests can count or fail fetches.
package protocol FileFetcher: Sendable {
  /// Downloads `url` to `destination` (replacing it). On failure or cancellation, resume data (if the server
  /// supports it) is stored in `resumeFile` and used by the next call. `progress` gets the bytes received so far
  /// for this file, on an arbitrary thread.
  func fetch(_ url: URL, to destination: URL, resumeFile: URL, progress: (@Sendable (Int64) -> Void)?) async throws
}

package struct HTTPStatusError: Error, Equatable, CustomStringConvertible {
  package let url: URL
  package let status: Int
  package var description: String { "HTTP \(status) for \(url.absoluteString)" }
}

/// `URLSession` download tasks, resumable through `resumeData` (stored atomically on error and on cancellation).
/// Works for https:// and file:// URLs. A default session, not a background one, so it works inside a library.
package struct URLSessionFetcher: FileFetcher {
  let session: URLSession

  package init(session: URLSession = .shared) { self.session = session }

  package func fetch(_ url: URL, to destination: URL, resumeFile: URL,
                     progress: (@Sendable (Int64) -> Void)?) async throws {
    try await Self.withResume(resumeFile) { data in
      try await attempt(url, destination, resumeFile, resumeData: data, progress: progress)
    }
  }

  /// The resume policy: with stored resume data, try it first; any failure of that attempt other than
  /// cancellation (bad resume data, HTTP 403/410/416 from an expired or changed URL, ...) deletes the resume file
  /// and starts over from scratch, once.
  static func withResume(_ resumeFile: URL, _ attempt: (Data?) async throws -> Void) async throws {
    if let data = try? Data(contentsOf: resumeFile) {
      do { return try await attempt(data) } catch is CancellationError { throw CancellationError() } catch {
        try? FileManager.default.removeItem(at: resumeFile)
      }
    }
    try await attempt(nil)
  }

  private func attempt(_ url: URL, _ destination: URL, _ resumeFile: URL, resumeData: Data?,
                       progress: (@Sendable (Int64) -> Void)?) async throws {
    let current = Mutex<URLSessionDownloadTask?>(nil)
    let saveResume: @Sendable (Data?) -> Void = { data in
      if let data { try? data.write(to: resumeFile, options: .atomic) }
    }
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
        let observation = Mutex<NSKeyValueObservation?>(nil)
        let done: @Sendable (URL?, URLResponse?, (any Error)?) -> Void = { tmp, response, error in
          observation.withLock { $0?.invalidate(); $0 = nil }
          if let error {
            let urlError = error as? URLError
            saveResume(urlError?.downloadTaskResumeData)
            return cont.resume(throwing: urlError?.code == .cancelled ? CancellationError() : error)
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
        if let progress {
          let o = task.progress.observe(\.completedUnitCount) { p, _ in progress(p.completedUnitCount) }
          observation.withLock { $0 = o }
        }
        current.withLock { $0 = task }
        task.resume()
        // A cancel that arrived before `current` was set found nothing to cancel; do it now.
        if Task.isCancelled { task.cancel(byProducingResumeData: saveResume) }
      }
    } onCancel: {
      current.withLock { $0 }?.cancel(byProducingResumeData: saveResume)
    }
  }
}

/// Fetch-and-verify for one manifest entry.
enum VerifiedDownload {
  /// `dir/<entry.path>` with the entry's size and SHA-256. A file that is already there and verifies is kept;
  /// otherwise it is downloaded from `base/<entry.path>` next to its destination, verified, then moved into place.
  /// A mismatch deletes the download and throws `.checksumMismatch(file: entry.path)`.
  static func fetch(_ entry: ManifestEntry, base: URL, into dir: URL, fetcher: any FileFetcher,
                    progress: (@Sendable (Int64) -> Void)? = nil) async throws {
    let dest = dir.appending(path: entry.path)
    if verifies(dest, entry) { return }
    let part = dest.appendingPathExtension("download")
    do {
      try await fetcher.fetch(base.appending(path: entry.path), to: part,
                              resumeFile: dest.appendingPathExtension("resume"), progress: progress)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw EmbeddingGemmaError.downloadFailed(underlying: error)
    }
    guard verifies(part, entry) else {
      try? FileManager.default.removeItem(at: part)
      throw EmbeddingGemmaError.checksumMismatch(file: entry.path)
    }
    do {
      try? FileManager.default.removeItem(at: dest)
      try FileManager.default.moveItem(at: part, to: dest)
    } catch { throw EmbeddingGemmaError.storage(error) }
  }

  static func verifies(_ file: URL, _ entry: ManifestEntry) -> Bool {
    guard let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize, Int64(size) == entry.size
    else { return false }
    return (try? Checksums.sha256(file: file)) == entry.sha256
  }
}
