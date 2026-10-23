import Foundation
import Synchronization

/// Process-wide bookkeeping that keeps installs and cleanups from stepping on each other.
enum StoreRegistry {
  /// One install task per (root, repo, revision); concurrent `ensureModel` calls await the same task.
  static let inFlight = Mutex<[String: Task<InstalledModel, any Error>]>([:])
  /// Revision folders loaded by live `EmbeddingGemma` instances in this process (refcounted by path).
  static let inUse = Mutex<[String: Int]>([:])

  static func key(_ url: URL) -> String { url.standardizedFileURL.resolvingSymlinksInPath().path }

  static func isInUse(_ folder: URL) -> Bool { inUse.withLock { ($0[key(folder)] ?? 0) > 0 } }
}

/// Marks a revision folder as in use for as long as it lives (an `EmbeddingGemma` actor holds one).
/// `ModelStore.removeRevisions(except:)` never deletes a leased folder. Only this process is tracked: another
/// process (an app extension) can still remove a folder this one has loaded; a loaded model keeps working, because
/// its files stay open.
package final class RevisionLease: Sendable {
  let path: String

  package init(_ folder: URL) {
    path = StoreRegistry.key(folder)
    StoreRegistry.inUse.withLock { $0[path, default: 0] += 1 }
  }

  deinit {
    StoreRegistry.inUse.withLock { counts in
      let n = (counts[path] ?? 1) - 1
      counts[path] = n > 0 ? n : nil
    }
  }
}

/// An exclusive `flock` on `root/.lock`, so installs and cleanups in other processes (app extensions) sharing the
/// folder wait for each other. Polled without blocking a thread, and cancellable.
enum FileLock {
  /// Runs `body` while holding the lock.
  static func withLock<T: Sendable>(_ url: URL, _ body: () async throws -> T) async throws -> T {
    let fd = open(url.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
    guard fd >= 0 else { throw EmbeddingGemmaError.storageFailed(underlying: POSIXError(.init(rawValue: errno) ?? .EIO)) }
    defer { close(fd) }  // closing the descriptor releases the lock
    while flock(fd, LOCK_EX | LOCK_NB) != 0 {
      let e = errno
      guard e == EWOULDBLOCK || e == EINTR else {
        throw EmbeddingGemmaError.storageFailed(underlying: POSIXError(.init(rawValue: e) ?? .EIO))
      }
      try await Task.sleep(for: .milliseconds(50))
    }
    return try await body()
  }
}

/// `.egcm-install.json`: written into every folder the store creates, so it only ever deletes its own folders.
struct InstallMarker: Codable, Equatable {
  static let fileName = ".egcm-install.json"
  var repo: String
  var revision: String
  var filesSHA256: String?
  var complete: Bool

  static func read(_ folder: URL) -> InstallMarker? {
    guard let data = try? Data(contentsOf: folder.appending(path: fileName)) else { return nil }
    return try? JSONDecoder().decode(InstallMarker.self, from: data)
  }

  func write(_ folder: URL) throws {
    try JSONEncoder().encode(self).write(to: folder.appending(path: Self.fileName), options: .atomic)
  }
}
