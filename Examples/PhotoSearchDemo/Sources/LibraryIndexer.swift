import EmbeddingGemma
import EmbeddingIndex
import Foundation
import ImageIO
import Observation
import Photos
import UIKit

/// A photo ready to embed. CGImage is immutable, so passing it between tasks is safe.
struct Photo: @unchecked Sendable {
  let id: String
  let image: CGImage
  let orientation: CGImagePropertyOrientation
}

/// Photos access: permission, the newest image assets, and 960 px images for embedding.
enum PhotoLibrary {
  /// Longest side of the image handed to the model (it resizes to its bucket anyway).
  static let embedPixels: CGFloat = 960

  /// True for full or limited access. Asks the first time.
  static func authorize() async -> Bool {
    let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    return status == .authorized || status == .limited
  }

  /// The newest `limit` images, newest first.
  static func newestImages(limit: Int) -> [PHAsset] {
    let options = PHFetchOptions()
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = limit
    let result = PHAsset.fetchAssets(with: .image, options: options)
    var assets: [PHAsset] = []
    assets.reserveCapacity(result.count)
    result.enumerateObjects { asset, _, _ in assets.append(asset) }
    return assets
  }

  /// The asset at most `maxPixels` on its longest side, or nil if Photos cannot deliver it.
  static func photo(_ asset: PHAsset, maxPixels: CGFloat = embedPixels) async -> Photo? {
    let options = PHImageRequestOptions()
    options.deliveryMode = .highQualityFormat
    options.resizeMode = .fast
    options.isNetworkAccessAllowed = true  // iCloud-optimized libraries keep only small local copies
    let id = asset.localIdentifier
    let image: UIImage? = await withCheckedContinuation { cont in
      PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: maxPixels, height: maxPixels),
                                            contentMode: .aspectFit, options: options) { image, info in
        // With .highQualityFormat the handler runs once, with the final image or an error.
        if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
        cont.resume(returning: image)
      }
    }
    guard let image, let cg = image.cgImage else { return nil }
    return Photo(id: id, image: cg, orientation: CGImagePropertyOrientation(image.imageOrientation))
  }
}

extension CGImagePropertyOrientation {
  init(_ o: UIImage.Orientation) {
    switch o {
    case .up: self = .up
    case .down: self = .down
    case .left: self = .left
    case .right: self = .right
    case .upMirrored: self = .upMirrored
    case .downMirrored: self = .downMirrored
    case .leftMirrored: self = .leftMirrored
    case .rightMirrored: self = .rightMirrored
    @unknown default: self = .up
    }
  }
}

/// Indexes the newest 1,000 photos into an `EmbeddingIndex` file in Documents and answers text queries.
/// Already-indexed photos are skipped, so later launches only embed new ones.
@MainActor @Observable
final class LibraryIndexer {
  enum Phase: Equatable {
    case starting, noPermission, loadingModel, indexing(done: Int, total: Int), ready(count: Int), failed(String)
  }

  static let photoLimit = 1_000
  static let indexURL = URL.documentsDirectory.appending(path: "photos.egix")

  private(set) var phase: Phase = .starting
  @ObservationIgnored private var model: EmbeddingGemma?
  @ObservationIgnored private var index: VectorIndex?

  func start() async {
    guard case .starting = phase else { return }
    guard await PhotoLibrary.authorize() else { phase = .noPermission; return }
    phase = .loadingModel
    do {
      let model = try await DemoModel.load()
      self.model = model
      let index = try Self.openIndex(revision: model.modelRevision)
      self.index = index
      let assets = PhotoLibrary.newestImages(limit: Self.photoLimit)
      let todo = assets.filter { index.vector(for: $0.localIdentifier) == nil }
      for (i, asset) in todo.enumerated() {
        phase = .indexing(done: i, total: todo.count)
        guard let photo = await PhotoLibrary.photo(asset) else { continue }
        let v = try await model.embed(image: photo.image, orientation: photo.orientation)
        try index.upsert(id: photo.id, vector: v)
        if i % 50 == 49 { try index.save() }
      }
      try index.save()
      phase = .ready(count: index.count)
    } catch {
      phase = .failed("\(error.localizedDescription)\n\(DemoModel.setupHint)")
    }
  }

  /// Asset identifiers of the best matches, best first.
  func search(_ text: String, k: Int = 60) async -> [String] {
    guard let model, let index, !text.trimmingCharacters(in: .whitespaces).isEmpty else { return [] }
    guard let q = try? await model.embed(query: text) else { return [] }
    return index.search(q, k: k).map(\.id)
  }

  /// The index file for this model revision; an index from another model is discarded and rebuilt.
  static func openIndex(revision: String) throws -> VectorIndex {
    do {
      return try VectorIndex(url: indexURL, dimensions: DemoModel.dimensions, modelRevision: revision)
    } catch VectorIndexError.revisionMismatch, VectorIndexError.dimensionMismatch {
      try FileManager.default.removeItem(at: indexURL)
      return try VectorIndex(url: indexURL, dimensions: DemoModel.dimensions, modelRevision: revision)
    }
  }
}
