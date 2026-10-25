import Photos
import SwiftUI
import UIKit

/// Search field over the indexed photos, results in a 3-column grid.
struct SearchView: View {
  @State private var library = LibraryIndexer()
  @State private var query = ""
  @State private var results: [String] = []

  private let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 3)

  var body: some View {
    NavigationStack {
      ScrollView {
        status.padding(.horizontal)
        LazyVGrid(columns: columns, spacing: 2) {
          ForEach(results, id: \.self) { id in
            Thumbnail(assetID: id).aspectRatio(1, contentMode: .fill).clipped()
          }
        }
      }
      .navigationTitle("Photo Search")
      .searchable(text: $query, prompt: "receipts, a dog on the beach, ...")
      .onSubmit(of: .search) { Task { results = await library.search(query) } }
      .task { await library.start() }
    }
  }

  @ViewBuilder private var status: some View {
    switch library.phase {
    case .starting: ProgressView()
    case .noPermission: Text("Allow access to Photos in Settings to search them.")
    case .loadingModel: ProgressView("Preparing the model (first launch compiles it)")
    case .indexing(let done, let total): ProgressView("Indexing \(done) of \(total)", value: Double(done),
                                                      total: Double(max(total, 1)))
    case .ready(let count): Text("\(count) photos indexed").font(.footnote).foregroundStyle(.secondary)
    case .failed(let message): Text(message).foregroundStyle(.red)
    }
  }
}

/// One result cell: a small square image of the asset.
struct Thumbnail: View {
  let assetID: String
  @State private var image: UIImage?

  var body: some View {
    Color.secondary.opacity(0.15)
      .overlay { if let image { Image(uiImage: image).resizable().scaledToFill() } }
      .task(id: assetID) { image = await Self.load(assetID) }
  }

  static func load(_ id: String) async -> UIImage? {
    guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil).firstObject else { return nil }
    let options = PHImageRequestOptions()
    options.deliveryMode = .highQualityFormat
    options.resizeMode = .fast
    options.isNetworkAccessAllowed = true
    return await withCheckedContinuation { cont in
      PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 300, height: 300),
                                            contentMode: .aspectFill, options: options) { image, info in
        if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
        cont.resume(returning: image)
      }
    }
  }
}
