# PhotoSearchDemo

A SwiftUI app that indexes the newest 1,000 photos on the phone with `EmbeddingGemma` + `EmbeddingIndex` and
searches them by text, fully on-device. Launched with `-benchmark` it runs the s6 benchmark instead
(`stages/s6_device/README.md`).

```
mise exec -- xcodegen generate     # or xcodegen generate; writes PhotoSearchDemo.xcodeproj (gitignored)
open PhotoSearchDemo.xcodeproj
```

- Signing: `Base.xcconfig` includes the gitignored `Team.xcconfig` (`DEVELOPMENT_TEAM`), written by
  `uv run --exact --group convert python -m stages.s6_device.team`.
- Model: bundled from `./Model` (gitignored), which `just stage 6` assembles from the s4 package. Without it the app
  uses the published Hugging Face model (`EmbeddingGemma.defaultRevision`, set by s7). After `Model/` changes,
  clean the build folder so Xcode re-signs the app.
