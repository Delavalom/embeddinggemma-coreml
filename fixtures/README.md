# Fixtures

The texts and images every stage compares on. Built by `python -m stages.s0_reference.build_fixtures`;
`MANIFEST.sha256` pins every file here (CI and `just stage 0` check it). Do not edit generated files by hand.

| File | What it is |
|---|---|
| `texts.jsonl` | 200 hand-written texts: 120 app queries (`kind: query`), 79 one-sentence documents and one paragraph of exactly 128 tokens (`kind: document`) |
| `images/` | 200 images, ids `i000`-`i199` (categories below) |
| `images.jsonl` | one row per image: `id`, `file`, `category`, `source_url`, `license`, `author`, `height`, `width`, plus `title` (Commons), `derived_from` (edits of a Commons photo), `kind` and `font` (synthetic) |
| `commons_sources.json` | the 100 pinned Commons files: URLs, license, author, Commons `sha1` and `timestamp`, and the `sha256` of the bytes downloaded |
| `MANIFEST.sha256` | sha256 of every other file here |
| `README.md` | this file (hand-written) |

## Images

- 100 photos from Wikimedia Commons whose `LicenseShortName` is `CC0` or `Public domain`: people, dogs, cats, food,
  landscapes, buildings, documents, cars. Longer side at most 1600 px, JPEG quality 85.
- 30 screenshots, 20 memes, 10 receipts rendered locally with macOS system fonts (synthetic, CC0).
- 20 blurry, 10 dark and 10 near-duplicate images made from the Commons photos (same license as their source).

**Colour:** every image is 8-bit RGB in **sRGB**, with no ICC profile and no EXIF. Commons photos that carried
an ICC profile (Adobe RGB, Display P3, ProPhoto and others) were converted to sRGB when built. The reference
processor ignores colour profiles, so the Swift preprocessor (s5) must draw images into an sRGB context to
reproduce these pixels.

## Licensing and people

CC0 and public-domain status covers copyright only. It does not clear personality or publicity rights of people
shown in the photos, nor trademarks. The people photos are used only to test that our conversion matches the
original model numerically (parity testing); they are not used for marketing, training, or any representation
about the people in them. If a person shown asks for removal, replace the file and rebuild.
