# group: reference
"""Build fixtures/: 200 images, images.jsonl and MANIFEST.sha256. texts.jsonl is hand-written and only read here.

    uv run --exact --group reference python -m stages.s0_reference.build_fixtures [--requery] [--accept-changes]

It builds into artifacts/fixtures_build/ and swaps the result into fixtures/ only at the end. If the new
MANIFEST.sha256 differs from the committed one it lists the changed files and refuses, unless --accept-changes.
Hand-written files (texts.jsonl, README.md) are copied, never generated.

Images (ids i000-i199):
  100 CC0 / public-domain Commons photos (8 categories, see commons.CATEGORIES), longer side <= 1600 px, JPEG quality 85
   30 screenshots (chat, tweet, settings) at 1170x2532, PNG
   20 memes (a Commons photo cropped to 1:1, 16:9, 4:5, 4:3 or 2:1 with Impact-style captions)
   20 blurry (Gaussian radius 6-12), 10 dark (brightness x 0.15), from Commons photos, longer side <= 1200 px
   10 near-duplicates (5% crop of a Commons photo, JPEG quality 60)
   10 receipts at 1000x2400
JPEG quality 92 unless stated. Seeded: the same Commons files (pinned in commons_sources.json) and the same
fonts give the same bytes. Photos are converted to sRGB and stored without ICC profiles or EXIF.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path

from egcm.fixtures import MANIFEST, load_texts, write_manifest
from egcm.paths import ARTIFACTS, FIXTURES
from stages.s0_reference import commons, synth

SEED = 20261022
MAX_SIDE = 1600  # Commons photos
# At quality 92 the 100 photos alone are 53 MB, so they are saved at 85 to keep fixtures/ under MAX_TOTAL_MB.
PHOTO_QUALITY = 85
DERIVED_MAX_SIDE = 1200  # memes, blurry and dark images, to keep fixtures/ under MAX_TOTAL_MB
MAX_TOTAL_MB = 60
HAND_WRITTEN = ("texts.jsonl", "README.md")
SOURCES = "commons_sources.json"
SYNTHETIC = {"license": "CC0", "author": "egcm build_fixtures (synthetic)", "source_url": None}

MEME_CAPTIONS = [
    ("when you finally clean", "your camera roll"), ("me pretending", "to read the receipt"),
    ("nobody:", "my phone at 3am taking pocket photos"), ("one does not simply", "delete 4000 screenshots"),
    ("this is fine", "only 98% storage used"), ("monday morning", "before coffee"),
    ("i said one photo", "took forty"), ("when the wifi", "finally connects"),
    ("trust me", "i'm a professional"), ("weekend plans", "vs reality"),
    ("that face when", "storage is full"), ("me explaining", "why i need 12 selfies"),
    ("when someone says", "just one more"), ("how it started", "how it's going"),
    ("i have no idea", "what i'm doing"), ("is this", "a meme"),
    ("when the food", "arrives"), ("expectation", "reality"),
    ("me on vacation", "taking pictures of everything"), ("not sure if blurry", "or art"),
]
MEME_ASPECTS = (1.0, 16 / 9, 0.8, 4 / 3, 2.0)
STORES = ["CORNER MARKET", "FRESH FOODS", "CITY PHARMACY", "BLUE DOOR CAFE", "HARDWARE PLUS",
          "GREEN GROCER", "PIZZA PALACE", "BOOK NOOK", "QUICK STOP GAS", "SUNRISE BAKERY"]
ITEMS = ["MILK 1L", "EGGS DOZEN", "BREAD WHOLE WHEAT", "COFFEE BEANS", "BANANAS", "APPLES GALA", "CHEDDAR",
         "PASTA PENNE", "TOMATO SAUCE", "OLIVE OIL", "YOGURT", "CHICKEN BREAST", "RICE 2KG", "ORANGE JUICE",
         "TOOTHPASTE", "SHAMPOO", "PAPER TOWELS", "BATTERIES AA", "CROISSANT", "LATTE LARGE", "SPARKLING WATER",
         "DARK CHOCOLATE", "AVOCADO", "SPINACH", "LIGHT BULB", "DISH SOAP", "CEREAL", "BUTTER", "HONEY", "TEA"]


def _rng(image_id: str) -> random.Random:
    return random.Random(f"{SEED}:{image_id}")


def _save(im, images_dir: Path, image_id: str, category: str, fmt: str = "JPEG", quality: int = 92) -> str:
    name = f"{image_id}_{category}.{'png' if fmt == 'PNG' else 'jpg'}"
    if fmt == "PNG":
        im.save(images_dir / name, "PNG", optimize=True)
    else:
        im.save(images_dir / name, "JPEG", quality=quality, optimize=True)
    return f"images/{name}"


def build(root: Path = FIXTURES, requery: bool = False, accept_changes: bool = False,
          staging: Path = ARTIFACTS / "fixtures_build") -> list[str]:
    """Rebuild fixtures/ and return the paths whose content changed (relative to fixtures/)."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in HAND_WRITTEN:
        if (root / name).is_file():
            shutil.copy2(root / name, staging / name)
    sources = copy.deepcopy(commons.pick_sources(root / SOURCES, requery=requery))
    commons.fill_revisions(sources)
    rows = render(staging, sources)
    (staging / SOURCES).write_text(json.dumps(sources, indent=1, ensure_ascii=False) + "\n")
    write_manifest(staging)
    _check(staging, rows)
    changed = manifest_changes(root / MANIFEST, staging / MANIFEST)
    if changed and (root / MANIFEST).is_file() and not accept_changes:
        shutil.rmtree(staging)
        raise SystemExit(f"{len(changed)} fixture files would change ({', '.join(changed[:10])}"
                         f"{', ...' if len(changed) > 10 else ''}). Rerun with --accept-changes to replace them.")
    if root.exists():
        old = root.with_name(root.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        root.rename(old)
        shutil.move(str(staging), str(root))
        shutil.rmtree(old)
    else:
        shutil.move(str(staging), str(root))
    print(f"{len(changed)} files changed" + (f": {', '.join(changed)}" if changed else ""))
    return changed


def manifest_changes(old: Path, new: Path) -> list[str]:
    """Paths added, removed or with a different hash between two manifests."""
    def entries(p: Path) -> set[tuple[str, str]]:
        if not p.is_file():
            return set()
        return {tuple(line.split("  ", 1)[::-1]) for line in p.read_text().splitlines() if line.strip()}

    return sorted({path for path, _ in entries(old) ^ entries(new)})


def render(root: Path, sources: list[dict]) -> list[dict]:
    """Write images/ and images.jsonl under root. Fills in each source's sha256."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    docs = [t["text"] for t in load_texts(root) if t["kind"] == "document"]
    cache = ARTIFACTS / "fixture_cache"
    rows: list[dict] = []
    photos: list[tuple[dict, object]] = []  # (row, image) of the Commons photos, reused by the edits

    def add(im, category, extra, fmt="JPEG", quality=92):
        image_id = f"i{len(rows):03d}"
        row = {"id": image_id, "file": _save(im, images_dir, image_id, category, fmt, quality), "category": category,
               **extra, "height": im.height, "width": im.width}
        rows.append(row)
        print(f"{image_id} {category:16s} {im.width}x{im.height} {extra.get('license')}", flush=True)
        return row

    for src in sources:
        im = commons.download_photo(src, MAX_SIDE, cache)
        extra = {"source_url": src["source_url"], "license": src["license"], "author": src["author"],
                 "title": src["title"]}
        row = add(im, src["category"], extra, quality=PHOTO_QUALITY)
        photos.append((row, im))

    kinds = (("chat", synth.chat_screenshot), ("tweet", synth.tweet_screenshot),
             ("settings", synth.settings_screenshot))
    for i in range(30):
        kind, fn = kinds[i % 3]
        rng = _rng(f"i{len(rows):03d}")
        im = fn(rng, rng.sample(docs, 8))
        add(im, "screenshots", {**SYNTHETIC, "kind": kind, "font": synth.font_name("sans")}, fmt="PNG")

    def pick(rng):
        return photos[rng.randrange(len(photos))]

    meme_sources = [p for p in photos if p[0]["category"] in ("people", "dogs", "cats")]
    for i in range(20):
        rng = _rng(f"i{len(rows):03d}")
        src_row, photo = meme_sources[rng.randrange(len(meme_sources))]
        top, bottom = MEME_CAPTIONS[i]
        im = synth.meme(rng, photo, MEME_ASPECTS[i % len(MEME_ASPECTS)], top, bottom, DERIVED_MAX_SIDE)
        add(im, "memes", _derived(src_row, font=synth.font_name("impact")))
    for _ in range(20):
        rng = _rng(f"i{len(rows):03d}")
        src_row, photo = pick(rng)
        im, radius = synth.blurry(rng, photo, DERIVED_MAX_SIDE)
        add(im, "blurry", _derived(src_row, blur_radius=radius))
    for _ in range(10):
        rng = _rng(f"i{len(rows):03d}")
        src_row, photo = pick(rng)
        add(synth.dark(photo, DERIVED_MAX_SIDE), "dark", _derived(src_row, brightness=0.15))
    for _ in range(10):
        rng = _rng(f"i{len(rows):03d}")
        src_row, photo = pick(rng)
        add(synth.near_duplicate(rng, photo), "near_duplicates", _derived(src_row), quality=60)
    for i in range(10):
        rng = _rng(f"i{len(rows):03d}")
        items = [(name, round(rng.uniform(0.5, 24.0), 2)) for name in rng.sample(ITEMS, rng.randint(6, 14))]
        add(synth.receipt(rng, items, STORES[i]), "receipts", {**SYNTHETIC, "font": synth.font_name("mono")})

    with open(root / "images.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def _derived(src_row: dict, **extra) -> dict:
    return {"source_url": src_row["source_url"], "license": src_row["license"], "author": src_row["author"],
            "derived_from": src_row["id"], **extra}


def _check(root: Path, rows: list[dict]) -> None:
    assert len(rows) == 200, len(rows)
    bad = [r["id"] for r in rows if r["license"] not in commons.ALLOWED_LICENSES]
    assert not bad, f"rows without a CC0 / public-domain license: {bad}"
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    largest = max((p.stat().st_size, p.name) for p in (root / "images").iterdir())
    print(f"{len(rows)} images, fixtures/ is {total / 1e6:.1f} MB, largest file {largest[1]} {largest[0] / 1e6:.2f} MB")
    assert total < MAX_TOTAL_MB * 1e6, f"fixtures/ is {total / 1e6:.1f} MB, over {MAX_TOTAL_MB} MB"
    assert largest[0] < 5e6, f"{largest[1]} is over the 5 MB per-file limit"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--requery", action="store_true", help="search Commons again instead of commons_sources.json")
    ap.add_argument("--accept-changes", action="store_true", help="replace fixtures even if the manifest changes")
    args = ap.parse_args()
    build(requery=args.requery, accept_changes=args.accept_changes)


if __name__ == "__main__":
    main()
