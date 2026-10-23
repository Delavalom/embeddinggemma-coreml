"""CC0 / public-domain photos from Wikimedia Commons for the fixtures.

The chosen files are pinned in `fixtures/commons_sources.json` the first time they are picked, so a rebuild
downloads the same files instead of searching again (pass `requery=True` to search afresh). Requests are
throttled to about one per second and carry a contact User-Agent, per the Wikimedia API etiquette.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "embeddinggemma-coreml fixtures builder (hi@delavalom.com)"
ALLOWED_LICENSES = {"CC0", "Public domain"}
THUMB_WIDTH = 1920  # a standard Commons thumbnail width; downscaled locally to MAX_SIDE
MIN_SHORT_SIDE, MIN_LONG_SIDE = 900, 1200
# Artworks and scans of prints are CC0 too, but we want camera photos.
NOT_A_PHOTO = re.compile(r"\bMET\b|DP-?\d|painting|drawing|engraving|lithograph|etching|illustration|poster|"
                         r"\bmap\b|diagram|logo|icon|svg", re.I)

# category -> (count, search phrases, in order). 100 photos in total.
CATEGORIES: dict[str, tuple[int, tuple[str, ...]]] = {
    "people": (13, ("people walking street", "portrait of a man", "portrait of a woman", "group of people")),
    "dogs": (13, ("dog", "puppy")),
    "cats": (12, ("cat", "kitten")),
    "food": (13, ("food dish", "meal plate", "breakfast", "fruit")),
    "landscapes": (13, ("landscape mountains", "beach sunset", "forest lake", "countryside")),
    "buildings": (12, ("building architecture", "church", "skyscraper", "house")),
    "documents": (12, ("handwritten letter", "document paper", "book page", "newspaper")),
    "cars": (12, ("car", "automobile street", "parked car")),
}

# Uploads often come in numbered series of near-identical shots ("... kitten 02.jpg", "... kitten 03.jpg").
# At most this many files share a series key, so a category is not one burst of the same scene.
MAX_PER_SERIES = 2

_last_request = 0.0


def series_key(title: str) -> str:
    """The first two words of a file title, ignoring digits and punctuation."""
    return " ".join(re.findall(r"[^\W\d_]+", title.lower().removeprefix("file:"))[:2])


def _get(url: str, tries: int = 5) -> bytes:
    """GET with a contact User-Agent, at most one request per second, retrying 429 and 5xx with backoff."""
    global _last_request
    for attempt in range(tries):
        wait = 1.0 - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(phrase: str, limit: int = 50) -> list[dict]:
    """CC0 / public-domain JPEG photos for a phrase, in search-rank order."""
    q = {
        "action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrnamespace": "6",
        "gsrlimit": str(limit), "gsrsearch": f"{phrase} filetype:bitmap incategory:CC-Zero",
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": str(THUMB_WIDTH),
        "iiextmetadatafilter": "LicenseShortName|Artist",
    }
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(q)))
    out = []
    for page in sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        w, h = info.get("width", 0), info.get("height", 0)
        if (info.get("mime") != "image/jpeg" or lic not in ALLOWED_LICENSES or NOT_A_PHOTO.search(page["title"])
                or min(w, h) < MIN_SHORT_SIDE or max(w, h) < MIN_LONG_SIDE):
            continue
        out.append({
            "title": page["title"],
            "source_url": info["descriptionurl"],
            "download_url": info.get("thumburl") or info["url"],
            "license": lic,
            "author": _strip_html(meta.get("Artist", {}).get("value", "")) or "unknown",
            "original_size": [h, w],
        })
    return out


def pick_sources(pinned: Path, requery: bool = False) -> list[dict]:
    """The 100 Commons photos, by category. Read from `pinned` when it exists, else searched and written there."""
    if pinned.is_file() and not requery:
        return json.loads(pinned.read_text())
    chosen: list[dict] = []
    seen: set[str] = set()
    series: dict[str, int] = {}
    for category, (count, phrases) in CATEGORIES.items():
        got = []
        for phrase in phrases:
            for c in search(phrase):
                key = series_key(c["title"])
                if len(got) < count and c["title"] not in seen and series.get(key, 0) < MAX_PER_SERIES:
                    seen.add(c["title"])
                    series[key] = series.get(key, 0) + 1
                    got.append({**c, "category": category})
            if len(got) == count:
                break
        if len(got) < count:
            raise RuntimeError(f"only {len(got)} CC0 photos found for {category!r}; add search phrases")
        chosen += got
    pinned.write_text(json.dumps(chosen, indent=1, ensure_ascii=False) + "\n")
    return chosen


def download_photo(src: dict, max_side: int, cache: Path | None = None) -> Image.Image:
    """The photo as an upright RGB image whose longer side is at most max_side.

    Downloaded bytes are kept in `cache` (gitignored, under artifacts/) so rebuilding does not download again."""
    if src["license"] not in ALLOWED_LICENSES:
        raise ValueError(f"{src['title']} has license {src['license']!r}")
    cached = cache / (hashlib.sha256(src["download_url"].encode()).hexdigest() + ".jpg") if cache else None
    if cached and cached.is_file():
        data = cached.read_bytes()
    else:
        data = _get(src["download_url"])
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    return im
