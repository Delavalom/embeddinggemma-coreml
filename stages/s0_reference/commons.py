"""CC0 / public-domain photos from Wikimedia Commons for the fixtures.

The chosen files are pinned in `fixtures/commons_sources.json` (title, URL, license, author, the file's sha1 and
timestamp on Commons, and the sha256 of the bytes we downloaded). A rebuild downloads the same URLs and refuses
bytes whose sha256 differs. `requery=True` searches afresh. Requests are throttled to about one per second,
send `maxlag=5`, retry transient failures honoring Retry-After, and carry a descriptive User-Agent (set
`EGCM_CONTACT` to append a contact address; none is stored in the source).
"""

from __future__ import annotations

import hashlib
import html
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSES = {"CC0", "Public domain"}
THUMB_WIDTH = 1920  # a standard Commons thumbnail width; downscaled locally to MAX_SIDE
MIN_SHORT_SIDE, MIN_LONG_SIDE = 900, 1200
TRIES = 5
# Artworks and scans of prints are CC0 too, but we want camera photos.
NOT_A_PHOTO = re.compile(r"\bMET\b|DP-?\d|painting|drawing|engraving|lithograph|etching|illustration|poster|"
                         r"\bmap\b|diagram|logo|icon|svg", re.I)
# Uploads often come in numbered series of near-identical shots ("... kitten 02.jpg", "... kitten 03.jpg").
# At most this many files share a series key, so a category is not one burst of the same scene.
MAX_PER_SERIES = 2

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

_RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionResetError, http.client.IncompleteRead)
_last_request = 0.0


def user_agent() -> str:
    ua = (f"embeddinggemma-coreml-fixtures/1.0 (https://github.com/Delavalom/embeddinggemma-coreml) "
          f"python-urllib/{sys.version_info.major}.{sys.version_info.minor}")
    contact = os.environ.get("EGCM_CONTACT", "").strip()
    return f"{ua} {contact}" if contact else ua


def _retry_after(headers, default: float) -> float:
    try:
        return max(0.0, float(headers.get("Retry-After"))) if headers and headers.get("Retry-After") else default
    except (TypeError, ValueError):
        return default


def _get(url: str, tries: int = TRIES, opener: Callable = urllib.request.urlopen,
         sleep: Callable[[float], None] = time.sleep) -> bytes:
    """GET with the User-Agent, at most one request per second.

    Retries HTTP 429 and 5xx (honoring Retry-After), URLError, timeouts, connection resets and truncated reads,
    with exponential backoff. Other HTTP errors, and the last failure, are raised."""
    global _last_request
    for attempt in range(tries):
        wait = 1.0 - (time.monotonic() - _last_request)
        if wait > 0:
            sleep(wait)
        _last_request = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
        try:
            with opener(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:  # a subclass of URLError, so it must come first
            if (e.code != 429 and e.code < 500) or attempt == tries - 1:
                raise
            sleep(_retry_after(e.headers, 2.0 ** (attempt + 1)))
        except _RETRYABLE:
            if attempt == tries - 1:
                raise
            sleep(2.0 ** (attempt + 1))
    raise AssertionError("unreachable")


def _api(params: dict, get: Callable[[str], bytes] = _get, sleep: Callable[[float], None] = time.sleep) -> dict:
    """One API call with maxlag=5; a maxlag error is retried after a pause."""
    q = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **params}
    for attempt in range(TRIES):
        data = json.loads(get(API + "?" + urllib.parse.urlencode(q)))
        if data.get("error", {}).get("code") != "maxlag":
            if "error" in data:
                raise RuntimeError(f"Commons API error: {data['error']}")
            return data
        sleep(5.0 * (attempt + 1))
    raise RuntimeError("Commons API stayed lagged (maxlag) after retries")


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def parse_search(data: dict) -> list[dict]:
    """CC0 / public-domain JPEG photos from an imageinfo API response, in search-rank order."""
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
            "sha1": info.get("sha1"),
            "timestamp": info.get("timestamp"),
        })
    return out


def search(phrase: str, limit: int = 50) -> list[dict]:
    return parse_search(_api({
        "generator": "search", "gsrnamespace": "6", "gsrlimit": str(limit),
        "gsrsearch": f"{phrase} filetype:bitmap incategory:CC-Zero", "prop": "imageinfo",
        "iiprop": "url|size|mime|sha1|timestamp|extmetadata", "iiurlwidth": str(THUMB_WIDTH),
        "iiextmetadatafilter": "LicenseShortName|Artist",
    }))


def series_key(title: str) -> str:
    """The first two words of a file title, ignoring digits and punctuation."""
    return " ".join(re.findall(r"[^\W\d_]+", title.lower().removeprefix("file:"))[:2])


def pick_sources(pinned: Path | None, requery: bool = False) -> list[dict]:
    """The 100 Commons photos, by category: read from `pinned` when it exists (unless requery), else searched."""
    if pinned is not None and pinned.is_file() and not requery:
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
    return chosen


def fill_revisions(sources: list[dict]) -> None:
    """Add the Commons sha1 and timestamp of each file that lacks them (pins the file revision we used)."""
    missing = [s for s in sources if not s.get("sha1")]
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        data = _api({"titles": "|".join(s["title"] for s in batch), "prop": "imageinfo",
                     "iiprop": "sha1|timestamp"})
        by_title = {p["title"]: (p.get("imageinfo") or [{}])[0] for p in data["query"]["pages"]}
        for s in batch:
            info = by_title.get(s["title"], {})
            s["sha1"], s["timestamp"] = info.get("sha1"), info.get("timestamp")


SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def to_srgb_rgb(im: Image.Image) -> Image.Image:
    """Upright, 8-bit RGB in sRGB, with no ICC profile or EXIF left: EXIF orientation applied, an embedded ICC
    profile converted to sRGB (perceptual intent), other modes (L, RGBA, P, CMYK) converted to RGB."""
    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "CMYK", "L"):
        im = im.convert("RGB")
    if icc:
        im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), SRGB, outputMode="RGB")
    out = im.convert("RGB")
    out.info.pop("icc_profile", None)
    out.info.pop("exif", None)
    return out


def _decode(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return to_srgb_rgb(im)


def download_photo(src: dict, max_side: int, cache: Path | None = None,
                   get: Callable[[str], bytes] = _get) -> Image.Image:
    """The photo as sRGB RGB, upright, longer side at most max_side.

    The bytes must match `src["sha256"]` when it is set; otherwise it is set from them. They are decoded before
    being cached, and the cache file (gitignored, under artifacts/) is written atomically."""
    if src["license"] not in ALLOWED_LICENSES:
        raise ValueError(f"{src['title']} has license {src['license']!r}")
    cached = cache / (hashlib.sha256(src["download_url"].encode()).hexdigest() + ".jpg") if cache else None
    if cached and cached.is_file():
        data = cached.read_bytes()
    else:
        data = get(src["download_url"])
    digest = hashlib.sha256(data).hexdigest()
    if src.get("sha256") and src["sha256"] != digest:
        raise ValueError(f"{src['title']}: downloaded bytes have sha256 {digest}, pinned {src['sha256']}. "
                         "The file changed on Commons; replace it or rebuild with --requery")
    im = _decode(data)
    if cached and not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(cached.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(cached)
    src["sha256"] = digest
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    return im
