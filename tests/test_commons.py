"""Offline tests for the Commons fetcher: parsing, picking, retries and photo decoding. No network."""

import hashlib
import http.client
import io
import json
import urllib.error
from pathlib import Path

import pytest
from PIL import Image, ImageCms, UnidentifiedImageError

from stages.s0_reference import commons

P3 = Path("/System/Library/ColorSync/Profiles/Display P3.icc")


def page(index, title, *, mime="image/jpeg", lic="CC0", w=2000, h=1500, artist="<a href='x'>Ann &amp; Bo</a>"):
    return {"index": index, "title": title, "imageinfo": [{
        "mime": mime, "width": w, "height": h, "descriptionurl": f"https://c/{title}", "url": f"https://u/{title}",
        "thumburl": f"https://t/{title}", "sha1": "s" + title, "timestamp": "2020-01-01T00:00:00Z",
        "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": artist}}}]}


def test_parse_search_filters_and_keeps_rank_order():
    data = {"query": {"pages": [
        page(3, "File:C.jpg"),
        page(1, "File:A.jpg", lic="Public domain"),
        page(2, "File:png.png", mime="image/png"),
        page(4, "File:Bad license.jpg", lic="CC BY-SA 4.0"),
        page(5, "File:Oil painting of a dog.jpg"),
        page(6, "File:Woman MET DP-1019.jpg"),
        page(7, "File:Small.jpg", w=800, h=600),
        page(8, "File:Narrow.jpg", w=1100, h=950),
    ]}}
    out = commons.parse_search(data)
    assert [c["title"] for c in out] == ["File:A.jpg", "File:C.jpg"]
    a = out[0]
    assert a["license"] == "Public domain" and a["author"] == "Ann & Bo" and a["download_url"] == "https://t/File:A.jpg"
    assert a["original_size"] == [1500, 2000] and a["sha1"] == "sFile:A.jpg"


def cand(title):
    return {"title": title, "source_url": "s", "download_url": "d", "license": "CC0", "author": "a"}


def test_pick_sources_dedups_titles_and_series(monkeypatch, tmp_path):
    results = {
        "dog": [cand("File:Dog 01.jpg"), cand("File:Dog 02.jpg"), cand("File:Dog 03.jpg"), cand("File:Puppy.jpg")],
        "puppy": [cand("File:Puppy.jpg"), cand("File:Hound.jpg")],
        "cat": [cand("File:Puppy.jpg"), cand("File:Cat.jpg")],
    }
    monkeypatch.setattr(commons, "search", lambda phrase: results[phrase])
    monkeypatch.setattr(commons, "CATEGORIES", {"dogs": (4, ("dog", "puppy")), "cats": (1, ("cat",))})
    out = commons.pick_sources(None)
    # "Dog 03" is the third of the "dog" series (limit 2); Puppy is taken once, by dogs
    assert [(c["category"], c["title"]) for c in out] == [
        ("dogs", "File:Dog 01.jpg"), ("dogs", "File:Dog 02.jpg"), ("dogs", "File:Puppy.jpg"),
        ("dogs", "File:Hound.jpg"), ("cats", "File:Cat.jpg")]
    pinned = tmp_path / "sources.json"
    pinned.write_text(json.dumps([cand("File:Pinned.jpg")]))
    assert commons.pick_sources(pinned)[0]["title"] == "File:Pinned.jpg"
    monkeypatch.setattr(commons, "CATEGORIES", {"cats": (3, ("cat",))})
    with pytest.raises(RuntimeError, match="only 2"):
        commons.pick_sources(None)


def test_series_key():
    assert commons.series_key("File:A mother cat of Meitei 04.jpg") == commons.series_key("File:A mother cat 12.jpg")
    assert commons.series_key("File:Landscape-mountains-sky (243).jpg") == "landscape mountains"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "x", headers, None)


def run_get(events, tries=5):
    """Run commons._get against a scripted opener; returns (result or exception, sleeps, calls)."""
    sleeps, calls = [], []

    def opener(req, timeout):
        calls.append(req.get_header("User-agent"))
        e = events[len(calls) - 1]
        if isinstance(e, BaseException):
            raise e
        return FakeResponse(e)

    try:
        out = commons._get("https://x", tries=tries, opener=opener, sleep=sleeps.append)
    except Exception as e:  # returned for the assertion
        out = e
    return out, [s for s in sleeps if s >= 1.0], calls


def test_get_retries_transient_failures_and_honors_retry_after(monkeypatch):
    monkeypatch.setattr(commons, "_last_request", -1e9)
    out, sleeps, calls = run_get([http_error(503, "7"), urllib.error.URLError("down"), TimeoutError(),
                                  ConnectionResetError(), http.client.IncompleteRead(b"x"), b"ok"], tries=6)
    assert out == b"ok" and len(calls) == 6
    assert sleeps[0] == 7.0  # Retry-After honored
    assert sleeps[1:] == [4.0, 8.0, 16.0, 32.0]  # exponential backoff for the rest


def test_get_does_not_retry_client_errors_and_gives_up(monkeypatch):
    monkeypatch.setattr(commons, "_last_request", -1e9)
    out, _, calls = run_get([http_error(404)])
    assert isinstance(out, urllib.error.HTTPError) and len(calls) == 1
    out, _, calls = run_get([http_error(429)] * 3, tries=3)
    assert isinstance(out, urllib.error.HTTPError) and out.code == 429 and len(calls) == 3


def test_user_agent_has_no_contact_unless_set(monkeypatch):
    monkeypatch.delenv("EGCM_CONTACT", raising=False)
    ua = commons.user_agent()
    assert ua.startswith("embeddinggemma-coreml-fixtures/1.0 (https://github.com/Delavalom/embeddinggemma-coreml)")
    assert "@" not in ua and "python-urllib/" in ua
    monkeypatch.setenv("EGCM_CONTACT", "me@example.org")
    assert commons.user_agent().endswith(" me@example.org")


def test_api_retries_maxlag():
    replies = [{"error": {"code": "maxlag"}}, {"query": {"pages": []}}]
    urls, sleeps = [], []

    def get(url):
        urls.append(url)
        return json.dumps(replies[len(urls) - 1]).encode()

    assert commons._api({"titles": "x"}, get=get, sleep=sleeps.append) == {"query": {"pages": []}}
    assert "maxlag=5" in urls[0] and len(sleeps) == 1
    with pytest.raises(RuntimeError, match="badtitle"):
        commons._api({}, get=lambda u: b'{"error": {"code": "badtitle"}}')


def encode(im, fmt="PNG", **kw):
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def src_for(**kw):
    return {"title": "File:T.jpg", "license": "CC0", "download_url": "https://t/T.jpg", **kw}


def fetch(data, src=None, max_side=1600, cache=None):
    calls = []

    def get(url):
        calls.append(url)
        return data

    src = src or src_for()
    return commons.download_photo(src, max_side, cache, get=get), calls, src


def test_download_rejects_other_licenses():
    with pytest.raises(ValueError, match="license"):
        commons.download_photo(src_for(license="CC BY 4.0"), 1600, get=lambda u: b"")


def test_download_applies_exif_orientation_6():
    im = Image.new("RGB", (40, 20), (0, 0, 255))
    im.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    out, _, _ = fetch(encode(im, exif=exif.tobytes()))
    assert out.size == (20, 40) and out.mode == "RGB"
    assert out.getpixel((19, 0)) == (255, 0, 0)  # the old top-left is now top-right
    assert "exif" not in out.info and not out.getexif()


@pytest.mark.parametrize("mode,fmt", [("L", "PNG"), ("RGBA", "PNG"), ("CMYK", "JPEG"), ("P", "PNG")])
def test_download_converts_modes_to_rgb(mode, fmt):
    im = Image.new("RGB", (30, 20), (10, 120, 200)).convert(mode)
    out, _, _ = fetch(encode(im, fmt))
    assert out.mode == "RGB" and out.size == (30, 20)


@pytest.mark.skipif(not P3.is_file(), reason="needs the macOS Display P3 profile")
def test_download_converts_icc_to_srgb_and_strips_it():
    im = Image.new("RGB", (16, 16), (40, 180, 90))
    out, _, _ = fetch(encode(im, icc_profile=P3.read_bytes()))
    expected = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(str(P3)), commons.SRGB, outputMode="RGB")
    assert out.getpixel((0, 0)) == expected.getpixel((0, 0)) != (40, 180, 90)
    assert "icc_profile" not in out.info


def test_download_downscales_to_max_side():
    out, _, _ = fetch(encode(Image.new("RGB", (400, 200))), max_side=100)
    assert out.size == (100, 50)


def test_download_cache_reuse_and_sha256_pin(tmp_path):
    data = encode(Image.new("RGB", (20, 10), (1, 2, 3)))
    out1, calls1, src = fetch(data, cache=tmp_path)
    assert len(calls1) == 1 and src["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(list(tmp_path.iterdir())) == 1 and not list(tmp_path.glob("*.tmp"))
    out2, calls2, _ = fetch(data, src=dict(src), cache=tmp_path)
    assert calls2 == [] and out2.tobytes() == out1.tobytes()
    with pytest.raises(ValueError, match="sha256"):
        fetch(data, src=src_for(sha256="0" * 64))


def test_download_does_not_cache_undecodable_bytes(tmp_path):
    with pytest.raises(UnidentifiedImageError):
        fetch(b"not an image", cache=tmp_path)
    assert list(tmp_path.iterdir()) == []
