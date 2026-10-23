"""Synthetic fixture renderers: same seed, same pixels; the documented sizes; fonts are required."""

import random

import pytest
from PIL import Image

from stages.s0_reference import synth

LINES = ["A golden retriever runs across a grassy park.", "A crumpled grocery store receipt lists milk.",
         "A screenshot shows a text message conversation about dinner plans."] * 3
PHOTO = Image.new("RGB", (1600, 1067), (90, 140, 60))


def render(fn, seed=7):
    return fn(random.Random(seed))


CASES = {
    "chat": (lambda r: synth.chat_screenshot(r, LINES), synth.SCREEN),
    "tweet": (lambda r: synth.tweet_screenshot(r, LINES), synth.SCREEN),
    "settings": (lambda r: synth.settings_screenshot(r, LINES), synth.SCREEN),
    "receipt": (lambda r: synth.receipt(r, [("MILK 1L", 1.25), ("EGGS", 3.5)], "STORE"), synth.RECEIPT),
    "meme_wide": (lambda r: synth.meme(r, PHOTO, 16 / 9, "top text", "bottom text", 1200), (1200, 675)),
    "meme_portrait": (lambda r: synth.meme(r, PHOTO, 0.8, "top", "bottom", 1200), (854, 1067)),
    "blurry": (lambda r: synth.blurry(r, PHOTO, 1200)[0], (1200, 800)),
    "dark": (lambda r: synth.dark(PHOTO, 1200), (1200, 800)),
    "near_duplicate": (lambda r: synth.near_duplicate(r, PHOTO), (1520, 1014)),
}


@pytest.mark.parametrize("name", CASES)
def test_deterministic_and_sized(name):
    fn, size = CASES[name]
    a, b = render(fn), render(fn)
    assert a.mode == "RGB" and a.size == size
    assert a.tobytes() == b.tobytes()


def test_blur_radius_in_range():
    assert all(6 <= synth.blurry(random.Random(s), PHOTO, 300)[1] <= 12 for s in range(20))


def test_missing_font_is_an_error(monkeypatch):
    monkeypatch.setitem(synth.FONT_FILES, "sans", ("/nonexistent/font.ttf",))
    synth._font_file.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            synth.font("sans", 12)
    finally:
        synth._font_file.cache_clear()
