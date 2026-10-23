"""Locally rendered fixture images (screenshots, memes, receipts) and the photo edits (blur, dark, near-duplicate).

Every function takes a seeded `random.Random`, so the same seed gives the same pixels on the same fonts.
Fonts are macOS system fonts; a missing font is an error. `font_name` records which file was used.
"""

from __future__ import annotations

import functools
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT_FILES = {
    "sans": ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"),
    "impact": ("/System/Library/Fonts/Supplemental/Impact.ttf", "/System/Library/Fonts/Supplemental/Arial Black.ttf"),
    "mono": ("/System/Library/Fonts/Supplemental/Courier New.ttf", "/System/Library/Fonts/Menlo.ttc"),
}
SCREEN = (1170, 2532)  # iPhone portrait, width x height
RECEIPT = (1000, 2400)


@functools.cache
def _font_file(kind: str) -> str:
    """The first loadable font file for `kind`. A missing font is an error: the pixels would silently change."""
    for f in FONT_FILES[kind]:
        if Path(f).is_file():
            try:
                ImageFont.truetype(f, 12)
                return f
            except OSError:
                continue
    raise FileNotFoundError(f"no {kind!r} font found; tried {FONT_FILES[kind]}")


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_file(kind), size)


def font_name(kind: str) -> str:
    """The font file `font(kind, ...)` uses."""
    return Path(_font_file(kind)).name


def _status_bar(d: ImageDraw.ImageDraw, rng: random.Random, dark: bool) -> None:
    fg = (255, 255, 255) if dark else (0, 0, 0)
    d.text((90, 50), f"{rng.randint(7, 11)}:{rng.randint(0, 59):02d}", font=font("sans", 52), fill=fg)
    for i in range(4):  # signal bars and battery
        d.rectangle((870 + i * 22, 90 - i * 12, 884 + i * 22, 100), fill=fg)
    d.rounded_rectangle((990, 62, 1080, 102), radius=10, outline=fg, width=4)
    d.rectangle((998, 70, 998 + rng.randint(30, 74), 94), fill=fg)


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def chat_screenshot(rng: random.Random, lines: list[str]) -> Image.Image:
    dark = rng.random() < 0.3
    bg, other = ((0, 0, 0), (38, 38, 40)) if dark else ((255, 255, 255), (233, 233, 235))
    im = Image.new("RGB", SCREEN, bg)
    d = ImageDraw.Draw(im)
    _status_bar(d, rng, dark)
    d.text((585, 200), rng.choice(["Mom", "Alex", "Sam", "Jordan", "Group: Weekend"]), anchor="mm",
           font=font("sans", 56), fill=(255, 255, 255) if dark else (0, 0, 0))
    d.line((0, 270, SCREEN[0], 270), fill=other, width=3)
    y, f = 330, font("sans", 46)
    for text in lines:
        mine = rng.random() < 0.5
        wrapped = _wrap(text, 30)
        h = 64 * len(wrapped) + 40
        w = min(820, max(len(s) for s in wrapped) * 25 + 70)
        x0 = SCREEN[0] - 60 - w if mine else 60
        d.rounded_rectangle((x0, y, x0 + w, y + h), radius=42, fill=(10, 132, 255) if mine else other)
        for i, s in enumerate(wrapped):
            d.text((x0 + 36, y + 22 + 64 * i), s, font=f,
                   fill=(255, 255, 255) if mine or dark else (0, 0, 0))
        y += h + 36
        if y > SCREEN[1] - 400:
            break
    d.rounded_rectangle((60, SCREEN[1] - 200, SCREEN[0] - 60, SCREEN[1] - 100), radius=50, outline=other, width=4)
    return im


def tweet_screenshot(rng: random.Random, lines: list[str]) -> Image.Image:
    im = Image.new("RGB", SCREEN, (255, 255, 255))
    d = ImageDraw.Draw(im)
    _status_bar(d, rng, False)
    y = 260
    for text in lines[:4]:
        color = tuple(rng.randint(40, 220) for _ in range(3))
        d.ellipse((50, y, 170, y + 120), fill=color)
        name = rng.choice(["Dana Park", "Camera Roll Tips", "Lee", "Morning News", "Chef Rosa", "Nadia K."])
        d.text((200, y + 6), name, font=font("sans", 46), fill=(15, 20, 25))
        d.text((200, y + 62), f"@{name.lower().replace(' ', '_').replace('.', '')} · {rng.randint(1, 23)}h",
               font=font("sans", 40), fill=(83, 100, 113))
        ty = y + 140
        for s in _wrap(text, 38):
            d.text((200, ty), s, font=font("sans", 46), fill=(15, 20, 25))
            ty += 62
        ty += 30
        stats = f"{rng.randint(0, 999)}   {rng.randint(0, 9999)}   {rng.randint(0, 99)}K"
        d.text((200, ty), stats, font=font("sans", 38), fill=(83, 100, 113))
        y = ty + 90
        d.line((0, y - 30, SCREEN[0], y - 30), fill=(239, 243, 244), width=3)
        if y > SCREEN[1] - 500:
            break
    return im


def settings_screenshot(rng: random.Random, lines: list[str]) -> Image.Image:
    im = Image.new("RGB", SCREEN, (242, 242, 247))
    d = ImageDraw.Draw(im)
    _status_bar(d, rng, False)
    d.text((60, 200), "Settings", font=font("sans", 90), fill=(0, 0, 0))
    labels = ["Airplane Mode", "Wi-Fi", "Bluetooth", "Cellular", "Notifications", "Sounds & Haptics",
              "Focus", "Screen Time", "General", "Control Center", "Display & Brightness", "Wallpaper",
              "Siri & Search", "Face ID & Passcode", "Battery", "Privacy & Security", "Photos", "Camera"]
    y = 360
    for group in range(4):
        rows = rng.randint(3, 5)
        d.rounded_rectangle((40, y, SCREEN[0] - 40, y + rows * 120), radius=30, fill=(255, 255, 255))
        for r in range(rows):
            label = labels[(group * 5 + r) % len(labels)]
            ry = y + r * 120
            icon = tuple(rng.randint(0, 255) for _ in range(3))
            d.rounded_rectangle((80, ry + 28, 144, ry + 92), radius=14, fill=icon)
            d.text((180, ry + 60), label, anchor="lm", font=font("sans", 46), fill=(0, 0, 0))
            if rng.random() < 0.5:  # a toggle
                on = rng.random() < 0.5
                track = (52, 199, 89) if on else (233, 233, 234)
                d.rounded_rectangle((930, ry + 30, 1090, ry + 90), radius=30, fill=track)
                cx = 1060 if on else 960
                d.ellipse((cx - 27, ry + 33, cx + 27, ry + 87), fill=(255, 255, 255))
            else:
                d.text((1090, ry + 60), ">", anchor="rm", font=font("sans", 46), fill=(196, 196, 199))
            if r < rows - 1:
                d.line((180, ry + 120, SCREEN[0] - 40, ry + 120), fill=(222, 222, 226), width=2)
        y += rows * 120 + 70
    for s in _wrap(rng.choice(lines), 48)[:3]:
        d.text((80, y), s, font=font("sans", 38), fill=(110, 110, 115))
        y += 52
    return im


def meme(rng: random.Random, photo: Image.Image, aspect: float, top: str, bottom: str, max_side: int) -> Image.Image:
    """Center-crop the photo to width/height = aspect, then add Impact-style top and bottom captions."""
    w, h = photo.size
    cw, ch = (w, round(w / aspect)) if w / h < aspect else (round(h * aspect), h)
    left, upper = (w - cw) // 2, (h - ch) // 2
    im = photo.crop((left, upper, left + cw, upper + ch))
    s = min(1.0, max_side / max(im.size))
    im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    for text, top_edge in ((top.upper(), True), (bottom.upper(), False)):
        size = max(24, im.width // 12)
        while size > 16 and d.textlength(text, font=font("impact", size)) > 0.94 * im.width:
            size -= 2
        y, anchor = (size // 3, "ma") if top_edge else (im.height - size // 3, "md")
        d.text((im.width // 2, y), text, anchor=anchor, font=font("impact", size), fill=(255, 255, 255),
               stroke_width=max(2, size // 14), stroke_fill=(0, 0, 0))
    return im


def receipt(rng: random.Random, items: list[tuple[str, float]], store: str) -> Image.Image:
    bg = tuple(rng.randint(90, 140) for _ in range(3))
    im = Image.new("RGB", RECEIPT, bg)
    paper = Image.new("RGB", (820, 2200), (250, 249, 244))
    d = ImageDraw.Draw(paper)
    big, f = font("mono", 64), font("mono", 40)
    d.text((410, 90), store, anchor="mm", font=big, fill=(20, 20, 20))
    d.text((410, 170), f"{rng.randint(10, 999)} Main St · Tel {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
           anchor="mm", font=font("mono", 30), fill=(40, 40, 40))
    stamp = f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}  {rng.randint(8, 21)}:{rng.randint(0, 59):02d}"
    d.text((60, 240), stamp, font=f, fill=(30, 30, 30))
    y = 330
    for name, price in items:
        d.text((60, y), name[:22], font=f, fill=(25, 25, 25))
        d.text((760, y), f"{price:.2f}", anchor="ra", font=f, fill=(25, 25, 25))
        y += 62
    sub = sum(p for _, p in items)
    tax = round(sub * 0.08, 2)
    d.line((60, y + 10, 760, y + 10), fill=(60, 60, 60), width=3)
    for label, v in (("SUBTOTAL", sub), ("TAX 8%", tax), ("TOTAL", sub + tax)):
        y += 70
        d.text((60, y), label, font=big if label == "TOTAL" else f, fill=(10, 10, 10))
        d.text((760, y), f"{v:.2f}", anchor="ra", font=big if label == "TOTAL" else f, fill=(10, 10, 10))
    y += 140
    d.text((410, y), rng.choice(["VISA **** ", "MASTERCARD **** "]) + f"{rng.randint(1000, 9999)}", anchor="mm",
           font=f, fill=(30, 30, 30))
    x = 110
    while x < 710:  # barcode
        bw = rng.choice((3, 5, 8))
        d.rectangle((x, y + 80, x + bw, y + 240), fill=(0, 0, 0))
        x += bw + rng.choice((3, 5, 7))
    d.text((410, y + 300), "THANK YOU FOR SHOPPING", anchor="mm", font=f, fill=(30, 30, 30))
    paper = paper.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=bg)
    im.paste(paper, ((RECEIPT[0] - paper.width) // 2, (RECEIPT[1] - paper.height) // 2))
    return im


def _fit(im: Image.Image, max_side: int) -> Image.Image:
    s = min(1.0, max_side / max(im.size))
    return im if s == 1.0 else im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)


def blurry(rng: random.Random, photo: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    radius = round(rng.uniform(6, 12), 2)
    return _fit(photo, max_side).filter(ImageFilter.GaussianBlur(radius)), radius


def dark(photo: Image.Image, max_side: int) -> Image.Image:
    return ImageEnhance.Brightness(_fit(photo, max_side)).enhance(0.15)


def near_duplicate(rng: random.Random, photo: Image.Image) -> Image.Image:
    """Crop 5% of each dimension, split at random between the two sides. The caller saves it at JPEG quality 60."""
    w, h = photo.size
    dx, dy = round(w * 0.05), round(h * 0.05)
    left, top = rng.randint(0, dx), rng.randint(0, dy)
    return photo.crop((left, top, left + w - dx, top + h - dy))
