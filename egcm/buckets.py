import math
from dataclasses import dataclass

PATCH, POOL = 16, 3
SIDE = PATCH * POOL  # 48

@dataclass(frozen=True)
class Bucket:
    budget: int; name: str; height: int; width: int
    @property
    def tokens(self) -> int: return (self.height // SIDE) * (self.width // SIDE)
    @property
    def grid(self) -> tuple[int, int]: return self.height // PATCH, self.width // PATCH
    @property
    def function_name(self) -> str: return f"embed_image_{self.budget}_{self.name}"

BUCKETS = (
    Bucket(70, "portrait", 480, 336), Bucket(70, "landscape", 336, 480), Bucket(70, "square", 384, 384),
    Bucket(280, "portrait", 960, 672), Bucket(280, "landscape", 672, 960), Bucket(280, "square", 768, 768),
)

def aspect_preserving_size(height: int, width: int, budget: int) -> tuple[int, int]:
    max_patches = budget * POOL * POOL
    target_px = max_patches * PATCH * PATCH
    f = math.sqrt(target_px / (height * width))
    th = int(math.floor(f * height / SIDE)) * SIDE
    tw = int(math.floor(f * width / SIDE)) * SIDE
    max_side = (max_patches // POOL**2) * SIDE
    if th == 0 and tw == 0:
        raise ValueError("image too small")
    if th == 0:
        th, tw = SIDE, min(int(math.floor(width / height)) * SIDE, max_side)
    elif tw == 0:
        tw, th = SIDE, min(int(math.floor(height / width)) * SIDE, max_side)
    return th, tw

def choose_bucket(height: int, width: int, budget: int) -> Bucket:
    r = width / height
    name = "portrait" if r < 0.83 else "landscape" if r > 1.2 else "square"
    return next(b for b in BUCKETS if b.budget == budget and b.name == name)
