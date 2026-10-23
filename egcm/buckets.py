"""Fixed image shapes for the Core ML functions, and the resize math of the reference processor."""

import math
from dataclasses import dataclass

PATCH, POOL = 16, 3
SIDE = PATCH * POOL  # 48
BUDGETS = (70, 280)  # soft tokens per image

# Aspect ratio r = width / height picks the bucket (see choose_bucket):
#   r < 0.6 tall | r < 0.83 portrait | r <= 1.2 square | r <= 1.67 landscape | otherwise wide
ASPECT_EDGES = (0.6, 0.83, 1.2, 1.67)


@dataclass(frozen=True)
class Bucket:
    budget: int
    name: str
    height: int
    width: int

    @property
    def tokens(self) -> int:
        return (self.height // SIDE) * (self.width // SIDE)

    @property
    def grid(self) -> tuple[int, int]:
        return self.height // PATCH, self.width // PATCH

    @property
    def function_name(self) -> str:
        return f"embed_image_{self.budget}_{self.name}"


BUCKETS = (
    Bucket(70, "tall", 576, 240),
    Bucket(70, "portrait", 480, 336),
    Bucket(70, "square", 384, 384),
    Bucket(70, "landscape", 336, 480),
    Bucket(70, "wide", 240, 576),
    Bucket(280, "tall", 1152, 528),
    Bucket(280, "portrait", 960, 672),
    Bucket(280, "square", 768, 768),
    Bucket(280, "landscape", 672, 960),
    Bucket(280, "wide", 528, 1152),
)

# The image sequence s0 recorded from the processor (image_token_layout.json) is
# [<bos>, <|image>, <|image|> x soft tokens, <image|>, <eos>]: soft tokens + 4.
IMAGE_SPECIAL_TOKENS = 4
# Longest sequence any function builds (284). Text functions are at most 128 tokens, so the image buckets set it.
MAX_SEQ = max(b.tokens for b in BUCKETS) + IMAGE_SPECIAL_TOKENS


def _check(height: int, width: int, budget: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}, got {budget}")


def aspect_preserving_size(height: int, width: int, budget: int) -> tuple[int, int]:
    """The size the reference processor resizes to: port of `get_aspect_ratio_preserving_size`
    in vendor/reference/image_processing_gemma4.py (tests compare the two on random sizes)."""
    _check(height, width, budget)
    max_patches = budget * POOL * POOL
    target_px = max_patches * PATCH * PATCH
    f = math.sqrt(target_px / (height * width))
    th = math.floor(f * height / SIDE) * SIDE
    tw = math.floor(f * width / SIDE) * SIDE
    max_side = (max_patches // POOL**2) * SIDE
    # Upstream raises if both sides are 0 and if the result exceeds target_px. Neither can happen: the area is
    # about target_px (>= 161,280), so both sides cannot be below 48; floored sides give th * tw <= f^2 * h * w
    # = target_px; and the one-side-zero branches below give at most SIDE * max_side = target_px.
    if th == 0:
        th, tw = SIDE, min(math.floor(width / height) * SIDE, max_side)
    elif tw == 0:
        tw, th = SIDE, min(math.floor(height / width) * SIDE, max_side)
    return th, tw


def choose_bucket(height: int, width: int, budget: int) -> Bucket:
    """The bucket for an image of the given size, from r = width / height."""
    _check(height, width, budget)
    r = width / height
    tall_below, portrait_below, square_to, landscape_to = ASPECT_EDGES
    if r < tall_below:
        name = "tall"
    elif r < portrait_below:
        name = "portrait"
    elif r <= square_to:
        name = "square"
    elif r <= landscape_to:
        name = "landscape"
    else:
        name = "wide"
    return next(b for b in BUCKETS if b.budget == budget and b.name == name)


def worst_case_stretch(bucket: Bucket) -> float:
    """Largest aspect-ratio distortion max(r/b, b/r) over the r range that maps to this bucket, b = W/H of the
    bucket. Infinite for tall and wide, which have no outer limit."""
    lo, hi = {
        "tall": (0.0, ASPECT_EDGES[0]),
        "portrait": (ASPECT_EDGES[0], ASPECT_EDGES[1]),
        "square": (ASPECT_EDGES[1], ASPECT_EDGES[2]),
        "landscape": (ASPECT_EDGES[2], ASPECT_EDGES[3]),
        "wide": (ASPECT_EDGES[3], math.inf),
    }[bucket.name]
    b = bucket.width / bucket.height
    return max((max(r / b, b / r) if 0 < r < math.inf else math.inf) for r in (lo, hi))
