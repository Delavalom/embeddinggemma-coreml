import ast
import math
import random
from pathlib import Path

import pytest

from egcm.buckets import (
    ASPECT_EDGES,
    BUCKETS,
    aspect_preserving_size,
    choose_bucket,
    worst_case_stretch,
)

UPSTREAM = Path(__file__).resolve().parents[1] / "vendor/reference/image_processing_gemma4.py"


def upstream_fn():
    """The unmodified upstream function, extracted from the vendored file (it can't be imported)."""
    fn = next(
        n for n in ast.parse(UPSTREAM.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_aspect_ratio_preserving_size"
    )
    ns = {"math": math}
    exec(compile(ast.Module([fn], []), str(UPSTREAM), "exec"), ns)
    return ns[fn.name]


def test_ten_buckets():
    assert len(BUCKETS) == 10
    assert {b.name for b in BUCKETS} == {"tall", "portrait", "square", "landscape", "wide"}
    assert len({b.function_name for b in BUCKETS}) == 10


def test_buckets_are_fixed_points_of_the_upstream_resize():
    up = upstream_fn()
    for b in BUCKETS:
        assert aspect_preserving_size(b.height, b.width, budget=b.budget) == (b.height, b.width)
        assert up(b.height, b.width, 16, b.budget * 9, 3) == (b.height, b.width)
        assert (b.height // 48) * (b.width // 48) == b.tokens <= b.budget


def test_each_bucket_is_chosen_by_its_own_shape():
    for b in BUCKETS:
        assert choose_bucket(b.height, b.width, b.budget) == b


def test_port_matches_upstream_on_random_sizes():
    up, rng = upstream_fn(), random.Random(0)
    for _ in range(2000):
        h, w = rng.randint(100, 8000), rng.randint(100, 8000)
        for budget in (70, 280):
            assert aspect_preserving_size(h, w, budget) == up(h, w, 16, budget * 9, 3)


def test_extreme_aspect_ratios_match_upstream():
    up = upstream_fn()
    for h, w in [(10, 100000), (100000, 10), (1, 1000), (1000, 1), (7, 5000)]:
        for budget in (70, 280):
            got = aspect_preserving_size(h, w, budget)
            assert got == up(h, w, 16, budget * 9, 3)
            assert 0 < got[0] * got[1] <= budget * 9 * 256


def test_matches_upstream_formula():
    # f = sqrt(630*256 / (3024*4032)) = 0.11501 -> ideal 347.8 x 463.7 -> floor to multiples of 48 = 336 x 432
    assert aspect_preserving_size(3024, 4032, budget=70) == (336, 432)


def test_choose():  # signature is (height, width, budget)
    assert choose_bucket(3024, 4032, 70).name == "landscape"  # w/h = 1.33
    assert choose_bucket(4032, 3024, 70).name == "portrait"
    assert choose_bucket(1000, 1000, 280).name == "square"
    assert choose_bucket(2532, 1170, 70).name == "tall"  # iPhone screenshot, 1170 wide x 2532 tall: r = 0.46
    assert choose_bucket(1170, 2532, 280).name == "wide"
    assert choose_bucket(1080, 1920, 70).name == "wide"  # 16:9, r = 1.78
    assert choose_bucket(2000, 3000, 70).name == "landscape"  # 3:2, r = 1.5
    assert choose_bucket(5000, 20000, 280).name == "wide"  # panorama


def test_boundaries():
    assert ASPECT_EDGES == (0.6, 0.83, 1.2, 1.67)

    def name(r):  # height 10000 keeps r exact
        return choose_bucket(10000, round(r * 10000), 70).name

    assert name(0.5999) == "tall" and name(0.6) == "portrait"  # r < 0.6 is tall
    assert name(0.8299) == "portrait" and name(0.83) == "square"  # r < 0.83 is portrait
    assert name(1.2) == "square" and name(1.2001) == "landscape"  # r <= 1.2 is square
    assert name(1.67) == "landscape" and name(1.6701) == "wide"  # r <= 1.67 is landscape


def test_stretch_of_bounded_buckets_is_small():
    for b in BUCKETS:
        if b.name in ("portrait", "square", "landscape"):
            assert worst_case_stretch(b) < 1.21, b
    assert all(math.isinf(worst_case_stretch(b)) for b in BUCKETS if b.name in ("tall", "wide"))


@pytest.mark.parametrize("h, w", [(0, 100), (100, 0), (0, 0), (-5, 10)])
def test_zero_or_negative_dimensions_raise(h, w):
    with pytest.raises(ValueError, match="positive"):
        aspect_preserving_size(h, w, 70)
    with pytest.raises(ValueError, match="positive"):
        choose_bucket(h, w, 70)


def test_unsupported_budget_raises():
    with pytest.raises(ValueError, match="budget"):
        aspect_preserving_size(100, 100, 100)
    with pytest.raises(ValueError, match="budget"):
        choose_bucket(100, 100, 100)
