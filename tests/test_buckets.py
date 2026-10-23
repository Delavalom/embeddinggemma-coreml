from egcm.buckets import BUCKETS, choose_bucket, aspect_preserving_size

def test_buckets_are_fixed_points():
    for b in BUCKETS:
        assert aspect_preserving_size(b.height, b.width, budget=b.budget) == (b.height, b.width)
        assert (b.height // 48) * (b.width // 48) == b.tokens

def test_choose():
    # signature is (height, width, budget)
    assert choose_bucket(3024, 4032, 70).name == "landscape"   # w/h=1.33
    assert choose_bucket(4032, 3024, 70).name == "portrait"
    assert choose_bucket(1000, 1000, 280).name == "square"
    assert choose_bucket(2532, 1170, 70).name == "portrait"    # iPhone screenshot, 1170 wide x 2532 tall

def test_matches_upstream_formula():
    # f = sqrt(630*256 / (3024*4032)) = 0.11501 -> ideal 347.8 x 463.7 -> floor to multiples of 48 = 336 x 432
    assert aspect_preserving_size(3024, 4032, budget=70) == (336, 432)
