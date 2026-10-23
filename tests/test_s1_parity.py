"""stages/s1_rewrite helpers: text length choice, image affixes from the layout, debug diffs."""

import numpy as np
import pytest
import torch

from egcm.buckets import BUCKETS
from egcm.model.embedder import TEXT_LENGTHS
from stages.s0_reference.artifacts import image_sequence
from stages.s1_rewrite.debug import diff, tokens
from stages.s1_rewrite.parity import embed_ids, image_affixes, text_length

SPECIAL = {"pad": 0, "bos": 2, "eos": 1, "boi": 255999, "image": 258880, "eoi": 258882}


def test_text_length_is_the_smallest_that_fits():
    assert [text_length(n) for n in (1, 16, 17, 32, 33, 64, 65, 128)] == [16, 16, 32, 32, 64, 64, 128, 128]
    assert TEXT_LENGTHS[-1] == 128
    with pytest.raises(ValueError):
        text_length(129)


def layout_for(bucket, ids):
    return {"special_token_ids": SPECIAL, "buckets": {bucket.function_name: {"input_ids": ids}}}


@pytest.mark.parametrize("bucket", [BUCKETS[0], BUCKETS[-1]])
def test_image_affixes_from_layout(bucket):
    ids = image_sequence(SPECIAL, bucket.tokens)
    assert image_affixes(layout_for(bucket, ids), bucket) == ([2, 255999], [258882, 1])


def test_image_affixes_reject_a_wrong_count_or_a_split_run():
    b = BUCKETS[0]
    with pytest.raises(AssertionError):
        image_affixes(layout_for(b, image_sequence(SPECIAL, b.tokens - 1)), b)
    split = image_sequence(SPECIAL, b.tokens)
    split.insert(5, 7)
    with pytest.raises(AssertionError):
        image_affixes(layout_for(b, split), b)


def test_embed_ids_is_channels_first():
    table = torch.arange(12.0).view(4, 3)
    e = embed_ids(table, [3, 1])
    assert e.shape == (1, 3, 1, 2)
    assert np.array_equal(tokens(e), table[[3, 1]].numpy())


def test_diff_is_relative_to_the_reference_scale():
    d = diff(np.array([1.0, 2.5]), np.array([1.0, 2.0]))
    assert d == {"max_abs": 0.5, "ref_max_abs": 2.0, "rel": 0.25}
    with pytest.raises(AssertionError):
        diff(np.zeros(2), np.zeros(3))
