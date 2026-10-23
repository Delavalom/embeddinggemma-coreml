"""egcm.probe on the tiny model: every site kind is recorded, maxima merge, and large values are flagged."""

import tiny
import torch

from egcm import probe


def test_probe_records_every_site_kind_and_flags_overflow():
    _, image, _ = tiny.rewrite(tiny.hf_state_dict(*tiny.reference()))
    pixels = torch.rand(1, 3, tiny.BUCKET.height, tiny.BUCKET.width)
    with torch.no_grad(), probe.probe(image, "image/") as s:
        image(pixels)
    kinds = {k.rsplit(":", 1)[1] for k in s}
    assert kinds == {"residual", "attn_out", "mlp_out", "mlp_inner", "ple_inner", "norm_in", "norm_out",
                     "embeddings", "patch_out"}
    assert all(k.startswith("image/") for k in s)
    assert sum(k.endswith(":residual") for k in s) == tiny.TEXT.layers + tiny.VISION.layers

    big = dict(s)
    big["image/tower.layers.0:residual"] = 4e4
    r = probe.report([s, big])
    assert r["overflow_risk"] == [["image/tower.layers.0:residual", 4e4]]
    assert r["max_by_kind"]["image/text:residual"] == 4e4
    assert any(k.startswith("image/vision:") for k in r["max_by_kind"])
    assert tuple(r["top"][0]) == ("image/tower.layers.0:residual", 4e4)
