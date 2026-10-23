import numpy as np
from egcm.metrics import cosine_rows, recall_at_k, spearman_topk

def test_cosine_identity():
    a = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    assert np.allclose(cosine_rows(a, a), 1.0, atol=1e-6)

def test_recall_perfect_and_partial():
    rng = np.random.default_rng(1)
    q, d = rng.normal(size=(4, 16)), rng.normal(size=(50, 16))
    assert recall_at_k(q, d, q, d, k=10) == 1.0
    assert recall_at_k(q, d, q, rng.normal(size=(50, 16)), k=10) < 0.6

def test_spearman_identical():
    rng = np.random.default_rng(2)
    q, d = rng.normal(size=(3, 16)), rng.normal(size=(60, 16))
    assert spearman_topk(q, d, q, d, k=50) > 0.999
