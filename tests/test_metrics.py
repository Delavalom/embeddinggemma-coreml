import numpy as np
import pytest

from egcm.metrics import cosine_rows, recall_at_k, spearman_topk, spearman_topk_per_query

Q = np.array([[1.0, 0.0]])  # one query; a doc's score is cos(angle) and falls as the angle grows


def docs(angles):
    a = np.asarray(angles, dtype=float)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def order_to_angles(order):
    """Angles that make `order` (best doc first) the ranking for Q."""
    angles = np.empty(len(order))
    angles[list(order)] = np.arange(len(order)) * 0.05
    return angles


def spearman(ref_order, new_order, k):
    return spearman_topk(ref=(Q, docs(order_to_angles(ref_order))), new=(Q, docs(order_to_angles(new_order))), k=k)


def test_cosine_identity():
    a = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    assert np.allclose(cosine_rows(a, a), 1.0, atol=1e-6)


def test_cosine_orthogonal_and_opposite():
    assert np.allclose(cosine_rows([[1, 0], [1, 0]], [[0, 2], [-3, 0]]), [0.0, -1.0])


def test_recall_perfect():
    rng = np.random.default_rng(1)
    q, d = rng.normal(size=(4, 16)), rng.normal(size=(50, 16))
    assert recall_at_k(ref=(q, d), new=(q, d), k=10) == 1.0


def test_recall_exactly_seven_of_ten():
    ref = docs(np.arange(20) * 0.01)  # docs 0..9 are the top 10
    new_angles = np.arange(20) * 0.01
    new_angles[[7, 8, 9]] += 1.0  # push 7, 8, 9 down; 10, 11, 12 move up into the top 10
    assert recall_at_k(ref=(Q, ref), new=(Q, docs(new_angles)), k=10) == pytest.approx(0.7)


def test_recall_is_the_mean_over_queries():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])  # q0 prefers small angles, q1 prefers large angles
    ref = docs([0.0, 0.1, 0.2, 0.3])  # k=1: q0 -> doc 0, q1 -> doc 3
    new = docs([0.0, 0.1, 0.35, 0.3])  # q0 -> doc 0 (recall 1), q1 -> doc 2 (recall 0)
    assert recall_at_k(ref=(q, ref), new=(q, new), k=1) == pytest.approx(0.5)


def test_spearman_identical_is_one():
    assert spearman([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], k=5) == pytest.approx(1.0)


def test_spearman_reversal_is_minus_one():
    assert spearman([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], k=5) == pytest.approx(-1.0)


def test_spearman_known_permutation_exact_rho():
    # new ranking [1, 0, 3, 2, 4]: d = (-1, 1, -1, 1, 0), sum d^2 = 4, rho = 1 - 6*4 / (5*24) = 0.8
    assert spearman([0, 1, 2, 3, 4], [1, 0, 3, 2, 4], k=5) == pytest.approx(0.8)


def test_spearman_intruder_lowers_it():
    # doc 5 enters at the top of new's ranking and pushes every reference doc down one place:
    # d = -1 for all five, rho = 1 - 6*5 / (5*24) = 0.75
    assert spearman([0, 1, 2, 3, 4, 5], [5, 0, 1, 2, 3, 4], k=5) == pytest.approx(0.75)
    assert spearman([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], k=5) > 0.75


def test_spearman_is_clamped_at_minus_one():
    # all five reference docs are pushed far down new's ranking, so the raw formula is far below -1
    order = [5, 6, 7, 8, 9, 10, 11, 4, 3, 2, 1, 0]
    assert spearman(list(range(12)), order, k=5) == -1.0


def test_spearman_p5_aggregation():
    # 20 queries: 19 identical rankings (rho 1) and one fully reversed (rho -1).
    # Sorted: [-1, 1, 1, ...]; the 5th percentile sits at index 0.05 * 19 = 0.95, so -1 + 0.95 * 2 = 0.9.
    n, k = 5, 5
    ref_d = docs(np.arange(n) * 0.05)
    # one doc set; queries 0..18 rank it like the reference, query 19 points the other way and reverses it
    ref_q = np.tile(Q, (20, 1))
    new_q = np.tile(Q, (20, 1))
    new_q[19] = [np.cos(1.5), np.sin(1.5)]  # nearly orthogonal to Q: nearer docs now score lower
    per_query = spearman_topk_per_query(ref=(ref_q, ref_d), new=(new_q, ref_d), k=k)
    assert per_query[:19] == pytest.approx(1.0) and per_query[19] == pytest.approx(-1.0)
    assert spearman_topk(ref=(ref_q, ref_d), new=(new_q, ref_d), k=k) == pytest.approx(0.9)


def test_k_larger_than_n_raises():
    d = docs(np.arange(5) * 0.1)
    with pytest.raises(ValueError, match="k"):
        recall_at_k(ref=(Q, d), new=(Q, d), k=10)
    with pytest.raises(ValueError, match="k"):
        spearman_topk(ref=(Q, d), new=(Q, d), k=10)


def test_k_must_be_at_least_two_for_spearman():
    d = docs(np.arange(5) * 0.1)
    with pytest.raises(ValueError):
        spearman_topk(ref=(Q, d), new=(Q, d), k=1)


def test_shape_mismatch_raises():
    d = docs(np.arange(10) * 0.1)
    with pytest.raises(ValueError):
        recall_at_k(ref=(Q, d), new=(Q, d[:8]), k=3)
    with pytest.raises(ValueError):
        recall_at_k(ref=(np.vstack([Q, Q]), d), new=(Q, d), k=3)


def test_keyword_only():
    d = docs(np.arange(10) * 0.1)
    with pytest.raises(TypeError):
        recall_at_k((Q, d), (Q, d), 3)  # noqa
