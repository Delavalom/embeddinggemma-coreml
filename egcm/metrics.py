"""Parity metrics. Embeddings are compared as (queries, docs) pairs: `ref=(q, d)` and `new=(q, d)`.

Aggregations differ per metric and are defined in each docstring; the `min_` / `max_` prefix on a gate key
in gates.toml only gives the comparison direction.
"""

from __future__ import annotations

import numpy as np


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosine_rows(a, b) -> np.ndarray:
    """Cosine similarity of row i of `a` and row i of `b`. Returns one value per row (not aggregated)."""
    return np.sum(_norm(a) * _norm(b), axis=-1)


def _check(ref, new, k: int) -> None:
    (rq, rd), (nq, nd) = ref, new
    if np.shape(rq)[0] != np.shape(nq)[0] or np.shape(rd)[0] != np.shape(nd)[0]:
        raise ValueError(
            f"ref and new must have the same number of queries and docs, "
            f"got ref=({np.shape(rq)[0]}, {np.shape(rd)[0]}) new=({np.shape(nq)[0]}, {np.shape(nd)[0]})"
        )
    if not 1 <= k <= np.shape(rd)[0]:
        raise ValueError(f"k must be between 1 and the number of docs ({np.shape(rd)[0]}), got k={k}")


def _scores(q, d):
    return _norm(q) @ _norm(d).T


def _ranking(scores):
    """Doc indices per query, best first. Stable, so ties go to the lower index."""
    return np.argsort(-scores, axis=1, kind="stable")


def recall_at_k(*, ref, new, k: int = 10) -> float:
    """Mean over queries of |top-k(ref) intersect top-k(new)| / k."""
    _check(ref, new, k)
    top_ref = _ranking(_scores(*ref))[:, :k]
    top_new = _ranking(_scores(*new))[:, :k]
    return float(np.mean([len(set(r) & set(n)) / k for r, n in zip(top_ref, top_new, strict=True)]))


def spearman_topk_per_query(*, ref, new, k: int = 50) -> np.ndarray:
    """Spearman rho per query between ref's top-k order and where those docs land in new's FULL ranking.

    With d_j = j - (position in new's full ranking of ref's j-th doc), rho = 1 - 6 * sum(d_j^2) / (k (k^2 - 1)).
    When new's top-k holds the same docs this is exactly Spearman's rho. A doc that new ranks above ref's
    top-k (an intruder) pushes ref's docs down, adds displacement, and lowers rho. Clamped to >= -1.
    """
    _check(ref, new, k)
    if k < 2:
        raise ValueError("spearman needs k >= 2")
    top_ref = _ranking(_scores(*ref))[:, :k]
    order_new = _ranking(_scores(*new))
    out = np.empty(len(top_ref))
    for i, (idx, order) in enumerate(zip(top_ref, order_new, strict=True)):
        position = np.empty(len(order), dtype=np.int64)
        position[order] = np.arange(len(order))
        d = np.arange(k) - position[idx]
        out[i] = max(-1.0, 1 - 6 * np.sum(d**2) / (k * (k * k - 1)))
    return out


def spearman_topk(*, ref, new, k: int = 50) -> float:
    """5th percentile (linear interpolation) over queries of `spearman_topk_per_query`: a floor, not a mean."""
    return float(np.percentile(spearman_topk_per_query(ref=ref, new=new, k=k), 5))
