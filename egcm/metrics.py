import numpy as np

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def cosine_rows(a, b) -> np.ndarray:
    return np.sum(_norm(a) * _norm(b), axis=-1)

def _topk(q, d, k):
    s = _norm(q) @ _norm(d).T
    return np.argsort(-s, axis=1)[:, :k], s

def recall_at_k(q_ref, d_ref, q_new, d_new, k=10) -> float:
    ref, _ = _topk(q_ref, d_ref, k)
    new, _ = _topk(q_new, d_new, k)
    return float(np.mean([len(set(r) & set(n)) / k for r, n in zip(ref, new)]))

def spearman_topk(q_ref, d_ref, q_new, d_new, k=50) -> float:
    ref, _ = _topk(q_ref, d_ref, k)
    _, s_new = _topk(q_new, d_new, k)
    out = []
    for i, idx in enumerate(ref):
        new_rank = np.argsort(np.argsort(-s_new[i, idx]))
        out.append(1 - 6 * np.sum((np.arange(k) - new_rank) ** 2) / (k * (k * k - 1)))
    return float(np.min(out))
