"""Attention masks in the layout AttentionCF expects."""

import torch

# Added to the logits of padded keys. Finite in fp16 (unlike finfo(float32).min, which upstream uses) and far below
# any real logit, so exp() of it is 0 in fp32 and fp16.
MASK_NEG = -1e4


def key_mask(valid: torch.Tensor) -> torch.Tensor:
    """valid: (1, S) float, 1 for real tokens and 0 for padding -> additive key-major mask (1, S, 1, 1).

    The mask is over keys only and broadcasts over queries, like upstream's bidirectional padding mask. Padded
    query positions still produce outputs; pooling ignores them. Batch is 1: `view(1, -1, 1, 1)` keeps shape ints
    out of the traced graph.
    """
    return ((1.0 - valid) * MASK_NEG).view(1, -1, 1, 1)
