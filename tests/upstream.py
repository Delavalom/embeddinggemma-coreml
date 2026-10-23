# Copyright 2026 the HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Delavalom, 2026: extracted the minimal functions from transformers v5.20.2 (modeling_gemma4.py,
# modeling_embedding_gemma2.py, image_processing_gemma4.py) into one test-only module; config lookups replaced by
# arguments, attention called through eager_attention_forward directly; maths unchanged. See NOTICE.

"""Upstream reference maths, copied from vendor/reference (transformers v5.20.2, Apache-2.0). Tests only.

The convert group has no transformers, so the minimal pure functions are copied here. Changes from upstream are
plumbing only: config lookups became arguments, `attention_scaling` is 1.0 (what upstream sets for the 'default'
and 'axial' rope types), and the attention classes call `eager_attention_forward` directly instead of going
through ALL_ATTENTION_FUNCTIONS. Every function body that does maths is as upstream wrote it.

Our additions: the `# --- <file>` section headers, the trailing comments naming each function's upstream source,
and explanatory comments marked `# (added)`.
"""

import torch
from torch import nn


# --- modeling_gemma4.py: Gemma4RMSNorm (EmbeddingGemma2RMSNorm is identical) ----------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


# --- modeling_embedding_gemma2.py: text RoPE ------------------------------------------------------------------
def text_inv_freq(base, dim):  # EmbeddingGemma2RotaryEmbedding.compute_default_rope_parameters
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq


def text_cos_sin(inv_freq, position_ids, attention_scaling=1.0):  # EmbeddingGemma2RotaryEmbedding.forward
    freqs = position_ids[..., None].float() * inv_freq.to(dtype=torch.float)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin


def rotate_half(x):  # identical in modeling_embedding_gemma2.py and modeling_gemma4.py
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_qk(q, k, cos, sin, unsqueeze_dim=1):  # modeling_embedding_gemma2.apply_rotary_pos_emb
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# --- image_processing_gemma4.py: Step 5, patch position ids ---------------------------------------------------
def patch_positions(patch_height, patch_width):
    patch_grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
    stacked_grid = torch.stack(patch_grid, dim=-1)
    return stacked_grid.reshape(patch_height * patch_width, 2)  # (added) (x, y) per patch, row-major


# --- modeling_gemma4.py: vision RoPE --------------------------------------------------------------------------
def axial_inv_freq(base, dim):  # Gemma4VisionRotaryEmbedding.compute_axial_rope_parameters
    spatial_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
    return inv_freq


def recomposition_frequencies(freq):  # Gemma4VisionRotaryEmbedding.recomposition_frequencies
    # in contrast to other 2D rope modules, interleave grids as H-H-W-W
    # (added) index 0 of the position ids is x (image_processing_gemma4 Step 5), so "freq_h" holds the x angles
    freq_h, freq_w = freq[:, :, 0], freq[:, :, 1]
    return torch.cat([freq_h, freq_h, freq_w, freq_w], dim=-1)


def vision_cos_sin(inv_freq, position_ids, attention_scaling=1.0):  # Gemma4VisionRotaryEmbedding.forward
    freqs = position_ids[..., None].float() * inv_freq.to(dtype=torch.float)
    cos = freqs.cos() * attention_scaling
    sin = freqs.sin() * attention_scaling
    cos = recomposition_frequencies(cos)
    sin = recomposition_frequencies(sin)
    return cos, sin


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):  # modeling_gemma4.apply_rotary_pos_emb
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2):  # modeling_gemma4
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        apply_rotary_pos_emb(x=x_parts[k], cos=cos_parts[k], sin=sin_parts[k], unsqueeze_dim=unsqueeze_dim)
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


# --- attention (identical in both files) ----------------------------------------------------------------------
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None,
                            **kwargs):
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class _Attention(nn.Module):
    """Projections and norms as in EmbeddingGemma2Attention / Gemma4VisionAttention (bias=False, scaling=1.0)."""

    def __init__(self, hidden, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = heads // kv_heads
        self.scaling = 1.0
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(dim=head_dim, eps=eps)
        self.k_norm = RMSNorm(dim=head_dim, eps=eps)
        self.v_norm = RMSNorm(dim=head_dim, eps=eps, with_scale=False)


class TextAttention(_Attention):
    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):  # EmbeddingGemma2Attention
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qk(query_states, key_states, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class VisionAttention(_Attention):
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None):
        # Gemma4VisionAttention.forward
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

