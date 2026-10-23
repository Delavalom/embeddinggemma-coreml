# Copyright 2026 the HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Delavalom, 2026: extracted the tower modules from transformers v5.20.2 (modeling_embedding_gemma2.py,
# modeling_gemma4.py, image_processing_gemma4.py) into one test-only module; config objects replaced by arguments,
# masks built inline, no output records or kernels; maths unchanged. See NOTICE.

"""Upstream towers, copied from vendor/reference (transformers v5.20.2, Apache-2.0). Tests only.

Attribute names are upstream's, so a state dict with Hugging Face key names (minus the `language_model.`,
`vision_tower.` or `embed_vision.` prefix) loads into these modules directly. Builds on tests/upstream.py.
Our additions: `Cfg`, the `# --- <file>` headers, the trailing comments naming each upstream source, and comments
marked `# (added)`.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import upstream as up
from torch import nn

GELU = nn.GELU(approximate="tanh")  # ACT2FN["gelu_pytorch_tanh"]


@dataclass
class Cfg:  # (added) the config fields the copied code reads
    hidden: int
    heads: int
    ffn: int
    eps: float = 1e-6


# --- modeling_embedding_gemma2.py -----------------------------------------------------------------------------
class MLP(nn.Module):  # EmbeddingGemma2MLP / Gemma4VisionMLP (vision wraps each linear, see ClippableLinear)
    def __init__(self, hidden, ffn, linear=nn.Linear):
        super().__init__()
        self.gate_proj = linear(hidden, ffn, bias=False)
        self.up_proj = linear(hidden, ffn, bias=False)
        self.down_proj = linear(ffn, hidden, bias=False)
        self.act_fn = GELU

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class TextPLE(nn.Module):  # EmbeddingGemma2TextPLE
    def __init__(self, hidden, layers, ple_dim, eps):
        super().__init__()
        self.num_hidden_layers = layers
        self.hidden_size_per_layer_input = ple_dim
        self.per_layer_model_projection = nn.Linear(hidden, layers * ple_dim, bias=False)
        self.per_layer_model_projection_scale = hidden**-0.5
        self.per_layer_projection_norm = up.RMSNorm(ple_dim, eps=eps)

    def forward(self, inputs_embeds):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(per_layer_projection)


class TextPLEBlock(nn.Module):  # EmbeddingGemma2TextPLEBlock
    def __init__(self, hidden, ple_dim, eps):
        super().__init__()
        self.act_fn = GELU
        self.per_layer_input_gate = nn.Linear(hidden, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, hidden, bias=False)
        self.post_per_layer_input_norm = up.RMSNorm(hidden, eps=eps)

    def forward(self, hidden_states, per_layer_input):
        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        return residual + hidden_states


class TextLayer(nn.Module):  # EmbeddingGemma2EncoderLayer
    def __init__(self, c: Cfg, ple_dim, kv_heads, head_dim):
        super().__init__()
        self.self_attn = up.TextAttention(c.hidden, c.heads, kv_heads, head_dim, c.eps)
        self.mlp = MLP(c.hidden, c.ffn)
        self.input_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.post_attention_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.pre_feedforward_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.post_feedforward_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.layer_scalar = nn.Buffer(torch.ones(1))
        self.ple_block = TextPLEBlock(c.hidden, ple_dim, c.eps)

    def forward(self, hidden_states, per_layer_input, attention_mask=None, position_embeddings=None):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.ple_block(hidden_states, per_layer_input)

        hidden_states *= self.layer_scalar
        return hidden_states


class TextModel(nn.Module):  # EmbeddingGemma2TextModel (inputs_embeds path; no padding mask: batch 1, unpadded)
    def __init__(self, c: Cfg, vocab, out_dim, ple_dim, layer_types, head_dim, kv_heads, theta):
        super().__init__()
        self.layer_types, self.head_dim, self.theta = layer_types, head_dim, theta
        self.embed_tokens = nn.Embedding(vocab, c.hidden)
        self.embed_scale = c.hidden**0.5
        self.layers = nn.ModuleList(
            [TextLayer(c, ple_dim, kv_heads[t], head_dim[t]) for t in layer_types]
        )
        self.norm = up.RMSNorm(c.hidden, eps=c.eps)
        self.ple = TextPLE(c.hidden, len(layer_types), ple_dim, c.eps)
        self.embedding_projection = nn.Linear(c.hidden, out_dim, bias=False)

    def embed(self, input_ids):  # EmbeddingGemma2TextScaledWordEmbedding.forward
        return self.embed_tokens(input_ids) * torch.tensor(self.embed_scale, dtype=self.embed_tokens.weight.dtype)

    def forward(self, inputs_embeds):
        per_layer_inputs = self.ple(inputs_embeds)
        position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.layer_types):
            inv = up.text_inv_freq(self.theta[layer_type], self.head_dim[layer_type])  # (added) rotary_emb
            position_embeddings[layer_type] = up.text_cos_sin(inv, position_ids)
        for i, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                per_layer_inputs[:, :, i, :],
                attention_mask=None,
                position_embeddings=position_embeddings[self.layer_types[i]],
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.embedding_projection(hidden_states)
        return hidden_states


class MultimodalEmbedder(nn.Module):  # EmbeddingGemma2MultimodalEmbedder
    def __init__(self, mm_hidden, text_hidden, eps):
        super().__init__()
        self.embedding_projection = nn.Linear(mm_hidden, text_hidden, bias=False)
        self.embedding_pre_projection_norm = up.RMSNorm(mm_hidden, eps=eps, with_scale=False)

    def forward(self, inputs_embeds):
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


# --- modeling_gemma4.py: vision -------------------------------------------------------------------------------
class ClippableLinear(nn.Module):  # Gemma4ClippableLinear with use_clipped_linears = False
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class PatchEmbedder(nn.Module):  # Gemma4VisionPatchEmbedder
    def __init__(self, hidden, patch, pos_size):
        super().__init__()
        self.input_proj = nn.Linear(3 * patch**2, hidden, bias=False)
        self.position_embedding_table = nn.Parameter(torch.ones(2, pos_size, hidden))

    def _position_embeddings(self, pixel_position_ids, padding_positions):
        clamped_positions = pixel_position_ids.clamp(min=0)
        x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
        position_embeddings = x_emb + y_emb
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values)
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings


class VisionLayer(nn.Module):  # Gemma4VisionEncoderLayer
    def __init__(self, c: Cfg, head_dim):
        super().__init__()
        self.self_attn = up.VisionAttention(c.hidden, c.heads, c.heads, head_dim, c.eps)
        for p in ("q", "k", "v", "o"):  # (added) Gemma4VisionAttention builds these as Gemma4ClippableLinear
            lin = getattr(self.self_attn, f"{p}_proj")
            setattr(self.self_attn, f"{p}_proj", ClippableLinear(lin.in_features, lin.out_features))
        self.mlp = MLP(c.hidden, c.ffn, linear=ClippableLinear)
        self.input_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.post_attention_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.pre_feedforward_layernorm = up.RMSNorm(c.hidden, eps=c.eps)
        self.post_feedforward_layernorm = up.RMSNorm(c.hidden, eps=c.eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _Encoder(nn.Module):  # (added) holds `layers` so keys read `encoder.layers.{i}.…`
    def __init__(self, c: Cfg, head_dim, layers):
        super().__init__()
        self.layers = nn.ModuleList([VisionLayer(c, head_dim) for _ in range(layers)])


def avg_pool_by_positions(hidden_states, pixel_position_ids, length):  # Gemma4VisionPooler._avg_pool_by_positions
    input_seq_len = hidden_states.shape[1]
    k = int((input_seq_len // length) ** 0.5)
    k_squared = k**2
    clamped_positions = pixel_position_ids.clamp(min=0)
    max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
    output = weights.transpose(1, 2) @ hidden_states.float()
    return output.to(hidden_states.dtype)


class VisionModel(nn.Module):  # Gemma4VisionModel (+ Gemma4VisionEncoder), one unpadded image, standardize off
    def __init__(self, c: Cfg, head_dim, layers, patch, pool, pos_size, theta):
        super().__init__()
        self.head_dim, self.pool, self.theta, self.hidden = head_dim, pool, theta, c.hidden
        self.patch_embedder = PatchEmbedder(c.hidden, patch, pos_size)
        self.encoder = _Encoder(c, head_dim, layers)

    def forward(self, pixel_values, pixel_position_ids):
        output_length = pixel_values.shape[-2] // (self.pool * self.pool)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        hidden_states = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        position_embeddings = up.vision_cos_sin(up.axial_inv_freq(self.theta, self.head_dim), pixel_position_ids)
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None,
                                  position_ids=pixel_position_ids)
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)  # Gemma4VisionPooler
        hidden_states = avg_pool_by_positions(hidden_states, pixel_position_ids, output_length)
        return hidden_states.float() * self.hidden**0.5


# --- image_processing_gemma4.py -------------------------------------------------------------------------------
def convert_image_to_patches(image, patch_size):
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image
