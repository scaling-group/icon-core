#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import copy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, ModuleList
from torch.nn import MultiheadAttention as BaseMultiheadAttention


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, batch_first=True, sdpa="default", device=None, dtype=None):
        """
        batch_first: useless, we always use batch_first=True
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.sdpa = sdpa

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        # Dropout
        self.dropout = dropout

    def forward(
        self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, average_attn_weights=False
    ):
        """
        attn_mask: (batch_size, num_heads, query_len, key_len) or (query_len, key_len) which can be broadcasted
        key_padding_mask: (batch_size, key_len)
        """
        batch_size, seq_len, _ = query.shape

        # Project inputs to Q, K, V using einops
        q = einops.rearrange(self.q_proj(query), "b s (h d) -> b h s d", h=self.num_heads)
        k = einops.rearrange(self.k_proj(key), "b s (h d) -> b h s d", h=self.num_heads)
        v = einops.rearrange(self.v_proj(value), "b s (h d) -> b h s d", h=self.num_heads)

        # Handle attention masks
        if key_padding_mask is None and attn_mask is None:
            effective_attn_mask = None
        elif key_padding_mask is None and attn_mask is not None:
            effective_attn_mask = attn_mask
        elif key_padding_mask is not None and attn_mask is None:
            effective_attn_mask = einops.rearrange(key_padding_mask, "b s -> b 1 1 s")
        elif key_padding_mask is not None and attn_mask is not None:
            effective_attn_mask = attn_mask * einops.rearrange(key_padding_mask, "b s -> b 1 1 s")
        else:
            raise ValueError("Invalid input")

        if self.sdpa == "default":
            attn_fn = F.scaled_dot_product_attention
        else:
            raise ValueError(f"Unknown scaled dot-product attention: {self.sdpa}")

        attn_output = attn_fn(q, k, v, attn_mask=effective_attn_mask, dropout_p=self.dropout)

        # Concatenate heads and project output using einops
        attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # Compute attention weights (optional)
            attn_weights = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim**0.5)
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            if average_attn_weights:
                attn_weights = einops.reduce(attn_weights, "b h s d -> b s d", "mean")
            return attn_output, attn_weights
        else:
            return attn_output, None


def get_mha(mha, d_model, nhead, dropout, factory_kwargs):
    if mha == "built-in":
        attn = BaseMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True, **factory_kwargs
        )
    elif mha == "custom":
        attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True, sdpa="default", **factory_kwargs
        )
    else:
        raise ValueError(f"Unknown multi-head attention: {mha}")

    return attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, ff=True, mha="built-in", device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = get_mha(mha, d_model, nhead, dropout, factory_kwargs)
        layer_norm_eps = 1e-5

        self.ff = ff

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        if ff:
            self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout2 = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.dropout3 = Dropout(dropout)
            self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        # fmt: off
        if need_weights:
            attn_out, weight = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )[0]
        # fmt: on

        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        if self.ff:
            ff_out = self.linear1(x)
            ff_out = self.activation(ff_out)
            ff_out = self.dropout2(ff_out)
            ff_out = self.linear2(ff_out)
            ff_out = self.dropout3(ff_out)
            x = self.norm2(x + ff_out)

        if need_weights:
            return x, weight
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()

        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        weights = []
        for i in range(self.num_layers):
            if need_weights:
                x, weight = self.layers[i](x, mask, src_key_padding_mask, need_weights=True)
                weights.append(weight)
            else:
                x = self.layers[i](x, mask, src_key_padding_mask, need_weights=False)
        if need_weights:
            return x, weights
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, ff=True, mha="built-in", device=None, dtype=None):
        """
        Transformer decoder layer, no self attention
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.cross_attn = get_mha(mha, d_model, nhead, dropout, factory_kwargs)
        layer_norm_eps = 1e-5

        self.ff = ff

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        if ff:
            self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout2 = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.dropout3 = Dropout(dropout)
            self.activation = nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        need_weights=False,
    ):
        x = tgt
        # fmt: off
        if need_weights:
            attn_out, weight = self.cross_attn(
                x, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out = self.cross_attn(
                x, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
        # fmt: on

        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        if self.ff:
            ff_out = self.linear1(x)
            ff_out = self.activation(ff_out)
            ff_out = self.dropout2(ff_out)
            ff_out = self.linear2(ff_out)
            ff_out = self.dropout3(ff_out)
            x = self.norm2(x + ff_out)

        if need_weights:
            return x, weight
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()

        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        need_weights=False,
    ):
        x = tgt
        weights = []
        for i in range(self.num_layers):
            # fmt: off
            if need_weights:
                x, weight = self.layers[i](
                    x, memory,
                    tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask,
                    need_weights=True
                )
                weights.append(weight)
            else:
                x = self.layers[i](
                    x, memory,
                    tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask,
                    need_weights=False
                )
            # fmt: on
        if need_weights:
            return x, weights
        return x


class TransformerEnDecoder(nn.Module):
    def __init__(self, layer, num_layers):
        """
        a special transformer.
        query: memory + tgt
        key + value: memory
        So we perform self-attension inside memory and cross-attension between memory and tgt, simultaneously.
        """
        super().__init__()

        # use TransformerDecoderLayer as the basic layer
        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        need_weights=False,
    ):
        x = torch.cat([memory, tgt], dim=1)
        weights = []
        for i in range(self.num_layers):
            if need_weights:
                x, weight = self.layers[i](
                    x,
                    x[:, : memory.shape[1], :],
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    need_weights=True,
                )
                weights.append(weight)
            else:
                x = self.layers[i](
                    x,
                    x[:, : memory.shape[1], :],
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    need_weights=False,
                )

        memory_out = x[:, : memory.shape[1], :]
        tgt_out = x[:, memory.shape[1] :, :]
        if need_weights:
            return memory_out, tgt_out, weights
        return memory_out, tgt_out
