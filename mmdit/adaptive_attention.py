from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, reduce, rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

Linear = partial(nn.Linear, bias = False)

# class

class AdaptiveAttention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        num_adaptive_weights = 1 # 1 becomes regular self attention with no gating
    ):
        """
        this idea was inspired by adaptive convs from gigagan https://arxiv.org/abs/2303.05511

        ein notation:
        b - batch
        n - sequence
        h - heads
        d - feature dimension
        w - adaptive weight
        """

        super().__init__()
        assert num_adaptive_weights >= 1

        has_gating = num_adaptive_weights > 1
        self.has_gating = has_gating

        dim_inner = dim_head * heads
        scale = dim_head ** -0.5
        self.scale = scale

        self.to_qkv = nn.Sequential(
            Linear(dim, dim_inner * num_adaptive_weights * 3),
            Rearrange('b n (qkv h d w) -> qkv b h n d w', qkv = 3, h = heads, w = num_adaptive_weights)
        )

        if has_gating:
            self.to_gates = nn.Sequential(
                Linear(dim, num_adaptive_weights),
                nn.Softmax(dim = -1)
            )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            Linear(dim_inner, dim * num_adaptive_weights),
            Rearrange('b n (d w) -> b n d w', w = num_adaptive_weights)
        )        

    def forward(
        self,
        x,
        mask = None
    ):
        has_gating = self.has_gating

        qkv = self.to_qkv(x)

        # token dependent choosing of which weight

        if has_gating:
            gates = self.to_gates(x)
            qkv_gates = rearrange(gates, 'b n w -> b 1 n 1 w')

            qkv = reduce(qkv * qkv_gates, '... w -> ...', 'sum')
        else:
            qkv = rearrange(qkv, '... 1 -> ...')

        # usual self attention logic

        q, k, v = qkv

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # again, adaptive weight on the outward projection
        # with gates from above

        out = self.to_out(out)

        if has_gating:
            out_gates = rearrange(gates, 'b n w -> b n 1 w')
            out = reduce(out * out_gates, '... w -> ...', 'sum')
        else:
            out = rearrange(out, '... 1 -> ...')

        return out

# example

if __name__ == '__main__':
    adaptive_attn = AdaptiveAttention(
        dim = 512,
        num_adaptive_weights = 4
    )

    text_tokens = torch.randn(1, 256, 512)
    image_tokens = torch.randn(1, 1024, 512)
    audio_tokens = torch.randn(1, 128, 512)

    tokens = torch.cat((text_tokens, image_tokens, audio_tokens), dim = -2)

    out = adaptive_attn(tokens)
