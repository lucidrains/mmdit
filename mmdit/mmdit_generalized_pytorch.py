from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import (
    RMSNorm,
    FeedForward
)

from mmdit.mmdit_pytorch import (
    JointAttention
)

from hyper_connections import (
    HyperConnections,
    Residual
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# adaptive layernorm
# aim for clarity in generalized version

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
 
        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)

            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        x = self.ln(x)

        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
            x = x * gamma + beta

        return x

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_modalities: tuple[int, ...],
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        softclamp = False,
        softclamp_value = 50.,
        num_residual_streams = 1,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # residuals / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.attn_residual_fns = ModuleList([residual_klass(num_residual_streams, dim = dim) for dim in dim_modalities])
        self.ff_residual_fns = ModuleList([residual_klass(num_residual_streams, dim = dim) for dim in dim_modalities])

        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)

            self.to_post_branch_gammas = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias, 1.)

        # joint modality attention

        attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
        self.attn_layernorms = ModuleList(attention_layernorms)

        self.joint_attn = JointAttention(
            dim = dim_joint_attn,
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn,
            softclamp = softclamp,
            softclamp_value = softclamp_value,
        )

        # feedforwards

        feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
        self,
        *,
        modality_tokens: tuple[Tensor, ...],
        modality_masks: tuple[Tensor | None, ...] | None = None,
        time_cond = None
    ):
        assert len(modality_tokens) == self.num_modalities
        assert not (exists(time_cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        ln_kwargs = dict()

        if self.has_cond:
            ln_kwargs = dict(cond = time_cond)

            gammas = self.to_post_branch_gammas(time_cond)
            attn_gammas, ff_gammas = gammas.chunk(2, dim = -1)

        # attention layernorms

        modality_tokens, modality_tokens_residual_fns = tuple(zip(*[residual_fn(modality_token) for residual_fn, modality_token in zip(self.attn_residual_fns, modality_tokens)]))

        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.attn_layernorms)]

        # attention

        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

        # post attention gammas

        if self.has_cond:
            attn_gammas = attn_gammas.split(self.dim_modalities, dim = -1)
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, attn_gammas)]

        # add attention residual

        modality_tokens = [add_attn_residual(tokens) for add_attn_residual, tokens in zip(modality_tokens_residual_fns, modality_tokens)]

        # handle feedforward adaptive layernorm

        modality_tokens, modality_tokens_residual_fns = tuple(zip(*[residual_fn(modality_token) for residual_fn, modality_token in zip(self.ff_residual_fns, modality_tokens)]))

        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.ff_layernorms)]

        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # post feedforward gammas

        if self.has_cond:
            ff_gammas = ff_gammas.split(self.dim_modalities, dim = -1)
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, ff_gammas)]

        # add feedforward residual

        modality_tokens = [add_residual_fn(tokens) for add_residual_fn, tokens in zip(modality_tokens_residual_fns, modality_tokens)]

        # returns

        return modality_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_modalities,
        final_norms = True,
        num_residual_streams = 4,
        **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        blocks = [MMDiTBlock(dim_modalities = dim_modalities, num_residual_streams = num_residual_streams, **block_kwargs) for _ in range(depth)]
        self.blocks = ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens: tuple[Tensor, ...],
        modality_masks: tuple[Tensor | None, ...] | None = None,
        time_cond = None
    ):

        modality_tokens = [self.expand_streams(modality) for modality in modality_tokens]

        for block in self.blocks:
            modality_tokens = block(
                time_cond = time_cond,
                modality_tokens = modality_tokens,
                modality_masks = modality_masks
            )

        modality_tokens = [self.reduce_streams(modality) for modality in modality_tokens]

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

        return tuple(modality_tokens)
