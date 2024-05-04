from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import FeedForward

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_inputs: Tuple[int],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(**attend_kwargs)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: Tuple[Tensor],
        masks: Optional[Tuple[Optional[Tensor]]] = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_text,
        dim_image,
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            self.cond_dims = (
                *((dim_text,) * 6),
                *((dim_image,) * 6)
            )

            self.to_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_cond, sum(self.cond_dims))
            )

        # handle adaptive norms

        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        # attention and feedforward

        self.attn = Attention(
            dim = dim_joint_attn,
            dim_inputs = (dim_text, dim_image),
            dim_head = dim_head,
            heads = heads
        )

        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        self.image_ff = FeedForward(dim_image, **ff_kwargs)

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None
    ):
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_pre_attn_beta,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_pre_ff_beta,
                text_post_ff_gamma,
                image_pre_attn_gamma,
                image_pre_attn_beta,
                image_post_attn_gamma,
                image_pre_ff_gamma,
                image_pre_ff_beta,
                image_post_ff_gamma,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm

        text_tokens_residual = text_tokens
        image_tokens_residual = image_tokens

        text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta

        # attention

        text_tokens, image_tokens = self.attn(
            inputs = (text_tokens, image_tokens),
            masks = (text_mask, None)
        )

        # condition attention output

        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            image_tokens = image_tokens * image_post_attn_gamma

        # add attention residual

        text_tokens = text_tokens + text_tokens_residual
        image_tokens = image_tokens + image_tokens_residual

        # handle feedforward adaptive layernorm

        text_tokens_residual = text_tokens
        image_tokens_residual = image_tokens

        text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta
            image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta

        # feedforward

        text_tokens = self.text_ff(text_tokens)
        image_tokens = self.image_ff(image_tokens)

        # condition feedforward output

        if self.has_cond:
            text_tokens = text_tokens * text_post_ff_gamma
            image_tokens = image_tokens * image_post_ff_gamma

        # add feedforward residual

        text_tokens = text_tokens + text_tokens_residual
        image_tokens = image_tokens + image_tokens_residual

        # return output of block

        return text_tokens, image_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        **block_kwargs
    ):
        super().__init__()
        self.blocks = ModuleList([])

        for _ in range(depth):
            block = MMDiTBlock(**block_kwargs)
            self.blocks.append(block)

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None
    ):
        for block in self.blocks:
            text_tokens, image_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                text_mask = text_mask
            )

        return text_tokens, image_tokens
