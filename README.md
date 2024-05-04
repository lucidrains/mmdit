<img src="./mmdit.png" width="300px"></img>

## MMDiT (wip)

Implementation of a single layer of the MMDiT, proposed by Esser et al. in <a href="https://arxiv.org/abs/2403.03206">Stable Diffusion 3</a>, in Pytorch and Jax

Besides a straight reproduction, will also generalize to > 2 modalities, as I can envision an MMDiT for images, audio, and text.

Will also offer an improvised variant of the attention that adaptively selects the weights to use through learned gating.

## Install

```bash
$ pip install mmdit
```

## Usage

```python
import torch
from mmdit import MMDiTBlock

# define mm dit block

block = MMDiTBlock(
    dim_joint_attn = 512,
    dim_cond = 256,
    dim_text = 768,
    dim_image = 512,
    qk_rmsnorm = True
)

# mock inputs

time_cond = torch.randn(1, 256)

text_tokens = torch.randn(1, 512, 768)
text_mask = torch.ones((1, 512)).bool()

image_tokens = torch.randn(1, 1024, 512)

# single block forward

text_tokens_next, image_tokens_next = block(
    time_cond = time_cond,
    text_tokens = text_tokens,
    text_mask = text_mask,
    image_tokens = image_tokens
)
```

## Citations

```bibtex
@article{Esser2024ScalingRF,
    title   = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
    author  = {Patrick Esser and Sumith Kulal and A. Blattmann and Rahim Entezari and Jonas Muller and Harry Saini and Yam Levi and Dominik Lorenz and Axel Sauer and Frederic Boesel and Dustin Podell and Tim Dockhorn and Zion English and Kyle Lacey and Alex Goodwin and Yannik Marek and Robin Rombach},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2403.03206},
    url     = {https://api.semanticscholar.org/CorpusID:268247980}
}
```
