<img src="./mmdit.png" width="400px"></img>

## MMDiT (wip)

Implementation of a single layer of the MMDiT, proposed by Esser et al. in <a href="https://arxiv.org/abs/2403.03206">Stable Diffusion 3</a>, in Pytorch and Jax

Besides a straight reproduction, will also generalize to > 2 modalities, as I can envision an MMDiT for images, audio, and text.

Will also offer an improvised variant of the attention that adaptively selects the weights to use through learned gating.

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
