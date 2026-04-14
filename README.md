# Vision Transformer (ViT) — Notebook reimplementation

This repository contains a concise reimplementation of the Vision Transformer described in, done for CS480:

Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" — https://openreview.net/pdf?id=YicbFdNTTy

- **Notebook**: [transformer.ipynb](transformer.ipynb)

**How this mirrors the paper**

- **Patch embedding**: Images are split into non-overlapping patches (paper default $P=16$). Each patch is flattened and projected to an embedding vector; sequence length is $N = \frac{HW}{P^2}$ and a learnable `CLS` token is prepended.
- **Positional embeddings**: A learned 1D positional embedding is added to the patch embeddings, matching the paper's scheme.
- **Transformer encoder**: The notebook implements stacked Transformer encoder blocks composed of LayerNorm, Multi-Head Self-Attention, residual connections, and an MLP (feed-forward) block with GELU activation — following the paper's block structure.
- **Classification head**: The final prediction uses the `CLS` token followed by a linear classification head, as in the original design.
- **Training & evaluation**: A minimal training loop (loss, optimizer, basic augmentation/evaluation) demonstrates how to train and validate the model on small datasets for experimentation.

**Limitations**

- This notebook is a compact reimplementation: it does not reproduce large-scale pretraining (ImageNet21k / JFT) or the full experimental sweep from the paper.
- Model sizes, batch sizes, and data pipelines are scaled down for runnable experiments on a single machine or Colab.
