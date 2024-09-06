# Kolmogorov–Arnold Transformer: A PyTorch Implementation

<p align="center">
<a href="https://arxiv.org/abs/2405.07992" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>
<a href="https://colab.research.google.com/drive/1DTJRsPczV0pOwmFhEjSWyI2NqQoR_u-K?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p align="center">
<img src="assets/KAT.png" width="400"> <br>
</p>

This is a PyTorch/GPU implementation of the paper **Kolmogorov–Arnold Transformer (KAT)**, which replace the MLP layers in vision transformer with KAN layers.

**Kolmogorov–Arnold Transformer**

 📝[[Paper](https://arxiv.org/abs/2407.06182)] </>[[code](https://github.com/Adamdad/kat)]

Xingyi Yang, Xinchao Wang

National University of Singapore

### Key Insight:

The KAT model integrates KANs into transformers for large-scale training scenarios such as ImageNet, showing significant performance improvements.

## Installation

```shell
# install torch and other things
pip install timm==1.0.3
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
pip install -e .
```
please refer to `https://github.com/Adamdad/rational_kat_cu.git` for the cuda rational function installation

## Model Checkpoints
Download pre-trained models or access training checkpoints:

|Model | Param| Top1 |Link|
| ---|---|---| ---|
|KAT-T| 5.7M | 74.6| [link](https://github.com/Adamdad/kat/releases/download/checkpoint/kat_small_patch16_224_32487885cf13d2c14e461c9016fac8ad43f7c769171f132530941e930aeb5fe2.pth)
|KAT-T (Pretrained)| 5.7M | 75.7| |
|KAT-S| 22.1M | 81.2| [link](https://github.com/Adamdad/kat/releases/download/checkpoint/kat_tiny_patch16_224_1f3ad3b2e69821f3d412f2924cf159a0e266f142d739cb68f68f796f5a0fe289.pth)
|KAT-S (Pretrained)|22.1M | 82.0|
| KAT-B|86.6M| 82.3 | [link](https://github.com/Adamdad/kat/releases/download/checkpoint/kat_base_patch16_224_abff874d925d756d15cde97303f772a3460ddbd44b9c53fb9ce5cf15be230fb6.pth)

## Model Training

```shell
bash scripts/train_kat_tiny_8x128.sh
```

## Acknowledgments
We extend our gratitude to the authors of [rational_activations]() for their contributions to CUDA rational function implementations that inspired parts of this work.