# Revisiting the Training of Generative Adversarial Networks via Non-Leaking Rotational Data Augmentation

Official code repository for the paper published in *Pattern Analysis and Applications*, volume 29, article 225 (2026).

This repository provides PyTorch implementations for the paper.

## Code overview

### `dcgan_and_gan/`

Our primary implementation using DCGAN and GAN, supporting multiple experiments.

- Minimal demonstration: `script_demo.sh`
- Full experiments: `script_full_experiment.sh`

For a simpler implementation, see `wgangp_wgandiv_dragan/` below.

### `wgangp_wgandiv_dragan/`

WGAN-gp, WGAN-div, and DRAGAN using ADA and our method.

- Full experiments: `script_experiment.sh`

### `synthetic_n-cyclic-rotation/`

Synthetic analysis of the convergence of our method.

### `styleGAN2ada_baseline/`

Baseline StyleGAN2-ADA.

- Experiment script: `script_stylegan2ada_baseline.sh`

### `styleGAN2ada_ours/`

StyleGAN2-ADA with our method.

- Experiment script: `script_stylegan2ada_ours.sh`



## Citation

If you use this work, please cite:

Nakamura, K., Sohn, BS., Korman, S. *et al.* Revisiting the training of generative adversarial networks via non-leaking rotational data augmentation. *Pattern Anal Applic* **29**, 225 (2026). [https://doi.org/10.1007/s10044-026-01809-6](https://doi.org/10.1007/s10044-026-01809-6)
