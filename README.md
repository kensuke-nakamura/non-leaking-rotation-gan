This repository is a placeholder for the code accompanying the paper, which is currently under review.

# Revisiting the Training of Generative Adversarial Networks via Non-Leaking Rotational Data Augmentation

We will present PyTorch implementations for the paper.

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
