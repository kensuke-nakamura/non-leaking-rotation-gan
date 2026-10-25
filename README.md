This repository is a placeholder for the code accompanying the paper, which is currently under review.

# Revisiting the Training of Generative Adversarial Networks via Non-Leaking Rotational Data Augmentation

We will present PyTorch implementations for the paper.

## Code overview

- `dcgan_and_gan/`: Our primary code using DCGAN and GAN, with `script_demo.sh` for a minimal demonstration and `script_full_experiment.sh` for the full experiments. This code was written for multiple experiments; see `wgangp_wgandiv_dragan/` for a simpler implementation.
- `wgangp_wgandiv_dragan/`: Code for WGAN-gp, WGAN-div, and DRAGAN using ADA and our method. Use `script_experiment.sh` for the full experiments.
- `synthetic_n-cyclic-rotation/`: Synthetic analysis of the convergence of our method.
- `styleGAN2ada_baseline/`: Baseline StyleGAN2-ADA. See `script_stylegan2ada_baseline.sh`.
- `styleGAN2ada_ours/`: StyleGAN2-ADA with our method. See `script_stylegan2ada_ours.sh`.
