# Eigen-SAM: Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization

This repository provides the official implementation of Eigen-SAM, introduced in our NeurIPS 2024 paper:  
**[Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization](https://openreview.net/pdf?id=JFUhBY34SC)**.

## Introduction
Eigen-SAM periodically estimates the top eigenvalue of the Hessian matrix and incorporates its orthogonal component to the gradient into the perturbation, thereby achieving a more effective top eigenvalue regularization effect.

## Hyperparameters

- `--sam`: Choose between `SGD`, `SAM`, or `Eigen-SAM` optimization.
- `--rho`: Perturbation magnitude for SAM optimizer.
- `--alpha`: Step size for Eigen-SAM projection adjustment.
- `--freq`: Frequency of updating the largest eigenvector (for Eigen-SAM).

## Dependencies

```bash
pip install -r requirements.txt
```

## Training example

```bash
python train.py --sam Eigen-SAM --freq 100 --epochs 200 --alpha 0.2 --batch_size 256 --dataset CIFAR100 --num_workers 4 --rho 0.1
```

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{luo2025explicit,
  title={Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization},
  author={Luo, Haocheng and Truong, Tuan and Pham, Tung and Harandi, Mehrtash and Phung, Dinh and Le, Trung},
  journal={arXiv preprint arXiv:2501.12666},
  year={2025}
}
