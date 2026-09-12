# Alt-CC-INR: Alternating Optimization Framework with Cross-Correlated Loss and Implicit Neural Representation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-Stable-EE4C2C.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper:
**"Alternating Optimization Framework with Cross-Correlated Loss and Implicit Neural Representation for Electromagnetic Inverse Scattering"**.

## Overview

Electromagnetic inverse scattering imaging of high-contrast targets is highly nonlinear and ill-posed. While joint-optimization neural solvers (like PINNs) suffer from severe gradient conflicts and empirical step-size bottlenecks, this work proposes a novel **Heterogeneous Optimization Paradigm (Alt-CC-INR)**. 

By strictly mapping distinct mathematical variables to their most suitable solvers, this framework profoundly improves reconstruction fidelity and robustness:
1. **Dynamic Physical Fields (Contrast Sources):** Solved via an analytical, exact zero-padded 2D-FFT accelerated Polak-Ribière Conjugate Gradient (PR-CG).
2. **Static Medium Profile (Permittivity/Conductivity):** Inferred via a stochastic deep learning optimizer (Adam) driving a continuous Implicit Neural Representation (INR) equipped with Fourier features.

## Key Features

- **Heterogeneous Alternating Engine:** Completely abandons purely empirical step-size stochastic descents for physical field evolution, preventing local minima trapping.
- **Cross-Correlated Global Regularizer:** Employs a dynamically annealed cross-correlated loss to forcefully couple predicted contrasts and contrast sources with far-field observations.
- **2D-FFT Operator Acceleration:** Bypasses massive memory overheads and achieves $\mathcal{O}(N^2 \log N)$ complexity for domain Green's function integration.
- **Comprehensive Ablation Baselines:** Contains built-in implementations of `CC-PINN`, `CC-CSI`, `Alt-INR`, and `CC-PINN (Alt)` (Alt-Adam-INR) to strictly validate the algorithmic contributions.

## Requirements

The code has been tested with the following environment:
- Python >= 3.8
- PyTorch >= 1.12.0 (CUDA enabled recommended)
- NumPy, SciPy, Matplotlib

Install the required dependencies:
```bash
pip install torch numpy scipy matplotlib
```

## Usage

The provided script `Alt-CC-INR.py` is an all-in-one robust benchmarking suite. It automatically performs data loading, multi-seed Monte Carlo executions (default: 11 runs), and performance evaluation (PSNR/SSIM), followed by automatic PDF report generation.

Simply run:
```bash
python Alt-CC-INR.py
```

### What does the script do?
1. **Data Generation:** Generates the Ground Truth matrices for highly challenging profiles (e.g., "Austria", "Bowtie-Cross") and calculates physics matrices (Green's kernels).
2. **Benchmarking:** Sequentially executes `CC-CSI`, `CC-PINN`, `Alt-CC-INR`, and the critical ablation baseline `Alt-Adam-INR`. 
3. **Visualization:** Automatically generates highly polished academic plots in the `robustness_test_sim/` or `robustness_test_hop/` directories:
   - Reconstructed Epsilon & Sigma images (`.pdf`)
   - Step-PSNR / Time-PSNR Convergence Curves
   - Final PSNR Statistical Boxplots across 11 Monte Carlo runs.

## Implemented Baselines in the Code

To ensure full transparency and reproducibility, the script includes the exact implementations of the following algorithms discussed in the manuscript:
- `run_cc_csi()`: Traditional purely discrete numerical iteration (CC-CSI).
- `run_cc_pinn_variant(classic_mode=False)`: Joint-optimization CC-PINN.
- `run_alt_cc_inr(classic_mode=False)`: Our proposed **Alt-CC-INR**.
- `run_alt_adam_inr()`: The decoupled Adam-only ablation baseline **CC-PINN (Alt)** to prove the necessity of analytical PR-CG.
- *(Ablation)* `run_source_inr_ablation()`: Contrast-Source parameterized INR baselines (comparing INR design philosophies).

## Citation

If you find this code or our framework useful for your research, please consider citing our paper:

```bibtex
@article{sun202Xaltccinr,
  title={Alternating Optimization Framework with Cross-Correlated Loss and Implicit Neural Representation for Electromagnetic Inverse Scattering},
  author={Sun, Shilong},
  journal={IEEE Transactions on Antennas and Propagation},
  year={202X},
  volume={--},
  number={--},
  pages={--}
}
```
*(Note: The citation will be updated once the manuscript is accepted and officially published.)*

## Contact
For any questions regarding the code or the paper, please open an issue in this repository or contact:
**Shilong Sun** - [sunshilong@nudt.edu.cn](mailto:sunshilong@nudt.edu.cn)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
