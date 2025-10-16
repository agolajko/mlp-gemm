
# MLP GEMM Custom CUDA Matrix Multiplication

Testing CUDA kernels for matrix multiplication with fused operations.

## Installation

```bash
pip install -e .
```

## Quick Start/ Benchmarking

```bash
python -m mygemm.bench --device cuda
```

## Kernel Variants

- **Plain**: Naive baseline implementation
- **Fused**: Bias + ReLU fused into single kernel
- **Optimized**: Tiled computation with shared memory, bank conflict resolution

## Development

```bash
# Rebuild after changing CUDA code
pip install -e . --force-reinstall --no-deps

# Debug mode
CUDA_LAUNCH_BLOCKING=1 python -m mygemm.bench
```

## Architecture

```
csrc/
├── mygemm_kernels.cu  # Naive & fused kernels
└── bank_extra.cu      # Optimized kernel
mygemm/
├── functional.py      # Autograd functions
├── modules.py         # nn.Module wrappers
└── bench.py           # Benchmarking
```

## Acknowledgments

Optimized kernel based on [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
