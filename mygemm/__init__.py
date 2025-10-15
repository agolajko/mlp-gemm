# mygemm/__init__.py
import os
import torch

# --- Default: pure-Python (CPU/MPS) fallbacks ---


def _sgemm_fallback(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A.matmul(B)


def _sgemm_bias_relu_fallback(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(A.matmul(B) + bias)


# Public API (these names are what the rest of the package should call)
sgemm = _sgemm_fallback
sgemm_bias_relu = _sgemm_bias_relu_fallback

# --- Try to load CUDA extension (Linux+NVIDIA only). If it works, rebind to torch.ops ---
USE_FAKE = os.getenv("MYGEMM_FAKE", "0") == "1"
if not USE_FAKE:
    try:
        import mygemm_cuda  # registers torch.ops.mygemm.sgemm* on CUDA
        # If import succeeded, prefer the real ops:
        sgemm = torch.ops.mygemm.sgemm
        sgemm_bias_relu = torch.ops.mygemm.sgemm_bias_relu
        if os.getenv("MYGEMM_DEBUG", "0") == "1":
            print("[mygemm] Using CUDA extension ops")
    except Exception as e:
        if os.getenv("MYGEMM_DEBUG", "0") == "1":
            print(
                f"[mygemm] CUDA extension unavailable, falling back. Reason: {type(e).__name__}: {e}")
