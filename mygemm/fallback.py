import torch


def sgemm_fake(A, B):
    # behaves like your CUDA op: expects 2D tensors, returns A @ B
    return A.matmul(B)


def sgemm_bias_relu_fake(A, B, bias):
    # (A @ B) + bias, then ReLU
    out = A.matmul(B) + bias
    return torch.relu(out)
