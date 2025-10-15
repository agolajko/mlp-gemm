# mygemm/functional.py
import torch
import mygemm
# from . import sgemm as _sgemm, sgemm_bias_relu as _sgemm_bias_relu


class SGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return mygemm.sgemm(A, B)

    @staticmethod
    def backward(ctx, dC):
        A, B = ctx.saved_tensors
        dA = dC.matmul(B.transpose(0, 1)) if ctx.needs_input_grad[0] else None
        dB = A.transpose(0, 1).matmul(dC) if ctx.needs_input_grad[1] else None
        return dA, dB


class SGemmBiasReluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias):
        Lin = mygemm.sgemm(A, B) + bias
        C = torch.relu(Lin)
        ctx.save_for_backward(A, B, bias, Lin)
        return C

    @staticmethod
    def backward(ctx, dC):
        A, B, bias, Lin = ctx.saved_tensors
        dZ = dC * (Lin > 0).to(dC.dtype)
        dA = dZ.matmul(B.transpose(0, 1)) if ctx.needs_input_grad[0] else None
        dB = A.transpose(0, 1).matmul(dZ) if ctx.needs_input_grad[1] else None
        db = dZ.sum(dim=0) if ctx.needs_input_grad[2] else None
        return dA, dB, db


def sgemm_autograd(A, B): return SGemmFn.apply(A, B)


def sgemm_bias_relu_autograd(
    A, B, bias): return SGemmBiasReluFn.apply(A, B, bias)
