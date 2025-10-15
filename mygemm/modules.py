import math
import torch
from .functional import sgemm_autograd, sgemm_bias_relu_autograd, sgemm_bank_extra_autograd


class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, fused=False, bank_extra=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused = fused
        self.bank_extra = bank_extra
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(
            torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [B, Din], weight: [Dout, Din]; we need [B,K]x[K,N] so B=Din^T
        Bmat = self.weight.t().contiguous()
        if self.fused and self.bias is not None:
            return sgemm_bias_relu_autograd(x, Bmat, self.bias)
        elif self.bank_extra:
            return sgemm_bank_extra_autograd(x, Bmat)
        out = sgemm_autograd(x, Bmat)
        if self.bias is not None:
            out = out + self.bias
        return out


class TinyMLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out, bias=True, fused=False, bank_extra=False):
        super().__init__()
        self.fc1 = MyLinear(d_in, d_hidden, bias=bias,
                            fused=fused, bank_extra=bank_extra)
        self.fc2 = MyLinear(d_hidden, d_out, bias=bias,
                            fused=False, bank_extra=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if not self.fc1.fused:
            x = self.relu(x)
        x = self.fc2(x)
        return x
