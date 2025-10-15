import torch
from mygemm.modules import TinyMLP

dev = "cpu"  # or "mps" on Apple Silicon
torch.manual_seed(0)

x = torch.randn(32, 64, device=dev, requires_grad=True)
y = torch.randint(0, 10, (32,), device=dev)

m = TinyMLP(64, 128, 10, fused=True).to(dev)
out = m(x)
loss = torch.nn.functional.cross_entropy(out, y)
loss.backward()

print(x.grad.shape)  # should be [32, 64]
