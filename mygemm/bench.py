# mygemm/bench.py
import time
import argparse
import torch
from typing import Optional


from mygemm.modules import TinyMLP


def pick_device(cli_choice: Optional[str] = None) -> torch.device:
    if cli_choice:
        return torch.device(cli_choice)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        # torch.mps.synchronize exists on recent PyTorch versions
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


def bench_step(model, x, iters=200, warmup=50, dev=None):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        synchronize(dev)
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None,
                   help="cuda|mps|cpu (auto if omitted)")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--din", type=int, default=1024)
    p.add_argument("--dhid", type=int, default=1024)
    p.add_argument("--dout", type=int, default=1024)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()

    dev = pick_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    print(f"device={dev.type}, dtype={dtype}, shapes: B={args.batch}, Din={args.din}, Dhid={args.dhid}, Dout={args.dout}")

    # Inputs
    x = torch.randn(args.batch, args.din, device=dev, dtype=dtype)

    # Stock PyTorch baseline
    mlp_stock = torch.nn.Sequential(
        torch.nn.Linear(args.din, args.dhid, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(args.dhid, args.dout, bias=True),
    ).to(dev, dtype=dtype)

    # Our modules: plain (separate bias/relu) vs fused (bias+relu in op)
    mlp_plain = TinyMLP(args.din, args.dhid, args.dout,
                        fused=False).to(dev, dtype=dtype)
    mlp_fused = TinyMLP(args.din, args.dhid, args.dout,
                        fused=True).to(dev, dtype=dtype)
    mlp_optimized = TinyMLP(args.din, args.dhid, args.dout,
                            fused=False, bank_extra=True).to(dev, dtype=dtype)

    # Copy weights so comparisons are apples-to-apples
    with torch.no_grad():
        mlp_plain.fc1.weight.copy_(mlp_stock[0].weight)
        mlp_plain.fc1.bias.copy_(mlp_stock[0].bias)
        mlp_plain.fc2.weight.copy_(mlp_stock[2].weight)
        mlp_plain.fc2.bias.copy_(mlp_stock[2].bias)
        mlp_fused.load_state_dict(mlp_plain.state_dict())
        mlp_optimized.load_state_dict(mlp_plain.state_dict())

    # Correctness (inference)
    with torch.inference_mode():
        y_ref = mlp_stock(x)
        y_plain = mlp_plain(x)
        y_fused = mlp_fused(x)
        y_optimized = mlp_optimized(x)
    print(f"Stock output shape: {y_ref.shape}")
    print(f"Plain output shape: {y_plain.shape}")

    def max_err(a, b):
        return (a - b).abs().max().item()

    print(f"plain max abs err: {max_err(y_ref, y_plain):.3e}")
    print(f"fused max abs err: {max_err(y_ref, y_fused):.3e}")
    print(f"optimized max abs err: {max_err(y_ref, y_optimized):.3e}")

    # Timing
    for name, m in [("stock", mlp_stock), ("plain", mlp_plain), ("fused", mlp_fused), ("optimized", mlp_optimized)]:
        dt = bench_step(m, x, iters=args.iters, warmup=args.warmup, dev=dev)
        print(f"{name:6s}: {dt*1e3:.3f} ms")


if __name__ == "__main__":
    main()
