"""
scripts/bench.py

Reproduces the computational efficiency numbers reported in Table 13 of the
paper (Parameters, FLOPs per sample, Peak GPU memory, Latency, Throughput).

Usage:
    python scripts/bench.py

Defaults match the paper: batch size 64, sequence length 56 frames, 17 COCO
joints, FP16 precision, single NVIDIA A100 (any FP16-capable CUDA device
will work; absolute numbers will differ from the paper if the device is not
an A100). The script averages 200 forward passes after 20 warmup iterations.

Numbers printed here should match Table 13 within rounding when run on an
NVIDIA A100 at FP16.
"""
from __future__ import annotations
import argparse
import time

import torch

from src.models.model_ssl import SSLBackbone, SSLReconstructor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=56)
    parser.add_argument("--joints", type=int, default=17)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--depth_pairs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    # Build the same architecture used throughout the paper.
    ssl = SSLReconstructor(d_model=args.d_model, nhead=4, depth_pairs=args.depth_pairs)
    model = SSLBackbone(ssl, l2_norm=True).to(device=device, dtype=dtype).eval()

    # --- Parameters ---
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:                  {n_params:>14,}")

    # --- FLOPs per sample (thop) ---
    try:
        from thop import profile
        probe = torch.randn(1, args.seq_len, args.joints, 2,
                            device=device, dtype=dtype)
        flops_per_sample, _ = profile(model, inputs=(probe,), verbose=False)
        print(f"FLOPs per sample:            {flops_per_sample / 1e6:>14.0f} MFLOPs")
    except ImportError:
        print("FLOPs per sample:            (install `thop` to measure)")

    # --- Peak GPU memory (inference) ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            x = torch.randn(args.batch_size, args.seq_len, args.joints, 2,
                            device=device, dtype=dtype)
            _ = model(x)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory (inference): {peak_mem_mb:>14.1f} MB")
    else:
        print("Peak GPU memory (inference): (CUDA device required)")

    # --- Latency and throughput ---
    x = torch.randn(args.batch_size, args.seq_len, args.joints, 2,
                    device=device, dtype=dtype)
    with torch.inference_mode():
        # Warmup
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Timed runs
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    latency_per_batch_ms = (elapsed / args.iters) * 1e3
    latency_per_sample_ms = latency_per_batch_ms / args.batch_size
    throughput = (args.batch_size * args.iters) / elapsed
    print(f"Latency per batch of {args.batch_size:<4d}   {latency_per_batch_ms:>14.2f} ms")
    print(f"Latency per sample:          {latency_per_sample_ms:>14.3f} ms")
    print(f"Throughput:                  {throughput:>14,.0f} samples/sec")


if __name__ == "__main__":
    main()
