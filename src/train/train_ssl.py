"""
SSL pretraining with masked-joint reconstruction.

Usage (CASIA-B):
    python -m src.train.train_ssl \
        --dataset casia --csv data/casia-b_pose_coco.csv \
        --seq_len 56 --stride 8 --epochs 50 --batch_size 64 \
        --save ckpts/ssl_casia_v2.pt

Usage (DenseGait):
    python -m src.train.train_ssl \
        --dataset densegait --csv data/densegait_pose_coco_filtered.csv \
        --poses_dir data/poses \
        --seq_len 56 --stride 4 --epochs 50 --batch_size 64 \
        --save ckpts/ssl_densegait_v2.pt
"""
from __future__ import annotations
import os, time, argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..datasets.dataset import CASIABPoseDataset
from ..datasets.densegait_dataset import DenseGaitPoseDataset
from ..datasets.ssl_dataset import MaskedJointSSL
from ..models.model_ssl import SSLReconstructor


def build_base_dataset(args):
    if args.dataset == "casia":
        ds = CASIABPoseDataset(
            csv_path=args.csv,
            seq_len=args.seq_len,
            stride=args.stride,
            conf_thr=args.conf_thr,
            limit_rows=args.limit_rows,
        )
    elif args.dataset == "densegait":
        ds = DenseGaitPoseDataset(
            csv_path=args.csv,
            poses_dir=args.poses_dir,
            seq_len=args.seq_len,
            stride=args.stride,
            conf_thr=args.conf_thr,
            limit_rows=args.limit_rows,
            limit_files=args.limit_files,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return ds


def train_one_epoch(model, loader, optim, device, scaler, use_amp,
                    log_every=50, save_every_steps=500,
                    resume_step=0, save_path=None, epoch=1,
                    best_loss=float("inf"), global_step=0):
    model.train()
    total_loss = 0.0
    total_steps = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        x = batch["input"].to(device, non_blocking=True)        # (B, L, 17, 2)
        y = batch["target"].to(device, non_blocking=True)       # (B, L, 17, 2)
        m = batch["mask"].to(device, non_blocking=True)         # (B, L, 17) bool

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred = model(x)
                diff_sq = (pred - y) ** 2
                mask_f = m.unsqueeze(-1).float()
                denom = mask_f.sum().clamp_min(1.0)
                loss = (diff_sq * mask_f).sum() / denom / 2.0
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            pred = model(x)
            diff_sq = (pred - y) ** 2
            mask_f = m.unsqueeze(-1).float()
            denom = mask_f.sum().clamp_min(1.0)
            loss = (diff_sq * mask_f).sum() / denom / 2.0
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        total_loss += float(loss.item())
        total_steps += 1
        global_step += 1

        # Skip steps already completed in a resumed epoch
        if global_step <= resume_step:
            continue

        # Mid-epoch checkpoint to Drive
        if save_path and global_step % save_every_steps == 0:
            ck = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
            }
            torch.save(ck, save_path.replace(".pt", "_resume.pt"))
            print(f"  [resume-ckpt] saved at global_step={global_step}")

        # Logging
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg = total_loss / total_steps
            print(f"  step {step+1:5d}/{len(loader):5d}  "
                  f"loss={loss.item():.5f}  avg={avg:.5f}  "
                  f"elapsed={elapsed:.1f}s")

    return total_loss / max(1, total_steps), global_step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["casia", "densegait"], required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--poses_dir", default=None, help="Only used for DenseGait")
    ap.add_argument("--seq_len", type=int, default=56)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--conf_thr", type=float, default=0.5)
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument("--mask_ratio", type=float, default=0.35)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--depth_pairs", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", required=True)
    ap.add_argument("--save_every", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ssl] device={device} | amp={bool(args.amp)}")

    # Data
    base = build_base_dataset(args)
    ds = MaskedJointSSL(base, mask_ratio=args.mask_ratio, seed=args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[ssl] dataset={args.dataset} | windows={len(base)} | batches/epoch={len(loader)}")

    # Model
    model = SSLReconstructor(
        d_model=args.d_model, nhead=args.nhead,
        depth_pairs=args.depth_pairs, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ssl] model params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    best_loss = float("inf")

    # Resume state
    start_epoch = 1
    global_step = 0
    resume_path = args.save.replace(".pt", "_resume.pt")

    if os.path.exists(resume_path):
        print(f"[ssl] Resuming from {resume_path}")
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        if ck["optim"] is not None:
            optim.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"]
        global_step = ck["global_step"]
        resume_step = ck["global_step"]
        best_loss   = ck["best_loss"]
        print(f"[ssl] Resumed at epoch {start_epoch}, step {global_step}, best_loss={best_loss:.5f}")
    else:
        resume_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        epoch_loss, global_step = train_one_epoch(
            model, loader, optim, device, scaler,
            use_amp=bool(args.amp),
            save_every_steps=500,
            resume_step=resume_step if epoch == start_epoch else 0,
            save_path=args.save,
            epoch=epoch,
            best_loss=best_loss,
            global_step=global_step,
        )
        print(f"[ssl] epoch {epoch:3d}  avg_loss={epoch_loss:.5f}")

        # Save end-of-epoch checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs or epoch_loss < best_loss:
            ck = {
                "model": model.state_dict(),
                "meta": {
                    "d_model": args.d_model, "nhead": args.nhead,
                    "depth_pairs": args.depth_pairs,
                    "epoch": epoch, "loss": epoch_loss,
                    "dataset": args.dataset, "seq_len": args.seq_len,
                    "mask_ratio": args.mask_ratio,
                },
            }
            torch.save(ck, args.save)
            print(f"[ssl] saved checkpoint -> {args.save}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss

    print(f"\n[ssl] done. best_loss={best_loss:.5f}")


if __name__ == "__main__":
    main()