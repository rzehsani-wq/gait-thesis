import torch
from src.models.model_ssl import SSLReconstructor, SSLBackbone, load_ssl_backbone
from src.utils.mask_utils import REGION_JOINT_INDICES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# === 1. Build a fresh reconstructor and check parameter count ===
ssl = SSLReconstructor(d_model=128, nhead=4, depth_pairs=2).to(device)
n = sum(p.numel() for p in ssl.parameters())
print(f"\nSSLReconstructor params: {n:,}")
# Expect: ~400k-600k. If much larger, something's wrong.

# === 2. Forward pass on each region ===
B, T = 2, 56
for region in ["R1", "R2", "ALL"]:
    jidx = REGION_JOINT_INDICES[region].to(device)
    J = jidx.numel()
    x = torch.randn(B, T, J, 2, device=device)
    out = ssl(x, joint_indices=jidx)
    print(f"\n{region:4s}: input {tuple(x.shape)}  ->  output {tuple(out.shape)}")
    print(f"      joint_indices: {jidx.tolist()}")
    assert out.shape == (B, T, J, 2), f"Shape mismatch for {region}"

# === 3. Backbone produces single pooled embedding ===
print("\nBackbone forward (for identification):")
bb = SSLBackbone(ssl, l2_norm=True).to(device)
for region in ["R1", "R2", "ALL"]:
    jidx = REGION_JOINT_INDICES[region].to(device)
    J = jidx.numel()
    x = torch.randn(B, T, J, 2, device=device)
    g = bb(x, joint_indices=jidx)
    print(f"  {region:4s}: -> {tuple(g.shape)}  norm/sample={g.norm(dim=1).tolist()}")
    assert g.shape == (B, 128), f"Backbone shape wrong for {region}"

# === 4. Test masked loss path (a single step of SSL) ===
print("\nSSL training step (one batch):")
x_full = torch.randn(B, T, 17, 2, device=device)
mask = torch.rand(B, T, 17, device=device) < 0.35
x_masked = x_full.clone()
x_masked[mask] = 0.0
pred = ssl(x_masked)
diff_sq = (pred - x_full) ** 2
mask_f = mask.unsqueeze(-1).float()
loss = (diff_sq * mask_f).sum() / mask_f.sum().clamp_min(1.0) / 2.0
print(f"  masked positions: {int(mask.sum())}/{int(mask.numel())} ({100*mask.float().mean():.1f}%)")
print(f"  loss: {loss.item():.5f}")
loss.backward()
print(f"  backward pass: OK")

# === 5. Verify load_ssl_backbone refuses to load the old broken ssl_best.pt ===
print("\nload_ssl_backbone refuses architecture mismatch:")
try:
    bb_old = load_ssl_backbone("ckpts/ssl_best.pt", d_model=128, depth_pairs=2)
    print("  ERROR: should have raised!")
except RuntimeError as e:
    print(f"  Refused correctly: {str(e)[:80]}...")
except FileNotFoundError:
    print("  (ssl_best.pt not present, skipping)")

# === 6. Verify load_ssl_backbone also refuses old ssl_casia.pt (different architecture) ===
print("\nOld ssl_casia.pt has incompatible architecture too (different module structure):")
try:
    bb_old2 = load_ssl_backbone("ckpts/ssl_casia.pt", d_model=128, depth_pairs=2)
    print("  ERROR: should have raised!")
except RuntimeError as e:
    print(f"  Refused correctly: {str(e)[:80]}...")
except FileNotFoundError:
    print("  (ssl_casia.pt not present, skipping)")

print("\nAll smoke tests passed.")