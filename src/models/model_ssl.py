"""
Pose transformer with factorized spatial-temporal attention.

Architecture (Factorized Spatial/Temporal Attention, Spatial-First):
  Input:  (B, T, J, 2) pose tensor — T frames, J joints, (x, y) per joint
  Token:  one per (frame, joint) pair, so T*J tokens per sequence
  Embed:  per-joint coord-embed + learned joint-ID embed + learned time-pos embed
  Stack:  4 layers alternating spatial / temporal attention (S-T-S-T)
    spatial layer:  attention within each frame, across J joints
    temporal layer: attention within each joint, across T frames
                    with relative temporal positional bias (RPE)
  Output: (B, T*J, d_model) features, mean-pooled to (B, d_model) for ID
"""
from __future__ import annotations
import os, math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Relative temporal positional bias (RPE)
# ---------------------------------------------------------------------------

class RelTemporalBias(nn.Module):
    """
    Learnable bias added to attention scores based on the relative temporal
    distance between two frames. Only used inside the temporal attention pass.
    Log-bucketed for distances beyond `max_exact`.
    """
    def __init__(self, num_heads: int, max_exact: int = 8, num_buckets: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.max_exact = max_exact
        self.num_buckets = num_buckets
        # +/- exact buckets covered, plus log-spaced beyond, plus center
        self.table = nn.Parameter(torch.zeros(num_heads, 2 * num_buckets + 1))
        nn.init.trunc_normal_(self.table, std=0.02)

    def _bucket(self, delta: torch.Tensor) -> torch.Tensor:
        # delta: (T, T), signed integer-valued
        sign = torch.sign(delta)
        absd = delta.abs()
        small = absd <= self.max_exact

        # exact part: bucket index = abs(delta), preserved sign
        exact_bucket = absd.clamp(max=self.max_exact)

        # log part for |delta| > max_exact
        # mapped into range [max_exact+1, num_buckets]
        log_input = (absd.float() / max(self.max_exact, 1)).clamp_min(1.0)
        log_bucket = self.max_exact + (
            (torch.log(log_input) / math.log(max(2.0, 2.0))) *
            (self.num_buckets - self.max_exact)
        ).long()
        log_bucket = log_bucket.clamp(max=self.num_buckets)

        bucket_abs = torch.where(small, exact_bucket, log_bucket).long()
        # signed bucket: 0..num_buckets for forward, 
        # num_buckets+1..2*num_buckets for backward, 
        # center is num_buckets
        # use center = num_buckets as the zero-offset
        signed = self.num_buckets + (sign.long() * bucket_abs)
        signed = signed.clamp(0, 2 * self.num_buckets)
        return signed  # (T, T)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Return (num_heads, T, T) additive attention bias.
        """
        idx = torch.arange(T, device=device)
        delta = idx[None, :] - idx[:, None]     # (T, T) signed
        bucket = self._bucket(delta)        # (T, T)
        # gather from table: (num_heads, 2*num_buckets+1) indexed by (T, T)
        bias = self.table[:, bucket]        # (num_heads, T, T)
        return bias


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    """
    Standard Multi-Head self-Attention. Accepts an optional additive bias.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, d). attn_bias: (num_heads, N, N) or None
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]        # each (B, H, N, hd)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale        # (B, H, N, N)
        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(0)        # broadcast over batch
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = attn @ v      # (B, H, N, hd)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class FFN(nn.Module):
    """
    Feed-Forward Network block: takes an input tensor, expands its features to capture complex patterns,
    applies a non-linear activation function (GELU), and then shrinks it back to its originl size.
    """
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, d_inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SpatialBlock(nn.Module):
    """
    Transformer Encoder Layer: treats each frame of a sequence independently and 
    calculates attention only among the spatial joints, like how the hand joint 
    relates to the elbow joint within the exact same frame.
    Attention within each frame, across J joints. No positional bias.
    """
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, T: int, J: int) -> torch.Tensor:
        # x: (B, T*J, d). Reshape to (B*T, J, d) so attention is over J only.
        B, N, D = x.shape
        assert N == T * J
        x_r = x.reshape(B, T, J, D).reshape(B * T, J, D)
        x_r = x_r + self.attn(self.norm1(x_r), attn_bias=None)
        x_r = x_r + self.ffn(self.norm2(x_r))
        x_r = x_r.reshape(B, T, J, D).reshape(B, T * J, D)
        return x_r


class TemporalBlock(nn.Module):
    """
    Attention within each joint, across T frames. Uses relative temporal bias(RPE).
    """
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.1, rpe_max_exact: int = 8, rpe_num_buckets: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.rpe = RelTemporalBias(num_heads, max_exact=rpe_max_exact, num_buckets=rpe_num_buckets)

    def forward(self, x: torch.Tensor, T: int, J: int) -> torch.Tensor:
        # x: (B, T*J, d). Reshape to (B*J, T, d) so attention is over T only.
        B, N, D = x.shape
        assert N == T * J
        # (B, T, J, D) -> (B, J, T, D) -> (B*J, T, D)
        x_r = x.reshape(B, T, J, D).permute(0, 2, 1, 3).contiguous().reshape(B * J, T, D)

        bias = self.rpe(T=T, device=x.device)               # (num_heads, T, T)
        x_r = x_r + self.attn(self.norm1(x_r), attn_bias=bias)
        x_r = x_r + self.ffn(self.norm2(x_r))

        # (B*J, T, D) -> (B, J, T, D) -> (B, T, J, D) -> (B, T*J, D)
        x_r = x_r.reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous().reshape(B, T * J, D)
        return x_r


# ---------------------------------------------------------------------------
# Backbone + reconstruction model
# ---------------------------------------------------------------------------

class FactorizedPoseTransformer(nn.Module):
    """
    Backbone: input (B, T, J, 2) -> features (B, T*J, d_model).

    The encoder is a stack of alternating SpatialBlock / TemporalBlock layers
    (spatial-first: S-T-S-T-...). Number of S blocks == number of T blocks.
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        depth_pairs: int = 2,       # number of (S, T) pairs => total layers = 2 * depth_pairs
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_joints: int = 17,
        max_frames: int = 128,
        rpe_max_exact: int = 8,
        rpe_num_buckets: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_joints = max_joints
        self.max_frames = max_frames

        # Per-joint coordinate embedding: (x, y) -> d_model
        self.coord_embed = nn.Linear(2, d_model)
        # Learned joint-ID embedding (one row per COCO-17 joint)
        self.joint_id_embed = nn.Embedding(max_joints, d_model)
        # Learned absolute time positional embedding (small additive prior;
        # most temporal info comes from the temporal-block RPE)
        self.time_pos_embed = nn.Embedding(max_frames, d_model)

        self.layers = nn.ModuleList()
        for _ in range(depth_pairs):
            self.layers.append(SpatialBlock(d_model, nhead, mlp_ratio, dropout))
            self.layers.append(TemporalBlock(
                d_model, nhead, mlp_ratio, dropout,
                rpe_max_exact=rpe_max_exact, rpe_num_buckets=rpe_num_buckets,
            ))

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                joint_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:              (B, T, J, 2)
        joint_indices:  (J,) long tensor of COCO-17 joint IDs for the joints
                        present in x. If None, assumes joints 0..J-1.

        Returns: (B, T*J, d_model) token features.
        """
        B, T, J, C = x.shape
        assert C == 2, f"Expected last dim=2, got {C}"
        assert T <= self.max_frames, f"T={T} exceeds max_frames={self.max_frames}"
        if joint_indices is None:
            joint_indices = torch.arange(J, device=x.device)
        assert joint_indices.numel() == J

        # Embed coordinates
        h = self.coord_embed(x)     # (B, T, J, d)
        # Add joint identity
        jid = self.joint_id_embed(joint_indices.to(x.device))       # (J, d)
        h = h + jid[None, None, :, :]
        # Add time positional embedding
        tid = torch.arange(T, device=x.device)
        tpe = self.time_pos_embed(tid)      # (T, d)
        h = h + tpe[None, :, None, :]

        # Flatten to T*J tokens
        h = h.reshape(B, T * J, self.d_model)

        # Apply factorized encoder
        for layer in self.layers:
            h = layer(h, T=T, J=J)

        h = self.norm_out(h)
        return h        # (B, T*J, d_model)


class SSLReconstructor(nn.Module):
    """
    Wraps the backbone with a per-token reconstruction head for masked-joint SSL.
    Predicts (x, y) for each (frame, joint) position from its contextualized
    token feature.
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        depth_pairs: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_joints: int = 17,
        max_frames: int = 128,
        rpe_max_exact: int = 8,
        rpe_num_buckets: int = 32,
    ):
        super().__init__()
        self.backbone = FactorizedPoseTransformer(
            d_model=d_model, nhead=nhead, depth_pairs=depth_pairs,
            mlp_ratio=mlp_ratio, dropout=dropout,
            max_joints=max_joints, max_frames=max_frames,
            rpe_max_exact=rpe_max_exact, rpe_num_buckets=rpe_num_buckets,
        )
        self.head = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, x: torch.Tensor,
                joint_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, J, 2) - with masked positions zeroed out
        Returns: (B, T, J, 2) reconstruction
        """
        B, T, J, _ = x.shape
        h = self.backbone(x, joint_indices=joint_indices)   # (B, T*J, d)
        out = self.head(h).reshape(B, T, J, 2)
        return out

    def encode(self, x: torch.Tensor,
               joint_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return per-token features without the reconstruction head.
        """
        return self.backbone(x, joint_indices=joint_indices)


class SSLBackbone(nn.Module):
    """
    Identification-time backbone. Wraps SSLReconstructor and returns a single
    pooled embedding per sequence (mean over all T*J tokens).
    """
    def __init__(self, ssl: SSLReconstructor,
                 proj_dim: Optional[int] = None,
                 l2_norm: bool = False):
        super().__init__()
        self.ssl = ssl
        d = ssl.d_model
        self.proj = nn.Linear(d, proj_dim) if proj_dim is not None else None
        self.d_model = proj_dim if proj_dim is not None else d
        self.l2_norm = bool(l2_norm)

    def forward(self, x: torch.Tensor,
                joint_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (B, T, J, 2) -> (B, T*J, d)
        feats = self.ssl.encode(x, joint_indices=joint_indices)
        # Mean-pool over all spatial-temporal tokens
        g = feats.mean(dim=1)                                # (B, d)
        if self.proj is not None:
            g = self.proj(g)
        if self.l2_norm:
            g = F.normalize(g, dim=-1)
        return g


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_ssl_backbone(
    ckpt_path: Optional[str],
    d_model: int = 128,
    nhead: int = 4,
    depth_pairs: int = 2,
    proj_dim: Optional[int] = None,
    l2_norm: bool = False,
    device: str | torch.device = "cpu",
    max_joints: int = 17,
    max_frames: int = 128,
    allow_partial: bool = False,
) -> SSLBackbone:
    """
    Build SSLBackbone and (optionally) load weights from a checkpoint.
    Refuses silent partial loads by default; pass allow_partial=True or set
    FORCE_LOAD=1 to override.
    """
    ckpt_str = "" if ckpt_path is None else str(ckpt_path).strip()
    use_random = (ckpt_str == "" or ckpt_str.lower() in {"none", "null", "scratch", "random"})

    ssl = SSLReconstructor(
        d_model=d_model, nhead=nhead, depth_pairs=depth_pairs,
        max_joints=max_joints, max_frames=max_frames,
    ).to(device)

    if use_random:
        print("[ckpt] no checkpoint -> using randomly initialized backbone")
        ssl.eval()
        return SSLBackbone(ssl, proj_dim=proj_dim, l2_norm=l2_norm).to(device)

    if not os.path.isfile(ckpt_str):
        raise FileNotFoundError(f"SSL checkpoint not found: {ckpt_str}")

    ck = torch.load(ckpt_str, map_location=device)
    state = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck

    try:
        ssl.load_state_dict(state, strict=True)
        n_total = len(ssl.state_dict())
        print(f"[ckpt] loaded {ckpt_str}: {n_total}/{n_total} keys (strict OK)")
    except RuntimeError:
        result = ssl.load_state_dict(state, strict=False)
        n_total = len(ssl.state_dict())
        n_loaded = n_total - len(result.missing_keys)
        print(f"[ckpt] STRICT LOAD FAILED for {ckpt_str}: loaded {n_loaded}/{n_total}")
        print(f"[ckpt] missing model keys ({len(result.missing_keys)}):")
        for k in result.missing_keys[:10]:
            print(f"    - {k}")
        if len(result.missing_keys) > 10:
            print(f"    ... and {len(result.missing_keys)-10} more")
        print(f"[ckpt] unexpected checkpoint keys ({len(result.unexpected_keys)}):")
        for k in result.unexpected_keys[:10]:
            print(f"    - {k}")
        if len(result.unexpected_keys) > 10:
            print(f"    ... and {len(result.unexpected_keys)-10} more")
        if not allow_partial and "FORCE_LOAD" not in os.environ:
            raise RuntimeError(
                f"Refusing to load {ckpt_str}: {len(result.missing_keys)} model "
                f"keys missing. Likely architecture mismatch. Pass allow_partial=True "
                f"or set FORCE_LOAD=1 to override."
            )
        print(f"[ckpt] WARNING: proceeding with partial load")

    ssl.eval()
    return SSLBackbone(ssl, proj_dim=proj_dim, l2_norm=l2_norm).to(device)