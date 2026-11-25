"""
train_id.py:
    Identification head training
    k-NN evaluation on CASIA-B (COCO-17 skeletons)

Features:
    - Region-aware inputs (R1, R2, or ALL with avg/concat fusion)
    - Subject- or clip-disjoint splits
    - Two validation protocols: softmax classifier or k-NN on embeddings
    - Optional L2 normalization of backbone embeddings
    - View/condition-aware k-NN (gallery/probe by 'nm','bg','cl', same-view matching)
    - Optional ArcFace metric head with warmup
    - Condition-balanced sampling (boost 'cl', mild 'bg')
    - Clean featurizer selection for k-NN: --knn_source {auto, raw, metric}
    - Pose augmentations (TemporalDropout, RandomJointMask, UpperBodyWiden)
"""

# --------------------------------------------------------
import argparse, os, csv, random, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# Project imports
from src.datasets.dataset import CASIABPoseDataset
from src.models.model_ssl import load_ssl_backbone
from src.models.id_model import IDHead
from src.models.metric_head import MetricHead
from src.utils.seed import set_seed
from src.utils.metrics import accuracy_topk
from src.utils.logger import append_csv, dump_json
from src.data.pose_augs import Compose, TemporalDropout, RandomJointMask, RandomUpperBodyWiden
from src.utils.casia_parsing import parse_cond_from_folder, parse_subject_from_folder, parse_view_from_folder, parse_seq_num_from_folder
from src.utils.mask_utils import REGIONS, REGION_JOINT_INDICES
from src.datasets.splits import (
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, get_split
)

# ---------- Utilities
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _parse_cond_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    # allow "nm,bg" or "nm bg"
    parts = []
    for token in s.replace(",", " ").split():
        token = token.strip().lower()
        if token:
            parts.append(token)
    return parts if parts else None


# ---------- Sequence-disjoint gallery/probe masks (Butler Item 3)
# CASIA-B nm sequences 1-4 are gallery, 5-6 are probe.
# For nm->bg / nm->cl the conditions are already disjoint, but we restrict
# the nm gallery to seqs 1-4 for consistency with the standard protocol.
GALLERY_NM_SEQS = frozenset({1, 2, 3, 4})
PROBE_NM_SEQS   = frozenset({5, 6})


def _build_gal_probe_masks(conds, folders, gallery_cond, probe_cond):
    """Return (gal_mask, pr_mask) boolean arrays with sequence-disjoint enforcement.

    - nm gallery windows are restricted to sequences 1-4.
    - nm probe  windows are restricted to sequences 5-6.
    - bg / cl windows are unrestricted (they are already separate sequences).
    """
    conds_arr = np.array(conds)
    seq_nums = np.array([parse_seq_num_from_folder(f) for f in folders])

    gal_targets = _parse_cond_list(gallery_cond)
    pr_targets  = _parse_cond_list(probe_cond)

    # --- gallery mask ---
    if gal_targets is None:
        gal_mask = np.ones(len(conds_arr), dtype=bool)
    else:
        gal_mask = np.isin(conds_arr, gal_targets)
    # restrict nm gallery to seqs 1-4
    nm_gal = (conds_arr == "nm") & gal_mask
    if nm_gal.any():
        gal_mask[nm_gal] = np.isin(seq_nums[nm_gal], list(GALLERY_NM_SEQS))

    # --- probe mask ---
    if pr_targets is None:
        pr_mask = np.ones(len(conds_arr), dtype=bool)
    else:
        pr_mask = np.isin(conds_arr, pr_targets)
    # restrict nm probe to seqs 5-6
    nm_pr = (conds_arr == "nm") & pr_mask
    if nm_pr.any():
        pr_mask[nm_pr] = np.isin(seq_nums[nm_pr], list(PROBE_NM_SEQS))

    return gal_mask, pr_mask


def _inspect_labelmap(train_loader, val_loader, num_classes):
    def _base(ds):
        return ds.dataset if isinstance(ds, tud.Subset) else ds
    tb = _base(train_loader.dataset)
    vb = _base(val_loader.dataset)
    tm = getattr(tb, "y_map", None)
    vm = getattr(vb, "y_map", None)
    
    print(f"[debug/y_map] train_len={len(tm) if tm else 'NA'} | val_len={len(vm) if vm else 'NA'}")
    print(f"[debug/y_map] same_object={tm is vm} | same_content={(tm == vm) if (tm and vm) else 'NA'}")
    print(f"[debug/y_map] num_classes={num_classes} | map_len={(len(tm) if tm else 'NA')}")
    
    # peek at val label range
    vmin, vmax = None, None
    for b in val_loader:
        y = b["y"]
        ymin, ymax = int(y.min()), int(y.max())
        vmin = ymin if vmin is None else min(vmin, ymin)
        vmax = ymax if vmax is None else max(vmax, ymax)
    print(f"[debug/y_map] val_label_range=[{vmin}, {vmax}]")

# ----------- Dataset wrapper to select region(s)
class IDRegionDataset(Dataset):
    """
    Wrap base dataset to expose region tensors + metadata.
    base[i] must yield:
      {"folder": str, "xy": Float(L,17,2)}
    The 'x' returned has shape (L, J, 2) where:
      R1 -> J=11 (head + arms)
      R2 -> J= 6 (legs)
      ALL -> J=17 (full skeleton)
    """
    def __init__(self, base: Dataset, subject_to_label: Dict[str, int],
                 region: str = "R1", pose_aug=None, is_train: bool = False):
        self.base = base
        self.y_map = subject_to_label
        self.region = region.upper()
        self.pose_aug = pose_aug
        self.is_train = bool(is_train)
        assert self.region in {"R1", "R2", "ALL"}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        folder = s["folder"]
        sid = parse_subject_from_folder(folder)
        y = self.y_map[sid]
        cond = parse_cond_from_folder(folder)
        view = parse_view_from_folder(folder)

        # Full normalized pose (L, 17, 2) -- new CASIA dataset exposes this.
        xy = s["xy"].float()

        # Apply pose augmentation on the FULL pose (consistent across regions)
        if self.pose_aug is not None and self.is_train:
            if i == 0:
                print("[aug] train split: pose augmentations active")
            x_np = xy.detach().cpu().numpy()
            x_np = self.pose_aug(x_np)
            x_np = np.nan_to_num(x_np, nan=0.0)
            xy = torch.from_numpy(x_np).to(xy.dtype)

        out = {"y": torch.tensor(y, dtype=torch.long),
               "folder": folder, "cond": cond, "view": int(view)}

        # Slice region from full pose
        if self.region == "R1":
            out["x"] = xy[:, REGIONS["R1"], :]      # (L, 11, 2)
        elif self.region == "R2":
            out["x"] = xy[:, REGIONS["R2"], :]      # (L,  6, 2)
        else:  # ALL
            out["x"] = xy       # (L, 17, 2)
        return out

# ---------- Loaders and Split
def build_id_loaders(
    csv_path: str,
    region: str = "R1",
    seq_len: int = 30,
    stride: int = 30,
    conf_thr: float = 0.5,
    min_clips_per_subject: int = 4,
    val_frac: float = 0.2,
    batch_size: int = 64,
    limit_rows: Optional[int] = None,
    limit_subjects: int = 0,
    limit_per_subject: int = 0,
    split_mode: str = "subject",
    seed: int = 0,
    pose_aug_train=None,
    protocol: str = "custom",
):
    base = CASIABPoseDataset(
        csv_path, seq_len=seq_len, stride=stride,
        conf_thr=conf_thr, limit_rows=limit_rows
    )

    # subject -> indices
    subject_to_indices = defaultdict(list)
    if hasattr(base, "seq") and hasattr(base.seq, "seq") and isinstance(base.seq.seq, list):
        for i, (folder, _xy, _conf) in enumerate(base.seq.seq):
            s = parse_subject_from_folder(folder)
            subject_to_indices[s].append(i)
    else:
        for i in range(len(base)):
            s = parse_subject_from_folder(base[i]["folder"])
            subject_to_indices[s].append(i)

    rng = np.random.RandomState(seed)

    train_idx, val_idx, test_idx = [], [], []
    sm = split_mode.lower()
    if sm == "subject":
        # --- Three-way split selected by protocol ---
        # custom: 99/15/10 (thesis); standard: 74/0/50 (field-standard, Reviewer 1 R1-1)
        _train_set, _val_set, _test_set = get_split(protocol)
        train_subjects = sorted(s for s in _train_set if s in subject_to_indices
                                and len(subject_to_indices[s]) >= min_clips_per_subject)
        val_subjects   = sorted(s for s in _val_set   if s in subject_to_indices
                                and len(subject_to_indices[s]) >= min_clips_per_subject)
        test_subjects  = sorted(s for s in _test_set  if s in subject_to_indices
                                and len(subject_to_indices[s]) >= min_clips_per_subject)

        # Disjointness assertion
        all_used = set(train_subjects) | set(val_subjects) | set(test_subjects)
        assert set(train_subjects).isdisjoint(val_subjects), "Train/val subject overlap"
        assert set(train_subjects).isdisjoint(test_subjects), "Train/test subject overlap"
        assert set(val_subjects).isdisjoint(test_subjects), "Val/test subject overlap"

        print(f"[split] train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)} subjects")
        print(f"[split] train_subjects={train_subjects}")
        print(f"[split] val_subjects={val_subjects}")
        print(f"[split] test_subjects={test_subjects}")

        for s in train_subjects:
            idxs = np.array(subject_to_indices[s], dtype=int)
            rng.shuffle(idxs)
            if limit_per_subject and limit_per_subject > 0:
                idxs = idxs[:limit_per_subject]
            train_idx.extend(idxs.tolist())
        for s in val_subjects:
            idxs = np.array(subject_to_indices[s], dtype=int)
            rng.shuffle(idxs)
            if limit_per_subject and limit_per_subject > 0:
                idxs = idxs[:limit_per_subject]
            val_idx.extend(idxs.tolist())
        for s in test_subjects:
            idxs = np.array(subject_to_indices[s], dtype=int)
            rng.shuffle(idxs)
            if limit_per_subject and limit_per_subject > 0:
                idxs = idxs[:limit_per_subject]
            test_idx.extend(idxs.tolist())

        subjects = sorted(set(train_subjects) | set(val_subjects) | set(test_subjects))
        subject_to_label = {s: i for i, s in enumerate(subjects)}

    elif sm == "clip":
        subjects = sorted([s for s, idxs in subject_to_indices.items()
                           if len(idxs) >= min_clips_per_subject])
        for s in subjects:
            idxs = np.array(subject_to_indices[s], dtype=int)
            rng.shuffle(idxs)
            if limit_per_subject > 0:
                idxs = idxs[:limit_per_subject]
            idxs = idxs.tolist()
            
            if not idxs:
                continue
            rng.shuffle(idxs)
            n_val = max(1, int(round(len(idxs) * val_frac))) if val_frac > 0 else 0
            if n_val >= len(idxs):
                n_val = max(1, len(idxs) - 1)
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
        if len(val_idx) == 0 and len(train_idx) > 1:
            val_idx.append(train_idx.pop())
        subject_to_label = {s: i for i, s in enumerate(subjects)}
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if len(train_idx) == 0:
        raise ValueError(f"Empty training split: train={len(train_idx)}")
    # Val may legitimately be empty under the standard protocol (74/0/50).

    train_ds = IDRegionDataset(Subset(base, train_idx), subject_to_label,
                               region=region, pose_aug=pose_aug_train, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Validation loader (None under standard protocol: no in-training val eval,
    # final checkpoint selected by last-step convention as in GaitSet/GaitGraph/GaitBase).
    val_loader = None
    if len(val_idx) > 0:
        val_ds   = IDRegionDataset(Subset(base, val_idx),   subject_to_label,
                                   region=region, pose_aug=None, is_train=False)
        val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Test loader (only for subject split; clip split has no held-out test set)
    test_loader = None
    if sm == "subject" and len(test_idx) > 0:
        test_ds = IDRegionDataset(Subset(base, test_idx), subject_to_label,
                                  region=region, pose_aug=None, is_train=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    return train_loader, val_loader, test_loader, len(subject_to_label)

# ---------- Forward region/fusion
def _forward_region(backbone, batch: Dict[str, Any], device,
                    region: str, fusion: str = "avg"):
    """
    Run the backbone with the right joint_indices for the chosen region.
    `fusion` is kept for backward CLI compatibility but ignored: the new model
    processes the full region as one input.
    """
    jidx = REGION_JOINT_INDICES[region.upper()].to(device)
    x = batch["x"].to(device)
    return backbone(x, joint_indices=jidx)

# ---------- Feature extraction (with meta)
@torch.no_grad()
def extract_features_with_meta(backbone, dl, device, region: str, fusion: str,
                               l2norm: bool = True, featurizer=None):
    feats, labels, conds, views, folders = [], [], [], [], []
    backbone.eval()
    for b in dl:
        y = b["y"].to(device, dtype=torch.long)
        z = _forward_region(backbone, b, device, region, fusion)
        if featurizer is not None:
            z = featurizer(z)
        if l2norm:
            z = F.normalize(z, dim=-1)
        feats.append(z.cpu())
        labels.append(y.cpu())
        conds.extend(list(b["cond"]))
        views.extend([int(v) for v in b["view"]])
        folders.extend(list(b["folder"]))
    backbone.train()
    X = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0)
    views_t = torch.tensor(views, dtype=torch.int64)
    return X, y, conds, views_t, folders

import json

def _dump_features_npz(X, y, conds, views, folders, ymap, out_npz):
    # Save embeddings + meta
    np.savez_compressed(
        out_npz,
        feats=X.numpy(),           # (N,D) float32
        y=y.numpy().astype(np.int64),
        cond=np.array(conds, dtype=object),
        view=views.numpy().astype(np.int64),
        folder=np.array(folders, dtype=object),
    )
    # Also write id_list.csv aligned with rows in feats
    csv_path = out_npz.replace(".npz", "_id_list.csv")
    inv = {v: k for k, v in ymap.items()}
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row","label","subject","cond","view","folder"])
        for i in range(y.shape[0]):
            w.writerow([i, int(y[i].item()), inv.get(int(y[i].item()), f"subj_{int(y[i].item())}"),
                        conds[i], int(views[i].item()), folders[i]])

def _materialize_knn_partition(X, y, conds, views, folders, *, gallery_cond, probe_cond,
                               view_match, gallery_per_subject, seed=0):
    # Reproduce the exact partitioning logic used by evaluate_knn()
    # Returns gallery_idx, probe_idx arrays that index rows of X/y/conds/views
    conds_list = list(conds) if not isinstance(conds, list) else conds
    conds_arr = np.array(conds_list)
    views = views.numpy().astype(int) if hasattr(views, 'numpy') else np.asarray(views, dtype=int)
    y_np  = y.numpy().astype(int) if hasattr(y, 'numpy') else np.asarray(y, dtype=int)

    gal_mask, pr_mask = _build_gal_probe_masks(conds_list, folders, gallery_cond, probe_cond)

    gal_idx = np.where(gal_mask)[0]
    pr_idx  = np.where(pr_mask)[0]

    y_gal, v_gal = y_np[gal_idx], views[gal_idx]
    y_pr         = y_np[pr_idx]

    common = np.intersect1d(np.unique(y_gal), np.unique(y_pr))
    if common.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    keep_g = np.isin(y_gal, common)
    keep_p = np.isin(y_pr,  common)
    gal_idx = gal_idx[keep_g]
    pr_idx  = pr_idx[keep_p]

    # cap gallery per subject (and per view if view_match=1)
    rng = np.random.RandomState(seed)
    chosen = []
    yg = y_np[gal_idx]
    vg = views[gal_idx]
    if int(view_match):
        for subj in np.unique(yg):
            m_subj = (yg == subj)
            for vv in np.unique(vg[m_subj]):
                idx = gal_idx[np.where(m_subj & (vg == vv))[0]]
                if gallery_per_subject > 0 and len(idx) > gallery_per_subject:
                    idx = rng.choice(idx, size=gallery_per_subject, replace=False)
                chosen.extend(idx.tolist())
    else:
        for subj in np.unique(yg):
            idx = gal_idx[np.where(yg == subj)[0]]
            if gallery_per_subject > 0 and len(idx) > gallery_per_subject:
                idx = rng.choice(idx, size=gallery_per_subject, replace=False)
            chosen.extend(idx.tolist())
    gal_final = np.array(sorted(chosen), dtype=int)
    return gal_final, pr_idx


# ---------- k-NN helpers
def _compute_centroids(Zg: torch.Tensor, yg: torch.Tensor, C: int) -> torch.Tensor:
    Ng, D = Zg.shape
    cent = torch.zeros(C, D, device=Zg.device, dtype=Zg.dtype)
    cnt  = torch.zeros(C, 1, device=Zg.device, dtype=Zg.dtype)
    cent.index_add_(0, yg, Zg)
    cnt.index_add_(0, yg, torch.ones(Ng, 1, device=Zg.device, dtype=Zg.dtype))
    cent = cent / cnt.clamp_min_(1.0)
    cent = F.normalize(cent, dim=1)
    return cent

def _knn_logits(Zq, Zg, yg, C, rule, K, centroids=None):
    Zq = F.normalize(Zq, dim=1)
    Zg = F.normalize(Zg, dim=1)
    r = (rule or "centroid").lower()
    if r == "centroid":
        if centroids is None:
            centroids = _compute_centroids(Zg, yg, C)
        return Zq @ centroids.t()
    S = Zq @ Zg.t()
    K_eff = 1 if r == "centroid" else int(K)
    if (K_eff <= 0) or (K_eff >= S.size(1)):
        topv, topi = S, torch.arange(S.size(1), device=S.device).view(1, -1).expand_as(S)
    else:
        topv, topi = S.topk(K_eff, dim=1)
    topy = yg[topi]
    logits = torch.zeros(Zq.size(0), C, device=Zq.device, dtype=Zq.dtype)
    if r == "vote":
        logits.scatter_add_(1, topy, torch.ones_like(topv))
    elif r in ("sim", "weighted", "wsim"):
        logits.scatter_add_(1, topy, topv)
    else:
        if centroids is None:
            centroids = _compute_centroids(Zg, yg, C)
        return Zq @ centroids.t()
    return logits


def _choose_featurizer(args, metric):
    src = getattr(args, "knn_source", "auto")
    if src == "raw":
        return None
    if src == "metric":
        if metric is None:
            raise ValueError("knn_source='metric' requires --use_arcface=1.")
        return lambda x: F.normalize(metric.proj(x), dim=1)
    # auto
    if metric is not None:
        return lambda x: F.normalize(metric.proj(x), dim=1)
    return None


@torch.no_grad()
def evaluate_knn(
    backbone, dl, device, region: str, fusion: str, *,
    gallery_cond: Optional[str] = None,
    probe_cond: Optional[str] = None,
    view_match: int = 1,
    view_exclude_identical: int = 0,
    gallery_per_subject: int = 1,
    rule: str = "centroid",
    K: int = 5,
    l2norm: bool = True,
    featurizer=None,
) -> Tuple[float, float]:

    X_t, y_t, conds, views_t, folders = extract_features_with_meta(
        backbone, dl, device, region, fusion, l2norm=l2norm, featurizer=featurizer
    )
    X = X_t.cpu().numpy()
    y = y_t.cpu().numpy().astype(int)
    views = views_t.cpu().numpy().astype(int)
    conds = list(conds)
    if len(X) == 0:
        return 0.0, 0.0

    def _l2n(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return a / n
    if l2norm:
        X = _l2n(X)

    gal_mask, pr_mask = _build_gal_probe_masks(conds, folders, gallery_cond, probe_cond)
    X_gal, y_gal, v_gal = X[gal_mask], y[gal_mask], views[gal_mask]
    X_pr,  y_pr,  v_pr  = X[pr_mask],  y[pr_mask],  views[pr_mask]

    print(f"[knn/debug] unique_conds={sorted(set(conds))} | gal={gal_mask.sum()} probe={pr_mask.sum()}")

    common = np.intersect1d(np.unique(y_gal), np.unique(y_pr))
    if common.size == 0:
        return 0.0, 0.0
    keep_g = np.isin(y_gal, common)
    keep_p = np.isin(y_pr,  common)
    X_gal, y_gal, v_gal = X_gal[keep_g], y_gal[keep_g], v_gal[keep_g]
    X_pr,  y_pr,  v_pr  = X_pr[keep_p],  y_pr[keep_p],  v_pr[keep_p]

    common = np.array(sorted(np.unique(common))).astype(int)
    remap = {int(old): int(new) for new, old in enumerate(common.tolist())}
    y_gal = np.array([remap[int(v)] for v in y_gal], dtype=int)
    y_pr  = np.array([remap[int(v)] for v in y_pr],  dtype=int)
    C = int(len(common))

    if len(X_gal) == 0 or len(X_pr) == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(0)
    keep = []
    if view_match:
        for subj in np.unique(y_gal):
            m_subj = (y_gal == subj)
            for vv in np.unique(v_gal[m_subj]):
                idx = np.where(m_subj & (v_gal == vv))[0]
                if gallery_per_subject > 0 and len(idx) > gallery_per_subject:
                    idx = rng.choice(idx, size=gallery_per_subject, replace=False)
                keep.extend(idx.tolist())
    else:
        for subj in np.unique(y_gal):
            idx = np.where(y_gal == subj)[0]
            if gallery_per_subject > 0 and len(idx) > gallery_per_subject:
                idx = rng.choice(idx, size=gallery_per_subject, replace=False)
            keep.extend(idx.tolist())
    keep = np.array(sorted(keep), dtype=int)
    X_gal, y_gal, v_gal = X_gal[keep], y_gal[keep], v_gal[keep]

    Zg = torch.from_numpy(X_gal).to(device)
    yg = torch.from_numpy(y_gal).to(device, dtype=torch.long)
    Vg = torch.from_numpy(v_gal).to(device)
    Zq_all = torch.from_numpy(X_pr).to(device)
    yq_all = torch.from_numpy(y_pr).to(device, dtype=torch.long)
    Vq_all = torch.from_numpy(v_pr).to(device)

    r = rule.lower()
    use_centroid = (r == "centroid")
    centroids = _compute_centroids(Zg, yg, C) if use_centroid else None

    top1 = top5 = 0
    for i in range(Zq_all.size(0)):
        Zq_i = Zq_all[i:i+1]
        if view_match:
            # same-view retrieval (thesis / custom protocol)
            mask = (Vg == Vq_all[i])
            if mask.sum().item() == 0:
                Zg_i, yg_i = Zg, yg
                centroids_i = centroids
            else:
                Zg_i, yg_i = Zg[mask], yg[mask]
                centroids_i = _compute_centroids(Zg_i, yg_i, C) if use_centroid else None
        elif view_exclude_identical:
            # cross-view retrieval excluding the probe's own view
            # (standard CASIA-B protocol, GaitSet/GaitGraph/GaitBase convention)
            mask = (Vg != Vq_all[i])
            if mask.sum().item() == 0:
                Zg_i, yg_i = Zg, yg
                centroids_i = centroids
            else:
                Zg_i, yg_i = Zg[mask], yg[mask]
                centroids_i = _compute_centroids(Zg_i, yg_i, C) if use_centroid else None
        else:
            # pooled across all views, including the probe's own
            Zg_i, yg_i = Zg, yg
            centroids_i = centroids

        logits = _knn_logits(Zq_i, Zg_i, yg_i, C, r, K, centroids=centroids_i)
        topk = logits.topk(k=min(5, logits.size(1)), dim=1).indices.squeeze(0).tolist()
        pred1 = topk[0] if topk else -1
        gt = int(yq_all[i].item())
        if pred1 == gt:
            top1 += 1
        if gt in topk:
            top5 += 1

    n = max(1, Zq_all.size(0))
    return float(top1 / n), float(top5 / n)

# ---------- Classifier evaluation
@torch.no_grad()
def evaluate(backbone, head, dl, device, region: str, fusion: str) -> Tuple[float, float]:
    backbone.eval(); head.eval()
    tot1 = tot5 = n = 0.0
    for b in dl:
        y = b["y"].to(device, dtype=torch.long)
        z = _forward_region(backbone, b, device, region, fusion)
        logits = head(z)
        B = y.size(0)
        n += B
        pred1 = logits.argmax(dim=1)
        tot1 += (pred1 == y).sum().item()
        C = logits.size(1)
        k = 5 if C >= 5 else C
        topk = logits.topk(k, dim=1).indices
        tot5 += topk.eq(y.view(-1, 1)).any(dim=1).sum().item()
    backbone.train(); head.train()
    return (tot1 / max(1, n), tot5 / max(1, n))

# ---------- Confusion matrix
@torch.no_grad()
def compute_confusion(backbone, head, dl, device, region, fusion, num_classes):
    backbone.eval(); head.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for b in dl:
        y = b["y"]
        z = _forward_region(backbone, b, device, region, fusion)
        logits = head(z)
        pred = logits.argmax(dim=1).detach().cpu()
        y_cpu = y.view(-1).to(torch.long).cpu()
        for t, p in zip(y_cpu.tolist(), pred.tolist()):
            conf[t, p] += 1
    backbone.train(); head.train()
    return conf


@torch.no_grad()
def compute_confusion_knn(
    backbone, dl, device, region, fusion, *,
    gallery_cond=None, probe_cond=None, view_match=1, view_exclude_identical=0,
    gallery_per_subject=1, rule="centroid", K=5, l2norm=True, featurizer=None,
):
    X_t, y_t, conds, views_t, folders = extract_features_with_meta(
        backbone, dl, device, region, fusion, l2norm=l2norm, featurizer=featurizer
    )
    X = X_t.cpu().numpy()
    y = y_t.cpu().numpy().astype(int)
    views = views_t.cpu().numpy().astype(int)
    conds = list(conds)

    C_all = (int(y.max()) + 1) if len(y) else 0
    if C_all == 0:
        return torch.zeros(0, 0, dtype=torch.long)

    def _l2n(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return a / n
    if l2norm:
        X = _l2n(X)

    gal_mask, pr_mask = _build_gal_probe_masks(conds, folders, gallery_cond, probe_cond)

    X_gal, y_gal, v_gal = X[gal_mask], y[gal_mask], views[gal_mask]
    X_pr,  y_pr,  v_pr  = X[pr_mask],  y[pr_mask],  views[pr_mask]

    common = np.intersect1d(np.unique(y_gal), np.unique(y_pr))
    if common.size == 0:
        return torch.zeros(C_all, C_all, dtype=torch.long)

    keep_g = np.isin(y_gal, common)
    keep_p = np.isin(y_pr,  common)
    X_gal, y_gal, v_gal = X_gal[keep_g], y_gal[keep_g], v_gal[keep_g]
    X_pr,  y_pr,  v_pr  = X_pr[keep_p],  y_pr[keep_p],  v_pr[keep_p]

    common = np.array(sorted(np.unique(common))).astype(int)
    remap = {int(old): int(new) for new, old in enumerate(common.tolist())}
    y_gal = np.array([remap[int(v)] for v in y_gal], dtype=int)
    y_pr  = np.array([remap[int(v)] for v in y_pr],  dtype=int)
    C = int(len(common))

    if len(X_gal) == 0 or len(X_pr) == 0:
        return torch.zeros(C_all, C_all, dtype=torch.long)

    rng = np.random.RandomState(0)
    keep = []
    if view_match:
        for subj in np.unique(y_gal):
            m_subj = (y_gal == subj)
            for vv in np.unique(v_gal[m_subj]):
                idx = np.where(m_subj & (v_gal == vv))[0]
                if gallery_per_subject > 0 and len(idx) > gallery_per_subject:
                    idx = rng.choice(idx, size=gallery_per_subject, replace=False)
                keep.extend(idx.tolist())
    else:
        for subj in np.unique(y_gal):
            idx = np.where(y_gal == subj)[0]
            if gallery_per_subject > 0 and len(idx) > gallery_per_subject:
                idx = rng.choice(idx, size=gallery_per_subject, replace=False)
            keep.extend(idx.tolist())
    keep = np.array(sorted(keep), dtype=int)
    if keep.size == 0:
        return torch.zeros(C_all, C_all, dtype=torch.long)
    X_gal, y_gal, v_gal = X_gal[keep], y_gal[keep], v_gal[keep]

    Zg = torch.from_numpy(X_gal).to(device)
    yg = torch.from_numpy(y_gal).to(device, dtype=torch.long)
    Vg = torch.from_numpy(v_gal).to(device)
    Zq_all = torch.from_numpy(X_pr).to(device)
    yq_all = torch.from_numpy(y_pr).to(device, dtype=torch.long)
    Vq_all = torch.from_numpy(v_pr).to(device)

    r = (rule or "centroid").lower()
    use_centroid = (r == "centroid")
    centroids = _compute_centroids(Zg, yg, C) if use_centroid else None

    conf = torch.zeros(C_all, C_all, dtype=torch.long)
    for i in range(Zq_all.size(0)):
        Zq_i = Zq_all[i:i+1]
        if view_match:
            mask = (Vg == Vq_all[i])
            if mask.any():
                Zg_i, yg_i = Zg[mask], yg[mask]
                centroids_i = _compute_centroids(Zg_i, yg_i, C) if use_centroid else None
            else:
                Zg_i, yg_i = Zg, yg
                centroids_i = centroids
        elif view_exclude_identical:
            mask = (Vg != Vq_all[i])
            if mask.any():
                Zg_i, yg_i = Zg[mask], yg[mask]
                centroids_i = _compute_centroids(Zg_i, yg_i, C) if use_centroid else None
            else:
                Zg_i, yg_i = Zg, yg
                centroids_i = centroids
        else:
            Zg_i, yg_i = Zg, yg
            centroids_i = centroids

        K_eff = 1 if use_centroid else min(int(K), Zg_i.size(0)) if Zg_i.size(0) > 0 else 1
        logits = _knn_logits(Zq_i, Zg_i, yg_i, C, r, K_eff, centroids=centroids_i)
        pred = int(logits.argmax(dim=1).item())
        gt = int(yq_all[i].item())
        conf[gt, pred] += 1

    return conf

# ---------- Main Training Loop
def run_training(args, device):
    print(f"[flags] pose_aug={int(getattr(args, 'pose_aug', 1))}   "
          f"balance_cond={int(getattr(args, 'balance_cond', 1))}   "
          f"use_arcface={int(getattr(args, 'use_arcface', 1))}")
    
    clip_max = float(getattr(args, "clip_grad", 1.0))  # e.g. 0.5–2.0 typical
    ce_smooth = float(getattr(args, "ce_smooth", 0.0))  # label smoothing

    
    # Build pose augmentation for train dataset (if enabled)
    pose_aug = None
    if int(getattr(args, "pose_aug", 1)) == 1:
        # Arms in COCO-17 space (full pose; aug runs before region slicing).
        upper_idx = [5, 6, 7, 8, 9, 10]
        pose_aug = Compose([
            TemporalDropout(drop_prob=0.20, mode="nan"),
            RandomJointMask(mask_ratio=0.25, groups=None, mode="zero"),
            RandomUpperBodyWiden(torso_idx=upper_idx, factor_range=(1.05, 1.18), center="mean"),
        ])
        print("[aug] Pose augmentations enabled (TemporalDropout, RandomJointMask, UpperBodyWiden)")
    
    # Data
    train_loader, val_loader, test_loader, num_classes = build_id_loaders(
    csv_path=args.csv,
    region=args.region,
    seq_len=args.seq_len,
    stride=args.stride,
    conf_thr=args.conf_thr,
    min_clips_per_subject=args.min_clips,
    val_frac=args.val_split,
    batch_size=args.batch_size,
    limit_rows=args.limit_rows,
    limit_subjects=getattr(args, "limit_subjects", 0),
    limit_per_subject=getattr(args, "limit_per_subject", 0),
    split_mode=getattr(args, "split_mode", "subject"),
    seed=args.seed,
    pose_aug_train=pose_aug,
    protocol=getattr(args, "protocol", "custom"),
    )

    val_info = f" | val={len(val_loader.dataset)}" if val_loader else " | val=None (standard protocol)"
    test_info = f" | test={len(test_loader.dataset)}" if test_loader else ""
    print(f"[DATA] train={len(train_loader.dataset)}{val_info}{test_info} | classes={num_classes}")
    if val_loader is not None:
        _inspect_labelmap(train_loader, val_loader, num_classes)

    # Condition-balanced sampler (oversample 'cl', mild 'bg')
    if int(getattr(args, "balance_cond", 1)) == 1:
        def _base(ds):
            return ds.dataset if isinstance(ds, tud.Subset) else ds
    
        def _extract_cond_list(idregion_ds):
            base_subset = idregion_ds.base
            base_ds = base_subset.dataset
            idxs = base_subset.indices
            conds = []

            has_fast = hasattr(base_ds, "seq") and hasattr(base_ds, "seq") and hasattr(base_ds.seq, "seq") and isinstance(base_ds.seq.seq, list)
            if has_fast:
                for i in idxs:
                    folder = base_ds.seq.seq[i][0]
                    c = parse_cond_from_folder(folder)
                    conds.append(c)
            else:
                for i in idxs:
                    folder = base_ds[i]["folder"]
                    c = parse_cond_from_folder(folder)
                    conds.append(c)
            return np.array(conds)

        try:
            train_id_ds = _base(train_loader.dataset)  # IDRegionDataset
            conds = _extract_cond_list(train_id_ds)
            w = np.ones(len(conds), dtype=np.float32)
            w[conds == "cl"] *= 3.0
            w[conds == "bg"] *= 1.2

            sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=0,
                drop_last=False,
            )
            print("[sampler] Enabled condition-balanced sampling: cl*3.0, bg*1.2")
        except Exception as e:
            print(f"[sampler] WARNING: falling back to default shuffle due to: {e}")
    else:
        print("[sampler] Disabled (balance_cond = 0)")

    # Consistency checks on label map
    def _base_map(ds): return ds.dataset if isinstance(ds, tud.Subset) else ds
    tm = getattr(_base_map(train_loader.dataset), "y_map", None)
    if val_loader is not None:
        vm = getattr(_base_map(val_loader.dataset),   "y_map", None)
        assert tm and vm and (tm == vm), "Label-map mismatch: train and val y_map differ."
    assert num_classes == len(tm), "num_classes != size of y_map."

    # Run directory & config dump
    stamp = time.strftime("%Y%m%d-%H%M%S")
    auto_name = f"id_{args.region.lower()}" + (f"_{args.fusion}" if args.region.upper() == 'ALL' else '')
    exp_name = args.exp_name if args.exp_name else f"{auto_name}_{stamp}"
    run_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    dump_json(os.path.join(run_dir, "config.json"), vars(args))

    # Save label map (for interpretability)
    try:
        ymap = val_loader.dataset.dataset.y_map   # IDRegionDataset.y_map
        inv = {v: k for k, v in ymap.items()}
        with open(os.path.join(run_dir, "label_map.csv"), "w", newline="") as f:
            wcsv = csv.writer(f); wcsv.writerow(["label", "subject"])
            for k in range(num_classes):
                wcsv.writerow([k, inv.get(k, f"subj_{k}")])
    except Exception:
        pass

    # Model
    backbone = load_ssl_backbone(
        args.ckpt,
        d_model=args.d_model,
        nhead=getattr(args, "nhead", 4),
        depth_pairs=getattr(args, "depth_pairs", 2),
        proj_dim=args.proj_dim,
        l2_norm=bool(getattr(args, "l2_norm", 0)),
        device=device,
        use_graph_mask=bool(getattr(args, "graph_mask", 0)),
        spatial_only=bool(getattr(args, "spatial_only", 0)),
    )

    # Trainability (explicit)
    backbone.requires_grad_(not bool(args.freeze_backbone))
    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    in_dim = getattr(backbone, "d_model", args.d_model)
    head_in = in_dim
    head = IDHead(in_dim=head_in, num_classes=num_classes).to(device)

    # Metric head (ArcFace) for training + projection featurizer for eval
    use_arc = int(getattr(args, "use_arcface", 0)) == 1
    metric = None
    if use_arc:
        metric = MetricHead(
            in_dim=head_in, 
            proj_dim=getattr(args, "proj_dim", 256), 
            num_classes=num_classes,
            s=float(getattr(args, "arc_s", 16.0)), 
            m=float(getattr(args, "arc_m", 0.35)),
        ).to(device)
        print(f"[arcface] MetricHead enabled (proj_dim={args.proj_dim}, s={args.arc_s}, m={args.arc_m}).")

    # Choose featurizer according to --knn_source and availability of metric head
    featurizer = _choose_featurizer(args, metric)
    print(f"[knn] feature source = {getattr(args, 'knn_source', 'auto')} "
          f"({'metric.proj' if featurizer is not None else 'raw z'})")

    # sanity on trainability
    n_bb_total = sum(p.numel() for p in backbone.parameters())
    n_bb_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[debug] backbone params: total={n_bb_total}, trainable={n_bb_train}")

    backbone.train(); head.train()
    if metric is not None: metric.train()

    # Optimizer
    param_groups = []
    if not args.freeze_backbone:
        param_groups.append({
            "params": backbone.parameters(),
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        })
    param_groups.append({
        "params": head.parameters(),
        "lr": args.lr_head,
        "weight_decay": args.weight_decay,
    })
    if metric is not None:
        param_groups.append({
            "params": metric.parameters(),
            "lr": args.lr_head,
            "weight_decay": args.weight_decay,
        })
    optim = torch.optim.AdamW(param_groups)

    # Warmup (only when ArcFace used)
    use_warmup = use_arc and int(getattr(args, "warmup_steps", 100)) > 0
    warmup_left = int(getattr(args, "warmup_steps", 100))

    # ArcFace scale schedule (if metric head exists)
    arc_s_target = float(getattr(args, "arc_s", 16.0))
    arc_s_min    = float(getattr(args, "arc_s_min", max(8.0, 0.5 * arc_s_target)))
    arc_s_warm   = int(getattr(args, "arc_s_warm", max(100, int(0.2 * args.steps))))  # steps to reach target


    # AMP
    use_amp = bool(args.amp) and torch.cuda.is_available()
    if use_amp:
        from torch import amp
        scaler = amp.GradScaler('cuda')
        def autocast_ctx(): return amp.autocast('cuda')
    else:
        scaler = None
        def autocast_ctx(): return nullcontext()

    # helper: dump features for a split
    def _maybe_dump(split_name: str, loader):
        if not int(getattr(args, "dump_features", 0)):
            return
        featurizer = _choose_featurizer(args, metric)  # respects knn_source
        X_t, y_t, conds, views_t, folders = extract_features_with_meta(
            backbone, loader, device, args.region, getattr(args, "fusion", None),
            l2norm=bool(getattr(args, "l2_norm", 1)), featurizer=featurizer
        )

        # y_map (label->subject string) for readable CSV
        def _base(ds): return ds.dataset if isinstance(ds, tud.Subset) else ds
        ymap = getattr(_base(loader.dataset), "y_map", None)
        if ymap is None:
            # fallback to val loader's map if present
            ymap = getattr(_base(val_loader.dataset), "y_map", {})

        out_dir = args.dump_dir if getattr(args, "dump_dir", "") else run_dir
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, f"features_{split_name}.npz")
        _dump_features_npz(X_t.cpu(), y_t.cpu(), conds, views_t.cpu(), folders, ymap, out_npz)
        print(f"[dump] wrote {out_npz}")

        if int(getattr(args, "dump_match_knn", 1)):
            gal_idx, pr_idx = _materialize_knn_partition(
                X_t.cpu(), y_t.cpu(), conds, views_t.cpu(), folders,
                gallery_cond=args.gallery_cond, probe_cond=args.probe_cond,
                view_match=int(args.knn_view_match),
                gallery_per_subject=(args.gallery_per_subject if getattr(args, "gallery_per_subject", 0) > 0 else args.knn_gallery),
                seed=0
            )
            np.savetxt(os.path.join(out_dir, f"gallery_idx_{split_name}.txt"), gal_idx, fmt="%d")
            np.savetxt(os.path.join(out_dir, f"probe_idx_{split_name}.txt"),   pr_idx,  fmt="%d")
            print(f"[dump] wrote gallery/probe indices for {split_name}")

    # export-only shortcut (no training)
    if int(getattr(args, "export_only", 0)) == 1:
        # optionally load a training checkpoint that contains backbone/head(/metric)
        export_ckpt = getattr(args, "export_ckpt", "")
        if export_ckpt:
            ck = torch.load(export_ckpt, map_location=device)
            if "backbone" in ck: backbone.load_state_dict(ck["backbone"])
            if "head"     in ck: head.load_state_dict(ck["head"])
            if (ck.get("metric") is not None) and (metric is not None):
                try: metric.load_state_dict(ck["metric"])
                except Exception: pass
            print(f"[export-only] loaded weights from {export_ckpt}")
        backbone.eval(); head.eval()
        if metric is not None: metric.eval()

        dump_splits = [s.strip() for s in str(getattr(args, "dump_splits","val")).split(",") if s.strip()]
        if "train" in dump_splits: _maybe_dump("train", train_loader)
        if "val"   in dump_splits: _maybe_dump("val",   val_loader)
        print("[export-only] done.")
        return
    
    # --- Training Loop
    best_top1 = -1.0
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    it = iter(train_loader)

    # --- Per-step training-loss log for the training-dynamics figure
    train_loss_csv = os.path.join(run_dir, "train_loss.csv")
    with open(train_loss_csv, "w") as _tf:
        _tf.write("step,loss\n")

    for step in range(1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(train_loader)
            b = next(it)

        y = b["y"].to(device, dtype=torch.long)

        if metric is not None:
            if use_warmup and warmup_left > 0:
                metric.s = arc_s_min
            else:
                # linear ramp from min -> target across arc_s_warm steps
                # start ramping the moment warmup ends
                steps_after_warm = step - int(args.warmup_steps) if use_warmup else step
                ramp_t = min(1.0, max(0.0, steps_after_warm / max(1, arc_s_warm)))
                metric.s = arc_s_min + (arc_s_target - arc_s_min) * ramp_t
        with autocast_ctx():
            with torch.set_grad_enabled(not args.freeze_backbone):
                z = _forward_region(backbone, b, device, args.region, args.fusion)
            
            # choose exactly one loss path
            if use_arc and (not use_warmup or warmup_left <= 0):
                arc_logits, _ = metric(z, y)
                logits = arc_logits
                loss = F.cross_entropy(arc_logits, y, label_smoothing=ce_smooth)
            else:
                logits = head(z)
                loss = F.cross_entropy(logits, y, label_smoothing=ce_smooth)

            if step == 1:
                print(f"[debug] z.requires_grad = {z.requires_grad}")

        if use_warmup and warmup_left > 0:
            warmup_left -= 1
            if warmup_left == 0:
                print("[warmup] Completed CE warmup; switching to ArcFace loss.")

        # Backward
        optim.zero_grad(set_to_none=True)

        if scaler is not None:  # AMP path
            scaler.scale(loss).backward()
            if clip_max > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optim.param_groups for p in g["params"] if p.requires_grad], 
                    max_norm=clip_max
                    )
            scaler.step(optim)
            scaler.update()
        else:  # no AMP
            loss.backward()
            if clip_max > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optim.param_groups for p in g["params"] if p.requires_grad],
                    max_norm=clip_max
                    )
            optim.step()

        # Per-step loss log for the training-dynamics figure
        with open(train_loss_csv, "a") as _tf:
            _tf.write(f"{step},{float(loss.item()):.6f}\n")

        if step % 10 == 0:
            with torch.no_grad():
                train_pred = logits.argmax(1)
                train_acc = (train_pred == y).float().mean().item()
            print(f"[{step:04d}] loss={loss.item():.4f} acc={train_acc:.3f}")
        
        if (step % args.eval_every == 0) or (step == args.steps):
            backbone.eval(); head.eval()
            if metric is not None: metric.eval()

            # Under standard protocol, val_loader is None: skip in-training eval,
            # just save a periodic checkpoint. Final test evaluation at the end.
            do_val_eval = (val_loader is not None)

            top1 = top5 = None
            if do_val_eval:
                if args.eval_protocol == "classifier":
                    top1, top5 = evaluate(backbone, head, val_loader, device, args.region, args.fusion)
                else:
                    gal_cap = args.gallery_per_subject if getattr(args, "gallery_per_subject", 0) > 0 else args.knn_gallery
                    top1, top5 = evaluate_knn(
                        backbone, val_loader, device, args.region, args.fusion,
                        gallery_cond=args.gallery_cond, probe_cond=args.probe_cond,
                        view_match=int(args.knn_view_match),
                        view_exclude_identical=int(getattr(args, "knn_view_exclude_identical", 0)),
                        gallery_per_subject=int(gal_cap),
                        rule=args.knn_rule, K=int(args.knn_k), l2norm=bool(args.l2_norm),
                        featurizer=featurizer,
                    )

                if num_classes >= 5:
                    print(f"[val@{step}] top1={top1:.4f} top5={top5:.4f}")
                else:
                    print(f"[val@{step}] top1={top1:.4f}")
                append_csv(metrics_csv, {
                    "step": step, "split": "val", "top1": top1,
                    "top5": top5 if num_classes >= 5 else top1, "num_classes": num_classes
                })
            else:
                # Standard protocol: val_loader is None. Run a monitoring test-set
                # eval for the training-dynamics figure (NOT used for checkpoint
                # selection; last.pt remains what the final test eval loads).
                if test_loader is not None:
                    featurizer_m = _choose_featurizer(args, metric)
                    gal_cap_m = args.gallery_per_subject if getattr(args, "gallery_per_subject", 0) > 0 else args.knn_gallery
                    tmon_top1, tmon_top5 = evaluate_knn(
                        backbone, test_loader, device, args.region, args.fusion,
                        gallery_cond=args.gallery_cond, probe_cond=args.probe_cond,
                        view_match=int(args.knn_view_match),
                        view_exclude_identical=int(getattr(args, "knn_view_exclude_identical", 0)),
                        gallery_per_subject=int(gal_cap_m),
                        rule=args.knn_rule, K=int(args.knn_k), l2norm=bool(args.l2_norm),
                        featurizer=featurizer_m,
                    )
                    print(f"[test-monitor@{step}] top1={tmon_top1:.4f} top5={tmon_top5:.4f}")
                    append_csv(metrics_csv, {
                        "step": step, "split": "test_monitor", "top1": tmon_top1,
                        "top5": tmon_top5, "num_classes": num_classes
                    })
                else:
                    print(f"[step {step}] standard protocol: no val/test eval; checkpoint saved to last.pt")

            # ---- checkpointing ----
            ckpt = {
                "step": step,
                "top1": top1 if top1 is not None else -1.0,
                "top5": top5 if top5 is not None else -1.0,
                "args": vars(args),
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "metric": (metric.state_dict() if metric is not None else None),
                }

            # always keep the latest
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))

            # update best (only when we have val-based selection)
            if do_val_eval and top1 is not None and top1 > best_top1:
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                dump_json(os.path.join(run_dir, "best.json"), {"step": step, "top1": top1})
                print(f"[val@{step}] new best → {top1:.4f}; saved best.pt")

            # periodic (reuse your confusion cadence if set)
            if args.confmat_every and (step % args.confmat_every == 0):
                torch.save(ckpt, os.path.join(run_dir, f"ckpt_step{step}.pt"))

            if do_val_eval and top1 is not None:
                best_top1 = max(best_top1, top1)

            need_conf = do_val_eval and (
                (args.confmat_every and args.confmat_every > 0 and (step % args.confmat_every == 0))
                or (args.save_final_confmat and step == args.steps)
            )
            if need_conf:
                import pandas as pd
                gal_cap = args.gallery_per_subject if getattr(args, "gallery_per_subject", 0) > 0 else args.knn_gallery
                if args.eval_protocol == "classifier" and metric is None:
                    conf = compute_confusion(backbone, head, val_loader, device, args.region, args.fusion, num_classes)
                else:
                    conf = compute_confusion_knn(
                        backbone, val_loader, device, args.region, args.fusion,
                        gallery_cond=args.gallery_cond, probe_cond=args.probe_cond,
                        view_match=int(args.knn_view_match),
                        view_exclude_identical=int(getattr(args, "knn_view_exclude_identical", 0)),
                        gallery_per_subject=int(gal_cap),
                        rule=args.knn_rule, K=int(args.knn_k), l2norm=bool(args.l2_norm),
                        featurizer=featurizer,
                    )
                if conf is None:
                    print(f"[val@{step}] confusion skipped (None from evaluator)")
                else:
                    conf_np = conf.cpu().numpy() if isinstance(conf, torch.Tensor) else np.asarray(conf)
                    if conf_np.ndim == 2:
                        df = pd.DataFrame(
                            conf_np,
                            index=[f"true_{i}" for i in range(conf_np.shape[0])],
                            columns=[f"pred_{i}" for i in range(conf_np.shape[1])],
                        )
                        out_path = os.path.join(run_dir, f"confmat_step{step}.csv")
                        df.to_csv(out_path, index=True)
                        true_counts = conf_np.sum(axis=1).tolist()
                        pred_counts = conf_np.sum(axis=0).tolist()
                        dump_json(
                            os.path.join(run_dir, f"confmat_step{step}_summary.json"),
                            {"true_counts": true_counts, "pred_counts": pred_counts}
                        )
                        print(f"[val@{step}] saved confusion matrix → {out_path}")
                    else:
                        print(f"[val@{step}] confusion skipped (ndim={conf_np.ndim})")

            if metric is not None: metric.train()
            head.train(); backbone.train()

            # dump features after each eval if requested
            dump_splits = [s.strip() for s in str(getattr(args, "dump_splits", "")).split(",") if s.strip()]
            if dump_splits:
                if "val" in dump_splits:
                    _maybe_dump("val", val_loader)
                if "train" in dump_splits:
                    _maybe_dump("train", train_loader)

    print(f"Best top1: {best_top1:.4f}")

    # ---------- Final evaluation on held-out test set ----------
    if test_loader is not None:
        print("\n[TEST] Evaluating on held-out test set (never used for checkpoint selection)...")

        # Checkpoint-selection strategy:
        # - custom protocol: use best.pt (val-selected), fall back to last.pt
        # - standard protocol: use last.pt directly (field convention,
        #   GaitSet/GaitGraph/GaitBase)
        protocol = getattr(args, "protocol", "custom")
        if protocol == "standard":
            ckpt_path = os.path.join(run_dir, "last.pt")
            ckpt_label = "last (standard protocol)"
        else:
            ckpt_path = os.path.join(run_dir, "best.pt")
            ckpt_label = "best (val-selected)"
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(run_dir, "last.pt")
                ckpt_label = "last (best.pt missing)"

        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device)
            backbone.load_state_dict(ck["backbone"])
            if "head" in ck:
                head.load_state_dict(ck["head"])
            if ck.get("metric") is not None and metric is not None:
                try: metric.load_state_dict(ck["metric"])
                except Exception: pass
            top1_str = f"{ck.get('top1'):.4f}" if isinstance(ck.get('top1'), float) and ck.get('top1') >= 0 else "n/a"
            print(f"[TEST] Loaded {ckpt_label} checkpoint from step {ck.get('step', '?')} (val top1={top1_str})")

        backbone.eval(); head.eval()
        if metric is not None: metric.eval()

        featurizer = _choose_featurizer(args, metric)
        gal_cap = args.gallery_per_subject if getattr(args, "gallery_per_subject", 0) > 0 else args.knn_gallery
        test_top1, test_top5 = evaluate_knn(
            backbone, test_loader, device, args.region, args.fusion,
            gallery_cond=args.gallery_cond, probe_cond=args.probe_cond,
            view_match=int(args.knn_view_match),
            view_exclude_identical=int(getattr(args, "knn_view_exclude_identical", 0)),
            gallery_per_subject=int(gal_cap),
            rule=args.knn_rule, K=int(args.knn_k), l2norm=bool(args.l2_norm),
            featurizer=featurizer,
        )
        print(f"[TEST] test_top1={test_top1:.4f} test_top5={test_top5:.4f}")
        append_csv(metrics_csv, {
            "step": "test", "split": "test", "top1": test_top1,
            "top5": test_top5, "num_classes": num_classes
        })

        # Dump test features if requested
        dump_splits = [s.strip() for s in str(getattr(args, "dump_splits", "")).split(",") if s.strip()]
        if "test" in dump_splits:
            _maybe_dump("test", test_loader)

# --------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train identification head with region masking + clothing/view-aware k-NN eval."
    )
    # Backbone / Head
    p.add_argument('--ckpt', type=str, default="",
                   help="Path to SSL checkpoint. Empty = random init.")
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--depth_pairs', type=int, default=2,
                   help="Number of (Spatial, Temporal) attention pairs.")
    p.add_argument('--proj_dim', type=int, default=256)
    p.add_argument('--l2_norm', type=int, default=1)
    p.add_argument('--freeze_backbone', type=int, default=1)
    p.add_argument('--lr_head', type=float, default=3e-4)

    # Training
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr_backbone', type=float, default=3e-6)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--amp', type=int, default=1)

    # Data
    p.add_argument('--csv', type=str, required=True, default="data/casia-b_pose_coco.csv")
    p.add_argument('--seq_len', type=int, default=30)
    p.add_argument('--stride', type=int, default=30)
    p.add_argument('--conf_thr', type=float, default=0.5)
    p.add_argument('--min_clips', type=int, default=4)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--limit_rows', type=int, default=None)

    # Regions
    p.add_argument('--region', choices=['R1', 'R2', 'ALL'], default='R1')
    p.add_argument('--fusion', choices=['avg', 'concat'], default='avg')

    # Splits / Limits
    p.add_argument('--split_mode', choices=['subject', 'clip'], default='subject')
    p.add_argument('--limit_subjects', type=int, default=-1,
                   help="0=use all subjects; else first N")
    p.add_argument('--limit_per_subject', type=int, default=9999,
                   help="0=all clips; else cap at N per subject")

    # Logging
    p.add_argument('--log_dir', type=str, default='runs')
    p.add_argument('--exp_name', type=str, default='exp')
    p.add_argument('--confmat_every', type=int, default=0)
    p.add_argument('--save_final_confmat', type=int, default=1)

    # Validation protocol
    p.add_argument('--eval_protocol', type=str, choices=['classifier','knn'], default='knn')
    p.add_argument('--knn_gallery', type=int, default=1,
                   help="Legacy gallery cap (kept for compatibility).")
    p.add_argument('--knn_k', type=int, default=5,
                   help="Top-K neighbours for k-NN rules (vote/sim). Ignored for 'centroid'.")
    p.add_argument('--knn_rule', type=str, choices=['centroid','vote','sim'], default='sim')
    p.add_argument('--knn_source', type=str, choices=['auto','raw','metric'], default='auto',
                   help="auto: metric if available else raw; raw: backbone z; metric: MetricHead projection")
    p.add_argument('--knn_view_match', type=int, default=0,
                   help="If 1, restrict candidates to same view id as probe.")
    p.add_argument('--knn_view_exclude_identical', type=int, default=0,
                   help="If 1, exclude same-view gallery items for each probe "
                        "(standard CASIA-B cross-view protocol). Ignored when "
                        "--knn_view_match=1.")
    p.add_argument('--protocol', choices=['custom', 'standard'], default='custom',
                   help="custom: 99/15/10 split with view-matched retrieval (thesis). "
                        "standard: 74/50 split with cross-view retrieval, field "
                        "convention (Reviewer 1 R1-1).")
    p.add_argument('--graph_mask', type=int, default=0,
                   help="If 1, restrict spatial attention to skeletally-"
                        "adjacent COCO-17 joint pairs (plus self-loops). "
                        "Implements the graph-prior ablation requested by "
                        "Reviewer 2 (R2-4). Must match the --graph_mask flag "
                        "used when the SSL checkpoint was pretrained for a "
                        "fair comparison. Default 0 (no mask).")
    p.add_argument('--spatial_only', type=int, default=0,
                   help="If 1, replace every TemporalBlock with a "
                        "SpatialBlock (stack becomes S-S-S-S instead of "
                        "S-T-S-T). Implements the spatial-only ablation "
                        "requested by Reviewer 3 (R3-3). MUST match the "
                        "--spatial_only flag used when the SSL checkpoint "
                        "was pretrained; a spatial-only checkpoint will not "
                        "load into the main architecture. Default 0.")

    # --- Feature export / export-only ---
    p.add_argument('--dump_features', type=int, default=0,
                   help="If 1, dump features+ids for the chosen split(s) after eval (or in export-only mode).")
    p.add_argument('--dump_splits', type=str, default="val",
                   help="Comma list of splits to dump: 'train', 'val', or 'train,val'.")
    p.add_argument('--dump_dir', type=str, default="",
                   help="Override output dir (default = run_dir).")
    p.add_argument('--dump_format', type=str, choices=['npz'], default='npz')

    # Match the exact KNN partition used during eval for deterministic gallery/probe sets
    p.add_argument('--dump_match_knn', type=int, default=1,
                   help="If 1, also save gallery/probe indices matching current KNN args.")

    # Export-only utility (no training loop)
    p.add_argument('--export_only', type=int, default=0,
                   help="If 1, skip training and just export features for requested splits.")
    p.add_argument('--export_ckpt', type=str, default="",
                   help="Optional: path to a train_id checkpoint (best.pt/last.pt) for export-only.")


    # Gallery/Probe
    p.add_argument('--gallery_cond', type=str, default="",
                   help="k-NN gallery conditions, e.g. 'nm' or 'nm,bg'. Empty = all.")
    p.add_argument('--probe_cond',   type=str, default="",
                   help="k-NN probe conditions, e.g. 'bg,cl'. Empty = all.")
    p.add_argument('--gallery_per_subject', type=int, default=0,
                   help="Cap gallery items per subject (and per view if --knn_view_match=1). 0 disables capping.")

    # Augment / sampling
    p.add_argument('--pose_aug', type=int, default=1,
                   help="Enable pose augmentations on train split (0/1).")
    p.add_argument('--balance_cond', type=int, default=1,
                   help="Enable condition-balanced sampler (cl*3, bg*1.2).")

    # Metric head (ArcFace)
    p.add_argument('--use_arcface', type=int, default=0,
                   help="Use ArcFace metric head for training + projection (0/1).")
    p.add_argument('--arc_s', type=float, default=16.0)     # scale
    p.add_argument('--arc_m', type=float, default=0.35)     # margin
    p.add_argument('--warmup_steps', type=int, default=100) # CE warmup before ArcFace
    p.add_argument('--arc_s_min', type=float, default=6.0,
                   help="Starting scale for ArcFace ramp; will linearly ramp to --arc_s.")
    p.add_argument('--arc_s_warm', type=int, default=400,
                   help="Steps to ramp ArcFace scale from arc_s_min to arc_s after warmup.")

    # Seed
    p.add_argument('--seed', type=int, default=3407)

    # ---- stability / schedule ----
    p.add_argument("--clip_grad", "--grad_clip", dest="clip_grad", type=float, default=1.0,
                   help="Max grad-norm; <=0 disables. (alias: --grad_clip [deprecated])")
    p.add_argument('--ce_smooth', dest='ce_smooth', type=float, default=0.0,
                   help='Label smoothing value for CE loss (e.g., 0.05).')
    p.add_argument("--freeze_first", type=int, default=0,
                   help="Keep backbone frozen for the first N steps, then unfreeze (0 = disabled).")

    return p.parse_args()

# ---------- MAIN
def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_training(args, device)

if __name__ == "__main__":
    main()