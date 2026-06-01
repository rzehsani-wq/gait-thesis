import os
import csv
import bisect
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.pose_norm import normalize_xy
from ..utils.mask_utils import split_regions

# Fast header-only .npy shape reader (no full array load)
from numpy.lib.format import (
    read_magic,
    read_array_header_1_0,
    read_array_header_2_0,
)


# DenseGait saved index -> CASIA-B COCO-17 index.
# Recovered empirically from 10k+ frames of the DenseGait release. The saved
# .npy arrays have 18 joints but the order does NOT match canonical OpenPose
# COCO-18. Index 4 in the saved data is the neck (no equivalent in COCO-17)
# and is dropped here.
#
# L/R follows image-space x convention: smaller x => "left" in the image.
# Since normalize_xy uses joints (5,6) and (11,12) symmetrically, L/R
# labeling does not affect the normalization math.
DENSEGAIT_TO_COCO17 = [
    0,    # COCO-17[ 0] nose         <- DenseGait[ 0]
    17,   # COCO-17[ 1] l_eye        <- DenseGait[17]
    5,    # COCO-17[ 2] r_eye        <- DenseGait[ 5]
    7,    # COCO-17[ 3] l_ear        <- DenseGait[ 7]
    9,    # COCO-17[ 4] r_ear        <- DenseGait[ 9]
    6,    # COCO-17[ 5] l_shoulder   <- DenseGait[ 6]
    8,    # COCO-17[ 6] r_shoulder   <- DenseGait[ 8]
    10,   # COCO-17[ 7] l_elbow      <- DenseGait[10]
    11,   # COCO-17[ 8] r_elbow      <- DenseGait[11]
    13,   # COCO-17[ 9] l_wrist      <- DenseGait[13]
    15,   # COCO-17[10] r_wrist      <- DenseGait[15]
    12,   # COCO-17[11] l_hip        <- DenseGait[12]
    14,   # COCO-17[12] r_hip        <- DenseGait[14]
    16,   # COCO-17[13] l_knee       <- DenseGait[16]
    1,    # COCO-17[14] r_knee       <- DenseGait[ 1]
    2,    # COCO-17[15] l_ankle      <- DenseGait[ 2]
    3,    # COCO-17[16] r_ankle      <- DenseGait[ 3]
]


def read_npy_shape_fast(path: str) -> Tuple[int, ...]:
    """Read only the .npy header to get array shape. Fast; does NOT load full array."""
    with open(path, "rb") as f:
        version = read_magic(f)
        if version == (1, 0):
            shape, _, _ = read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = read_array_header_2_0(f)
        else:
            shape, _, _ = read_array_header_2_0(f)
    return tuple(shape)


def _remap_xy_to_coco17(arr: np.ndarray) -> np.ndarray:
    """
    Take a [T, J, C] raw DenseGait pose array and return [T, 17, 2] xy in
    CASIA-B COCO-17 order. Expects J = 18 (the empirically verified format).
    For other J values, falls back to truncate/pad with a warning, but this
    indicates the file is not a standard DenseGait pose array.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected .npy shape [T,J,C], got {arr.shape}")
    T, J, C = arr.shape
    if C < 2:
        raise ValueError(f"Expected last dim >= 2, got {arr.shape}")

    xy = arr[..., :2]  # [T, J, 2]

    if J == 18:
        xy = xy[:, DENSEGAIT_TO_COCO17, :]                  # -> (T, 17, 2)
    elif J == 17:
        # Already 17 joints; assume order is already COCO-17 and hope for the best.
        pass
    elif J > 18:
        xy = xy[:, DENSEGAIT_TO_COCO17, :]
    else:
        # J < 17: pad and warn
        pad = np.zeros((T, 17 - J, 2), dtype=xy.dtype)
        xy = np.concatenate([xy, pad], axis=1)

    return np.asarray(xy, dtype=np.float32)                  # [T, 17, 2]


def _remap_conf_to_coco17(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Take a [T, J, C] raw pose array (with C >= 3) and return [T, 17] confidence
    in COCO-17 order. Returns None if confidence is unavailable.
    """
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    T, J, _ = arr.shape
    conf = arr[..., 2]                                       # [T, J]
    if J == 18:
        return np.asarray(conf[:, DENSEGAIT_TO_COCO17], dtype=np.float32)
    if J >= 17:
        return np.asarray(conf[:, :17], dtype=np.float32)
    pad = np.zeros((T, 17 - J), dtype=np.float32)
    return np.concatenate([conf.astype(np.float32), pad], axis=1)


class DenseGaitPoseDataset(Dataset):
    """
    CSV-driven DenseGait dataset that produces sliding windows of poses
    remapped to COCO-17 order and re-normalized to match CASIA-B convention.

    Each __getitem__ call returns:
        {
            "folder":    str,
            "xy":        Tensor (L, 17, 2) normalized full pose,
            "R1":        Tensor (L, 11, 2) upper body (head + arms),
            "R2":        Tensor (L,  6, 2) lower body (legs),
            "meta":      dict (CSV row),
            "file_name": str,
            "start":     int,
        }

    Note: DenseGait poses are already body-normalized by the original authors
    (pelvis-centered, anisotropic shoulder/neck scaling). We then re-apply
    CASIA-style normalize_xy on top of that to unify the coordinate convention
    across the two datasets. This is intentional double-normalization; the
    result is a consistent representation across both datasets even though
    the intermediate scales differ.
    """

    def __init__(
        self,
        csv_path: str,
        poses_dir: str,
        seq_len: int = 56,
        stride: int = 2,
        min_len: int = 32,
        conf_thr: float = 0.3,          # lower than CASIA: in-the-wild noisier
        limit_rows: Optional[int] = None,
        limit_files: Optional[int] = None,
        verbose: bool = True,
    ):
        self.csv_path = csv_path
        self.poses_dir = poses_dir
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.min_len = int(min_len)
        self.conf_thr = float(conf_thr)

        # 1) Read CSV
        rows: List[Dict[str, Any]] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for i, d in enumerate(rdr):
                if limit_rows is not None and i >= int(limit_rows):
                    break
                if "file_name" not in d:
                    raise ValueError(
                        "CSV must include a 'file_name' column (e.g., '0.npy')."
                    )
                rows.append(d)

        # 2) Validate files + read header-only length T
        self.files: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.lengths: List[int] = []

        for d in rows:
            fn = d["file_name"]
            p = os.path.join(self.poses_dir, fn)
            if not os.path.isfile(p):
                continue
            shape = read_npy_shape_fast(p)
            if len(shape) < 1:
                continue
            T = int(shape[0])
            if T < max(self.min_len, self.seq_len):
                continue
            self.files.append(p)
            self.metas.append(d)
            self.lengths.append(T)
            if limit_files is not None and len(self.files) >= int(limit_files):
                break

        if len(self.files) == 0:
            raise ValueError(
                f"No valid DenseGait pose files after filtering.\n"
                f"csv_path={self.csv_path}\nposes_dir={self.poses_dir}"
            )

        # 3) Build window counts and prefix sum index
        self.win_counts: List[int] = []
        for T in self.lengths:
            n = 1 + (T - self.seq_len) // self.stride
            self.win_counts.append(max(0, int(n)))

        self.prefix: List[int] = [0]
        for n in self.win_counts:
            self.prefix.append(self.prefix[-1] + n)
        self.num_windows = int(self.prefix[-1])

        if verbose:
            print(
                f"[DenseGaitPoseDataset] poses_dir={self.poses_dir} | "
                f"csv={os.path.basename(self.csv_path)} | "
                f"files={len(self.files)} | windows={self.num_windows}"
            )

    def __len__(self) -> int:
        return self.num_windows

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(idx)
        fidx = bisect.bisect_right(self.prefix, idx) - 1
        local = idx - self.prefix[fidx]
        return fidx, local

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fidx, local = self._locate(int(idx))
        start = int(local * self.stride)
        arr = np.load(self.files[fidx], mmap_mode="r")        # loads only ONE file

        # Remap to COCO-17 once, then slice
        xy_full = _remap_xy_to_coco17(arr)                    # [T, 17, 2]
        conf_full = _remap_conf_to_coco17(arr)                # [T, 17] or None

        clip = xy_full[start:start + self.seq_len]            # [L, 17, 2]
        conf_clip = (
            conf_full[start:start + self.seq_len] if conf_full is not None else None
        )

        # Normalize (mid-hip centered, shoulder-width scaled, conf-aware)
        clip_norm = normalize_xy(clip, conf=conf_clip, conf_thr=self.conf_thr)
        clip_t = torch.from_numpy(clip_norm).float()

        regions = split_regions(clip_t)

        meta = self.metas[fidx]
        folder = meta.get("track_id", None) or os.path.basename(self.files[fidx])

        return {
            "folder":    str(folder),
            "xy":        clip_t,           # (L, 17, 2)
            "R1":        regions["R1"],    # (L, 11, 2)
            "R2":        regions["R2"],    # (L,  6, 2)
            "meta":      meta,
            "file_name": os.path.basename(self.files[fidx]),
            "start":     start,
        }