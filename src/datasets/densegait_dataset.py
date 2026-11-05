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

# AlphaPose/OpenPose COCO-18 order -> COCO-17 order (same as CASIA-B COCO-17)
# COCO-18 (OpenPose): [nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear]
# COCO-17 (CASIA-B): [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
COCO18_TO_COCO17 = [
    0,   # nose
    15,  # left_eye
    14,  # right_eye
    17,  # left_ear
    16,  # right_ear
    5,   # left_shoulder
    2,   # right_shoulder
    6,   # left_elbow
    3,   # right_elbow
    7,   # left_wrist
    4,   # right_wrist
    11,  # left_hip
    8,   # right_hip
    12,  # left_knee
    9,   # right_knee
    13,  # left_ankle
    10,  # right_ankle
]

def read_npy_shape_fast(path: str) -> Tuple[int, ...]:
    """
    Read only the .npy header to get array shape (fast; does NOT load the full array).
    """
    with open(path, "rb") as f:
        version = read_magic(f)
        if version == (1, 0):
            shape, _, _ = read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = read_array_header_2_0(f)
        else:
            # Fallback for unexpected versions: try 2.0 header
            shape, _, _ = read_array_header_2_0(f)
    return tuple(shape)

def to_xy_for_masks(arr: np.ndarray, joint_indices: Optional[List[int]] = None) -> torch.Tensor:
    """
    Convert DenseGait pose array -> torch.FloatTensor [T, 17, 2] in CASIA-B COCO-17 order.
    Handles AlphaPose/OpenPose COCO-18 by remapping.

    After this, split_regions(xy) is valid
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected .npy shape [T,J,C], got {arr.shape}")

    T, J, C = arr.shape
    if C < 2:
        raise ValueError(f"Expected last dim >= 2, got {arr.shape}")

    xy = arr[..., :2]  # [T,J,2]
    J = xy.shape[1]

    required_joints = 17

    # Remap COCO-18 (OpenPose/AlphaPose) -> COCO-17 (CASIA-B order)
    if J == 18:
        xy = xy[:, COCO18_TO_COCO17, :]     # -> (T, 17, 2) in CASIA-B COCO-17 order
    elif J > required_joints:
        xy = xy[:, :required_joints, :]
    elif J < required_joints:
        pad = np.zeros((T, required_joints - J, 2), dtype=xy.dtype)
        xy = np.concatenate([xy, pad], axis=1)

    # Optional selection AFTER standardization (indices are now COCO-17 space)
    if joint_indices is not None:
        raise ValueError(
            "DenseGait: joint_indices is not supported. "
            "Poses are standardize to CASIA-B COCO-17 internally. "
            "Set joint_indices = None."
        )

    return torch.from_numpy(xy).float()     # [T, 17, 2] (unless joint_indices changes it)


class DenseGaitPoseDataset(Dataset):
    """
    CSV-driven DenseGait dataset that produces sliding windows and returns
    the exact keys expected by MaskedRegionSSL:
        {
          "folder": str,
          "R1": Tensor [L,12,2],
          "R2": Tensor [L,<=12,2],
          # optional:
          "meta": dict,
          "file_name": str,
          "start": int
        }

    Fast init:
      - reads CSV rows (optionally limited)
      - reads only .npy headers to get T (shape[0])
      - builds window indexing via prefix sums

    Reviewer 1 Comment 3 (R1-3) support: `subsample_n` + `subsample_seed`
    deterministically shuffle the CSV rows and keep only the first
    `subsample_n` valid tracklets, letting us compare DenseGait-SSL at
    matched clip-count to CASIA-B. Preserves the full-dataset default so
    existing runs are unaffected.
    """

    def __init__(
        self,
        csv_path: str,
        poses_dir: str,
        seq_len: int = 56,
        stride: int = 2,
        min_len: int = 32,
        joint_indices: Optional[List[int]] = None,
        conf_thr: float = 0.3,              # lower threshold than CASIA (in-the-wild noise)
        limit_rows: Optional[int] = None,   # smoke test: only read first N CSV rows
        limit_files: Optional[int] = None,  # smoke test: cap number of valid files
        subsample_n: Optional[int] = None,  # R1-3: keep N valid tracklets, seeded random
        subsample_seed: int = 0,            # R1-3: shuffle seed for subsample_n
        verbose: bool = True,
    ):
        self.csv_path = csv_path
        self.poses_dir = poses_dir
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.min_len = int(min_len)
        self.joint_indices = joint_indices
        self.conf_thr = float(conf_thr)

        # 1) Read CSV
        rows: List[Dict[str, Any]] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for i, d in enumerate(rdr):
                if limit_rows is not None and i >= int(limit_rows):
                    break
                if "file_name" not in d:
                    raise ValueError("CSV must include a 'file_name' column (e.g., '0.npy').")
                rows.append(d)

        # 1b) R1-3: deterministic random subsample.
        # If subsample_n is set, shuffle rows by subsample_seed and treat
        # subsample_n as the target valid-file cap (interacts with limit_files
        # by taking the stricter of the two bounds).
        if subsample_n is not None:
            rng = np.random.default_rng(int(subsample_seed))
            idx = np.arange(len(rows))
            rng.shuffle(idx)
            rows = [rows[i] for i in idx]
            cap_from_subsample = int(subsample_n)
            effective_cap = (cap_from_subsample
                             if limit_files is None
                             else min(int(limit_files), cap_from_subsample))
            limit_files = effective_cap
            if verbose:
                print(f"[DenseGaitPoseDataset] subsample enabled: "
                      f"shuffled {len(rows)} rows with seed={subsample_seed}, "
                      f"capping valid files at {effective_cap}.")

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
            n = max(0, int(n))
            self.win_counts.append(n)

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
        """
        Map global window idx -> (file_idx, local_window_idx)
        """
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(idx)
        fidx = bisect.bisect_right(self.prefix, idx) - 1
        local = idx - self.prefix[fidx]
        return fidx, local

    def __getitem__(self, idx):
        fidx, local = self._locate(int(idx))
        start = int(local * self.stride)
        arr = np.load(self.files[fidx], mmap_mode="r")     # loads only ONE file

        xy_full = to_xy_for_masks(arr, self.joint_indices)  # [T, 17, 2]
        clip = xy_full[start:start + self.seq_len].numpy()   # [L, 17, 2]

        # Extract confidence if available, normalize
        # DenseGait stores xyc, so confidence is the 3rd channel of the original array
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            # Need to remap confidences using the same COCO-18 -> COCO-17 map
            if arr.shape[1] == 18:
                from ..datasets.densegait_dataset import COCO18_TO_COCO17
                conf_full = arr[:, COCO18_TO_COCO17, 2]
            else:
                conf_full = arr[:, :17, 2] if arr.shape[1] >= 17 else None

            if conf_full is not None:
                conf_clip = conf_full[start: start + self.seq_len]
            else:
                conf_clip = None
        else:
            conf_clip = None

        clip_normalized = normalize_xy(clip, conf=conf_clip, conf_thr=self.conf_thr)
        # lower threshold for in-the-wild, since in-the-wild pose detections are noisier: we'll lose too many frames at 0.5
        clip = torch.from_numpy(clip_normalized).float()  # (L, 17, 2)

        regions = split_regions(clip)

        meta = self.metas[fidx]
        folder = meta.get("track_id", None) or os.path.basename(self.files[fidx])

        return{
            "folder": str(folder),
            "xy": clip,                   # full (L, 17, 2) COCO-17 pose for MaskedJointSSL
            "R1": regions["R1"],
            "R2": regions["R2"],
            "meta": meta,
            "file_name": os.path.basename(self.files[fidx]),
            "start": start,
        }