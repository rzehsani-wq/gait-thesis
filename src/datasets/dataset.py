from typing import Optional, Tuple, List
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.pose_norm import normalize_xy
from ..utils.mask_utils import split_regions

N_JOINTS = 17
COCO_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _find_semantic_cols(df: pd.DataFrame):
    """
    Return three lists of column names (len=17) for x, y, conf (conf may be None).
    """
    cols = list(df.columns)
    lower2orig = {c.lower(): c for c in cols}

    def get_col(base: str, suffix: str) -> Optional[str]:
        key = f"{base}_{suffix}".lower()
        return lower2orig.get(key, None)

    xcols, ycols, ccols = [], [], []
    missing = {"x": [], "y": []}

    for name in COCO_NAMES:
        x = get_col(name, "x")
        y = get_col(name, "y")
        c = get_col(name, "conf")  # optional

        if x is None: missing["x"].append(name)
        if y is None: missing["y"].append(name)

        xcols.append(x)
        ycols.append(y)
        ccols.append(c)

    if missing["x"] or missing["y"]:
        raise ValueError(
            "Dataset loader could not find required columns.\n"
            f"Missing X for: {missing['x']}\n"
            f"Missing Y for: {missing['y']}\n"
            "Expected columns like 'nose_x', 'left_eye_y', 'right_hip_conf', etc."
        )
    return xcols, ycols, ccols


def _derive_folder_from_image_name(df: pd.DataFrame) -> pd.Series:
    if "image_name" not in df.columns:
        raise ValueError(
            "Expected a column named 'image_name' with paths like "
            "'./001-bg-01-000/000001.jpg'"
        )
    return df["image_name"].astype(str).apply(
        lambda p: os.path.dirname(p.rstrip("/\\")) + "/"
    )


class CASIABPoseSequences:
    """
    Group rows by folder (from image_name) and create sliding windows.
    """

    def __init__(self, csv_path: str, seq_len: int = 30, stride: int = 30,
                 limit_rows: Optional[int] = None):
        self.df = pd.read_csv(csv_path, nrows=limit_rows)
        xcols, ycols, ccols = _find_semantic_cols(self.df)

        X = self.df[xcols].to_numpy(dtype=np.float32)  # (N, 17)
        Y = self.df[ycols].to_numpy(dtype=np.float32)  # (N, 17)
        xy = np.stack([X, Y], axis=-1)                  # (N, 17, 2)

        if all(c is not None for c in ccols):
            conf = self.df[ccols].to_numpy(dtype=np.float32)  # (N, 17)
            conf_found = True
        else:
            conf = np.ones((len(self.df), N_JOINTS), dtype=np.float32)
            conf_found = False

        self.df["folder"] = _derive_folder_from_image_name(self.df)

        # Groups in original order
        groups: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for folder, g in self.df.groupby("folder", sort=False):
            idx = g.index.to_numpy()
            groups.append((folder, xy[idx], conf[idx]))

        # Sliding windows
        self.seq: List[Tuple[str, np.ndarray, np.ndarray]] = []
        self.seq_len, self.stride = int(seq_len), int(stride)
        for folder, xy_g, conf_g in groups:
            T = len(xy_g)
            if T < self.seq_len:
                continue
            for s in range(0, T - self.seq_len + 1, self.stride):
                self.seq.append(
                    (folder, xy_g[s:s + self.seq_len], conf_g[s:s + self.seq_len])
                )

        print(
            f"[dataset] schema=semantic (nose_x, ...) | "
            f"conf={'found' if conf_found else 'missing->ones'} | "
            f"folders={self.df['folder'].nunique()} | "
            f"windows={len(self.seq)}"
        )

    def __len__(self): return len(self.seq)
    def __getitem__(self, i: int): return self.seq[i]


class CASIABPoseDataset(Dataset):
    """
    CASIA-B pose dataset returning normalized COCO-17 poses split into upper/lower body.

    __getitem__ returns:
        folder: str
        xy:  (L, 17, 2) float — full normalized pose
        R1:  (L, 11, 2)       — upper body (head + arms)
        R2:  (L,  6, 2)       — lower body (legs)
    """

    def __init__(self, csv_path: str, seq_len: int = 30, stride: int = 30,
                 conf_thr: float = 0.5, limit_rows: Optional[int] = None):
        self.seq = CASIABPoseSequences(
            csv_path, seq_len=seq_len, stride=stride, limit_rows=limit_rows
        )
        self.conf_thr = float(conf_thr)

    def __len__(self): return len(self.seq)

    def __getitem__(self, i):
        folder, xy, conf = self.seq[i]
        xy_norm = normalize_xy(xy, conf=conf, conf_thr=self.conf_thr)
        xy_t = torch.from_numpy(xy_norm).float()
        parts = split_regions(xy_t)
        return {
            "folder": folder,
            "xy": xy_t,        # (L, 17, 2)
            "R1": parts["R1"], # (L, 11, 2)
            "R2": parts["R2"], # (L,  6, 2)
        }