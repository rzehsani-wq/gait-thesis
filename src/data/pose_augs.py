import random
import numpy as np

class TemporalDropout:
    """
    Randomly drop frames to simulate brief occlusion / tracker failure.
    """
    def __init__(self, drop_prob=0.15, mode="nan"):
        self.drop_prob = float(drop_prob)
        assert mode in ("nan", "hold")
        self.mode = mode

    def __call__(self, seq: np.ndarray) -> np.ndarray:  # [T,J,C]
        assert seq.ndim == 3
        T = seq.shape[0]
        keep = np.ones(T, dtype=bool)
        for t in range(T):
            if random.random() < self.drop_prob:
                keep[t] = False
        if keep.all():
            return seq
        out = seq.copy()
        if self.mode == "nan":
            out[~keep] = np.nan
        else:  # "hold"
            last = None
            for t in range(T):
                if keep[t]:
                    last = out[t].copy()
                else:
                    out[t] = 0.0 if last is None else last
        return out

class RandomJointMask:
    """
    Mask a subset of joints across the whole window to mimic sustained occlusion.
    """
    def __init__(self, mask_ratio=0.20, groups=None, mode="zero"):
        self.mask_ratio = float(mask_ratio)
        self.groups = groups
        assert mode in ("zero", "nan")
        self.mode = mode

    def __call__(self, seq: np.ndarray) -> np.ndarray:  # [T,J,C]
        assert seq.ndim == 3
        T, J, C = seq.shape
        out = seq.copy()
        if self.groups:
            # Choose a subset of groups
            k = max(1, int(len(self.groups) * self.mask_ratio))
            chosen = random.sample(self.groups, k=k)
            mask_idx = [i for g in chosen for i in g]
        else:
            k = max(1, int(J * self.mask_ratio))
            mask_idx = random.sample(range(J), k=k)
        if self.mode == "zero":
            out[:, mask_idx, :] = 0.0
        else:
            out[:, mask_idx, :] = np.nan
        return out

class RandomUpperBodyWiden:
    """
    Radially expands upper-body joints to mimic bulky clothing.
    """
    def __init__(self, torso_idx, factor_range=(1.05, 1.15), center="mean"):
        self.torso_idx = list(torso_idx)
        self.factor_range = (float(factor_range[0]), float(factor_range[1]))
        self.center = center

    def __call__(self, seq: np.ndarray) -> np.ndarray:  # [T,J,C]
        assert seq.ndim == 3
        out = seq.copy()
        T, J, C = out.shape
        if C < 2:
            return out
        s = random.uniform(*self.factor_range)
        for t in range(T):
            if self.center == "mean":
                cx = np.nanmean(out[t, :, 0])
                cy = np.nanmean(out[t, :, 1])
            else:
                # Fallback center: mean of hips/torso still approximated by overall mean
                cx = np.nanmean(out[t, :, 0])
                cy = np.nanmean(out[t, :, 1])
            for j in self.torso_idx:
                dx = out[t, j, 0] - cx
                dy = out[t, j, 1] - cy
                out[t, j, 0] = cx + s * dx
                out[t, j, 1] = cy + s * dy
        return out

class Compose:
    def __init__(self, augs):
        self.augs = list(augs)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        for a in self.augs:
            seq = a(seq)
        return seq
