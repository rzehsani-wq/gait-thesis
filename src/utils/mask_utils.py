"""
Region partitioning for COCO-17 skeletons.

R1 = upper body (head + arms) = 11 joints
R2 = lower body (hips + legs) = 6 joints
"""
from typing import Dict
import torch

# COCO-17 indices
# 0 Nose
# 1 Left Eye        2 Right Eye
# 3 Left Ear        4 Right Ear
# 5 Left Shoulder   6 Right Shoulder
# 7 Left Elbow      8 Right Elbow
# 9 Left Wrist      10 Right Wrist
# 11 Left Hip       12 Right Hip
# 13 Left Knee      14 Right Knee
# 15 Left Ankle     16 Right Ankle

COCO17 = list(range(17))

GROUPS: Dict[str, list] = {
    "head": [0, 1, 2, 3, 4],                 # nose. eyes, ears (5 joints)
    "arms": [5, 6, 7, 8, 9, 10],              # shoulders, elbows, wrists (6 joints)
    "legs": [11, 12, 13, 14, 15, 16],         # hips, knees, ankles (6 joints)
    # Mid-hip is a virtual mid-hip we append at index 17 (computed on the fly)
}

REGIONS: Dict[str, list] = {
    "R1": GROUPS["head"] + GROUPS["arms"],  # upper body: 5 + 6 = 11 joints
    "R2": GROUPS["legs"],   # lower body: 6 joints
    "ALL": COCO17,  # all 17 joints
}

# Joint indices as torch tensors, ready to be passed to the model.
REGION_JOINT_INDICES: Dict[str, torch.Tensor] = {
    "R1": torch.tensor(REGIONS["R1"], dtype=torch.long),
    "R2": torch.tensor(REGIONS["R2"], dtype=torch.long),
    "ALL": torch.tensor(REGIONS["ALL"], dtype=torch.long),
}

def split_regions(xy: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Split a COCO-17 pose into R1 (upper body) and R2 (lower body).

    Args:
        xy: (L, 17, 2) pose tensor

    Returns:
        dict with:
            "R1": (L, 11, 2) — head + arms
            "R2": (L,  6, 2) — legs
    """
    if xy.dim() != 3 or xy.shape[1] != 17:
        raise ValueError(f"split_regions expects (L, 17, 2), got {tuple(xy.shape)}")
    r1 = xy[:, REGIONS["R1"], :]
    r2 = xy[:, REGIONS["R2"], :]
    return {"R1": r1, "R2": r2}


def compute_midhip(xy: torch.Tensor) -> torch.Tensor:
    """
    xy: (L, 17, 2) -> returns (L, 1, 2) — the midpoint between left and right hip.
    """
    return 0.5 * (xy[:, 11:12, :] + xy[:, 12:13, :])