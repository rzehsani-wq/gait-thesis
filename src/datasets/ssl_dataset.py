"""
Masked-joint SSL wrapper. Wraps a base dataset that returns full poses ('xy')
and produces (masked_input, target, mask) triplets for the SSL pretraining
objective.

Pretext task:
    Randomly mask `mask_ratio` of (frame, joint) positions per sample.
    Input is the pose with masked positions set to 0.
    Target is the full pose.
    Loss is MSE on the masked positions only (handled in train_ssl.py).
"""
from typing import Dict, Any
import torch
from torch.utils.data import Dataset


class MaskedJointSSL(Dataset):
    def __init__(self, base: Dataset, mask_ratio: float = 0.35,
                 mask_token_value: float = 0.0, seed: int = 0):
        self.base = base
        self.mask_ratio = float(mask_ratio)
        self.mask_value = float(mask_token_value)
        # Generator so different workers get different masks but same dataset.
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.base[i]
        if "xy" not in s:
            raise KeyError(
                "MaskedJointSSL requires base dataset to return 'xy' "
                "(the full L, 17, 2 pose). Got keys: " + str(list(s.keys()))
            )

        full = s["xy"]      # (L, 17, 2)
        L, J, _ = full.shape

        # Random boolean mask over (L, J) positions
        mask = torch.rand(L, J) < self.mask_ratio
        # Ensure at least one position is masked (so loss isn't NaN)
        if not mask.any():
            mask[0, 0] = True

        masked = full.clone()
        masked[mask] = self.mask_value

        return {
            "folder": s.get("folder", ""),
            "input":  masked,    # (L, 17, 2)
            "target": full,      # (L, 17, 2)
            "mask":   mask,      # (L, 17) bool - True = position was masked
        }