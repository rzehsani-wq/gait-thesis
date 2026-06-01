import numpy as np
from typing import Optional


# Maximum absolute coordinate value after normalization. Anything beyond this
# is almost certainly a bad pose detection (joint flying off the body during
# occlusion). Clipping prevents these outliers from dominating gradient updates.
# A range of ±8 comfortably covers all plausible joint positions for any subject.
_CLIP_RANGE = 8.0


def normalize_xy(xy: np.ndarray, conf: Optional[np.ndarray] = None,
                 conf_thr: float = 0.5) -> np.ndarray:
    """
    Mid-hip centered, shoulder-width scaled normalization for COCO-17 poses,
    with per-sequence std rescaling and outlier clipping.

    Pipeline:
      1. Center each frame on its mid-hip (with carry-forward for invalid frames).
      2. Scale the whole sequence by the median shoulder width (robust to outliers).
      3. Rescale to unit per-sequence std (for cross-dataset consistency).
      4. Clip outliers to +/- _CLIP_RANGE.

    Args:
        xy: (L, 17, 2) pose sequence.
        conf: optional (L, 17) per-joint confidence scores.
        conf_thr: confidence threshold for treating a joint as valid.

    Returns:
        Normalized (L, 17, 2) array.
    """
    xy = xy.copy().astype(np.float32)
    L = xy.shape[0]

    # ----- 1) Confidence masks for hips (centering) and shoulders (scaling) -----
    if conf is not None:
        hip_valid = (conf[:, 11] >= conf_thr) & (conf[:, 12] >= conf_thr)
        sh_valid = (conf[:, 5] >= conf_thr) & (conf[:, 6] >= conf_thr)
    else:
        hip_valid = np.ones(L, dtype=bool)
        sh_valid = np.ones(L, dtype=bool)

    # ----- 2) Per-frame mid-hip centering with carry-forward -----
    # If a frame has invalid hips, reuse the most recent valid mid-hip.
    # This prevents bad detections from yanking the whole frame off-center.
    mids = np.zeros((L, 2), dtype=np.float32)
    last_valid = None
    for t in range(L):
        if hip_valid[t]:
            mids[t] = 0.5 * (xy[t, 11] + xy[t, 12])
            last_valid = mids[t].copy()
        elif last_valid is not None:
            mids[t] = last_valid
        else:
            # No valid hip yet -- fall back to whatever we have.
            mids[t] = 0.5 * (xy[t, 11] + xy[t, 12])
    xy -= mids[:, None, :]

    # ----- 3) Sequence-level scale: median shoulder width -----
    # Median is more robust than mean against outlier detections.
    if sh_valid.any():
        sh_vecs = xy[sh_valid, 6, :] - xy[sh_valid, 5, :]
        widths = np.linalg.norm(sh_vecs, axis=1)
        scale = float(np.median(widths))
    else:
        widths = np.linalg.norm(xy[:, 6, :] - xy[:, 5, :], axis=1)
        scale = float(np.median(widths))
    if scale < 1e-6:
        scale = 1.0
    xy /= scale

    # ----- 4) Final per-sequence std rescale for cross-dataset consistency -----
    # CASIA and DenseGait have different upstream pre-normalization (DenseGait
    # was already normalized by Cosma & Radoi). Rescaling to unit std here
    # aligns the two distributions so the model sees comparable inputs at
    # SSL pretraining time and fine-tuning time.
    final_std = float(np.sqrt((xy ** 2).mean() + 1e-8))
    if final_std > 1e-6:
        xy /= final_std

    # ----- 5) Clip outlier joints -----
    # After per-sequence rescaling, the typical magnitude is O(1). Anything
    # beyond +/- 8 is a bad detection, not a real pose. Clipping prevents
    # these from contributing extreme gradients during training.
    xy = np.clip(xy, -_CLIP_RANGE, _CLIP_RANGE)

    return xy