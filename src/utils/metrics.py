from typing import List
import torch

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, ks=(1,5)) -> List[float]:
    """
    Safe top-k: clamps ks to <= num_classes. Returns a list of accuracies in order of ks.
    If num_classes < 5, top-5 effectively becomes top-num_classes.
    """
    B, C = logits.shape
    if C <= 0 or B <= 0:
        return [0.0 for _ in ks]
    kmax = min(C, max(ks))
    topk_idx = logits.topk(kmax, dim=1).indices  # (B, kmax)
    out = []
    for k in ks:
        k = min(k, C)
        correct_any = topk_idx[:, :k].eq(target.view(-1,1)).any(dim=1).float().mean().item()
        out.append(correct_any)
    return out
