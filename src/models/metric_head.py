import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Normalize(nn.Module):
    def forward(self, x, eps=1e-6):
        return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

class ArcMarginProduct(nn.Module):
    """
    ArcFace margin.
    """
    def __init__(self, in_features, out_features, s=32.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(input, F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.clamp(cosine ** 2, 0, 1)).clamp_min(1e-6))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class MetricHead(nn.Module):
    """
    Linear BN projection + L2 + ArcFace logits.
    """
    def __init__(self, in_dim, proj_dim=256, num_classes=60, s=32.0, m=0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
        )
        self.l2 = L2Normalize()
        self.arc = ArcMarginProduct(proj_dim, num_classes, s=s, m=m)

    def forward(self, feats, labels):
        z = self.proj(feats)          # [B,proj_dim]
        z = self.l2(z)                # L2-normed
        logits = self.arc(z, labels)  # ArcFace logits
        return logits, z              # return z for kNN export
    
    def embed(self, feats):
        # projection used for retrieval (no margin, just proj+bn+l2)
        z = self.proj(feats)
        return self.l2(z)
