"""
Acquisition Functions for Active Learning in FLIM
==================================================
All functions receive a list of image tensors (or a DataLoader) and
return a score per image. Higher score = more informative = select next.

Functions:
    entropy_map(pred)         → spatial entropy map
    entropy_score(pred)       → scalar score per image (mean entropy)
    bald_score(preds)         → BALD (Bayesian Active Learning by Disagreement)
    least_confidence(pred)    → 1 - max(p, 1-p) per pixel, averaged
    rank_pool(model, pool)    → rank unlabeled pool by entropy
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Sequence


def entropy_map(pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pixel-wise binary entropy of a segmentation probability map.

    Parameters
    ----------
    pred : Tensor [B, 1, H, W]  — sigmoid probability

    Returns
    -------
    ent : Tensor [B, 1, H, W]
    """
    p = pred.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def entropy_score(pred: torch.Tensor) -> torch.Tensor:
    """Mean entropy per image. Shape: [B]"""
    return entropy_map(pred).mean(dim=[1, 2, 3])


def least_confidence(pred: torch.Tensor) -> torch.Tensor:
    """
    Least confidence: 1 - |p - 0.5| * 2.
    Highest when p≈0.5 (most uncertain).
    Shape: [B]
    """
    return (1 - (pred - 0.5).abs() * 2).mean(dim=[1, 2, 3])


def bald_score(
    preds: Sequence[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    BALD (Houlsby et al. 2011) — measures information gain.
    Requires multiple stochastic forward passes (MC-Dropout or ensemble).

    BALD = H(E[p]) - E[H(p)]
         = entropy of mean prediction - mean entropy of individual predictions

    Parameters
    ----------
    preds : list of Tensor [B, 1, H, W], len = T (n_passes)

    Returns
    -------
    score : Tensor [B]
    """
    stacked = torch.stack(preds, dim=0)           # [T, B, 1, H, W]
    mean_p = stacked.mean(dim=0)                   # [B, 1, H, W]

    h_mean = entropy_map(mean_p).mean(dim=[1, 2, 3])          # H(E[p])
    mean_h = torch.stack([
        entropy_map(p).mean(dim=[1, 2, 3]) for p in preds
    ]).mean(dim=0)                                             # E[H(p)]

    return h_mean - mean_h                         # BALD score [B]


@torch.no_grad()
def rank_pool(
    model: torch.nn.Module,
    pool: list[torch.Tensor],
    device: str = "cuda",
    method: str = "entropy",
    mc_passes: int = 10,
    batch_size: int = 4,
) -> list[int]:
    """
    Score and rank all images in the unlabeled pool.

    Parameters
    ----------
    model    : FLIM decoder (UncertaintyDecoder or plain decoder)
    pool     : list of image tensors [1, C, H, W]
    method   : 'entropy' | 'least_confidence' | 'bald'
    mc_passes: T for BALD (ignored for other methods)

    Returns
    -------
    indices sorted by score descending (most informative first)
    """
    model.eval()
    scores = []

    for i in range(0, len(pool), batch_size):
        batch = torch.cat(pool[i : i + batch_size], dim=0).to(device)

        if method == "bald":
            model.train()
            mc_preds = []
            with torch.no_grad():
                for _ in range(mc_passes):
                    out = model(batch)
                    p = out[0] if isinstance(out, tuple) else out
                    mc_preds.append(p.cpu())
            model.eval()
            s = bald_score(mc_preds)
        else:
            with torch.no_grad():
                out = model(batch)
                pred = out[0] if isinstance(out, tuple) else out
                pred = pred.cpu()
            if method == "least_confidence":
                s = least_confidence(pred)
            else:
                s = entropy_score(pred)

        scores.extend(s.tolist())

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked, scores
