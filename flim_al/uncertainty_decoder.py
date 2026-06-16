"""
UncertaintyDecoder
==================
Wraps any FLIM segmentation decoder (backprop_decoder, labeled_marker, etc.)
and adds a parallel uncertainty head that estimates aleatoric uncertainty.

Architecture:
    Shared backbone (FLIM encoder features)
        ├── Segmentation head  → P(y|x)          (original decoder output)
        └── Uncertainty head   → log σ²(x)        (learned per-pixel log-variance)

Loss:
    NLL = 0.5 * exp(-log_var) * BCE(pred, target) + 0.5 * log_var
    = heteroscedastic loss (Kendall & Gal, NeurIPS 2017)

Usage:
    model = UncertaintyDecoder(base_decoder, in_channels=64)
    seg, log_var = model(features)
    loss = model.nll_loss(seg, log_var, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    """Lightweight conv head: in_channels → 1 (log-variance per pixel)."""

    def __init__(self, in_channels: int, mid_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UncertaintyDecoder(nn.Module):
    """
    Wraps a FLIM decoder and adds an uncertainty head.

    Parameters
    ----------
    base_decoder : nn.Module
        Any FLIM decoder that exposes .features(x) → feature map
        OR is used as a black-box with a feature hook.
    in_channels : int
        Number of channels in the feature map fed to the uncertainty head.
    hook_layer : str | None
        If provided, registers a forward hook on base_decoder.<hook_layer>
        to capture intermediate features. If None, uses the full decoder
        output (1-channel sigmoid) and uncertainty is estimated from that.
    """

    def __init__(
        self,
        base_decoder: nn.Module,
        in_channels: int = 1,
        hook_layer: str | None = None,
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.uncertainty_head = UncertaintyHead(in_channels)
        self._features: torch.Tensor | None = None

        if hook_layer is not None:
            layer = dict(base_decoder.named_modules()).get(hook_layer)
            if layer is None:
                raise ValueError(f"Layer '{hook_layer}' not found in base_decoder")
            layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        seg : Tensor [B, 1, H, W]  — segmentation probability (sigmoid)
        log_var : Tensor [B, 1, H, W] — log aleatoric variance
        """
        seg = self.base_decoder(x)

        feat = self._features if self._features is not None else seg
        log_var = self.uncertainty_head(feat)

        self._features = None  # reset hook buffer
        return seg, log_var

    @staticmethod
    def nll_loss(
        seg: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Heteroscedastic NLL loss (Kendall & Gal 2017).
        L = 0.5 * exp(-log_var) * BCE(seg, target) + 0.5 * log_var
        """
        bce = F.binary_cross_entropy(seg.clamp(eps, 1 - eps), target, reduction="none")
        precision = torch.exp(-log_var)
        loss = 0.5 * precision * bce + 0.5 * log_var
        return loss.mean()

    @torch.no_grad()
    def predict_uncertainty(self, x: torch.Tensor):
        """
        Returns
        -------
        seg : Tensor  — segmentation map
        uncertainty : Tensor  — aleatoric std per pixel (exp(0.5 * log_var))
        """
        seg, log_var = self.forward(x)
        uncertainty = torch.exp(0.5 * log_var)
        return seg, uncertainty


# ── MC-Dropout uncertainty (epistemic) ──────────────────────────────────────

def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 20,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Epistemic uncertainty via MC-Dropout.
    Enables dropout at inference and runs n_passes forward passes.

    Returns
    -------
    mean_pred : Tensor [B, 1, H, W]
    epistemic_var : Tensor [B, 1, H, W]
    """
    model.train()  # enables dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(x.to(device))
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu())
    model.eval()

    stacked = torch.stack(preds, dim=0)          # [T, B, 1, H, W]
    mean_pred = stacked.mean(dim=0)
    epistemic_var = stacked.var(dim=0)
    return mean_pred, epistemic_var
