"""
inference.py — Inferência: roda modelo treinado em imagem nova
==============================================================
Carrega encoder treinado → gera saliency map → aplica Otsu+AF ou DT →
retorna máscara binária + overlay colorido como base64 PNG.
"""
from __future__ import annotations

import base64
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import REPO_ROOT, FLIMPY_PATH, DT_BIN, DT_AVAILABLE

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLIMPY_PATH))


def run_inference(
    image_path: str | Path,
    encoder_path: str | Path,
    gt_path: str | Path | None = None,
    use_dt: bool = False,
    overlay_color: tuple[int, int, int] = (255, 50, 50),
    overlay_alpha: float = 0.45,
) -> dict:
    """
    Roda inferência em uma imagem.

    Parâmetros
    ----------
    image_path   : caminho para imagem PNG de entrada
    encoder_path : caminho para encoder .pth treinado
    gt_path      : (opcional) máscara GT para calcular Fβ/IoU
    use_dt       : se True, usa DT pós-processamento (precisa do binário)
    overlay_color: RGB da sobreposição de segmentação
    overlay_alpha: opacidade da sobreposição

    Retorna
    -------
    dict com:
        overlay_b64 : PNG como base64 (imagem + sobreposição vermelha)
        mask_b64    : máscara binária como base64 PNG
        fb          : Fβ (None se sem GT)
        iou         : IoU (None se sem GT)
        has_gt      : bool
    """
    # TODO: implementar usando:
    #   from flim_al.al_encoder_experiment import image_to_lab, evaluate_decoder
    #   + _official_filter_and_binarize para Otsu+AF
    #   + _run_dt para DT (se use_dt=True e DT_AVAILABLE)

    raise NotImplementedError("TODO: implementar inference.run_inference()")


def _apply_overlay(
    orig_img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Sobrepõe máscara binária na imagem original com cor semitransparente.
    Retorna array RGB uint8.
    """
    out = orig_img.copy().astype(np.float32)
    fg  = mask.astype(bool)

    for c, cv in enumerate(color):
        out[fg, c] = out[fg, c] * (1 - alpha) + cv * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def _to_b64(arr: np.ndarray, mode: str = "RGB") -> str:
    """Converte array numpy para base64 PNG."""
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _fb_iou(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> tuple[float, float]:
    """Calcula Fβ e IoU entre predição e GT binárias."""
    eps   = 1e-8
    tp    = float(np.sum((pred == 1) & (gt == 1)))
    fp    = float(np.sum((pred == 1) & (gt == 0)))
    fn    = float(np.sum((pred == 0) & (gt == 1)))
    pr    = tp / (tp + fp + eps)
    rc    = tp / (tp + fn + eps)
    fb    = (1 + beta2) * pr * rc / (beta2 * pr + rc + eps)
    iou   = tp / (tp + fp + fn + eps) if (tp + fp + fn) > 0 else 1.0
    return round(fb, 4), round(iou, 4)
