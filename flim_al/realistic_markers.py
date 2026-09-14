#!/usr/bin/env python3
"""
realistic_markers.py
====================
Gerador de markers sintéticos que imita a anotação real do usuário.

Por que
-------
O gerador antigo (`marker_generator.generate_markers_from_gt`) sorteia 100
pontos de foreground e 300 de background uniformemente sobre a máscara. O
k-means do FLIM estima os filtros a partir dos patches na VIZINHANÇA de cada
seed: um ponto isolado no meio do objeto só carrega textura homogênea, ao
passo que uma pincelada carrega gradiente e contexto espacial. Foi essa
diferença que degradou o encoder no Experimento A e travou a fase F4, onde os
saliency maps ficaram uniformes e a entropia de todo o pool zerou.

Calibração
----------
Medido nos 31 arquivos de markers reais (`analyze_real_markers.py`):

    componentes conexas    81 px cada, bbox 11×11  → disco de raio 5
    n_fg                   mediana 335  ≈ 4 discos × 81
    n_bg                   mediana 723  ≈ 9 discos × 81
    vizinho mais próximo   1.0 px       → contíguo, não pontilhado
    profundidade fg        ~12 px DENTRO da borda do objeto
    distância bg           ~121 px FORA do objeto

Parte dos traços de background é arrastada (componentes de até 4629 px), então
o gerador suporta pinceladas em linha além de toques isolados.

Uso:
    from flim_al.realistic_markers import generate_realistic_markers
    seeds = generate_realistic_markers("label/000002.png")
    save_markers(seeds, "out/000002-seeds.txt")
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Valores medidos nos markers reais — ver docstring
BRUSH_RADIUS = 5      # disco de 81 px
N_FG_DABS = 4
N_BG_DABS = 9
FG_DEPTH = 12         # px dentro da borda
BG_MIN_DIST = 20      # px fora da borda (o disco não pode encostar no objeto)
DRAG_PROB = 0.25      # fração das pinceladas de bg que são arrastadas
DRAG_LEN = 40         # comprimento do arrasto, em px


def _disk(radius: int) -> np.ndarray:
    """Offsets (dr, dc) do disco discreto — 81 px para raio 5."""
    r = int(radius)
    dr, dc = np.mgrid[-r:r + 1, -r:r + 1]
    m = (dr ** 2 + dc ** 2) <= r ** 2 + r * 0.5
    return np.stack([dr[m], dc[m]], axis=1)


def _paint(centers, shape, radius: int) -> np.ndarray:
    """Pinta discos nos centros e devolve os pixels marcados como [col, row]."""
    off = _disk(radius)
    H, W = shape
    pts = set()
    for cy, cx in centers:
        rr = np.clip(cy + off[:, 0], 0, H - 1)
        cc = np.clip(cx + off[:, 1], 0, W - 1)
        pts.update(zip(rr.tolist(), cc.tolist()))
    if not pts:
        return np.empty((0, 2), dtype=int)
    arr = np.array(sorted(pts))
    return np.stack([arr[:, 1], arr[:, 0]], axis=1)   # [col, row]


def _drag(center, shape, length: int, rng) -> list:
    """Arrasto: centros ao longo de uma direção aleatória, passo de 1 px."""
    ang = rng.uniform(0, 2 * np.pi)
    dy, dx = np.sin(ang), np.cos(ang)
    H, W = shape
    out = []
    for t in range(0, int(length)):
        y = int(round(center[0] + dy * t))
        x = int(round(center[1] + dx * t))
        if not (0 <= y < H and 0 <= x < W):
            break
        out.append((y, x))
    return out or [tuple(center)]


def generate_realistic_markers(
    gt_path: str,
    n_fg_dabs: int = N_FG_DABS,
    n_bg_dabs: int = N_BG_DABS,
    radius: int = BRUSH_RADIUS,
    fg_depth: int = FG_DEPTH,
    bg_min_dist: int = BG_MIN_DIST,
    drag_prob: float = DRAG_PROB,
    drag_len: int = DRAG_LEN,
    seed: int = 42,
) -> dict:
    """
    Devolve o mesmo dict de `marker_generator.generate_markers_from_gt`
    (`fg_seeds`, `bg_seeds`, `H`, `W`), compatível com `save_markers`.

    Foreground: `n_fg_dabs` toques a ~`fg_depth` px dentro da borda.
    Background: `n_bg_dabs` toques a ≥`bg_min_dist` px do objeto, dos quais
    uma fração `drag_prob` é arrastada por `drag_len` px.
    """
    from scipy.ndimage import distance_transform_edt

    rng = np.random.default_rng(seed)
    gt = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8) > 127
    H, W = gt.shape

    # ── foreground: toques a fg_depth px dentro da borda ─────────────────────
    fg_seeds = np.empty((0, 2), dtype=int)
    if gt.any():
        din = distance_transform_edt(gt)
        # candidatos: profundidade próxima do alvo, sem estourar o objeto
        target = min(fg_depth, max(1.0, din.max() - 1))
        band = (din >= max(1.0, target - 3)) & (din <= target + 3)
        if not band.any():                      # objeto fino: usa o mais fundo
            band = din >= max(1.0, din.max() * 0.6)
        ys, xs = np.where(band)
        if len(ys):
            k = min(n_fg_dabs, len(ys))
            pick = rng.choice(len(ys), size=k, replace=False)
            fg_seeds = _paint([(ys[i], xs[i]) for i in pick], gt.shape, radius)

    # ── background: toques longe do objeto, alguns arrastados ────────────────
    bg_seeds = np.empty((0, 2), dtype=int)
    if (~gt).any():
        dout = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, 1e3)
        far = dout >= bg_min_dist
        if not far.any():
            far = ~gt
        ys, xs = np.where(far)
        if len(ys):
            k = min(n_bg_dabs, len(ys))
            pick = rng.choice(len(ys), size=k, replace=False)
            centers = []
            for i in pick:
                c = (ys[i], xs[i])
                if rng.random() < drag_prob:
                    centers.extend(_drag(c, gt.shape, drag_len, rng))
                else:
                    centers.append(c)
            bg_seeds = _paint(centers, gt.shape, radius)
            # o arrasto não pode invadir o objeto
            if gt.any() and len(bg_seeds):
                keep = ~gt[bg_seeds[:, 1], bg_seeds[:, 0]]
                bg_seeds = bg_seeds[keep]

    return {"fg_seeds": fg_seeds, "bg_seeds": bg_seeds, "H": H, "W": W}


def generate_and_save_realistic(image_name: str, gt_folder: str,
                                out_marker_folder: str, seed: int = 42,
                                **kw) -> str:
    """Wrapper: gera e grava o seeds.txt. Devolve o caminho."""
    import os
    from flim_al.marker_generator import save_markers
    gt_path = os.path.join(gt_folder, f"{image_name}.png")
    out_path = os.path.join(out_marker_folder, f"{image_name}-seeds.txt")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)
    save_markers(generate_realistic_markers(gt_path, seed=seed, **kw), out_path)
    return out_path
