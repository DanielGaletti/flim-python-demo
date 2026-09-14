#!/usr/bin/env python3
"""
analyze_real_markers.py
=======================
Caracteriza os markers REAIS (desenhados pelos usuários do paper) para
calibrar o gerador sintético com dado, não com palpite.

O gerador atual sorteia 100 pontos de foreground e 300 de background de forma
uniforme sobre a máscara. O k-means do FLIM aprende filtros a partir dos
patches na VIZINHANÇA de cada seed, então a estrutura espacial importa: um
traço contíguo cruzando a borda carrega gradiente; um ponto isolado no meio
do objeto carrega textura homogênea.

Mede, por imagem anotada:
    n_fg, n_bg                 quantos seeds de cada classe
    vizinho mais próximo       mediana da distância ao seed mais próximo
                               (≈1 px ⇒ traço contíguo; alto ⇒ pontos soltos)
    componentes conexas        quantos traços distintos
    distância à borda do GT    onde o usuário desenha em relação ao objeto

Uso:
    cd flim_ad && python ../flim_al/analyze_real_markers.py
"""
from __future__ import annotations

import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from flim_al.paper_selection import real_marker_index  # noqa: E402

LABEL = "datasets/schistossoma-eggs/label"


def read_seeds(path):
    fg, bg = [], []
    with open(path) as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            (fg if int(p[3]) == 1 else bg).append((int(p[0]), int(p[1])))
    return np.array(fg), np.array(bg)


def nn_median(pts: np.ndarray) -> float:
    """Mediana da distância ao vizinho mais próximo dentro do conjunto."""
    if len(pts) < 2:
        return float("nan")
    sub = pts if len(pts) <= 1500 else pts[
        np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def n_components(pts: np.ndarray, shape, radius: int = 2) -> int:
    """Traços distintos: componentes conexas do conjunto de seeds dilatado."""
    from skimage import measure, morphology
    if len(pts) == 0:
        return 0
    m = np.zeros(shape, dtype=bool)
    m[np.clip(pts[:, 1], 0, shape[0] - 1), np.clip(pts[:, 0], 0, shape[1] - 1)] = True
    m = morphology.binary_dilation(m, morphology.disk(radius))
    return int(measure.label(m, connectivity=2).max())


def dist_to_boundary(pts: np.ndarray, gt: np.ndarray) -> float:
    """Mediana da distância |assinada| de cada seed à borda do objeto."""
    from scipy.ndimage import distance_transform_edt
    if len(pts) == 0 or gt.sum() == 0:
        return float("nan")
    din = distance_transform_edt(gt)
    dout = distance_transform_edt(~gt)
    signed = np.where(gt, din, -dout)
    v = signed[np.clip(pts[:, 1], 0, gt.shape[0] - 1),
               np.clip(pts[:, 0], 0, gt.shape[1] - 1)]
    return float(np.median(v))


def main():  # noqa: D103
    idx = real_marker_index(REPO)
    print(f"markers reais encontrados: {len(idx)}\n")
    print(f"{'imagem':<10}{'n_fg':>7}{'n_bg':>7}{'nn_fg':>8}{'nn_bg':>8}"
          f"{'comp_fg':>9}{'comp_bg':>9}{'d_borda_fg':>12}{'d_borda_bg':>12}")
    print("-" * 82)

    acc = {k: [] for k in ("n_fg", "n_bg", "nn_fg", "nn_bg",
                           "cf", "cb", "dbf", "dbb")}
    for img_id in sorted(idx):
        lp = os.path.join(LABEL, f"{img_id}.png")
        if not os.path.exists(lp):
            continue
        gt = np.array(Image.open(lp).convert("L")) > 127
        fg, bg = read_seeds(idx[img_id])
        nnf, nnb = nn_median(fg), nn_median(bg)
        cf, cb = n_components(fg, gt.shape), n_components(bg, gt.shape)
        dbf, dbb = dist_to_boundary(fg, gt), dist_to_boundary(bg, gt)
        print(f"{img_id:<10}{len(fg):>7}{len(bg):>7}{nnf:>8.2f}{nnb:>8.2f}"
              f"{cf:>9}{cb:>9}{dbf:>12.1f}{dbb:>12.1f}")
        for k, v in zip(acc, (len(fg), len(bg), nnf, nnb, cf, cb, dbf, dbb)):
            if v == v:
                acc[k].append(v)

    print("-" * 82)
    med = {k: st.median(v) if v else float("nan") for k, v in acc.items()}
    print(f"{'MEDIANA':<10}{med['n_fg']:>7.0f}{med['n_bg']:>7.0f}"
          f"{med['nn_fg']:>8.2f}{med['nn_bg']:>8.2f}{med['cf']:>9.0f}"
          f"{med['cb']:>9.0f}{med['dbf']:>12.1f}{med['dbb']:>12.1f}")

    print("\n=== leitura ===")
    print(f"  vizinho mais próximo ~{med['nn_fg']:.1f} px (fg) e "
          f"{med['nn_bg']:.1f} px (bg)")
    if med["nn_fg"] < 2.0:
        print("    → seeds CONTÍGUOS: o usuário desenha traços, não pontos.")
    print(f"  {med['cf']:.0f} traços de fg e {med['cb']:.0f} de bg por imagem")
    print(f"  fg fica ~{med['dbf']:.1f} px DENTRO da borda; "
          f"bg ~{abs(med['dbb']):.1f} px FORA")

    print("\n=== o gerador atual, para comparação ===")
    from flim_al.marker_generator import generate_markers_from_gt
    ex = sorted(idx)[0]
    d = generate_markers_from_gt(os.path.join(LABEL, f"{ex}.png"),
                                 n_fg=100, n_bg=300)
    gt = np.array(Image.open(os.path.join(LABEL, f"{ex}.png")).convert("L")) > 127
    print(f"  {len(d['fg_seeds'])} fg + {len(d['bg_seeds'])} bg  |  "
          f"nn_fg={nn_median(d['fg_seeds']):.2f}  "
          f"nn_bg={nn_median(d['bg_seeds']):.2f}  |  "
          f"comp_fg={n_components(d['fg_seeds'], gt.shape)}  "
          f"comp_bg={n_components(d['bg_seeds'], gt.shape)}")


if __name__ == "__main__":
    main()
