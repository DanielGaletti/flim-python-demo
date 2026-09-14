#!/usr/bin/env python3
"""
fig_tutorial.py — a figura que ensina a anotar
==============================================
Três formas de anotar a MESMA imagem, com o Fβ que cada uma produz. Os riscos
são gerados pelo mesmo `generate_realistic_markers` usado nos experimentos,
então o que se vê é o que foi medido — não um desenho ilustrativo.

Rodar (de dentro de flim_ad/):
    python ../flim_al/fig_tutorial.py --img 000675 --out ../figs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402

ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"
VERDE = (0.12, 0.62, 0.29)
VERM = (0.91, 0.33, 0.25)

# Cada caso: título, Fβ medido, e os parâmetros do gerador que o produzem.
# Os Fβ vêm da comparação em três imagens documentada no texto.
CASOS = [
    ("CERTO\nobjeto na borda · 4 toques\nfundo longe · 9 toques", 0.665,
     dict(n_fg_dabs=4, n_bg_dabs=9, fg_depth=12, bg_min_dist=60,
          drag_prob=0.0)),
    ("ERRADO\ntoques no CENTRO do ovo", 0.000,
     dict(n_fg_dabs=4, n_bg_dabs=9, fg_depth=40, bg_min_dist=60,
          drag_prob=0.0)),
    ("ERRADO\ntraço longo arrastado", 0.000,
     dict(n_fg_dabs=40, n_bg_dabs=90, fg_depth=12, bg_min_dist=20,
          drag_prob=1.0, drag_len=60)),
]


def _disco(base, r0, c0, cor, raio=4):
    """Desenha o toque com o raio real do pincel (5 px), não 1 pixel."""
    h, w = base.shape[:2]
    for dr in range(-raio, raio + 1):
        for dc in range(-raio, raio + 1):
            if dr * dr + dc * dc <= raio * raio:
                r, c = int(r0) + dr, int(c0) + dc
                if 0 <= r < h and 0 <= c < w:
                    base[r, c] = cor


def desenha(img_id: str, kw: dict) -> tuple[np.ndarray, int]:
    base = np.array(Image.open(os.path.join(ORIG, f"{img_id}.png"))
                    .convert("RGB"), dtype=np.float32) / 255.0
    d = generate_realistic_markers(os.path.join(LABEL, f"{img_id}.png"),
                                   seed=3, **kw)
    h, w = base.shape[:2]
    for seeds, cor in ((d["bg_seeds"], VERM), (d["fg_seeds"], VERDE)):
        for c, r in seeds:
            _disco(base, r, c, cor, raio=1 if len(seeds) > 2000 else 4)
    # contorno da anotação de referência, em magenta, para orientar o olhar
    gt = np.array(Image.open(os.path.join(LABEL, f"{img_id}.png"))
                  .convert("L")) > 0
    e = gt.copy()
    for _ in range(2):
        e &= (np.roll(e, 1, 0) & np.roll(e, -1, 0)
              & np.roll(e, 1, 1) & np.roll(e, -1, 1))
    base[gt & ~e] = (1.0, 0.0, 0.85)
    return np.clip(base, 0, 1), len(d["fg_seeds"]) + len(d["bg_seeds"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="000675")
    ap.add_argument("--out", default="../figs")
    a = ap.parse_args()

    # recorte em volta do ovo, para a diferença borda-vs-centro ficar visível
    gt = np.array(Image.open(os.path.join(LABEL, f"{a.img}.png"))
                  .convert("L")) > 0
    ys, xs = np.nonzero(gt)
    m = 55
    r0, r1 = max(0, ys.min() - m), ys.max() + m
    c0, c1 = max(0, xs.min() - m), xs.max() + m

    fig, ax = plt.subplots(2, 3, figsize=(11.4, 7.6),
                           gridspec_kw={"height_ratios": [1, 1]})
    for i, (titulo, fb, kw) in enumerate(CASOS):
        arr, n = desenha(a.img, kw)
        cor = "#1e9e4a" if fb > 0.3 else "#e8543f"
        for linha, im in ((0, arr), (1, arr[r0:r1, c0:c1])):
            eixo = ax[linha, i]
            eixo.imshow(im)
            eixo.set_xticks([]); eixo.set_yticks([])
            for sp in eixo.spines.values():
                sp.set_edgecolor(cor); sp.set_linewidth(2.5)
        ax[0, i].set_title(titulo, fontsize=11.5, color=cor,
                           fontweight="bold", pad=8)
        ax[1, i].set_xlabel(f"{n} px marcados  ·  Fβ = {fb:.3f}", fontsize=11.5,
                            color=cor, fontweight="bold", labelpad=6)
    ax[0, 0].set_ylabel("imagem inteira", fontsize=10.5, color="#5b6678")
    ax[1, 0].set_ylabel("ampliado no ovo", fontsize=10.5, color="#5b6678")

    fig.legend(handles=[Patch(facecolor=VERDE, label="toque de objeto"),
                        Patch(facecolor=VERM, label="toque de fundo"),
                        Patch(facecolor=(1, 0, 0.85), label="onde o ovo está")],
               loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Mesma imagem, mesma quantidade de tinta — só muda ONDE",
                 fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "tutorial_anotacao.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"[salvo] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
