#!/usr/bin/env python3
"""
make_seg_figure.py
==================
As figuras de segmentação por modelo, a partir das máscaras que
`final_model_comparison.py --examples` deixou em out/final_comparison/seg/.

Convenção visual (a mesma do artigo): **contorno da ground truth em magenta**,
predição preenchida em ciano translúcido. Assim dá para ler os dois erros
separadamente — vazamento (ciano fora do magenta) e buraco (magenta sem
ciano).

Gera:
    fig8_seg_por_modelo.png    linhas = decoders, colunas = imagens · só o AL
    fig9_antes_depois.png      linhas = decoders, colunas = artigo | AL

Uso (de dentro de flim_ad/):
    python ../flim_al/make_seg_figure.py --split 1 --out ../figs
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

ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"

DECODERS = [
    ("labeled_marker",              "FLIM$_{lm}$"),
    ("decoder_2",                   "FLIM$_{pb}$"),
    ("decoder_3",                   "FLIM$_{mb}$"),
    ("hybrid_decoder",              "FLIM$_{lt}$"),
    ("vanilla_adaptive_decoder",    "FLIM$_{ts}$"),
    ("decoder_attention",           "FLIM$_{at}$"),
    ("vanilla_adaptive_decoder_wt", "FLIM$_{ts*}$"),
]

MAGENTA = (1.0, 0.0, 0.85)
CIANO = (0.10, 0.85, 0.95)
NL = chr(10)


def _contour(mask: np.ndarray, w: int = 2) -> np.ndarray:
    """Borda de uma máscara binária, por erosão com dilatação de vizinhança."""
    m = mask.astype(bool)
    e = m.copy()
    for _ in range(w):
        e &= (np.roll(e, 1, 0) & np.roll(e, -1, 0)
              & np.roll(e, 1, 1) & np.roll(e, -1, 1))
    return m & ~e


def overlay(fname: str, pred: np.ndarray | None) -> np.ndarray:
    img = np.array(Image.open(os.path.join(ORIG, fname)).convert("RGB"),
                   dtype=np.float32) / 255.0
    if pred is not None and pred.any():
        a = 0.55
        img[pred.astype(bool)] = (1 - a) * img[pred.astype(bool)] + a * np.array(CIANO)
    gt_path = os.path.join(LABEL, fname)
    if os.path.exists(gt_path):
        gt = np.array(Image.open(gt_path).convert("L")) > 0
        img[_contour(gt)] = MAGENTA
    return np.clip(img, 0, 1)


def papeis(seg_root: str, split: int) -> dict[str, str]:
    """consenso / disputa / falha, se pick_examples.py já rodou."""
    import csv
    p = os.path.join(seg_root, f"split{split}_exemplos.csv")
    if not os.path.exists(p):
        return {}
    return {r["fname"]: r["papel"] for r in csv.DictReader(open(p))}


def deltas(seg_root: str, split: int) -> dict[str, float]:
    """Fβ(AL) − Fβ(artigo) no decoder de referência, por imagem."""
    import csv
    p = os.path.join(seg_root, f"split{split}_exemplos.csv")
    if not os.path.exists(p):
        return {}
    return {r["fname"]: float(r.get("delta_lm", 0) or 0)
            for r in csv.DictReader(open(p))}


def load_pred(seg_root: str, split: int, arm: str, dec: str,
              fname: str) -> np.ndarray | None:
    p = os.path.join(seg_root, f"split{split}", arm, dec, fname)
    if not os.path.exists(p):
        return None
    return (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)


def _legend(fig):
    fig.legend(handles=[Patch(facecolor=CIANO, label="predição do modelo"),
                        Patch(facecolor=MAGENTA, label="contorno da anotação")],
               loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))


def fig_por_modelo(seg_root, split, imgs, out):
    """
    Transposta de propósito: decoders nas COLUNAS, imagens nas LINHAS.

    O arranjo natural (um decoder por linha) produz um grid retrato de 7x4 que,
    reduzido para caber num slide 16:9, fica pequeno demais para se ler. Com os
    eixos trocados o grid vira 4x7, que é largo — e é o slide que tem de mandar
    na forma da figura, não o contrário.
    """
    pap = papeis(seg_root, split)
    nl, nc = len(imgs), len(DECODERS)
    fig, ax = plt.subplots(nl, nc, figsize=(1.85 * nc, 1.95 * nl))
    ax = np.atleast_2d(ax)
    for r, fn in enumerate(imgs):
        for c, (dec, nome) in enumerate(DECODERS):
            a = ax[r, c]
            a.imshow(overlay(fn, load_pred(seg_root, split, "al", dec, fn)))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(nome, fontsize=14, pad=6)
            if c == 0:
                rot = pap.get(fn, "")
                a.set_ylabel(fn.replace(".png", "") + (NL + rot if rot else ""),
                             fontsize=11, rotation=0, ha="right", va="center",
                             labelpad=38)
    fig.suptitle("Segmentação final após active learning, por modelo "
                 f"(split {split}, teste Z2)", fontsize=15)
    _legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[salvo] {out}")


def par_antes_depois(seg_root, split, imgs):
    """
    Uma imagem que o AL melhorou e uma que ele piorou. Mostrar só os ganhos
    seria escolher a evidência.
    """
    d = deltas(seg_root, split)
    if not d:
        return imgs[:2]
    ganho = max((f for f in imgs if d.get(f, 0) > 0), key=lambda f: d[f],
                default=None)
    perda = min((f for f in imgs if d.get(f, 0) < 0), key=lambda f: d[f],
                default=None)
    par = [f for f in (ganho, perda) if f]
    return par if len(par) == 2 else sorted(
        imgs, key=lambda f: -abs(d.get(f, 0)))[:2]


def fig_antes_depois(seg_root, split, imgs, out, titulo=""):
    """
    Uma imagem por figura: duas linhas (T do artigo, T do AL) por sete
    decoders. Quatro linhas numa figura só ficariam altas demais para o slide.
    """
    d = deltas(seg_root, split)
    linhas = [(fn, arm, rot) for fn in imgs[:1]
              for arm, rot in (("paper", "T do artigo"), ("al", "T do AL"))]
    nl, nc = len(linhas), len(DECODERS)
    fig, ax = plt.subplots(nl, nc, figsize=(1.85 * nc, 1.95 * nl))
    ax = np.atleast_2d(ax)
    for r, (fn, arm, rot) in enumerate(linhas):
        for c, (dec, nome) in enumerate(DECODERS):
            a = ax[r, c]
            a.imshow(overlay(fn, load_pred(seg_root, split, arm, dec, fn)))
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(nome, fontsize=14, pad=6)
            if c == 0:
                marca = (NL + f"Fβ do lm {d.get(fn, 0):+.2f}"
                         if arm == "al" and d else "")
                a.set_ylabel(fn.replace(".png", "") + NL + rot + marca,
                             fontsize=10.5, rotation=0, ha="right",
                             va="center", labelpad=44)
    fig.suptitle(titulo or f"Antes e depois do active learning (split {split})",
                 fontsize=15)
    _legend(fig)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[salvo] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root", default="out/final_comparison/seg")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--out", default="../figs")
    a = ap.parse_args()

    ref = os.path.join(a.seg_root, f"split{a.split}", "al", "labeled_marker")
    if not os.path.isdir(ref):
        raise SystemExit(f"sem máscaras em {ref} — rode final_model_comparison "
                         "--examples antes")
    pap = papeis(a.seg_root, a.split)
    imgs = sorted((f for f in os.listdir(ref) if f.endswith(".png")),
                  key=lambda f: (["consenso", "disputa", "falha",
                                  "mudanca"].index(pap[f])
                                 if f in pap else 9, f))
    print(f"imagens de exemplo: {[(f, pap.get(f, '?')) for f in imgs]}")
    os.makedirs(a.out, exist_ok=True)
    # fig8 mostra o contraste entre decoders; as de 'mudanca' vão para a fig9
    fig8 = [f for f in imgs if pap.get(f) != "mudanca"] or imgs
    fig_por_modelo(a.seg_root, a.split, fig8,
                   os.path.join(a.out, "fig8_seg_por_modelo.png"))

    d = deltas(a.seg_root, a.split)
    par = par_antes_depois(a.seg_root, a.split, imgs)
    for fn, nome in zip(par, ("fig9a_al_melhorou.png", "fig9b_al_piorou.png")):
        verbo = "melhorou" if d.get(fn, 0) > 0 else "piorou"
        fig_antes_depois(
            a.seg_root, a.split, [fn], os.path.join(a.out, nome),
            titulo=(f"O AL {verbo} esta imagem: {fn.replace('.png', '')}, "
                    f"ΔFβ do lm = {d.get(fn, 0):+.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
