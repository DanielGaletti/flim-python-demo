#!/usr/bin/env python3
"""
make_figures.py
===============
Gera as figuras da apresentação/dissertação a partir dos dados reais.

    cd flim_ad && python ../flim_al/make_figures.py --out ../figs

Figuras:
  fig1_pipeline.png   — FLIM ponta a ponta: imagem → markers → saliency →
                        Otsu+filtro de área → predição vs GT
  fig2_markers.png    — o que é um marker: manual (usuário) vs sintético (GT)
  fig3_selecao.png    — o que o AL vê: mapa de entropia e as imagens mais/menos
                        incertas do pool
  fig4_regiao.png     — Region AL: superpixels SLIC → regiões incertas → seeds
  fig5_curva.png      — Fβ vs budget K, com a faixa de ruído e a linha do pool
  fig6_ruido.png      — por que a semeadura pareada importa
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"
MARKERS = "data/schisto/user_A/split1/markers"
SAL = "out/saliencies/schisto/user_A/test/split1/labeled_marker/layer_3"
RUNS = "out/al_backprop_results"

FG, BG = "#e8543f", "#2f7fd4"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


# ── utilidades ────────────────────────────────────────────────────────────────

def read_seeds(path):
    """seeds.txt do FLIM → (fg[N,2], bg[M,2]) em [col,row]."""
    fg, bg = [], []
    with open(path) as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            col, row, cls = int(p[0]), int(p[1]), int(p[3])
            (fg if cls == 1 else bg).append((col, row))
    return np.array(fg), np.array(bg)


def otsu_af(sal_u8, area_range=(1000, 9000)):
    """Pipeline oficial de pós-processamento: Otsu + filtro por área."""
    from skimage.filters import threshold_otsu
    from skimage import measure
    if sal_u8.sum() == 0:
        return np.zeros_like(sal_u8, dtype=np.uint8)
    b = (sal_u8 > threshold_otsu(sal_u8)).astype(np.uint8)
    lab = measure.label(b, background=0, connectivity=2)
    out = b.copy()
    for c in range(1, lab.max() + 1):
        a = (lab == c).sum()
        if a < area_range[0] or a > area_range[1]:
            out[lab == c] = 0
    return out


def img(name):
    return np.array(Image.open(os.path.join(ORIG, name)).convert("RGB"))


def gt(name):
    return np.array(Image.open(os.path.join(LABEL, name)).convert("L")) > 127


def sal(name):
    return np.array(Image.open(os.path.join(SAL, name)).convert("L"))


def ax_off(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc")
    if title:
        ax.set_title(title)


def contour(ax, mask, color, lw=1.4):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


# ── Figura 1: pipeline ────────────────────────────────────────────────────────

def fig_pipeline(out, name):
    im, g, s = img(name), gt(name), sal(name)
    pred = otsu_af(s)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    ax_off(axes[0], "1. Imagem de entrada")
    axes[0].imshow(im)

    ax_off(axes[1], "2. Encoder FLIM → saliency")
    axes[1].imshow(s, cmap="inferno")

    ax_off(axes[2], "3. Otsu + filtro de área")
    axes[2].imshow(pred, cmap="gray")

    ax_off(axes[3], "4. Predição vs anotação")
    axes[3].imshow(im)
    contour(axes[3], g, "#39b54a", 2.0)
    contour(axes[3], pred.astype(bool), "#e8543f", 1.6)
    axes[3].legend(handles=[Patch(color="#39b54a", label="ground truth"),
                            Patch(color="#e8543f", label="predição")],
                   loc="lower right", fontsize=7, framealpha=.9)

    tp = (pred.astype(bool) & g).sum()
    fp = (pred.astype(bool) & ~g).sum()
    fn = (~pred.astype(bool) & g).sum()
    eps = 1e-8
    pr, rc = tp / (tp + fp + eps), tp / (tp + fn + eps)
    fb = 1.3 * pr * rc / (0.3 * pr + rc + eps)
    iou = tp / (tp + fp + fn + eps)

    ax_off(axes[4], "5. Erro por pixel")
    err = np.zeros((*g.shape, 3))
    err[pred.astype(bool) & g] = [0.22, 0.71, 0.29]      # acerto
    err[pred.astype(bool) & ~g] = [0.91, 0.33, 0.25]     # falso positivo
    err[~pred.astype(bool) & g] = [0.99, 0.79, 0.20]     # falso negativo
    axes[4].imshow(err)
    axes[4].set_xlabel(f"Fβ = {fb:.3f}   IoU = {iou:.3f}", fontsize=9)
    axes[4].legend(handles=[Patch(color="#39b54a", label="acerto"),
                            Patch(color="#e8543f", label="falso positivo"),
                            Patch(color="#fdca33", label="falso negativo")],
                   loc="lower right", fontsize=7, framealpha=.9)

    fig.suptitle(f"Pipeline FLIM ponta a ponta  —  imagem {name}", y=1.03,
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(out, "fig1_pipeline.png"))
    plt.close(fig)
    print("  fig1_pipeline.png")


# ── Figura 2: markers ─────────────────────────────────────────────────────────

def fig_markers(out):
    names = sorted(f.replace("-seeds.txt", "") for f in os.listdir(MARKERS))[:3]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))

    for j, nm in enumerate(names):
        fn = f"{nm}.png"
        fg, bg = read_seeds(os.path.join(MARKERS, f"{nm}-seeds.txt"))
        ax = axes[0, j]
        ax_off(ax, f"Manual — {nm}  ({len(fg) + len(bg)} seeds)")
        ax.imshow(img(fn))
        if len(bg):
            ax.scatter(bg[:, 0], bg[:, 1], s=1.2, c=BG, alpha=.55)
        if len(fg):
            ax.scatter(fg[:, 0], fg[:, 1], s=1.2, c=FG, alpha=.85)

    from flim_al.marker_generator import generate_markers_from_gt
    pool = sorted(os.listdir(LABEL))[:80]
    picked, k = [], 0
    while len(picked) < 3 and k < len(pool):
        if gt(pool[k]).sum() > 500:
            picked.append(pool[k])
        k += 1
    for j, fn in enumerate(picked):
        d = generate_markers_from_gt(os.path.join(LABEL, fn), n_fg=100, n_bg=300)
        ax = axes[1, j]
        n = len(d["fg_seeds"]) + len(d["bg_seeds"])
        ax_off(ax, f"Sintético — {fn[:-4]}  ({n} seeds)")
        ax.imshow(img(fn))
        if len(d["bg_seeds"]):
            ax.scatter(d["bg_seeds"][:, 0], d["bg_seeds"][:, 1], s=2.5, c=BG, alpha=.55)
        if len(d["fg_seeds"]):
            ax.scatter(d["fg_seeds"][:, 0], d["fg_seeds"][:, 1], s=2.5, c=FG, alpha=.85)

    axes[0, 0].set_ylabel("anotação do usuário", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("gerado do ground truth", fontsize=11, fontweight="bold")
    fig.legend(handles=[Patch(color=FG, label="foreground (ovo)"),
                        Patch(color=BG, label="background")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.suptitle("Markers: traços contíguos do usuário vs pontos aleatórios do GT",
                 fontsize=13, fontweight="bold", y=.98)
    fig.tight_layout(rect=[0, .02, 1, .95])
    fig.savefig(os.path.join(out, "fig2_markers.png"))
    plt.close(fig)
    print("  fig2_markers.png")


# ── Figura 3: o que o AL enxerga ──────────────────────────────────────────────

def fig_selecao(out):
    paths = sorted(glob.glob(os.path.join(SAL, "*.png")))
    eps = 1e-6
    scored = []
    for p in paths[:250]:
        a = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        a = np.clip(a, eps, 1 - eps)
        ent = float((-(a * np.log(a) + (1 - a) * np.log(1 - a))).mean())
        scored.append((ent, os.path.basename(p)))
    scored.sort(reverse=True)
    top, bot = scored[:3], scored[-3:]

    fig, axes = plt.subplots(3, 3, figsize=(11, 10.6))
    ent_map = None
    for j, (e, fn) in enumerate(top):
        a = np.array(Image.open(os.path.join(SAL, fn)).convert("L"),
                     dtype=np.float32) / 255.0
        a = np.clip(a, eps, 1 - eps)
        ent_map = -(a * np.log(a) + (1 - a) * np.log(1 - a))
        ax_off(axes[0, j], f"{fn[:-4]}  —  entropia {e:.4f}")
        axes[0, j].imshow(img(fn))
        ax_off(axes[1, j])
        axes[1, j].imshow(ent_map, cmap="viridis")
    for j, (e, fn) in enumerate(bot):
        ax_off(axes[2, j], f"{fn[:-4]}  —  entropia {e:.4f}")
        axes[2, j].imshow(img(fn))

    axes[0, 0].set_ylabel("MAIS incertas\n(AL escolhe)", fontsize=10,
                          fontweight="bold", color="#c0392b")
    axes[1, 0].set_ylabel("mapa de entropia", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("MENOS incertas\n(AL descarta)", fontsize=10,
                          fontweight="bold", color="#2f7fd4")
    fig.suptitle("O sinal que guia o AL: entropia do saliency map (sem usar GT)",
                 fontsize=13, fontweight="bold", y=.98)
    fig.tight_layout(rect=[0, 0, 1, .955])
    fig.savefig(os.path.join(out, "fig3_selecao.png"))
    plt.close(fig)
    print("  fig3_selecao.png")


# ── Figura 4: Region AL ───────────────────────────────────────────────────────

def fig_regiao(out, name):
    from skimage.segmentation import slic, mark_boundaries
    im, g, s = img(name), gt(name), sal(name).astype(np.float32) / 255.0
    sp = slic(im, n_segments=300, compactness=10.0, start_label=0)

    eps = 1e-6
    sc = np.clip(s, eps, 1 - eps)
    ent = -(sc * np.log(sc) + (1 - sc) * np.log(1 - sc))
    scores = {int(r): float(ent[sp == r].mean()) for r in np.unique(sp)}
    top = sorted(scores, key=lambda r: scores[r], reverse=True)[:20]

    sel = np.isin(sp, top)
    labels, fg_pts, bg_pts = {}, [], []
    rng = np.random.default_rng(42)
    for r in top:
        m = sp == r
        is_fg = (m & g).sum() / max(m.sum(), 1) >= 0.15
        rows, cols = np.where(m)
        idx = rng.choice(len(rows), size=min(15, len(rows)), replace=False)
        (fg_pts if is_fg else bg_pts).extend(
            [(int(cols[i]), int(rows[i])) for i in idx])
        labels[r] = is_fg

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    ax_off(axes[0], f"1. SLIC — {len(np.unique(sp))} superpixels")
    axes[0].imshow(mark_boundaries(im, sp, color=(1, 1, 0), mode="thin"))

    ax_off(axes[1], "2. Entropia média por região")
    hm = np.zeros(sp.shape)
    for r, v in scores.items():
        hm[sp == r] = v
    axes[1].imshow(hm, cmap="viridis")

    ax_off(axes[2], f"3. Top-{len(top)} regiões consultadas")
    ov = im.copy()
    ov[sel] = (0.45 * ov[sel] + 0.55 * np.array([255, 220, 0])).astype(np.uint8)
    axes[2].imshow(ov)

    ax_off(axes[3], "4. Resposta do especialista → seeds")
    axes[3].imshow(im)
    if bg_pts:
        a = np.array(bg_pts); axes[3].scatter(a[:, 0], a[:, 1], s=5, c=BG)
    if fg_pts:
        a = np.array(fg_pts); axes[3].scatter(a[:, 0], a[:, 1], s=5, c=FG)
    n_fg = sum(labels.values())
    axes[3].set_xlabel(f"{n_fg} regiões 'ovo' · {len(top) - n_fg} 'fundo'", fontsize=9)

    fig.suptitle(f"Region AL: anotar regiões em vez da imagem inteira  —  {name}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(out, "fig4_regiao.png"))
    plt.close(fig)
    print("  fig4_regiao.png")


# ── Figura 5: curva de resultados ─────────────────────────────────────────────

def _runs(m):
    f = glob.glob(os.path.join(RUNS, m, "*_runs.csv"))
    return list(csv.DictReader(open(f[0]))) if f else []


def fig_curva(out):
    methods = [("coreset", "#2f7fd4"), ("badge", "#8e44ad"),
               ("entropy", "#e8543f"), ("region_entropy", "#e0a800")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))

    rand_done = False
    for m, c in methods:
        rows = _runs(m)
        if not rows:
            continue
        ks = sorted({int(r["budget"]) for r in rows if r["arm"] == "al"})
        al_mu = [st.mean([float(r["fb"]) for r in rows
                          if r["arm"] == "al" and int(r["budget"]) == k]) for k in ks]
        al_sd = [st.stdev([float(r["fb"]) for r in rows
                           if r["arm"] == "al" and int(r["budget"]) == k]) for k in ks]
        a1.errorbar(ks, al_mu, yerr=al_sd, marker="o", color=c, label=m,
                    capsize=3, lw=2)
        if not rand_done:
            rd_mu = [st.mean([float(r["fb"]) for r in rows
                              if r["arm"] == "rand" and int(r["budget"]) == k]) for k in ks]
            rd_sd = [st.stdev([float(r["fb"]) for r in rows
                               if r["arm"] == "rand" and int(r["budget"]) == k]) for k in ks]
            a1.errorbar(ks, rd_mu, yerr=rd_sd, marker="s", color="#666666",
                        label="random", capsize=3, lw=2, ls="--")
            rand_done = True
        full = [float(r["fb"]) for r in rows if r["arm"] == "full"]
        if full and m == "coreset":
            mu, sd = st.mean(full), st.stdev(full)
            a1.axhspan(mu - sd, mu + sd, color="#39b54a", alpha=.13)
            a1.axhline(mu, color="#39b54a", lw=2, ls=":",
                       label=f"pool inteiro (~190 imgs) = {mu:.3f}")

    a1.set_xlabel("Budget K (imagens anotadas)")
    a1.set_ylabel("Fβ  (β² = 0.3)")
    a1.set_title("Desempenho vs orçamento de anotação")
    a1.set_xticks([3, 5, 10])
    a1.grid(alpha=.25)
    a1.legend(fontsize=8, loc="lower right")

    # ΔFβ pareado
    for m, c in methods:
        rows = _runs(m)
        if not rows:
            continue
        ks = sorted({int(r["budget"]) for r in rows if r["arm"] == "al"})
        dm, ds = [], []
        for k in ks:
            pairs = []
            for sp_ in sorted({r["split"] for r in rows}):
                for sd_ in sorted({r["seed"] for r in rows}):
                    a = [float(r["fb"]) for r in rows if r["arm"] == "al"
                         and int(r["budget"]) == k and r["split"] == sp_ and r["seed"] == sd_]
                    d = [float(r["fb"]) for r in rows if r["arm"] == "rand"
                         and int(r["budget"]) == k and r["split"] == sp_ and r["seed"] == sd_]
                    if a and d:
                        pairs.append(a[0] - d[0])
            dm.append(st.mean(pairs))
            ds.append(st.stdev(pairs) / np.sqrt(len(pairs)))
        a2.errorbar(ks, dm, yerr=ds, marker="o", color=c, label=m, capsize=3, lw=2)

    noise = []
    for m, _ in methods[:1]:
        rows = _runs(m)
        for sp_ in sorted({r["split"] for r in rows}):
            v = [float(r["fb"]) for r in rows if r["arm"] == "full" and r["split"] == sp_]
            if len(v) > 1:
                noise.append(st.stdev(v))
    if noise:
        n = st.mean(noise)
        a2.axhspan(-n, n, color="#999999", alpha=.22,
                   label=f"piso de ruído (±{n:.3f})")
    a2.axhline(0, color="black", lw=1)
    a2.set_xlabel("Budget K (imagens anotadas)")
    a2.set_ylabel("ΔFβ pareado  (AL − Random)")
    a2.set_title("Ganho sobre seleção aleatória")
    a2.set_xticks([3, 5, 10])
    a2.grid(alpha=.25)
    a2.legend(fontsize=8)

    fig.suptitle("Experimento B — encoder FLIM fixo, decoder 1×1 por backprop "
                 "(3 splits × 5 sementes pareadas)", fontsize=12.5,
                 fontweight="bold", y=1.02)
    fig.savefig(os.path.join(out, "fig5_curva.png"))
    plt.close(fig)
    print("  fig5_curva.png")


# ── Figura 6: por que a semeadura pareada importa ─────────────────────────────

def fig_ruido(out):
    rows = _runs("coreset")
    if not rows:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    splits = sorted({r["split"] for r in rows})
    for i, sp_ in enumerate(splits):
        v = [float(r["fb"]) for r in rows if r["arm"] == "full" and r["split"] == sp_]
        a1.scatter([i] * len(v), v, s=70, color="#2f7fd4", zorder=3)
        if len(v) > 1:
            a1.errorbar(i, st.mean(v), yerr=st.stdev(v), color="#c0392b",
                        capsize=8, lw=2, zorder=2)
    a1.set_xticks(range(len(splits)))
    a1.set_xticklabels([f"Split {s}" for s in splits])
    a1.set_ylabel("Fβ")
    a1.set_title("Mesmas imagens, só muda a inicialização\n(pool inteiro, 3 sementes)")
    a1.grid(alpha=.25, axis="y")

    ks = [3, 5, 10]
    unp, par = [], []
    for k in ks:
        al = [float(r["fb"]) for r in rows if r["arm"] == "al" and int(r["budget"]) == k]
        rd = [float(r["fb"]) for r in rows if r["arm"] == "rand" and int(r["budget"]) == k]
        unp.append(st.stdev([a - b for a in al for b in rd]))
        pairs = []
        for sp_ in sorted({r["split"] for r in rows}):
            for sd_ in sorted({r["seed"] for r in rows}):
                a = [float(r["fb"]) for r in rows if r["arm"] == "al"
                     and int(r["budget"]) == k and r["split"] == sp_ and r["seed"] == sd_]
                d = [float(r["fb"]) for r in rows if r["arm"] == "rand"
                     and int(r["budget"]) == k and r["split"] == sp_ and r["seed"] == sd_]
                if a and d:
                    pairs.append(a[0] - d[0])
        par.append(st.stdev(pairs))
    x = np.arange(len(ks))
    a2.bar(x - .19, unp, .38, label="não pareado", color="#e8543f")
    a2.bar(x + .19, par, .38, label="pareado (mesma semente)", color="#2f7fd4")
    a2.set_xticks(x); a2.set_xticklabels([f"K={k}" for k in ks])
    a2.set_ylabel("desvio-padrão de ΔFβ")
    a2.set_title("Comparação pareada reduz o ruído da medida")
    a2.legend(fontsize=9); a2.grid(alpha=.25, axis="y")

    fig.suptitle("Controle do ruído de inicialização (método: CoreSet)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.savefig(os.path.join(out, "fig6_ruido.png"))
    plt.close(fig)
    print("  fig6_ruido.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figs")
    ap.add_argument("--exemplo", default="", help="imagem para figs 1 e 4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    name = a.exemplo
    if not name:
        for c in sorted(os.listdir(SAL)):
            if os.path.exists(os.path.join(LABEL, c)) and 2000 < gt(c).sum() < 9000:
                name = c
                break
    print(f"gerando figuras em {a.out} (exemplo: {name})")

    fig_pipeline(a.out, name)
    fig_markers(a.out)
    fig_selecao(a.out)
    fig_regiao(a.out, name)
    fig_curva(a.out)
    fig_ruido(a.out)
    print("pronto")


if __name__ == "__main__":
    main()
