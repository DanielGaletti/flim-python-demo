#!/usr/bin/env python3
"""
fig_gerador.py — superpixel contra contorno previsto, lado a lado

A figura do gargalo e da intervencao. Mostra, na mesma imagem e com o mesmo
orcamento:

  - os candidatos que o SLIC produz, e quantos deles ATRAVESSAM a fronteira
    verdadeira do objeto;
  - os candidatos que o gerador de contorno produz, e a mesma contagem.

E esse contraste que sustenta o argumento central do capitulo de resultados: o
criterio apenas ORDENA a lista que recebe, entao se quase nenhum candidato da
lista cruza a fronteira, nenhuma escolha de escore resolve. As arestas do SLIC
seguem as bordas da imagem por construcao, e e justamente por isso que seus
pedacos voltam a ficar de um lado so.

As duas contagens sao medidas aqui, na imagem desenhada, e impressas na
figura. Nao sao os 2% a 12% do texto, que vem da campanha inteira: este painel
ilustra UMA imagem e deve ser lido assim.

Os candidatos de contorno saem de `flim_al.region_al.candidatos_de_contorno`,
a mesma funcao dos experimentos.

Uso
    python scripts/fig_gerador.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al.region_al import candidatos_de_contorno      # noqa: E402

ORIG = os.path.join(RAIZ, "flim_ad/datasets/schistossoma-eggs/orig")
LABEL = os.path.join(RAIZ, "flim_ad/datasets/schistossoma-eggs/label")


def atravessa(m, gt):
    """O candidato tem pixel dos dois lados da fronteira verdadeira?"""
    dentro = (m & gt).sum()
    fora = (m & ~gt).sum()
    return dentro > 0 and fora > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "DISSERTACAO",
                                                  "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from skimage.segmentation import slic

    alvo = None
    for f in sorted(glob.glob(os.path.join(ORIG, "*.png")))[:40]:
        base = os.path.splitext(os.path.basename(f))[0]
        lf = os.path.join(LABEL, base + ".png")
        if not os.path.isfile(lf):
            continue
        g = np.array(Image.open(lf).convert("L")) > 127
        if g.sum() > 800:
            alvo = (f, base, g)
            break
    if alvo is None:
        print("nenhum exemplo com objeto")
        return 1
    fimg, base, gt = alvo
    img = np.array(Image.open(fimg).convert("RGB"))

    # SLIC sobre a imagem, como no experimento
    seg = slic(img, n_segments=150, compactness=10, start_label=0)
    sps = [seg == i for i in np.unique(seg)]
    sp_cruza = [m for m in sps if atravessa(m, gt)]

    # Predicao aproximada para o gerador de contorno. Aqui ela e o proprio
    # gabarito levemente erodido: a figura e sobre a GEOMETRIA dos candidatos,
    # e usar o gabarito evita que um encoder ruim vire o assunto do painel.
    pred = ndimage.binary_erosion(gt, iterations=2, border_value=0)
    cands = candidatos_de_contorno(pred.astype(float), raio=6, semente=0)
    ct = list(cands.values())
    ct_cruza = [m for m in ct if atravessa(m, gt)]

    pct_sp = 100.0 * len(sp_cruza) / max(len(sps), 1)
    pct_ct = 100.0 * len(ct_cruza) / max(len(ct), 1)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "figure.dpi": 160, "savefig.bbox": "tight",
        "savefig.facecolor": "white"})

    ys, xs = np.nonzero(gt)
    m_ = 70
    y0, y1 = max(0, ys.min() - m_), min(gt.shape[0], ys.max() + m_)
    x0, x1 = max(0, xs.min() - m_), min(gt.shape[1], xs.max() + m_)
    rec = lambda A: A[y0:y1, x0:x1]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8))

    ax[0].imshow(rec(img))
    ax[0].contour(rec(gt), [0.5], colors="#19c37d", linewidths=2.0)
    ax[0].set_title("Imagem e fronteira verdadeira")

    ax[1].imshow(rec(img))
    for mm in sps:
        ax[1].contour(rec(mm), [0.5], colors="#f2c744", linewidths=0.45)
    for mm in sp_cruza:
        ax[1].contour(rec(mm), [0.5], colors="#e8543f", linewidths=1.9)
    ax[1].contour(rec(gt), [0.5], colors="#19c37d", linewidths=2.0)
    ax[1].set_title(f"Candidatos por SLIC: {len(sps)}\n"
                    f"atravessam a fronteira: {len(sp_cruza)} "
                    f"({pct_sp:.0f}%)")

    ax[2].imshow(rec(img))
    for mm in ct:
        ax[2].contour(rec(mm), [0.5], colors="#5b8def", linewidths=0.9)
    for mm in ct_cruza:
        ax[2].contour(rec(mm), [0.5], colors="#e8543f", linewidths=1.9)
    ax[2].contour(rec(gt), [0.5], colors="#19c37d", linewidths=2.0)
    ax[2].set_title(f"Candidatos por contorno previsto: {len(ct)}\n"
                    f"atravessam a fronteira: {len(ct_cruza)} "
                    f"({pct_ct:.0f}%)  [predicao ideal]")

    for e in ax:
        e.set_xticks([])
        e.set_yticks([])
    fig.suptitle("O gargalo esta na LISTA, nao no escore: "
                 "um criterio so ordena o que recebe",
                 fontsize=12, fontweight="bold")
    # A ressalva vai DENTRO da figura, nao so na legenda do LaTeX: o painel da
    # direita usa predicao ideal, e com a predicao real do encoder os discos
    # caem no interior, porque o modelo sub-segmenta e o contorno previsto fica
    # por dentro do verdadeiro. Ver `limitacao_confirmada_do_metodo` na campanha
    # do gerador.
    fig.text(0.5, -0.03,
             "O painel da direita usa predicao ideal e mostra a GEOMETRIA do "
             "gerador. Com a predicao real do encoder os discos caem no "
             "interior,\nporque o modelo sub-segmenta e o contorno previsto "
             "fica por dentro do verdadeiro. O ganho medido acontece apesar "
             "disso.",
             ha="center", fontsize=8.5, style="italic")

    destino = os.path.join(a.out, "gerador_candidatos.png")
    fig.savefig(destino)
    plt.close(fig)
    print(f"exemplo {base}")
    print(f"  SLIC    : {len(sp_cruza)}/{len(sps)} atravessam ({pct_sp:.1f}%)")
    print(f"  contorno: {len(ct_cruza)}/{len(ct)} atravessam ({pct_ct:.1f}%)")
    print(f"gravado em {os.path.relpath(destino, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
