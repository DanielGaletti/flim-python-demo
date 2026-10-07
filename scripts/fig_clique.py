#!/usr/bin/env python3
"""
fig_clique.py — como o clique do especialista simulado e posicionado

Responde em imagem a pergunta que o texto responde em palavras: o clique NAO
vai no centroide da regiao de erro, vai no maximo da transformada de distancia
dentro da maior componente conexa de erro.

A diferenca importa e a figura mostra por que: o centroide de uma regiao
concava pode cair FORA dela, e nesse caso o clique rotularia o pixel errado.
O maximo da transformada de distancia e, por construcao, o ponto mais interior
da componente.

A regra desenhada aqui e lida de `flim_al.il_flim.proximo_clique`, que e a que
roda nos experimentos; a figura chama a propria funcao em vez de reimplementar
a logica, para nao haver chance de a ilustracao divergir do codigo medido.

Uso
    cd flim_ad && python ../scripts/fig_clique.py
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

from flim_al.il_flim import proximo_clique          # noqa: E402

ORIG = os.path.join(RAIZ, "flim_ad/datasets/schistossoma-eggs/orig")
LABEL = os.path.join(RAIZ, "flim_ad/datasets/schistossoma-eggs/label")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "DISSERTACAO",
                                                  "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Um exemplo com objeto. A predicao e simulada por erosao do gabarito, o
    # que e honesto para ILUSTRAR a geometria do clique: a figura e sobre ONDE
    # o clique cai dada uma predicao, nao sobre a qualidade de predicao nenhuma.
    alvo = None
    for f in sorted(glob.glob(os.path.join(ORIG, "*.png")))[:40]:
        base = os.path.splitext(os.path.basename(f))[0]
        lf = os.path.join(LABEL, base + ".png")
        if not os.path.isfile(lf):
            continue
        g = np.array(Image.open(lf).convert("L")) > 127
        if g.sum() > 800:
            alvo = (f, lf, base, g)
            break
    if alvo is None:
        print("nenhum exemplo com objeto")
        return 1
    fimg, _, base, gt = alvo
    img = np.array(Image.open(fimg).convert("RGB"))

    pred = ndimage.binary_erosion(gt, iterations=6, border_value=0)
    erro = gt & ~pred                      # falso negativo: falta objeto

    lab, n = ndimage.label(erro)
    areas = ndimage.sum(erro, lab, index=range(1, n + 1))
    comp = lab == (int(np.argmax(areas)) + 1)
    dt = ndimage.distance_transform_edt(comp)

    x, y, rotulo = proximo_clique(pred, gt)
    cy, cx = ndimage.center_of_mass(comp)          # o centroide, para contraste
    dentro = bool(comp[int(round(cy)), int(round(cx))])

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "figure.dpi": 160, "savefig.bbox": "tight",
        "savefig.facecolor": "white"})

    ys, xs = np.nonzero(gt)
    m = 40
    y0, y1 = max(0, ys.min() - m), min(gt.shape[0], ys.max() + m)
    x0, x1 = max(0, xs.min() - m), min(gt.shape[1], xs.max() + m)
    rec = lambda A: A[y0:y1, x0:x1]

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.3))

    ax[0].imshow(rec(img))
    ax[0].contour(rec(gt), [0.5], colors="#19c37d", linewidths=1.8)
    ax[0].contour(rec(pred), [0.5], colors="#e8543f", linewidths=1.8)
    ax[0].set_title("1. Predicao (vermelho) contra\nreferencia (verde)")

    ax[1].imshow(rec(erro), cmap="gray")
    ax[1].set_title(f"2. Erro = referencia \\ predicao\n{n} componente(s) conexa(s)")

    ax[2].imshow(rec(dt), cmap="magma")
    ax[2].set_title("3. Transformada de distancia\nna MAIOR componente")

    ax[3].imshow(rec(img))
    ax[3].contour(rec(comp), [0.5], colors="#888888", linewidths=1.0)
    ax[3].plot(x - x0, y - y0, "o", ms=13, mfc="#19c37d", mec="white", mew=2,
               label="clique: maximo da\ntransformada de distancia")
    ax[3].plot(cx - x0, cy - y0, "X", ms=13, mfc="#e8543f", mec="white", mew=2,
               label=f"centroide ({'dentro' if dentro else 'FORA'} da regiao)")
    ax[3].set_title(f"4. Onde o clique cai\nrotulo = {rotulo} (objeto)")
    ax[3].legend(loc="lower center", fontsize=7.5, framealpha=0.92)

    for e in ax:
        e.set_xticks([])
        e.set_yticks([])
    fig.suptitle("Posicionamento do clique do especialista simulado "
                 "(protocolo de Xu et al., 2016)",
                 fontsize=12, fontweight="bold")

    destino = os.path.join(a.out, "il_clique.png")
    fig.savefig(destino)
    plt.close(fig)
    print(f"exemplo {base}: {n} componente(s) de erro; "
          f"clique em ({x}, {y}) com rotulo {rotulo}")
    print(f"centroide da componente em ({cx:.0f}, {cy:.0f}), "
          f"{'DENTRO' if dentro else 'FORA'} dela")
    print(f"gravado em {os.path.relpath(destino, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
