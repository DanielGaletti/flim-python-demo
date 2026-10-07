#!/usr/bin/env python3
"""
fig_dominios.py — painel dos dominios levantados, usados e nao usados

A secao de levantamento do Capitulo 3 cita seis conjuntos e explica por que
apenas tres entraram nos experimentos: o que determina o comportamento dos
metodos avaliados nao e a modalidade de aquisicao, e sim a FRACAO DA IMAGEM
OCUPADA PELO OBJETO, que controla a razao entre candidatos a filtro e canais
disponiveis. Esta figura mostra os seis lado a lado para o leitor ver a
diversidade de modalidade e, ao mesmo tempo, ler a fracao de objeto de cada
um.

A fracao impressa e medida na mascara do EXEMPLO mostrado, nao no conjunto
inteiro. A legenda no LaTeX precisa dizer isso: um exemplo nao e estatistica
de dataset, e apresenta-lo como se fosse seria exatamente o tipo de salto que
o resto do trabalho evita.

As imagens vem de `DISSERTACAO/figuras/nao_usadas/`, onde ficaram guardadas
desde o levantamento da qualificacao.

Uso
    python scripts/fig_dominios.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDADAS = os.path.join(RAIZ, "DISSERTACAO", "figuras", "nao_usadas")
sys.stdout.reconfigure(encoding="utf-8")

# (rotulo, arquivo da imagem, arquivo da mascara, desafio que o texto atribui)
DOMINIOS = [
    ("Kvasir-SEG", "example_kvsair.jpg", "example_kvsair_mask_1.png",
     "bordas irregulares de polipos,\nartefatos de iluminacao"),
    ("ISIC 2018", "example_isic_1.jpg", "example_isic_mask_1.png",
     "cor e textura variaveis,\nsombras e pelos"),
    ("BraTS", "example_brats_1.jpg", "example_brats_mask_1.png",
     "volumetrico em fatias 2D,\ncanal monocromatico"),
    ("LiTS", "example_lits_1.jpg", "example_lits_mask_1.png",
     "contraste baixo\nentre tecidos"),
    ("BUSI", "example_busi_1.jpg", "example_busi_mask_1.png",
     "ruido caracteristico\nda ultrassonografia"),
]

# O REFUGE2 e citado no levantamento e NAO entra nesta figura. O par
# `example_refuge_1.jpg` / `example_refuge_mask_1.png` guardado no repositorio
# nao corresponde: o contorno da mascara cai no canto inferior esquerdo e o
# disco optico da imagem esta em cima, e a fracao medida sai em 98,5%, isto e,
# a mascara esta invertida alem de deslocada. Mostrar essa sobreposicao seria
# publicar uma ilustracao errada. Para reincluir o dominio, e preciso obter do
# conjunto original um par imagem/mascara que case.

USADOS = {"BraTS"}        # dos seis levantados, so o BraTS entrou nos experimentos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "DISSERTACAO",
                                                  "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "figure.dpi": 165, "savefig.bbox": "tight",
        "savefig.facecolor": "white"})

    n = len(DOMINIOS)
    cols = 3
    linhas = (n + cols - 1) // cols
    fig, axs = plt.subplots(linhas, cols, figsize=(4.0 * cols, 4.2 * linhas))
    axs = np.atleast_1d(axs).ravel()
    for sobra in axs[n:]:
        sobra.axis("off")

    faltando = []
    for k, (rotulo, fi, fm, desafio) in enumerate(DOMINIOS):
        pi, pm = os.path.join(GUARDADAS, fi), os.path.join(GUARDADAS, fm)
        ax = axs[k]
        if not (os.path.isfile(pi) and os.path.isfile(pm)):
            faltando.append(rotulo)
            ax.axis("off")
            continue

        img = np.array(Image.open(pi).convert("RGB"))
        msk = np.array(Image.open(pm).convert("L").resize(
            (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
        frac = 100.0 * msk.mean()

        ax.imshow(img)
        ax.contour(msk, [0.5], colors="#19c37d", linewidths=2.0)
        marca = "  [usado]" if rotulo in USADOS else ""
        ax.set_title(f"{rotulo}{marca}\n{desafio}", fontsize=9)
        ax.set_xlabel(f"objeto: {frac:.1f}% dos pixels neste exemplo",
                      fontsize=8.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  {rotulo:12s} objeto = {frac:5.1f}% (neste exemplo)")

    fig.suptitle("Dominios levantados para a investigacao",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.02,
             "A fracao de objeto e medida na mascara DESTE exemplo, nao no "
             "conjunto inteiro. Dos dominios levantados, apenas o BraTS "
             "entrou nos experimentos,\nao lado de parasitas e conjuntivite, "
             "porque os tres cobrem a fracao de objeto em faixas bem "
             "separadas.",
             ha="center", fontsize=8.5, style="italic")

    destino = os.path.join(a.out, "dominios_levantados.png")
    fig.savefig(destino)
    plt.close(fig)
    if faltando:
        print(f"  SEM EXEMPLO: {', '.join(faltando)}")
    print(f"gravado em {os.path.relpath(destino, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
