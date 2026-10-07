#!/usr/bin/env python3
"""
fig_datasets.py — painel com exemplos dos tres conjuntos de dados

Uma figura para o capitulo de materiais, mostrando lado a lado uma imagem de
cada conjunto com o contorno da anotacao de referencia sobreposto, e abaixo a
mascara sozinha. Serve para o leitor ver DE IMEDIATO tres coisas que o texto
afirma e que sao faceis de duvidar:

  - os tres sao BINARIOS, objeto contra fundo, e nao multiclasse;
  - o objeto ocupa fracao minima da imagem, que e por que a acuracia passa de
    0,97 em quase tudo e nao separa braco nenhum;
  - os dominios sao visualmente muito diferentes entre si, o que e o que torna
    informativo um metodo falhar nos tres.

A fracao de objeto impressa em cada painel e medida na propria mascara, nao
copiada de lugar nenhum.

Uso
    python scripts/fig_datasets.py [--out DISSERTACAO/figuras]
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

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (rotulo, diretorio de imagens, diretorio de mascaras, resolucao declarada)
CONJUNTOS = [
    ("Parasitas (ovos de Schistosoma)",
     "flim_ad/datasets/schistossoma-eggs/orig",
     "flim_ad/datasets/schistossoma-eggs/label",
     "400x400 RGB, 1220 imagens"),
    ("BraTS (tumor cerebral)",
     "data/brats/orig", "data/brats/label",
     "240x240 tom de cinza, 3753 imagens"),
    ("Conjuntivite",
     "data/conjunctiva/images", "data/conjunctiva/labels",
     "1079x863 RGB, 83 imagens"),
]

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def carrega(dir_img, dir_msk):
    """Primeiro par (imagem, mascara) com objeto nao vazio."""
    arquivos = sorted(f for f in glob.glob(os.path.join(dir_img, "*"))
                      if f.lower().endswith(EXTS))
    for f in arquivos[:60]:
        base = os.path.splitext(os.path.basename(f))[0]
        cands = [c for c in glob.glob(os.path.join(dir_msk, base + ".*"))
                 if c.lower().endswith(EXTS)]
        if not cands:
            continue
        m = np.array(Image.open(cands[0]).convert("L")) > 127
        # Precisa ter objeto: 49% do pool do schisto nao tem, e um painel com
        # mascara vazia nao ilustra nada.
        if m.sum() > 50:
            return np.array(Image.open(f).convert("RGB")), m, base
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "DISSERTACAO",
                                                  "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "figure.dpi": 160, "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

    dados = []
    for rotulo, di, dm, res in CONJUNTOS:
        img, msk, base = carrega(os.path.join(RAIZ, di),
                                 os.path.join(RAIZ, dm))
        if img is None:
            print(f"  SEM EXEMPLO: {rotulo} ({di})")
            continue
        dados.append((rotulo, res, img, msk, base))

    if not dados:
        print("nenhum conjunto encontrado")
        return 1

    fig, axs = plt.subplots(2, len(dados), figsize=(4.0 * len(dados), 8.2))
    if len(dados) == 1:
        axs = axs.reshape(2, 1)

    for j, (rotulo, res, img, msk, base) in enumerate(dados):
        frac = 100.0 * msk.mean()

        axs[0, j].imshow(img)
        axs[0, j].contour(msk, levels=[0.5], colors="#19c37d", linewidths=1.8)
        axs[0, j].set_title(f"{rotulo}\n{res}")
        axs[0, j].set_xlabel(f"{base}", fontsize=8)

        axs[1, j].imshow(msk, cmap="gray", vmin=0, vmax=1)
        axs[1, j].set_title("anotacao de referencia")
        axs[1, j].set_xlabel(f"objeto ocupa {frac:.2f}% dos pixels",
                             fontsize=9, fontweight="bold")
        print(f"  {rotulo}: {base}, objeto = {frac:.2f}% dos pixels")

        for i in (0, 1):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    fig.suptitle("Os tres conjuntos: segmentacao BINARIA, objeto pequeno, "
                 "dominios distintos", fontsize=12, fontweight="bold")
    destino = os.path.join(a.out, "datasets.png")
    fig.savefig(destino)
    plt.close(fig)
    print(f"\ngravado em {os.path.relpath(destino, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
