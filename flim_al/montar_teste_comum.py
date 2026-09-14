#!/usr/bin/env python3
"""
montar_teste_comum.py
=====================
Constrói o conjunto de teste único usado por TODOS os braços da comparação.

Por que isso precisa existir. A avaliação removia do teste apenas as imagens
que os braços *daquela execução* haviam treinado. Como o T do artigo difere
entre usuários e o T do AL difere entre sementes, cada execução acabava com um
teste ligeiramente diferente — o usuário A em 366 imagens, o B em 365. Uma
diferença de Fβ entre eles passaria a misturar dois efeitos: o modelo e o
conjunto em que foi medido.

Aqui a subtração é feita uma vez, sobre a união de tudo que qualquer braço de
qualquer usuário e qualquer semente vai treinar. O arquivo resultante é passado
a todas as execuções via `--test_list`.

Uso (de dentro de flim_ad/):
    python ../flim_al/montar_teste_comum.py \\
        --selection_csv out/paper_selection_v2/selection_entropy_real.csv \\
        --seeds 0 1 2 3 4 5 6 7 8 \\
        --out out/final_comparison/teste_comum.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from flim_al.final_model_comparison import SPLITS, al_selection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection_csv", required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(range(9)))
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--users", nargs="+", default=["A", "B"])
    ap.add_argument("--criterion", default="entropy")
    ap.add_argument("--out", default="out/final_comparison/teste_comum.txt")
    a = ap.parse_args()

    test = [l.strip() for l in open(f"{SPLITS}/test.txt") if l.strip()]
    treinadas: set[str] = set()

    for u in a.users:
        for s in a.splits:
            for p in glob.glob(f"data/schisto/user_{u}/split{s}/markers/"
                               "*-seeds.txt"):
                treinadas.add(os.path.basename(p).replace("-seeds.txt", "")
                              + ".png")
    for s in a.splits:
        for sd in a.seeds:
            try:
                ids, _ = al_selection(s, a.criterion, sd, a.selection_csv)
            except SystemExit:
                print(f"  aviso: sem seleção para split {s} semente {sd}")
                continue
            treinadas |= {i + ".png" for i in ids}

    comum = [f for f in test if f not in treinadas]
    removidas = sorted(set(test) & treinadas)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(comum) + "\n")

    print(f"teste original      {len(test)}")
    print(f"imagens de treino   {len(treinadas)} (em qualquer braço)")
    print(f"removidas do teste  {len(removidas)}  {removidas}")
    print(f"teste comum         {len(comum)}")
    print(f"[salvo] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
