#!/usr/bin/env python3
"""
medir_vazamento.py
==================
Quanto o vazamento de 000675/000917 mexe nos números já calculados?

Dois dos 31 markers reais do Schisto estão em `test.txt`. `000917` entra no T
do artigo do split 1; `000675` entra no T do artigo do split 1 **e** na semente
1 dos três splits. Alguns braços, portanto, foram avaliados em imagens que
treinaram, e outros não.

A correção é analítica e exata, porque a métrica é uma média simples por
imagem:

    Fβ(sem as vazadas) = (n·Fβ(com) − Σ Fβ das vazadas) / (n − k)

Basta medir o Fβ das imagens vazadas em cada braço — 2 imagens por braço, não
366. O custo é de minutos, contra as 4 horas de reexecutar tudo.

Uso (de dentro de flim_ad/):
    python ../flim_al/medir_vazamento.py --user A
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import statistics as st
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import retrain_encoder  # noqa: E402
from flim_al.final_model_comparison import (  # noqa: E402
    DECODERS, LABEL, ORIG, SPLITS, al_selection, build_marker_dir,
)
from flim_al.pick_examples import fb_por_imagem  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="A")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--criterion", default="entropy")
    ap.add_argument("--selection_csv", default="")
    ap.add_argument("--csv_glob",
                    default="out/final_comparison/per_model_u{u}_s*.csv")
    a = ap.parse_args()

    test = [l.strip() for l in open(f"{SPLITS}/test.txt") if l.strip()]
    n_test = len(test)
    test_set = set(test)

    linhas = []
    for f in sorted(glob.glob(a.csv_glob.format(u=a.user))):
        linhas += list(csv.DictReader(open(f)))
    if not linhas:
        raise SystemExit(f"sem CSVs em {a.csv_glob.format(u=a.user)}")

    # quais braços tocaram imagens do teste, e quais
    vazadas: dict[tuple[str, str], list[str]] = {}
    for r in linhas:
        ids = r["train_ids"].split("|")
        v = [i for i in ids if f"{i}.png" in test_set]
        if v:
            vazadas[(r["split"], r["arm"])] = v
    if not vazadas:
        print("nenhum braço treinou em imagem do teste — nada a corrigir")
        return 0

    print(f"teste com {n_test} imagens\n")
    print("braços que treinaram em imagem do teste:")
    for (s, arm), v in sorted(vazadas.items()):
        print(f"  split {s} · {arm:10s} → {v}")
    print()

    work = tempfile.mkdtemp(prefix="vaz_")
    correcoes: dict[str, list[float]] = defaultdict(list)
    try:
        for (s, arm), v in sorted(vazadas.items()):
            split = int(s)
            arch = f"data/schisto/user_{a.user}/split{split}/arch2D.json"
            if arm == "paper":
                mdir = f"data/schisto/user_{a.user}/split{split}/markers"
            else:
                sd = int(arm.replace("al_seed", ""))
                ids, _ = al_selection(split, a.criterion, sd,
                                      a.selection_csv or None)
                mdir = build_marker_dir(ids, os.path.join(work,
                                                          f"m_{s}_{arm}"))
            enc = os.path.join(work, f"enc_{s}_{arm}.pth")
            retrain_encoder(arch, mdir, ORIG, LABEL, a.device, enc)

            blocos = {r["decoder"]: int(r["block"]) for r in linhas
                      if r["split"] == s and r["arm"] == arm}
            alvo = [f"{i}.png" for i in v]
            for dec, nome in DECODERS:
                if dec not in blocos:
                    continue
                fbs = fb_por_imagem(enc, dec, blocos[dec], alvo)
                soma = sum(fbs.values())
                antes = next(float(r["fb"]) for r in linhas
                             if r["split"] == s and r["arm"] == arm
                             and r["decoder"] == dec)
                depois = (n_test * antes - soma) / (n_test - len(alvo))
                correcoes[nome].append(depois - antes)
                print(f"  split {s} · {arm:10s} · {nome:9s} "
                      f"Fβ das vazadas={[round(x, 3) for x in fbs.values()]}  "
                      f"{antes:.4f} → {depois:.4f}  ({depois - antes:+.4f})")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nimpacto por modelo (média das correções aplicadas):")
    pior = 0.0
    for nome, cs in correcoes.items():
        m = st.mean(cs)
        pior = max(pior, max(abs(x) for x in cs))
        print(f"  {nome:9s} {m:+.5f}  (máx |Δ| = {max(abs(x) for x in cs):.5f})")
    print(f"\nmaior correção individual: {pior:.5f} de Fβ")
    print("Como a margem de equivalência é δ = 0.02, uma correção abaixo de "
          "0.002 não muda nenhuma conclusão; acima disso, reexecutar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
