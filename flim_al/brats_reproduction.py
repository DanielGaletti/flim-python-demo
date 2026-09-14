#!/usr/bin/env python3
"""
brats_reproduction.py
=====================
A reprodução mais limpa que este repositório consegue produzir.

Motivo: pela Tabela I do artigo, o pós-processamento do BraTS é
**Otsu + filtro de área [100, 20000], sem Dynamic Trees** — ao contrário do
Parasites, onde os modelos FLIM recebem DT. Como o binário
`iftSMansoniDelineation` não roda nesta imagem, a reprodução do Parasites
carrega um Δ de −0.080 de origem conhecida mas não removível. No BraTS não há
etapa faltando: os números saem diretamente comparáveis às colunas BraTS das
Tabelas III e IV.

Duas outras diferenças que importam para a dissertação:

    imagens     3753, todas com marker real desenhado por um usuário — no
                Schisto há 31 markers reais para 1220 imagens, e o AL sobre o
                pool completo precisou de markers sintéticos.
    foreground  100% do val tem objeto anotado — no Schisto 49% não tem, e é
                dessa patologia que vem a crítica ao passo 7 do Algoritmo 1.

Ou seja: o BraTS testa se as conclusões sobrevivem quando as duas maiores
fraquezas do experimento principal desaparecem.

Uso (de dentro de flim_ad/):
    python ../flim_al/brats_reproduction.py --split 1
    python ../flim_al/brats_reproduction.py --split all --n_test 0   # teste inteiro
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, retrain_encoder,
)
from flim_al.final_model_comparison import DECODERS, al_selection  # noqa: E402

BR = str(REPO / "data" / "brats")
ORIG = os.path.join(BR, "orig")
LABEL = os.path.join(BR, "label")
MARKERS = os.path.join(BR, "markers")
SPLITS = os.path.join(BR, "Splits-50_50")
ARCH = str(REPO / "arch_best_brats.json")
AREA = (100, 20000)              # Tabela I do artigo, linha BraTS

FIELDS = ["dataset", "split", "arm", "decoder", "paper_name", "block",
          "n_train", "train_ids", "fb", "dice", "mae", "iou", "fb_val"]

# Tabela IV do artigo, coluna BraTS, usuário A (teste Z2)
PAPER_TEST_A = {
    "FLIM_lm": (0.018, 0.678), "FLIM_pb": (0.019, 0.711),
    "FLIM_mb": (0.021, 0.712), "FLIM_lt": (0.017, 0.731),
    "FLIM_ts": (0.017, 0.720), "FLIM_at": (0.021, 0.688),
}


def ler(path: str) -> list[str]:
    """Os arquivos de split do BraTS têm terminação CRLF."""
    return [l.strip() for l in open(path) if l.strip()]


def marker_dir_para(img_ids: list[str], dest: str) -> str:
    """
    Os 3753 markers moram num diretório só; o retrain_encoder treina com TUDO
    que encontra no diretório que recebe. Então isola-se os de T numa cópia.
    """
    os.makedirs(dest, exist_ok=True)
    faltando = []
    for i in img_ids:
        src = os.path.join(MARKERS, f"{i}-seeds.txt")
        if not os.path.exists(src):
            faltando.append(i)
            continue
        shutil.copy(src, os.path.join(dest, f"{i}-seeds.txt"))
    if faltando:
        raise SystemExit(f"markers ausentes: {faltando}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="1", help="1, 2, 3 ou all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n_val", type=int, default=120)
    ap.add_argument("--n_test", type=int, default=600,
                    help="0 = conjunto de teste inteiro (1877 imagens)")
    ap.add_argument("--arms", nargs="+", default=["paper"],
                    choices=["paper", "al"])
    ap.add_argument("--al_seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--al_csv",
                    default="out/paper_selection_brats/selection_entropy_"
                            "real.csv")
    ap.add_argument("--out", default="out/brats/repro.csv")
    a = ap.parse_args()

    splits = [1, 2, 3] if a.split == "all" else [int(a.split)]
    work = tempfile.mkdtemp(prefix="brats_")
    try:
        for s in splits:
            train = [f.replace(".png", "") for f in
                     ler(os.path.join(SPLITS, f"split{s}-train.txt"))]
            val_all = ler(os.path.join(SPLITS, f"split{s}-val.txt"))
            test_all = ler(os.path.join(SPLITS, f"split{s}-test.txt"))

            rng = np.random.default_rng(0)
            val = list(rng.choice(val_all, size=min(a.n_val, len(val_all)),
                                  replace=False))
            test = (test_all if a.n_test == 0 else
                    list(np.random.default_rng(1).choice(
                        test_all, size=min(a.n_test, len(test_all)),
                        replace=False)))

            # um braço por semente do Algoritmo 1, pelo mesmo motivo do
            # Schisto: escolher a melhor semente daria ao AL uma vantagem que o
            # braço do artigo não tem
            tarefas = []
            if "paper" in a.arms:
                tarefas.append(("paper", train))
            if "al" in a.arms:
                for sd in a.al_seeds:
                    ids, fbv = al_selection(s, "entropy", sd, a.al_csv)
                    print(f"split {s} · semente {sd}: |T|={len(ids)} "
                          f"(Fβ val = {fbv:.4f})")
                    tarefas.append((f"al_seed{sd}", ids))

            for arm, ids in tarefas:
                print(f"\n=== BraTS split {s} · {arm} · |T|={len(ids)} · "
                      f"val={len(val)} · teste={len(test)} ===", flush=True)
                linhas = _rodar(arm, ids, s, val, test, a, work)
                _salvar(a.out, linhas)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def _salvar(out: str, linhas: list[dict]) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    novo = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if novo:
            w.writeheader()
        w.writerows(linhas)
    print(f"  [salvo] {out}", flush=True)


def _rodar(arm, ids, s, val, test, a, work) -> list[dict]:
    mdir = marker_dir_para(ids, os.path.join(work, f"m_s{s}_{arm}"))
    enc = os.path.join(work, f"enc_s{s}_{arm}.pth")
    retrain_encoder(ARCH, mdir, ORIG, LABEL, a.device, enc)

    linhas = []
    for dec, nome in DECODERS:
        t0 = time.time()
        por_bloco = {}
        for b in a.blocks:
            try:
                por_bloco[b] = evaluate_decoder(
                    enc, dec, b, val, ORIG, LABEL, a.device,
                    area_range=AREA)["fb"]
            except Exception as e:
                print(f"    bloco {b} falhou: {e}", flush=True)
        if not por_bloco:
            continue
        bb = max(por_bloco, key=por_bloco.get)
        m = evaluate_decoder(enc, dec, bb, test, ORIG, LABEL, a.device,
                             area_range=AREA)
        linhas.append({
            "dataset": "brats", "split": s, "arm": arm,
            "decoder": dec, "paper_name": nome, "block": bb,
            "n_train": len(ids), "train_ids": "|".join(ids),
            "fb": round(m["fb"], 4), "dice": round(m["dice"], 4),
            "mae": round(m["mae"], 4), "iou": round(m["iou"], 4),
            "fb_val": round(por_bloco[bb], 4),
        })
        # a comparação com o artigo só faz sentido no braço do artigo: o
        # braço de AL treina com outras imagens
        ref = PAPER_TEST_A.get(nome) if arm == "paper" else None
        comp = (f"  artigo={ref[1]:.3f} Δ={m['fb'] - ref[1]:+.3f}" if ref
                else "")
        print(f"  {nome:9s} b={bb}  val={por_bloco[bb]:.3f}  "
              f"teste Fβ={m['fb']:.3f}{comp}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return linhas


if __name__ == "__main__":
    sys.exit(main())
