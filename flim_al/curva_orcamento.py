#!/usr/bin/env python3
"""
curva_orcamento.py
==================
A curva que responde a pergunta da dissertação: **aplicando AL, os resultados
melhoram?**

Até aqui a resposta era "não" — mas medida só em orçamento pequeno. O braço de
AL termina com |T| ≈ 3.3 porque o backtracking do Algoritmo 1 corta cedo, e com
paciência chega a 6.6. A tendência sobe:

    |T| = 3.3  →  Fβ 0.738
    |T| = 6.6  →  Fβ 0.755
    artigo (supervisionado, 3 imagens) → Fβ 0.788

O pool tem 31 markers reais e nunca foi usado além de 7. Esta curva vai até o
fim: se ela cruzar 0.788, existe um orçamento a partir do qual um critério sem
ground truth SUPERA a seleção supervisionada — e a resposta vira sim, com o
número de imagens necessário.

Se não cruzar, o teto do pool fica medido, que também é resposta.

Ordenação por CoreSet (k-center guloso sobre as features do encoder), que foi o
critério mais confiável no teste por decoder. Determinístico dado o encoder
inicial, então uma execução por ponto basta.

Uso (de dentro de flim_ad/):
    python ../flim_al/curva_orcamento.py --split 1
    python ../flim_al/curva_orcamento.py --split all --decoders todos
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, retrain_encoder,
)
from flim_al.coreset_badge import (  # noqa: E402
    coreset_select, extract_encoder_features,
)
from flim_al.final_model_comparison import (  # noqa: E402
    DECODERS, LABEL, ORIG, real_marker_index,
)
from flim_al.paper_selection import imagem_medoide  # noqa: E402

AREA = (1000, 9000)
FIELDS = ["split", "k", "decoder", "paper_name", "block", "imagens",
          "fb", "dice", "mae", "iou", "segundos"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="1", help="1, 2, 3 ou all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ks", nargs="+", type=int,
                    default=[3, 5, 8, 12, 16, 20, 25, 31])
    ap.add_argument("--blocks", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--decoders", default="ref",
                    choices=["ref", "todos"],
                    help="'ref' usa só o labeled_marker (a curva); 'todos' "
                         "roda os sete decoders (para o ponto final)")
    ap.add_argument("--n_val", type=int, default=120)
    ap.add_argument("--test_list",
                    default="out/final_comparison_v3/teste_comum.txt")
    ap.add_argument("--out", default="out/curva/curva_orcamento.csv")
    a = ap.parse_args()

    splits = [1, 2, 3] if a.split == "all" else [int(a.split)]
    test = [l.strip() for l in open(a.test_list) if l.strip()]
    decs = DECODERS if a.decoders == "todos" else [("labeled_marker",
                                                    "FLIM_lm")]
    idx = real_marker_index()
    pool = sorted(f"{k}.png" for k in idx)
    print(f"pool de {len(pool)} markers reais · teste com {len(test)} imagens")

    work = tempfile.mkdtemp(prefix="curva_")
    try:
        for s in splits:
            arch = f"data/schisto/user_A/split{s}/arch2D.json"
            val_all = [l.strip() for l in open(
                "datasets/schistossoma-eggs/Splits-5train-70_30/"
                f"split{s}-val.txt") if l.strip()]
            treinadas = set(pool)
            val_pool = [f for f in val_all if f not in treinadas]
            rng = np.random.default_rng(0)
            val = list(rng.choice(val_pool,
                                  size=min(a.n_val, len(val_pool)),
                                  replace=False))

            # Encoder-semente: a imagem medoide, escolhida sem modelo e sem
            # ground truth. Serve só para dar features ao CoreSet — o k-center
            # precisa de um espaço onde medir distância.
            inicial = imagem_medoide(pool, ORIG)
            d0 = os.path.join(work, f"m0_s{s}")
            os.makedirs(d0, exist_ok=True)
            shutil.copy(idx[inicial[:-4]],
                        os.path.join(d0, f"{inicial[:-4]}-seeds.txt"))
            e0 = os.path.join(work, f"e0_s{s}.pth")
            retrain_encoder(arch, d0, ORIG, LABEL, a.device, e0)

            enc0 = torch.load(e0, map_location="cpu", weights_only=False)
            feats = extract_encoder_features(
                enc0, [os.path.join(ORIG, f) for f in pool], 2, "cpu")

            # Ordem CoreSet: k-center guloso a partir da medoide. Uma ordem
            # só serve todos os K — o prefixo de tamanho K é a seleção de K.
            i0 = pool.index(inicial)
            ordem = [i0] + list(coreset_select(
                feats, budget=len(pool) - 1, labeled_indices=[i0]))
            ordem_f = [pool[i] for i in ordem]
            print(f"\nsplit {s}: inicial={inicial[:-4]} · "
                  f"ordem CoreSet dos {len(ordem_f)} markers", flush=True)

            linhas = []
            for k in a.ks:
                if k > len(ordem_f):
                    continue
                sel = ordem_f[:k]
                mdir = os.path.join(work, f"m_s{s}_k{k}")
                shutil.rmtree(mdir, ignore_errors=True)
                os.makedirs(mdir, exist_ok=True)
                for f in sel:
                    shutil.copy(idx[f[:-4]],
                                os.path.join(mdir, f"{f[:-4]}-seeds.txt"))
                enc = os.path.join(work, f"e_s{s}_k{k}.pth")
                retrain_encoder(arch, mdir, ORIG, LABEL, a.device, enc)

                for dec, nome in decs:
                    t0 = time.time()
                    por_b = {}
                    for b in a.blocks:
                        try:
                            por_b[b] = evaluate_decoder(
                                enc, dec, b, val, ORIG, LABEL, a.device,
                                area_range=AREA)["fb"]
                        except Exception as e:
                            print(f"    bloco {b}: {e}", flush=True)
                    if not por_b:
                        continue
                    bb = max(por_b, key=por_b.get)
                    m = evaluate_decoder(enc, dec, bb, test, ORIG, LABEL,
                                         a.device, area_range=AREA)
                    linhas.append({
                        "split": s, "k": k, "decoder": dec,
                        "paper_name": nome, "block": bb,
                        "imagens": "|".join(x[:-4] for x in sel),
                        "fb": round(m["fb"], 4), "dice": round(m["dice"], 4),
                        "mae": round(m["mae"], 4), "iou": round(m["iou"], 4),
                        "segundos": round(time.time() - t0, 1),
                    })
                    print(f"  K={k:>3}  {nome:9s} b={bb}  "
                          f"teste Fβ={m['fb']:.4f}"
                          f"  ({time.time() - t0:.0f}s)", flush=True)

                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                novo = not os.path.exists(a.out)
                with open(a.out, "a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    if novo:
                        w.writeheader()
                    w.writerows(linhas)
                linhas = []
            print(f"  [salvo] {a.out}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
