#!/usr/bin/env python3
"""
orcamento_seeds.py
==================
A pergunta que a dissertação precisa responder para justificar o título.

O experimento de seleção de imagens mostrou que o critério sem ground truth
empata com a seleção supervisionada do artigo — bom, mas não é "melhoria do
FLIM". A decomposição do Experimento A, porém, mediu algo acionável: **a
geometria dos seeds pesa 2.1× a seleção das imagens**. Se onde o especialista
desenha importa mais que qual imagem ele abre, então o eixo de melhoria não é
QUAL, é ONDE — e o orçamento deveria ser contado em pinceladas, não em imagens.

Daí a pergunta:

    Dado um orçamento fixo de B pinceladas, o especialista deve anotar
    POUCAS imagens com muito traço, ou MUITAS imagens com pouco traço?

O artigo responde implicitamente "3 a 5 imagens, anotadas à vontade" — mas
nunca mede o custo em pinceladas, então nunca compara alocações a esforço
constante. É exatamente a comparação que falta.

Desenho: orçamento × alocação × critério de escolha da imagem.

    orçamento    12, 24, 48, 96 pinceladas no total
    alocação     2 imagens (densa) · 4 (média) · 8 (esparsa)
    critério     aleatório (controle) vs entropia (AL, sem ground truth)

Fβ medido no teste comum, com o decoder de referência (`labeled_marker`).
Um decoder só, de propósito: o eixo em estudo é a anotação, e sete decoders
multiplicariam o custo por sete sem responder a pergunta.

Uso (de dentro de flim_ad/):
    python ../flim_al/orcamento_seeds.py --split 1
    python ../flim_al/orcamento_seeds.py --split all --out out/orcamento.csv
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
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, load_input, retrain_encoder,
)
from flim_al.final_model_comparison import LABEL, ORIG, SPLITS  # noqa: E402
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from pyflim import layers  # noqa: E402

AREA = (1000, 9000)
DECODER = "labeled_marker"
BLOCO = 2

# Proporção entre pinceladas de objeto e de fundo, medida nos 31 markers reais
# (~4 toques de objeto para ~9 de fundo). Mantida fixa para que a variável em
# estudo seja o ORÇAMENTO, não a razão objeto/fundo.
RAZAO_BG = 9 / 4

FIELDS = ["split", "orcamento", "alocacao", "n_imgs", "dabs_por_img",
          "criterio", "imagens", "n_seeds", "fb", "dice", "mae", "iou",
          "segundos"]


def imagens_com_ovo(candidatas: list[str], minimo: int = 800) -> list[str]:
    """
    Só entram imagens com ovo anotado. Não é conveniência: 49% do pool do
    Schisto não tem objeto, e um encoder treinado só com fundo colapsa — a
    predição vira tudo-fundo e o Fβ trava em 0.4788. Misturar esse regime ao
    experimento mediria a patologia, não a alocação de esforço.
    """
    out = []
    for f in candidatas:
        lp = os.path.join(LABEL, f)
        if os.path.exists(lp) and \
                (np.array(Image.open(lp).convert("L")) > 0).sum() >= minimo:
            out.append(f)
    return out


@torch.no_grad()
def ranquear_por_entropia(enc: str, cands: list[str]) -> list[str]:
    """Ordena candidatas pela entropia média do saliency — sem ground truth."""
    model = torch.load(enc, map_location="cpu", weights_only=False)
    model.device = "cpu"
    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            model.layers[l].marker_labels = ml.to("cpu")
    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type=DECODER, multi_layer=False)

    eps, escores = 1e-6, []
    for f in cands:
        x, _, _ = load_input(os.path.join(ORIG, f))
        y_hat, _ = model.forward(x, decoder_layer=[BLOCO - 1])
        pred = y_hat[0].float()
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        p = np.clip(pred.squeeze().numpy() / 255.0, eps, 1 - eps)
        escores.append((float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean()), f))
    escores.sort(reverse=True)
    return [f for _, f in escores]


def montar_markers(imgs: list[str], dabs: int, dest: str, seed: int) -> int:
    """Desenha `dabs` pinceladas de objeto (e RAZAO_BG× de fundo) por imagem."""
    os.makedirs(dest, exist_ok=True)
    total = 0
    for i, f in enumerate(imgs):
        d = generate_realistic_markers(
            os.path.join(LABEL, f), n_fg_dabs=dabs,
            n_bg_dabs=max(1, int(round(dabs * RAZAO_BG))), seed=seed + i)
        total += len(d["fg_seeds"]) + len(d["bg_seeds"])
        save_markers(d, os.path.join(dest, f"{f[:-4]}-seeds.txt"))
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="1", help="1, 2, 3 ou all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--orcamentos", nargs="+", type=int,
                    default=[12, 24, 48, 96])
    ap.add_argument("--alocacoes", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_cand", type=int, default=40)
    ap.add_argument("--test_list",
                    default="out/final_comparison_v3/teste_comum.txt")
    ap.add_argument("--n_test", type=int, default=0,
                    help="subamostra do teste; 0 = completo. As condições "
                         "compartilham o MESMO subconjunto, então o erro de "
                         "amostragem é comum e cancela nas diferenças entre "
                         "elas — que é o que este experimento mede.")
    ap.add_argument("--out", default="out/orcamento/orcamento_seeds.csv")
    a = ap.parse_args()

    splits = [1, 2, 3] if a.split == "all" else [int(a.split)]
    test = [l.strip() for l in open(a.test_list) if l.strip()]
    test_set = set(test)
    if a.n_test and a.n_test < len(test):
        test = sorted(np.random.default_rng(7).choice(
            test, size=a.n_test, replace=False).tolist())
    print(f"teste: {len(test)} imagens (pool de {len(test_set)})")

    work = tempfile.mkdtemp(prefix="orc_")
    try:
        for s in splits:
            arch = f"data/schisto/user_A/split{s}/arch2D.json"
            val = [l.strip() for l in open(f"{SPLITS}/split{s}-val.txt")
                   if l.strip()]
            # candidatas: fora do teste, com ovo
            cands = imagens_com_ovo(
                [f for f in val if f not in test_set])[:a.n_cand]
            print(f"\nsplit {s}: {len(cands)} candidatas com ovo")
            if len(cands) < max(a.alocacoes):
                print("  candidatas insuficientes — pulando")
                continue

            # Encoder-semente para o critério de entropia: treinado com UMA
            # imagem sorteada. É o que existe na primeira rodada de um AL real
            # — o critério não pode depender de um modelo que ainda não há.
            base_dir = os.path.join(work, f"base_s{s}")
            montar_markers([cands[0]], 8, base_dir, seed=99)
            enc_base = os.path.join(work, f"base_s{s}.pth")
            retrain_encoder(arch, base_dir, ORIG, LABEL, a.device, enc_base)
            ordem_ent = ranquear_por_entropia(enc_base, cands[1:])

            linhas = []
            for orc in a.orcamentos:
                for n_img in a.alocacoes:
                    dabs = orc // n_img
                    if dabs < 1:
                        continue
                    for criterio in ("aleatorio", "entropia"):
                        for sd in a.seeds:
                            if criterio == "entropia":
                                escolhidas = ordem_ent[:n_img]
                            else:
                                rng = np.random.default_rng(1000 + sd)
                                escolhidas = list(rng.choice(
                                    cands[1:], size=n_img, replace=False))
                            mdir = os.path.join(work, f"m_{s}_{orc}_{n_img}_"
                                                      f"{criterio}_{sd}")
                            shutil.rmtree(mdir, ignore_errors=True)
                            n_seeds = montar_markers(escolhidas, dabs, mdir,
                                                     seed=sd * 17)
                            enc = os.path.join(work, "e.pth")
                            t0 = time.time()
                            retrain_encoder(arch, mdir, ORIG, LABEL, a.device,
                                            enc)
                            m = evaluate_decoder(enc, DECODER, BLOCO, test,
                                                 ORIG, LABEL, a.device,
                                                 area_range=AREA)
                            linhas.append({
                                "split": s, "orcamento": orc,
                                "alocacao": f"{n_img}img x {dabs}dab",
                                "n_imgs": n_img, "dabs_por_img": dabs,
                                "criterio": criterio,
                                "imagens": "|".join(x[:-4] for x in escolhidas),
                                "n_seeds": n_seeds,
                                "fb": round(m["fb"], 4),
                                "dice": round(m["dice"], 4),
                                "mae": round(m["mae"], 4),
                                "iou": round(m["iou"], 4),
                                "segundos": round(time.time() - t0, 1),
                            })
                            print(f"  B={orc:>3} {n_img}x{dabs:<3} "
                                  f"{criterio:<9} sd{sd}  "
                                  f"seeds={n_seeds:<5} Fβ={m['fb']:.4f}",
                                  flush=True)
                            # o critério de entropia é determinístico: uma
                            # semente basta, as outras repetiriam a linha
                            if criterio == "entropia":
                                break

            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            novo = not os.path.exists(a.out)
            with open(a.out, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if novo:
                    w.writeheader()
                w.writerows(linhas)
            print(f"  [salvo] {a.out}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
