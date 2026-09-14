#!/usr/bin/env python3
"""
pick_examples.py
================
Escolhe as imagens de exemplo que valem um slide, e regenera as máscaras só
para elas.

O `final_model_comparison.py` pega as primeiras imagens do teste cujo ovo
sobrevive ao filtro de área. O resultado é honesto e inútil: são os casos
fáceis, e os sete decoders produzem máscaras indistinguíveis. Uma figura de
comparação em que tudo parece igual não compara nada.

Aqui a escolha é feita pelo que a figura precisa mostrar:

    consenso    todos os decoders acertam — mostra que o pipeline funciona
    disputa     o desvio de Fβ entre decoders é máximo — mostra a diferença
    falha       nem o melhor decoder resolve — mostra o limite do método

Uso (de dentro de flim_ad/):
    python ../flim_al/pick_examples.py --split 1 --device cuda:0
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
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    _official_filter_and_binarize, image_to_lab, retrain_encoder,
)
from flim_al.final_model_comparison import (  # noqa: E402
    DECODERS, LABEL, ORIG, SPLITS, al_selection, build_marker_dir,
    save_segmentations,
)
from pyflim import layers  # noqa: E402

BETA2 = 0.3


def fb(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    eps = 1e-8
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt.astype(bool)).sum())
    fn = float((~pred.astype(bool) & gt).sum())
    pr, rc = tp / (tp + fp + eps), tp / (tp + fn + eps)
    return (1 + BETA2) * pr * rc / (BETA2 * pr + rc + eps)


@torch.no_grad()
def fb_por_imagem(enc: str, decoder_type: str, block: int,
                  fnames: list[str]) -> dict[str, float]:
    model = torch.load(enc, map_location="cpu", weights_only=False)
    model.device = "cpu"
    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            model.layers[l].marker_labels = ml.to("cpu")
    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type=decoder_type,
        multi_layer=False)

    out = {}
    for fn in fnames:
        p = os.path.join(ORIG, fn)
        gtp = os.path.join(LABEL, fn)
        if not (os.path.exists(p) and os.path.exists(gtp)):
            continue
        img = Image.open(p).convert("RGB")
        h, w = img.size[1], img.size[0]
        x = torch.tensor(image_to_lab(np.array(img, dtype=np.uint8))
                         .transpose(2, 0, 1)).unsqueeze(0)
        y_hat, _ = model.forward(x, decoder_layer=[block - 1])
        pred = y_hat[0].float()
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        sal = pred.squeeze().numpy().astype(np.uint8)
        if sal.shape[0] != h or sal.shape[1] != w:
            sal = np.array(Image.fromarray(sal).resize((w, h), Image.BILINEAR),
                           dtype=np.uint8)
        binm = _official_filter_and_binarize(sal, (1000, 9000))
        gt = (np.array(Image.open(gtp).convert("L")) > 0).astype(np.uint8)
        out[fn] = fb(binm, gt)
    return out


def blocos(split: int, arm: str) -> dict[str, int]:
    """Os blocos que a validação escolheu na execução principal."""
    b = {}
    for f in glob.glob(f"out/final_comparison/per_model_s{split}.csv"):
        for r in csv.DictReader(open(f)):
            if r["arm"] == arm:
                b[r["decoder"]] = int(r["block"])
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--criterion", default="entropy")
    ap.add_argument("--n_cand", type=int, default=45)
    ap.add_argument("--seg_root", default="out/final_comparison/seg")
    a = ap.parse_args()

    bl_al = blocos(a.split, "al")
    bl_paper = blocos(a.split, "paper")
    if not bl_al:
        raise SystemExit(f"sem blocos do braço al no CSV do split {a.split} — "
                         "a execução principal ainda não terminou")

    test = [l.strip() for l in open(f"{SPLITS}/test.txt") if l.strip()]
    # candidatos: com ovo anotado, senão a comparação vira só fundo vazio
    cand = []
    for f in test:
        p = os.path.join(LABEL, f)
        if os.path.exists(p) and (np.array(Image.open(p).convert("L")) > 0).sum() > 400:
            cand.append(f)
        if len(cand) >= a.n_cand:
            break
    print(f"candidatos: {len(cand)}")

    paper_ids = sorted(os.path.basename(p).replace("-seeds.txt", "")
                       for p in glob.glob(f"data/schisto/user_A/split{a.split}"
                                          "/markers/*-seeds.txt"))
    al_ids, _ = al_selection(a.split, a.criterion)

    work = tempfile.mkdtemp(prefix="pick_")
    try:
        encs = {}
        arch = f"data/schisto/user_A/split{a.split}/arch2D.json"
        for arm, ids in (("paper", paper_ids), ("al", al_ids)):
            mdir = (f"data/schisto/user_A/split{a.split}/markers"
                    if arm == "paper"
                    else build_marker_dir(ids, os.path.join(work, "m_al")))
            e = os.path.join(work, f"enc_{arm}.pth")
            retrain_encoder(arch, mdir, ORIG, LABEL, a.device, e)
            encs[arm] = e

        # Fβ por imagem, braço AL, todos os decoders
        tab: dict[str, dict[str, float]] = {}
        for dec, nome in DECODERS:
            if dec not in bl_al:
                continue
            tab[dec] = fb_por_imagem(encs["al"], dec, bl_al[dec], cand)
            print(f"  {nome:9s} média {st.mean(tab[dec].values()):.3f}")

        ref = "labeled_marker"
        # o mesmo decoder no braço do artigo, para medir o que o AL mudou
        lm_paper = fb_por_imagem(encs["paper"], ref, bl_paper[ref], cand)

        linhas = []
        for fn in cand:
            vals = [tab[d][fn] for d in tab if fn in tab[d]]
            if len(vals) < len(tab):
                continue
            linhas.append((fn, st.mean(vals), st.pstdev(vals), tab[ref][fn],
                           tab[ref][fn] - lm_paper.get(fn, tab[ref][fn])))

        # fig8 quer contraste ENTRE decoders; fig9 quer contraste ENTRE braços.
        # São critérios diferentes, então são seleções diferentes.
        consenso = max((l for l in linhas if l[2] < 0.05), key=lambda l: l[1],
                       default=None)
        disputa = sorted(linhas, key=lambda l: -l[2])[:2]
        falha = min(linhas, key=lambda l: l[3])
        mudanca = sorted(linhas, key=lambda l: -abs(l[4]))[:2]

        escolhidas, rotulos = [], {}
        for l, tag in ([(consenso, "consenso")] if consenso else []) + \
                [(disputa[0], "disputa"), (disputa[1], "disputa"),
                 (falha, "falha"),
                 (mudanca[0], "mudanca"), (mudanca[1], "mudanca")]:
            if l and l[0] not in escolhidas:
                escolhidas.append(l[0])
                rotulos[l[0]] = tag
        print("\nescolhidas:")
        for fn in escolhidas:
            l = next(x for x in linhas if x[0] == fn)
            print(f"  {fn}  {rotulos[fn]:9s} média={l[1]:.3f} "
                  f"desvio={l[2]:.3f} lm={l[3]:.3f} Δlm(al−artigo)={l[4]:+.3f}")

        with open(f"{a.seg_root}/split{a.split}_exemplos.csv", "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fname", "papel", "fb_medio", "desvio", "fb_lm",
                        "delta_lm"])
            for fn in escolhidas:
                l = next(x for x in linhas if x[0] == fn)
                w.writerow([fn, rotulos[fn], round(l[1], 4), round(l[2], 4),
                            round(l[3], 4), round(l[4], 4)])

        # regenera as máscaras só para as escolhidas, nos dois braços
        for arm, bl in (("paper", bl_paper), ("al", bl_al)):
            for dec, _ in DECODERS:
                if dec not in bl:
                    continue
                d = os.path.join(a.seg_root, f"split{a.split}", arm, dec)
                shutil.rmtree(d, ignore_errors=True)
                save_segmentations(encs[arm], dec, bl[dec], escolhidas, d)
        print(f"\n[ok] máscaras regeneradas em {a.seg_root}/split{a.split}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
