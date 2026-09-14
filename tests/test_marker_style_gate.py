"""
Teste-portão: o gerador de markers realistas evita o colapso do encoder?

A fase F4 falhou porque, com markers pontilhados, o encoder colapsava, o
saliency map ficava uniforme e a entropia de TODO o pool zerava — sem variação
nenhuma, o critério de seleção não tem o que rankear e o experimento vira ruído.

Este teste treina dois encoders com AS MESMAS imagens, mudando só o estilo do
marker, e compara:

    Fβ no val            desempenho
    entropia do pool     se ~0, o saliency é uniforme → critério cego
    saliências vazias    fração de mapas totalmente pretos

Se o estilo novo não melhorar a entropia, o problema não eram os markers e não
adianta reexecutar a F4.

Rodar (de dentro de flim_ad/):
    python ../tests/test_marker_style_gate.py --device cuda:0
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, generate_pool_saliencies, retrain_encoder,
)
from flim_al.marker_generator import generate_markers_from_gt, save_markers  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402

ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"


def pool_entropy_stats(sal_dir, fnames):
    eps = 1e-6
    ents, empty = [], 0
    for fn in fnames:
        p = os.path.join(sal_dir, fn)
        if not os.path.exists(p):
            continue
        a = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        if a.max() == 0:
            empty += 1
        a = np.clip(a, eps, 1 - eps)
        ents.append(float((-(a * np.log(a) + (1 - a) * np.log(1 - a))).mean()))
    if not ents:
        return 0.0, 0.0, 1.0
    return float(np.mean(ents)), float(np.std(ents)), empty / len(ents)


def build(style, images, mdir, seed=42):
    os.makedirs(mdir, exist_ok=True)
    for img_id in images:
        gt = os.path.join(LABEL, f"{img_id}.png")
        d = (generate_realistic_markers(gt, seed=seed) if style == "realistic"
             else generate_markers_from_gt(gt, n_fg=100, n_bg=300, seed=seed))
        save_markers(d, os.path.join(mdir, f"{img_id}-seeds.txt"))
    return mdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--n_pool", type=int, default=60)
    a = ap.parse_args()

    arch = f"data/schisto/user_A/split{a.split}/arch2D.json"
    if not os.path.exists(arch):
        print(f"SKIP: {arch} não encontrado — rode de dentro de flim_ad/")
        return 0

    with open(f"datasets/schistossoma-eggs/Splits-5train-70_30/split{a.split}-val.txt") as fh:
        val = [l.strip() for l in fh if l.strip()]

    # imagens de treino: as primeiras com foreground suficiente, fora do val
    val_set = set(val)
    cand = [f for f in sorted(os.listdir(ORIG))
            if f.endswith(".png") and f not in val_set]
    images = []
    for f in cand:
        lp = os.path.join(LABEL, f)
        if not os.path.exists(lp):
            continue
        if (np.array(Image.open(lp).convert("L")) > 127).sum() > 1500:
            images.append(f[:-4])
        if len(images) >= a.n_images:
            break

    rng = np.random.default_rng(0)
    val_s = list(rng.choice(val, size=min(a.n_val, len(val)), replace=False))
    pool_s = [f for f in cand if f not in set(i + ".png" for i in images)][:a.n_pool]

    print(f"treino: {images}")
    print(f"val: {len(val_s)} | pool p/ entropia: {len(pool_s)}\n")
    print(f"{'estilo':<12}{'Fβ':>9}{'entropia média':>17}{'desvio':>10}{'vazias':>9}")
    print("-" * 58)

    work = tempfile.mkdtemp(prefix="gate_")
    res = {}
    try:
        for style in ("points", "realistic"):
            mdir = build(style, images, os.path.join(work, f"{style}_markers"))
            enc = os.path.join(work, f"{style}.pth")
            retrain_encoder(arch, mdir, ORIG, LABEL, a.device, enc)
            m = evaluate_decoder(enc, "labeled_marker", 3, val_s, ORIG, LABEL, a.device)
            sal = os.path.join(work, f"{style}_sal")
            generate_pool_saliencies(enc, pool_s, ORIG, a.device, sal,
                                     "labeled_marker", 3)
            e_mu, e_sd, frac = pool_entropy_stats(sal, pool_s)
            res[style] = (m["fb"], e_mu, e_sd, frac)
            print(f"{style:<12}{m['fb']:>9.4f}{e_mu:>17.5f}{e_sd:>10.5f}{frac:>8.0%}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    p_fb, p_e, p_sd, p_fr = res["points"]
    r_fb, r_e, r_sd, r_fr = res["realistic"]
    print()
    print(f"  ΔFβ            {r_fb - p_fb:+.4f}")
    print(f"  Δentropia      {r_e - p_e:+.5f}  (desvio {p_sd:.5f} → {r_sd:.5f})")
    print(f"  Δsaliências vazias {r_fr - p_fr:+.0%}")
    print()
    # Dois critérios independentes. O primeiro decide se o EXPERIMENTO volta a
    # ser interpretável; o segundo, se o encoder não piorou no caminho. Um
    # gerador que devolve entropia mas derruba o Fβ resolve metade do problema
    # e cria outra — por isso os dois são reportados separadamente.
    sinal = r_sd > max(p_sd * 2, 1e-4) or r_e > max(p_e * 2, 1e-4)
    nao_piorou = r_fb >= p_fb - 0.02

    print(f"  sinal no saliency  {'✓' if sinal else '✗'}  "
          f"(desvio da entropia {p_sd:.5f} → {r_sd:.5f})")
    print(f"  Fβ preservado      {'✓' if nao_piorou else '✗'}  "
          f"({p_fb:.4f} → {r_fb:.4f})")
    print()
    if sinal and nao_piorou:
        print("  ✓ PASSOU nos dois critérios.")
        return 0
    if sinal:
        print("  ~ PARCIAL: o saliency voltou a variar, mas o Fβ caiu.")
        print("    A F4 fica interpretável; o Experimento A, porém, herda um")
        print("    encoder pior. Verificar em mais splits antes de concluir —")
        print("    uma configuração só não distingue efeito de ruído.")
        return 2
    print("  ✗ REPROVOU: a entropia do pool segue achatada.")
    print("    O colapso não vinha (só) do estilo do marker.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
