"""
Quantas pinceladas o gerador precisa?

O sweep anterior mostrou que markers realistas (4 toques de fg, 9 de bg)
devolvem variação ao saliency map mas derrubam o Fβ em 0.11 a 0.32 — pior em
6 de 6 configurações. Hipótese: contiguidade e DIVERSIDADE DE AMOSTRAGEM são
eixos distintos. O gerador antigo amostra 100 lugares distintos (ruim em
estrutura, bom em cobertura); o novo amostra 4 (bom em estrutura, ruim em
cobertura). O k-means estima os filtros a partir dos patches ao redor de cada
seed, então poucos lugares ⇒ poucos protótipos distintos.

Se a hipótese valer, aumentar o número de pinceladas — mantendo cada uma
contígua — recupera o Fβ sem perder o sinal.

Rodar (de dentro de flim_ad/):
    python ../tests/test_dab_count_sweep.py --device cuda:0
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

sys.path.insert(0, str(REPO / "tests"))
from test_marker_style_gate import pool_entropy_stats  # noqa: E402

ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--n_val", type=int, default=120)
    ap.add_argument("--n_pool", type=int, default=40)
    ap.add_argument("--dabs", nargs="+", type=int, default=[4, 12, 25, 50])
    a = ap.parse_args()

    print(f"{'config':<22}{'Fβ':>9}{'entropia':>11}{'desvio':>10}{'n_seeds':>9}")
    print("-" * 62)

    for split in a.splits:
        arch = f"data/schisto/user_A/split{split}/arch2D.json"
        if not os.path.exists(arch):
            print(f"SKIP split {split}")
            continue
        with open(f"datasets/schistossoma-eggs/Splits-5train-70_30/"
                  f"split{split}-val.txt") as fh:
            val = [l.strip() for l in fh if l.strip()]
        val_set = set(val)
        cand = [f for f in sorted(os.listdir(ORIG))
                if f.endswith(".png") and f not in val_set]
        images = []
        for f in cand:
            lp = os.path.join(LABEL, f)
            if os.path.exists(lp) and \
                    (np.array(Image.open(lp).convert("L")) > 127).sum() > 1500:
                images.append(f[:-4])
            if len(images) >= a.n_images:
                break
        rng = np.random.default_rng(0)
        val_s = list(rng.choice(val, size=min(a.n_val, len(val)), replace=False))
        pool_s = [f for f in cand
                  if f not in set(i + ".png" for i in images)][:a.n_pool]

        variants = [("points", None)] + [("realistic", d) for d in a.dabs]
        for style, dabs in variants:
            work = tempfile.mkdtemp(prefix="dab_")
            try:
                mdir = os.path.join(work, "m")
                os.makedirs(mdir)
                total = 0
                for img_id in images:
                    gt = os.path.join(LABEL, f"{img_id}.png")
                    if style == "points":
                        d = generate_markers_from_gt(gt, n_fg=100, n_bg=300)
                    else:
                        d = generate_realistic_markers(
                            gt, n_fg_dabs=dabs,
                            n_bg_dabs=max(2, int(dabs * 2.25)))
                    total += len(d["fg_seeds"]) + len(d["bg_seeds"])
                    save_markers(d, os.path.join(mdir, f"{img_id}-seeds.txt"))
                enc = os.path.join(work, "e.pth")
                retrain_encoder(arch, mdir, ORIG, LABEL, a.device, enc)
                m = evaluate_decoder(enc, "labeled_marker", 3, val_s,
                                     ORIG, LABEL, a.device)
                sal = os.path.join(work, "sal")
                generate_pool_saliencies(enc, pool_s, ORIG, a.device, sal,
                                         "labeled_marker", 3)
                e_mu, e_sd, _ = pool_entropy_stats(sal, pool_s)
                tag = f"s{split} {style}" + (f" {dabs}d" if dabs else "")
                print(f"{tag:<22}{m['fb']:>9.4f}{e_mu:>11.5f}{e_sd:>10.5f}"
                      f"{total // len(images):>9}")
            finally:
                shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
