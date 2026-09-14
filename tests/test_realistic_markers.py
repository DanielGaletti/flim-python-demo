"""
Valida o gerador de markers realistas contra os markers REAIS dos 31 arquivos
anotados. O critério não é opinião: as estatísticas geradas têm que cair na
faixa do que o usuário de fato desenhou.

Rodar (de dentro de flim_ad/):
    python ../tests/test_realistic_markers.py
"""
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from flim_al.analyze_real_markers import (  # noqa: E402
    dist_to_boundary, n_components, nn_median, read_seeds,
)
from flim_al.marker_generator import generate_markers_from_gt  # noqa: E402
from flim_al.paper_selection import real_marker_index  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402

LABEL = "datasets/schistossoma-eggs/label"


def stats(fg, bg, gt):
    return {
        "n_fg": len(fg), "n_bg": len(bg),
        "nn_fg": nn_median(fg), "nn_bg": nn_median(bg),
        "comp_fg": n_components(fg, gt.shape), "comp_bg": n_components(bg, gt.shape),
        "d_fg": dist_to_boundary(fg, gt), "d_bg": dist_to_boundary(bg, gt),
    }


def summarize(rows, key):
    v = [r[key] for r in rows if r[key] == r[key]]
    return st.median(v) if v else float("nan")


def main():
    idx = real_marker_index(REPO)
    imgs = [i for i in sorted(idx) if os.path.exists(os.path.join(LABEL, f"{i}.png"))]
    if not imgs:
        print("SKIP: rode de dentro de flim_ad/")
        return 0

    real, novo, velho = [], [], []
    for img_id in imgs:
        lp = os.path.join(LABEL, f"{img_id}.png")
        gt = np.array(Image.open(lp).convert("L")) > 127
        if not gt.any():
            continue
        fr, br = read_seeds(idx[img_id])
        real.append(stats(fr, br, gt))
        d = generate_realistic_markers(lp, seed=hash(img_id) % 10000)
        novo.append(stats(d["fg_seeds"], d["bg_seeds"], gt))
        o = generate_markers_from_gt(lp, n_fg=100, n_bg=300)
        velho.append(stats(o["fg_seeds"], o["bg_seeds"], gt))

    keys = ["n_fg", "n_bg", "nn_fg", "nn_bg", "comp_fg", "comp_bg", "d_fg", "d_bg"]
    print(f"{'métrica':<10}{'REAL':>10}{'novo':>10}{'antigo':>10}   veredicto")
    print("-" * 60)
    ok = True
    # tolerâncias: fator 2 para contagens, valor absoluto para estrutura
    checks = {
        "n_fg":    ("fator", 2.5), "n_bg":   ("fator", 2.5),
        "nn_fg":   ("abs", 0.5),   "nn_bg":  ("abs", 1.5),
        "comp_fg": ("fator", 3.0), "comp_bg": ("fator", 4.0),
        "d_fg":    ("abs", 10.0),  "d_bg":   ("fator", 3.0),
    }
    for k in keys:
        r, n, v = summarize(real, k), summarize(novo, k), summarize(velho, k)
        mode, tol = checks[k]
        if mode == "fator":
            good = (r != 0) and (1 / tol) <= (n / r if r else 9) <= tol
        else:
            good = abs(n - r) <= tol
        ok &= good
        print(f"{k:<10}{r:>10.2f}{n:>10.2f}{v:>10.2f}   {'OK' if good else 'FORA'}")

    print()
    print(f"  contiguidade (nn≈1 px): real={summarize(real,'nn_fg'):.2f}  "
          f"novo={summarize(novo,'nn_fg'):.2f}  antigo={summarize(velho,'nn_fg'):.2f}")
    assert ok, "gerador novo fora da faixa dos markers reais"
    assert summarize(novo, "nn_fg") < 1.5, "seeds gerados não são contíguos"
    assert summarize(velho, "nn_fg") > 2.0, "esperado: gerador antigo pontilhado"
    print("\nOK: o gerador novo reproduz a estatística da anotação real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
