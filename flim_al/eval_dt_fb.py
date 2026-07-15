"""
eval_dt_fb.py
=============
Computa Fβ pixel-level (β²=0.3) a partir dos masks gerados pelo
iftSMansoniDelineation (Dynamic Trees).

Replica FLIMMetrics.fmeasure(bin_sal, label, beta2=0.3) de metrics.py.

Uso:
  cd flim_ad
  python3 ../flim_al/eval_dt_fb.py \\
      --dt_masks  out/saliencies_delination/schisto/user_A/test/split1/labeled_marker/layer_3/masks \\
      --gt_dir    datasets/schistossoma-eggs/label \\
      --val_list  data/schisto/user_A/split1/test1.csv
"""

import argparse
import os
import numpy as np
from PIL import Image

EPS = 1e-8


def _pixel_fb(pred_bin: np.ndarray, gt_bin: np.ndarray, beta2: float = 0.3) -> float:
    """
    Fβ pixel-level — cópia de FLIMMetrics.fmeasure(bin_sal, label, beta2=0.3).
    Entrada: arrays binários uint8 {0,1}.
    """
    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 1.0   # ambos vazios → correto

    tp = float(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = float(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (gt_bin == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return (1 + beta2) * prec * rec / (EPS + beta2 * prec + rec)


def _pixel_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Dice oficial (1.0 para ambos vazios)."""
    gs, ps = gt_bin.sum(), pred_bin.sum()
    if gs == 0 and ps == 0: return 1.0
    if gs == 0 or  ps == 0: return 0.0
    return 2.0 * float((gt_bin * pred_bin).sum()) / (gs + ps)


def compute_fb_from_masks(
    masks_dir: str,
    gt_dir: str,
    fnames: list[str],
    beta2: float = 0.3,
) -> dict:
    """
    Para cada imagem em fnames:
      - Se existe mask em masks_dir/ → usa como predição binária
      - Caso contrário → predição = zeros (DT não gerou máscara)

    Retorna médias: {fb, dice, n_images, n_masks_found}
    """
    fbs, dices = [], []
    n_found = 0

    for fname in fnames:
        gt_path   = os.path.join(gt_dir, fname)
        mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(gt_path):
            continue

        gt_bin = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)

        if os.path.exists(mask_path):
            pred_bin = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
            n_found += 1
        else:
            pred_bin = np.zeros_like(gt_bin)

        fbs.append(_pixel_fb(pred_bin, gt_bin, beta2))
        dices.append(_pixel_dice(pred_bin, gt_bin))

    return {
        "fb":            float(np.mean(fbs))   if fbs   else 0.0,
        "dice":          float(np.mean(dices)) if dices else 0.0,
        "n_images":      len(fbs),
        "n_masks_found": n_found,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt_masks",  required=True,
                   help="Dir com masks do DT (pode ser dir/masks/ ou dir/)")
    p.add_argument("--gt_dir",    required=True)
    p.add_argument("--val_list",  required=True,
                   help="CSV/TXT com lista de imagens (uma por linha)")
    p.add_argument("--beta2",     type=float, default=0.3)
    args = p.parse_args()

    # Resolve sub-dir masks/ se existir
    masks_dir = args.dt_masks
    candidate = os.path.join(masks_dir, "masks")
    if os.path.isdir(candidate):
        masks_dir = candidate
        print(f"[eval_dt_fb] Usando {masks_dir}")

    with open(args.val_list) as f:
        fnames = [l.strip().replace("images/", "").strip()
                  for l in f if l.strip()]

    r = compute_fb_from_masks(masks_dir, args.gt_dir, fnames, args.beta2)

    print(f"\n{'='*50}")
    print(f"Imagens avaliadas : {r['n_images']}")
    print(f"Masks DT encontradas: {r['n_masks_found']} / {r['n_images']}")
    print(f"Fβ  (β²={args.beta2}): {r['fb']:.4f}")
    print(f"DICE:                {r['dice']:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
