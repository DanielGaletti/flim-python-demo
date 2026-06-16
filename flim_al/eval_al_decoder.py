"""
eval_al_decoder.py
==================
Avalia o UncertaintyDecoder treinado com AL e gera métricas
comparáveis com as do pipeline FLIM-AD original.

Métricas geradas: DICE, Fβ, MAE, uncertainty_mean
Saída: out/al_results/<markers>/split<N>/<decoder>/eval_metrics.csv

Uso:
    python flim_al/eval_al_decoder.py \
        --markers schisto/user_A \
        --decoder backprop_decoder \
        --splits 1 2 3 \
        --al_dir out/al_results \
        --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import glob
import csv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from flim_al.uncertainty_decoder import UncertaintyDecoder
from flim_al.train_al_decoder import SalisencyDataset, ConvDecoder


def dice_score(pred_bin, mask):
    inter = (pred_bin * mask).sum()
    union = pred_bin.sum() + mask.sum()
    return (2 * inter / (union + 1e-8)).item()


def fscore_beta(pred_bin, mask, beta=1.0):
    tp = (pred_bin * mask).sum()
    fp = (pred_bin * (1 - mask)).sum()
    fn = ((1 - pred_bin) * mask).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return ((1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-8)).item()


def mae_score(pred, mask):
    return (pred - mask).abs().mean().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--markers",   default="schisto/user_A")
    p.add_argument("--decoder",   default="backprop_decoder")
    p.add_argument("--splits",    nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--al_dir",    default="out/al_results")
    p.add_argument("--dataset_home", default="/workspace/flim-python-demo/flim_ad/datasets/")
    p.add_argument("--layer",     type=int, default=4)
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
    return p.parse_args()


@torch.no_grad()
def eval_split(model, dataset, device):
    model.eval()
    rows = []
    for i in range(len(dataset)):
        sal, mask, idx = dataset[i]
        sal  = sal.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        seg, unc = model.predict_uncertainty(sal)
        pred_bin = (seg > 0.5).float()
        rows.append({
            "dice":  dice_score(pred_bin, mask),
            "fscore": fscore_beta(pred_bin, mask),
            "mae":   mae_score(seg, mask),
            "unc_mean": unc.mean().item(),
        })
    return rows


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    size = tuple(args.image_size)

    all_results = []

    for split in args.splits:
        model_path = os.path.join(
            args.al_dir, args.markers, f"split{split}", args.decoder,
            f"model_round10.pt"  # last round checkpoint
        )
        # fallback to highest round available
        if not os.path.exists(model_path):
            candidates = sorted(glob.glob(os.path.join(
                args.al_dir, args.markers, f"split{split}", args.decoder,
                "model_round*.pt"
            )))
            if not candidates:
                print(f"  No checkpoint for split{split} — skipping")
                continue
            model_path = candidates[-1]

        print(f"Split {split}: loading {model_path}")

        base  = ConvDecoder(in_ch=1)
        model = UncertaintyDecoder(base, in_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        saliency_dir = (
            f"out/saliencies_delination/{args.markers}/test/split{split}"
            f"/{args.decoder}/layer_{args.layer}/filtered_saliencies/1000-9000"
        )
        mask_dir = os.path.join(
            args.dataset_home, "schisto",
            args.markers.split("/")[-1], "test", f"split{split}", "masks"
        )

        if not os.path.exists(saliency_dir):
            print(f"  Saliency dir not found — skipping")
            continue

        ds = SalisencyDataset(saliency_dir, mask_dir, size=size)
        rows = eval_split(model, ds, device)

        dice_m  = np.mean([r["dice"] for r in rows])
        fs_m    = np.mean([r["fscore"] for r in rows])
        mae_m   = np.mean([r["mae"] for r in rows])
        unc_m   = np.mean([r["unc_mean"] for r in rows])

        print(f"  DICE={dice_m:.3f}  Fβ={fs_m:.3f}  MAE={mae_m:.3f}  unc={unc_m:.4f}")

        all_results.append({
            "split": split,
            "dice": round(dice_m, 4),
            "fscore": round(fs_m, 4),
            "mae": round(mae_m, 4),
            "unc_mean": round(unc_m, 5),
        })

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir = os.path.join(args.al_dir, args.markers, args.decoder)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "dice", "fscore", "mae", "unc_mean"])
        w.writeheader()
        w.writerows(all_results)

    # ── Summary ───────────────────────────────────────────────────────────
    if all_results:
        print(f"\nMédia splits:")
        print(f"  DICE={np.mean([r['dice'] for r in all_results]):.3f}")
        print(f"  Fβ  ={np.mean([r['fscore'] for r in all_results]):.3f}")
        print(f"  MAE ={np.mean([r['mae'] for r in all_results]):.3f}")
        print(f"\nSalvo em {csv_path}")


if __name__ == "__main__":
    main()
