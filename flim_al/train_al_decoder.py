"""
train_al_decoder.py
====================
Treina um UncertaintyDecoder sobre um decoder FLIM existente
usando Active Learning para seleção de amostras de treino.

Uso:
    python flim_al/train_al_decoder.py \
        --dataset schisto \
        --dataset_home /workspace/flim-python-demo/flim_ad/datasets/ \
        --markers schisto/user_A \
        --decoder backprop_decoder \
        --layer 4 \
        --splits 1 2 3 \
        --acquisition entropy \
        --query_size 5 \
        --n_rounds 10 \
        --n_epochs 30 \
        --device cuda:0 \
        --save_dir out/al_results/schisto/user_A/backprop_decoder
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import glob

# ── path setup ───────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al import ALLoop, UncertaintyDecoder


# ── Dataset ──────────────────────────────────────────────────────────────────

class SalisencyDataset(Dataset):
    """
    Loads pre-computed FLIM saliency maps as (saliency, mask) pairs.
    Saliency maps = output of run_flim_decoders saved in out/saliencies_*/
    """

    def __init__(
        self,
        saliency_dir: str,
        mask_dir: str,
        size: tuple[int, int] = (256, 256),
    ):
        self.saliency_paths = sorted(glob.glob(os.path.join(saliency_dir, "*.png")))
        self.mask_dir = mask_dir
        self.size = size

        if not self.saliency_paths:
            raise FileNotFoundError(f"No .png saliencies in {saliency_dir}")

    def __len__(self):
        return len(self.saliency_paths)

    def __getitem__(self, idx: int):
        sal_path = self.saliency_paths[idx]
        fname = os.path.basename(sal_path)
        mask_path = os.path.join(self.mask_dir, fname)

        sal = Image.open(sal_path).convert("L").resize(self.size)
        sal = torch.tensor(np.array(sal), dtype=torch.float32).unsqueeze(0) / 255.0

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L").resize(self.size)
            mask = (torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) > 127).float()
        else:
            mask = torch.zeros(1, *self.size)

        return sal, mask, idx


# ── Minimal decoder wrapper ───────────────────────────────────────────────────

class IdentityDecoder(nn.Module):
    """
    Pass-through for cases where the saliency IS already the decoder output.
    The uncertainty head learns on top of that.
    """
    def forward(self, x):
        return torch.sigmoid(x)


class ConvDecoder(nn.Module):
    """
    Small conv decoder to refine saliency maps.
    Replaces the FLIM decoder when running AL standalone.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 1),            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_flim_decoder(decoder_path: str, device: str) -> nn.Module:
    """Load a saved FLIM decoder (.pt) onto CPU then move to device."""
    model = torch.load(decoder_path, map_location="cpu", weights_only=False)
    if hasattr(model, "to"):
        model = model.to(device)
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="schisto")
    p.add_argument("--dataset_home", default="/workspace/flim-python-demo/flim_ad/datasets/")
    p.add_argument("--markers",      default="schisto/user_A")
    p.add_argument("--decoder",      default="backprop_decoder")
    p.add_argument("--layer",        type=int, default=4)
    p.add_argument("--splits",       nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--acquisition",  default="entropy",
                   choices=["entropy", "least_confidence", "bald"])
    p.add_argument("--query_size",   type=int, default=5)
    p.add_argument("--n_rounds",     type=int, default=10)
    p.add_argument("--n_epochs",     type=int, default=30)
    p.add_argument("--seed_size",    type=int, default=3,
                   help="Initial labeled pool size per split")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--save_dir",     default="out/al_results")
    p.add_argument("--decoder_path", default=None,
                   help="Path to .pt FLIM decoder. If None, uses ConvDecoder.")
    p.add_argument("--image_size",   nargs=2, type=int, default=[256, 256])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for split in args.splits:
        print(f"\n{'='*60}\nSplit {split}\n{'='*60}")

        saliency_dir = (
            f"out/saliencies_delination/{args.markers}/test/split{split}"
            f"/{args.decoder}/layer_{args.layer}/filtered_saliencies/1000-9000"
        )
        mask_dir = os.path.join(
            args.dataset_home, args.dataset,
            args.markers.split("/")[-1], "test", f"split{split}", "masks"
        )

        if not os.path.exists(saliency_dir):
            print(f"  Saliency dir not found: {saliency_dir} — skipping")
            continue

        size = tuple(args.image_size)
        full_ds = SalisencyDataset(saliency_dir, mask_dir, size=size)
        print(f"  Dataset: {len(full_ds)} samples")

        n = len(full_ds)
        seed_idx = list(range(min(args.seed_size, n)))
        pool_idx = list(range(len(seed_idx), n))

        from torch.utils.data import Subset
        labeled_ds   = full_ds   # ALLoop manages indices
        unlabeled_ds = full_ds

        # ── Build model ────────────────────────────────────────────────────
        if args.decoder_path and os.path.exists(args.decoder_path):
            base = load_flim_decoder(args.decoder_path, device)
            in_ch = 1
        else:
            base = ConvDecoder(in_ch=1)
            in_ch = 1

        model = UncertaintyDecoder(base, in_channels=in_ch)

        save_dir = os.path.join(
            args.save_dir, args.markers, f"split{split}", args.decoder
        )

        # ── Run AL loop ────────────────────────────────────────────────────
        al = ALLoop(
            model=model,
            labeled_dataset=labeled_ds,
            unlabeled_dataset=unlabeled_ds,
            acquisition=args.acquisition,
            query_size=args.query_size,
            n_rounds=args.n_rounds,
            n_epochs=args.n_epochs,
            device=device,
            save_dir=save_dir,
        )
        # Override initial labeled/unlabeled indices
        al.labeled_indices   = seed_idx
        al.unlabeled_indices = pool_idx

        history = al.run()

        # ── Save final metrics CSV ─────────────────────────────────────────
        csv_path = os.path.join(save_dir, "al_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("round,labeled_size,train_loss,proxy_dice\n")
            for h in history:
                f.write(f"{h['round']},{h['labeled_size']},{h['train_loss']},{h['proxy_dice']}\n")
        print(f"  Metrics → {csv_path}")


if __name__ == "__main__":
    main()
