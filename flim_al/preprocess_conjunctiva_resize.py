#!/usr/bin/env python3
"""
preprocess_conjunctiva_resize.py
=================================
Redimensiona todas as imagens do dataset conjunctiva para tamanho fixo.

Uso (dentro do container em flim_ad/):
    python3 ../flim_al/preprocess_conjunctiva_resize.py [--target_w 640 --target_h 512]

Modifica in-place:
    flim_al/datasets/conjunctiva/orig/   (LANCZOS)
    flim_al/datasets/conjunctiva/label/  (NEAREST — preserva binário)
"""
import argparse
import os
from pathlib import Path
from PIL import Image


def resize_dataset(orig_dir: str, label_dir: str, target_w: int, target_h: int) -> None:
    orig_paths  = sorted(Path(orig_dir).glob("*.png"))
    label_paths = sorted(Path(label_dir).glob("*.png"))

    print(f"Target size: {target_w}×{target_h} (W×H)")
    print(f"Found {len(orig_paths)} orig images, {len(label_paths)} label images")

    # ── Orig images ──────────────────────────────────────────────────────────
    already_ok = 0
    resized    = 0
    for p in orig_paths:
        img = Image.open(p)
        if img.size == (target_w, target_h):
            already_ok += 1
            continue
        img_r = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_r.save(p)
        resized += 1

    print(f"  orig: {resized} resizadas, {already_ok} já corretas")

    # ── Label masks ──────────────────────────────────────────────────────────
    already_ok = 0
    resized    = 0
    for p in label_paths:
        img = Image.open(p).convert("L")        # grayscale
        if img.size == (target_w, target_h):
            already_ok += 1
            continue
        img_r = img.resize((target_w, target_h), Image.Resampling.NEAREST)  # preserva binário
        img_r.save(p)
        resized += 1

    print(f"  label: {resized} resizadas, {already_ok} já corretas")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_w", type=int, default=640)
    p.add_argument("--target_h", type=int, default=512)
    p.add_argument("--dataset_dir",
                   default=os.path.join(os.path.dirname(__file__), "datasets", "conjunctiva"))
    args = p.parse_args()

    orig_dir  = os.path.join(args.dataset_dir, "orig")
    label_dir = os.path.join(args.dataset_dir, "label")

    for d in [orig_dir, label_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Diretório não encontrado: {d}")

    resize_dataset(orig_dir, label_dir, args.target_w, args.target_h)
    print("Done.")


if __name__ == "__main__":
    main()
