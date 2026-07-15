"""
quick_test.py — Valida os fixes (Otsu + formato markers) sem treinar novo encoder.
Usa o encoder original (3 imagens) e avalia os 7 decoders no val set.

Resultado esperado (próximo ao paper FLIM Tabela 3, user_A split1):
  labeled_marker  ~ 0.697
  vanilla         ~ 0.628
  vanilla_wt      ~ 0.464
  decoder_2       ~ 0.700
  decoder_3       ~ 0.715
  attention       ~ 0.514
  hybrid          ~ 0.639

Rodar de dentro de flim_ad/:
  python3 ../flim_al/quick_test.py
"""

import os, sys, csv
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def threshold_otsu(img: np.ndarray) -> float:
    """Otsu's threshold — implementado sem skimage."""
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    best, thresh = 0.0, 0
    w0 = sum_bg = 0.0
    for t in range(256):
        w0    += hist[t]
        w1     = total - w0
        if w0 == 0 or w1 == 0:
            continue
        sum_bg += t * hist[t]
        mu0    = sum_bg / w0
        mu1    = (hist @ np.arange(256) - sum_bg) / w1
        score  = w0 * w1 * (mu0 - mu1) ** 2
        if score > best:
            best, thresh = score, t
    return float(thresh)


def label_components(bin_arr: np.ndarray):
    """BFS labeling de componentes conectadas (4-conectividade)."""
    labeled = np.zeros_like(bin_arr, dtype=np.int32)
    cur = 0
    H, W = bin_arr.shape
    for i in range(H):
        for j in range(W):
            if bin_arr[i, j] > 0 and labeled[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labeled[i, j] = cur
                while stack:
                    r, c = stack.pop()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<H and 0<=nc<W and bin_arr[nr,nc]>0 and labeled[nr,nc]==0:
                            labeled[nr,nc] = cur
                            stack.append((nr,nc))
    return labeled, cur

REPO   = Path(__file__).resolve().parent.parent
# Usa pyflim base (sem faiss/monai/transformers) — forward retorna tensor direto
sys.path.insert(0, str(REPO))

from pyflim import layers

ENCODER_PATH  = "out/trained_models/schisto/user_A/split1/flim_encoder_split1.pth"
ORIG_FOLDER   = "datasets/schisto/images/"
LABEL_FOLDER  = "datasets/schisto/truelabels/"
VAL_CSV       = "data/schisto/user_A/split1/val1.csv"
TARGET_LAYER  = 3
AREA_RANGE    = (1000, 9000)

DECODERS = [
    "labeled_marker",
    "vanilla_adaptive_decoder",
    "vanilla_adaptive_decoder_wt",
    "decoder_2",
    "decoder_3",
    "decoder_attention",
    "hybrid_decoder",
]

PAPER = {
    "labeled_marker":            0.697,
    "vanilla_adaptive_decoder":  0.628,
    "vanilla_adaptive_decoder_wt": 0.464,
    "decoder_2":                 0.700,
    "decoder_3":                 0.715,
    "decoder_attention":         0.514,
    "hybrid_decoder":            0.639,
}

def filter_components(pred_bin, area_range):
    labeled, n = label_components(pred_bin)
    out = pred_bin.copy()
    for c in range(1, n + 1):
        area = (labeled == c).sum()
        if area < area_range[0] or area > area_range[1]:
            out[labeled == c] = 0
    return out


@torch.no_grad()
def eval_decoder(model, decoder_type, val_fnames, layer_idx):
    import traceback
    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type=decoder_type, multi_layer=False,
    )

    fbs = []
    for fname in val_fnames[:30]:   # primeiras 30 imagens para teste rápido
        orig_path  = os.path.join(ORIG_FOLDER, fname)
        label_path = os.path.join(LABEL_FOLDER, fname)
        if not (os.path.exists(orig_path) and os.path.exists(label_path)):
            continue

        img   = Image.open(orig_path).convert("RGB")
        orig_h, orig_w = img.size[1], img.size[0]
        arr   = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x     = torch.tensor(arr).unsqueeze(0)

        try:
            # pyflim base: forward(X, decoder_layer=int) → tensor (H,W) ou (1,1,H,W)
            out = model.forward(x, decoder_layer=layer_idx)
            if out is None:
                continue
            pred_np = out.float().squeeze().numpy()
            if pred_np.max() <= 1.0:
                pred_np = pred_np * 255.0
            pred_u8  = pred_np.astype(np.uint8)

            if pred_u8.shape[0] != orig_h or pred_u8.shape[1] != orig_w:
                pil = Image.fromarray(pred_u8).resize((orig_w, orig_h), Image.BILINEAR)
                pred_u8 = np.array(pil, dtype=np.uint8)

            if pred_u8.max() > pred_u8.min():
                pred_bin = (pred_u8 > threshold_otsu(pred_u8)).astype(np.float32)
            else:
                pred_bin = np.zeros_like(pred_u8, dtype=np.float32)

            pred_bin = filter_components(pred_bin, AREA_RANGE)
            gt = (np.array(Image.open(label_path).convert("L")) > 127).astype(np.float32)

            pt = torch.tensor(pred_bin)
            gt_t = torch.tensor(gt)
            tp = (pt * gt_t).sum()
            fp = (pt * (1 - gt_t)).sum()
            fn = ((1 - pt) * gt_t).sum()
            pr = tp / (tp + fp + 1e-8)
            rc = tp / (tp + fn + 1e-8)
            fb = (2 * pr * rc / (pr + rc + 1e-8)).item()
            fbs.append(fb)
        except Exception as e:
            print(f"    ERRO em {fname}: {e}")
            traceback.print_exc()

    return float(np.mean(fbs)) if fbs else 0.0


def main():
    if not os.path.exists(ENCODER_PATH):
        print(f"ERRO: encoder não encontrado: {ENCODER_PATH}")
        print("Rode de dentro de flim_ad/")
        return

    # Val set
    with open(VAL_CSV) as f:
        val_fnames = [os.path.basename(row[0].strip()) for row in csv.reader(f) if row]
    print(f"Val set: {len(val_fnames)} imagens (testando primeiras 30)")

    # Carrega encoder
    model = torch.load(ENCODER_PATH, map_location="cpu", weights_only=False)
    model.device = "cpu"
    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None:
            print(f"  Layer {l}: marker_labels unique = {ml.unique().tolist()}")
            model.layers[l].marker_labels = ml.to("cpu")

    layer_idx = TARGET_LAYER - 1

    print(f"\n{'Decoder':<30} {'Fβ obtido':>10} {'Fβ paper':>10} {'Δ':>8}")
    print("-" * 62)

    for decoder_type in DECODERS:
        fb = eval_decoder(model, decoder_type, val_fnames, layer_idx)
        paper_fb = PAPER.get(decoder_type, 0.0)
        delta = fb - paper_fb
        status = "✓" if abs(delta) < 0.1 else "?"
        print(f"{decoder_type:<30} {fb:>10.3f} {paper_fb:>10.3f} {delta:>+8.3f}  {status}")

    print("\nExpectativa: todos os decoders com Fβ > 0.0 e próximos do paper.")


if __name__ == "__main__":
    main()
