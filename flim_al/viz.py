"""
viz.py
======
Geração de grids de comparação entre decoders FLIM.

Grid: [Original | labeled_marker | backprop | decoder_2 | decoder_3 |
        decoder_attention | hybrid | vanilla | vanilla_wt | AL result | GT]

Uso:
    from flim_al.viz import generate_comparison_grid, generate_entropy_overlay

    # Grid de comparação de todos os decoders
    generate_comparison_grid(
        fnames=["000002.png", "000003.png"],
        orig_folder="datasets/schistossoma-eggs/orig",
        label_folder="datasets/schistossoma-eggs/label",
        saliency_base="out/saliencies/schisto/user_A/test/split1",
        al_weights_path="out/al_flim_curve/.../layer3_weight.pth",  # opcional
        encoder_path="out/trained_models/.../flim_encoder_split1.pth",
        target_layer=3,
        output_path="out/viz/split1",
        device="cuda:0",
        n_images=5,
    )

    # Overlay de entropia sobre a imagem original
    generate_entropy_overlay(...)
"""

from __future__ import annotations

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image


# Decoder layout: (folder_name, layer_subfolder, display_name)
DECODER_CONFIG = [
    ("labeled_marker",          "layer_3", "Saliency\n(marker)"),
    ("backprop_decoder",        "layer_4", "Backprop\n(original)"),
    ("decoder_2",               "layer_3", "Decoder 2"),
    ("decoder_3",               "layer_3", "Decoder 3"),
    ("decoder_attention",       "layer_3", "Attention"),
    ("hybrid_decoder",          "layer_3", "Hybrid"),
    ("vanilla_adaptive_decoder","layer_3", "Vanilla\nAD"),
    ("vanilla_adaptive_decoder_wt","layer_3","Vanilla\nAD-WT"),
]


def _load_gray(path: str) -> np.ndarray | None:
    if path and os.path.exists(path):
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return None


def _load_rgb(path: str) -> np.ndarray | None:
    if path and os.path.exists(path):
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return None


@torch.no_grad()
def _predict_al(
    encoder_path: str,
    al_weights_path: str,
    img_path: str,
    target_layer: int,
    device: str,
) -> np.ndarray:
    """Roda o decoder AL sobre uma imagem e retorna predição [0,255]."""
    model = torch.load(encoder_path, map_location=device, weights_only=False)
    model.eval()
    decoder_w = torch.load(al_weights_path, map_location=device, weights_only=True)

    img_arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    orig_h, orig_w = img_arr.shape[:2]
    x = torch.tensor(img_arr.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)

    for l in range(model.architecture.nlayers):
        if not model.use_bias:
            x = model.normalization(x, model.layers[l].normalization_parameters)
        x = model.layers[l].conv(x)
        x = model.layers[l].activation(x)
        x = model.layers[l].pool(x)
        if l == target_layer:
            break

    pred = F.conv2d(x, decoder_w.to(device), padding=0, stride=1)
    pred = torch.sigmoid(pred)
    pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
    return (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)


def generate_comparison_grid(
    fnames: list[str],
    orig_folder: str,
    label_folder: str,
    saliency_base: str,
    output_path: str,
    al_weights_path: str | None = None,
    encoder_path: str | None = None,
    target_layer: int = 3,
    device: str = "cpu",
    n_images: int = 8,
    colormap: str = "jet",
) -> list[str]:
    """
    Gera grids de comparação para as primeiras n_images imagens.

    Colunas: Original | [decoders...] | [AL result*] | GT
    (* apenas se al_weights_path e encoder_path forem fornecidos)

    Returns
    -------
    saved_paths : lista de arquivos PNG salvos
    """
    os.makedirs(output_path, exist_ok=True)

    # Descobrir quais decoders existem no saliency_base
    active_decoders = []
    for folder, layer, name in DECODER_CONFIG:
        d = os.path.join(saliency_base, folder, layer)
        if os.path.isdir(d):
            active_decoders.append((d, name))

    has_al = (al_weights_path is not None and encoder_path is not None
              and os.path.exists(al_weights_path))

    n_cols = 1 + len(active_decoders) + (1 if has_al else 0) + 1

    cmap = plt.get_cmap(colormap)
    saved = []

    for fname in fnames[:n_images]:
        orig_path  = os.path.join(orig_folder,  fname)
        label_path = os.path.join(label_folder, fname)

        orig  = _load_rgb(orig_path)
        gt    = _load_gray(label_path)

        if orig is None:
            print(f"  [viz] skip {fname}: imagem não encontrada")
            continue

        fig = plt.figure(figsize=(n_cols * 2.8, 3.5))
        gs  = GridSpec(1, n_cols, figure=fig, wspace=0.04)

        col = 0

        # ── Original ──────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(orig)
        ax.set_title("Original", fontsize=7, pad=2)
        ax.axis("off")
        col += 1

        # ── Decoders ──────────────────────────────────────────────────────────
        for sal_dir, dec_name in active_decoders:
            ax = fig.add_subplot(gs[0, col])
            sal = _load_gray(os.path.join(sal_dir, fname))
            if sal is not None:
                ax.imshow(sal, cmap=colormap, vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_title(dec_name, fontsize=6, pad=2)
            ax.axis("off")
            col += 1

        # ── AL Result ─────────────────────────────────────────────────────────
        if has_al:
            ax = fig.add_subplot(gs[0, col])
            try:
                al_pred = _predict_al(
                    encoder_path, al_weights_path, orig_path, target_layer, device
                )
                ax.imshow(al_pred, cmap=colormap, vmin=0, vmax=255)
            except Exception as e:
                ax.text(0.5, 0.5, f"ERR\n{str(e)[:20]}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color="red")
            ax.set_title("AL\nBackprop", fontsize=7, pad=2)
            ax.axis("off")
            col += 1

        # ── GT ────────────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        if gt is not None:
            ax.imshow(gt, cmap="gray", vmin=0, vmax=255)
        ax.set_title("GT", fontsize=7, pad=2)
        ax.axis("off")

        out_file = os.path.join(output_path, fname.replace(".png", "_compare.png"))
        plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(out_file)
        print(f"  [viz] Salvo: {out_file}")

    return saved


def generate_entropy_overlay(
    fnames: list[str],
    orig_folder: str,
    saliency_folder: str,    # pasta com saliency maps (labeled_marker/layer_3)
    output_path: str,
    patch_size: int = 64,
    top_k_patches: int = 5,
    n_images: int = 8,
) -> list[str]:
    """
    Gera overlays mostrando regiões de alta incerteza (Region AL).

    Para cada imagem: [Original | Entropy Map | Annotation Mask (patches)]
    """
    from flim_al.region_al import compute_entropy_mask

    os.makedirs(output_path, exist_ok=True)
    saved = []

    for fname in fnames[:n_images]:
        orig_path = os.path.join(orig_folder, fname)
        sal_path  = os.path.join(saliency_folder, fname)

        orig = _load_rgb(orig_path)
        if orig is None:
            continue

        H, W = orig.shape[:2]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

        # Original
        axes[0].imshow(orig)
        axes[0].set_title("Original", fontsize=8)
        axes[0].axis("off")

        # Entropy map
        if os.path.exists(sal_path):
            arr = np.array(Image.open(sal_path).convert("L"), dtype=np.float32) / 255.0
            p = np.clip(arr, 1e-6, 1 - 1e-6)
            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            axes[1].imshow(entropy, cmap="hot", vmin=0, vmax=np.log(2))
            axes[1].set_title("Entropy Map\n(incerteza por pixel)", fontsize=7)
        else:
            axes[1].text(0.5, 0.5, "N/A", ha="center", va="center",
                         transform=axes[1].transAxes, color="gray")
            axes[1].set_title("Entropy Map", fontsize=8)
        axes[1].axis("off")

        # Annotation mask overlay
        axes[2].imshow(orig)
        if os.path.exists(sal_path):
            mask = compute_entropy_mask(
                sal_path, patch_size=patch_size, top_k_patches=top_k_patches
            )
            # Resize mask to orig size if needed
            if mask.shape != (H, W):
                mask = np.array(
                    Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        (W, H), Image.NEAREST
                    ), dtype=np.float32
                ) / 255.0
            # Red overlay on uncertain patches
            overlay = np.zeros((H, W, 4), dtype=np.float32)
            overlay[:, :, 0] = 1.0  # red
            overlay[:, :, 3] = mask * 0.4  # alpha
            axes[2].imshow(overlay)
            cov = float(mask.mean()) * 100
            axes[2].set_title(
                f"Annotation Mask\n(top-{top_k_patches} patches, {cov:.1f}% pixels)",
                fontsize=7
            )
        axes[2].axis("off")

        plt.tight_layout(pad=0.3)
        out_file = os.path.join(output_path, fname.replace(".png", "_entropy.png"))
        plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(out_file)
        print(f"  [viz] Entropy overlay: {out_file}")

    return saved


def generate_al_curve_plot(
    csv_path: str,
    output_path: str,
    metric: str = "fb",
) -> str:
    """
    Plota curva AL vs Random (e outros métodos se disponíveis no CSV).

    O CSV deve ter colunas: split, budget, acquisition,
    al_fb/al_dice/al_mae, rand_fb/rand_dice/rand_mae.

    Returns
    -------
    Path do PNG salvo.
    """
    import csv
    from collections import defaultdict

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[viz] CSV vazio — nenhum plot gerado")
        return ""

    # Agrupa por acquisition + budget → média dos splits
    data = defaultdict(lambda: defaultdict(list))
    acqs = set()
    for row in rows:
        acq = row.get("acquisition", "entropy")
        b   = int(row["budget"])
        data[acq][b].append(float(row[f"al_{metric}"]))
        data[acq]["rand_" + str(b)].append(float(row[f"rand_{metric}"]))
        acqs.add(acq)

    budgets = sorted(set(int(r["budget"]) for r in rows))

    os.makedirs(output_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"entropy": "#e63946", "coreset": "#457b9d", "badge": "#2a9d8f",
              "region_entropy": "#e9c46a"}
    labels = {"entropy": "Entropy (AL)", "coreset": "CoreSet (AL)",
              "badge": "BADGE (AL)", "region_entropy": "Region Entropy (AL)"}

    rand_plotted = False
    for acq in sorted(acqs):
        al_vals  = [np.mean(data[acq][b]) for b in budgets]
        rand_vals = [np.mean(data[acq][f"rand_{b}"]) for b in budgets]

        color = colors.get(acq, "#666666")
        ax.plot(budgets, al_vals, "o-", color=color,
                label=labels.get(acq, acq), linewidth=2, markersize=5)

        if not rand_plotted:
            ax.plot(budgets, rand_vals, "s--", color="#aaaaaa",
                    label="Random baseline", linewidth=1.5, markersize=4, alpha=0.7)
            rand_plotted = True

    ax.set_xlabel("Budget K (imagens anotadas)", fontsize=11)
    ax.set_ylabel(f"F-beta" if metric == "fb" else metric.upper(), fontsize=11)
    ax.set_title("AL vs Random — Curva de Aprendizado Ativo\n(FLIM backprop_decoder, schisto/user_A)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    out_file = os.path.join(output_path, f"al_curve_{metric}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [viz] Curva AL: {out_file}")
    return out_file
