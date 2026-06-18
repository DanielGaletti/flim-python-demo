"""
al_flim_backprop.py
====================
AL correto integrado ao pipeline FLIM real (backprop_decoder).

Fluxo:
  1. Saliency maps do labeled_marker como proxy de incerteza (sem GT)
  2. Entropy score → ranking das imagens do pool
  3. Para cada budget K:
       AL:     top-K por entropy → fit_backprop_decoder → Fβ no val set
       Random: K aleatório (avg n_seeds) → fit_backprop_decoder → Fβ
  4. Curva AL vs Random usando o decoder real do FLIM (1x1 conv sobre features)

Referência interna: src/train_backprop_decoder.py + pyflim/flim.py::fit_backprop_decoder

Uso:
  cd flim_ad
  python3 ../flim_al/al_flim_backprop.py \\
      --dataset_home /workspace/flim-python-demo/flim_ad/datasets \\
      --markers schisto/user_A \\
      --splits 1 2 3 \\
      --budgets 3 5 10 20 30 50 \\
      --target_layer 3 \\
      --n_epochs 100 \\
      --n_seeds 5 \\
      --device cuda:0 \\
      --save_dir out/al_flim_curve
"""

import argparse, os, sys, csv, random, glob
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

REPO   = Path(__file__).resolve().parent.parent
FLIMPY = REPO / "flim_ad" / "libs" / "flim-python"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FLIMPY))
# Permite rodar de dentro de flim_ad/ também
sys.path.insert(0, str(FLIMPY / "pyflim"))

from pyflim import layers, data as flimdata
from flim_al.acquisition import entropy_score


# ── Scoring do pool ───────────────────────────────────────────────────────────

def score_saliencies(sal_dir: str, device: str) -> tuple[list[str], list[float]]:
    """
    Calcula entropy dos saliency maps do labeled_marker (proxy de incerteza).
    Retorna (filenames_com_extensao, scores).
    """
    paths = sorted(glob.glob(os.path.join(sal_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"Sem saliency maps em {sal_dir}")

    fnames  = [os.path.basename(p) for p in paths]   # "img001.png"
    scores  = []
    batch_size = 32

    for i in range(0, len(paths), batch_size):
        tensors = []
        for p in paths[i:i + batch_size]:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            tensors.append(torch.tensor(arr).unsqueeze(0).unsqueeze(0))
        batch = torch.cat(tensors, dim=0).to(device)
        scores.extend(entropy_score(batch).cpu().tolist())

    return fnames, scores


# ── Treino do backprop_decoder com subconjunto AL ────────────────────────────

def train_backprop_on_subset(
    encoder_path: str,
    selected_fnames: list[str],   # filenames COM extensão, ex: "img001.png"
    orig_folder: str,
    label_folder: str,
    target_layer: int,
    output_path: str,
    n_epochs: int,
    device: str,
    lr: float = 1e-2,
    wd:  float = 1e-2,
) -> str:
    """
    Treina o backprop_decoder nas imagens selecionadas.
    Retorna o path do arquivo de pesos salvo.
    """
    model = torch.load(encoder_path, map_location=device, weights_only=False)
    model.device = device

    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1,
        adaptation_function="robust_weights",
        filter_by_size=False,
        device=device,
        adj_radius=1.5,
        decoder_type="backprop_decoder",
        multi_layer=False,
    )

    train_ds = flimdata.FLIMData(
        orig_folder=orig_folder,
        images_list=selected_fnames,      # lista de "img.png"
        label_folder=label_folder,
        orig_ext=".png",
        label_ext=".png",
        marker_folder=None,               # sem markers — usa images_list direto
        transform=flimdata.transforms.Compose([flimdata.ToTensor()]),
        bits=8,
        convert_gray_to_lab=False,
    )

    os.makedirs(output_path, exist_ok=True)
    model.fit_backprop_decoder(
        train_ds,
        output_path,
        target_layer=target_layer,
        epochs=n_epochs,
        lr=lr,
        wd=wd,
        use_scheduler=False,
    )

    weights_path = os.path.join(output_path, f"layer{target_layer}_weight.pth")
    return model, weights_path


# ── Inferência ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_backprop(
    encoder_path: str,
    weights_path: str,
    eval_fnames: list[str],
    orig_folder: str,
    label_folder: str,
    target_layer: int,
    device: str,
) -> tuple[float, float, float]:
    """
    Roda inferência manual replicando a lógica de fit_backprop_decoder:
      features = FLIM encoder até target_layer
      pred = F.conv2d(features, decoder_weights) → relu → normalize → interpolate
    """
    model = torch.load(encoder_path, map_location=device, weights_only=False)
    model.device = device
    model.eval()

    decoder_weights = torch.load(weights_path, map_location=device, weights_only=True)

    dices, fbs, maes = [], [], []

    for fname in eval_fnames:
        orig_path  = os.path.join(orig_folder,  fname)
        label_path = os.path.join(label_folder, fname)
        if not (os.path.exists(orig_path) and os.path.exists(label_path)):
            continue

        img  = Image.open(orig_path).convert("RGB")
        orig_h, orig_w = img.size[1], img.size[0]
        arr  = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x    = torch.tensor(arr).unsqueeze(0).to(device)

        # Forward FLIM encoder até target_layer (mesma lógica de fit_backprop_decoder)
        for l in range(model.architecture.nlayers):
            if not model.use_bias:
                x = model.normalization(x, model.layers[l].normalization_parameters)
            x = model.layers[l].conv(x)
            x = model.layers[l].activation(x)
            x = model.layers[l].pool(x)
            if l == target_layer:
                break

        # Decoder: replicar run do backprop_decoder
        pred = F.conv2d(x, decoder_weights, padding=0, stride=1)
        pred = F.relu(pred)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=True)

        mask = Image.open(label_path).convert("L")
        mask_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        gt = torch.tensor(mask_arr).unsqueeze(0).unsqueeze(0).to(device)

        pred_bin = (pred > 0.5).float()
        inter = (pred_bin * gt).sum()
        union = pred_bin.sum() + gt.sum()
        dice  = (2 * inter / (union + 1e-8)).item()
        tp = (pred_bin * gt).sum()
        fp = (pred_bin * (1 - gt)).sum()
        fn = ((1 - pred_bin) * gt).sum()
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        fb = (2 * pr * rc / (pr + rc + 1e-8)).item()
        mae = (pred - gt).abs().mean().item()

        dices.append(dice); fbs.append(fb); maes.append(mae)

    if not dices:
        return 0.0, 0.0, 1.0
    return float(np.mean(dices)), float(np.mean(fbs)), float(np.mean(maes))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_home",  default="/workspace/flim-python-demo/flim_ad/datasets")
    p.add_argument("--markers",       default="schisto/user_A")
    p.add_argument("--splits",        nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--proxy_layer",   type=int, default=3)
    p.add_argument("--target_layer",  type=int, default=3)
    p.add_argument("--budgets",       nargs="+", type=int, default=[3, 5, 10, 20, 30, 50])
    p.add_argument("--n_epochs",      type=int, default=100)
    p.add_argument("--n_seeds",       type=int, default=5)
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--save_dir",      default="out/al_flim_curve")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    orig_folder  = os.path.join(args.dataset_home, "schistossoma-eggs", "orig")
    label_folder = os.path.join(args.dataset_home, "schistossoma-eggs", "label")
    os.makedirs(args.save_dir, exist_ok=True)

    all_rows = []

    for split in args.splits:
        print(f"\n{'='*60}\nSplit {split}\n{'='*60}")

        sal_dir = (
            f"out/saliencies/{args.markers}/test/split{split}"
            f"/labeled_marker/layer_{args.proxy_layer}"
        )
        if not os.path.exists(sal_dir):
            print(f"  Saliency dir não encontrado: {sal_dir} — skip")
            continue

        fnames, scores = score_saliencies(sal_dir, device)
        N = len(fnames)
        al_ranking = sorted(range(N), key=lambda i: scores[i], reverse=True)
        print(f"  Pool: {N} imagens | top-3 mais incertas: {[fnames[i] for i in al_ranking[:3]]}")

        enc_path = (
            f"out/trained_models/{args.markers}/split{split}"
            f"/flim_encoder_split{split}.pth"
        )
        if not os.path.exists(enc_path):
            print(f"  Encoder não encontrado: {enc_path} — skip")
            continue

        # Conjunto de validação: split{N}-val.txt
        val_list_path = f"datasets/schistossoma-eggs/Splits-5train-70_30/split{split}-val.txt"
        if os.path.exists(val_list_path):
            with open(val_list_path) as f:
                val_fnames = [l.strip() for l in f if l.strip()]
        else:
            # Fallback: todo o pool (610 imagens de teste)
            val_fnames = fnames
        print(f"  Val set: {len(val_fnames)} imagens")

        budgets = sorted(set([b for b in args.budgets if b <= N] + [N]))

        for budget in budgets:
            pct = budget / N * 100
            print(f"\n  Budget={budget} ({pct:.1f}%)")

            # ── AL: top-K por entropy ─────────────────────────────────────
            al_fnames = [fnames[i] for i in al_ranking[:budget]]

            al_out = os.path.join(
                args.save_dir, args.markers, f"split{split}", f"budget{budget}", "al"
            )
            _, al_weights = train_backprop_on_subset(
                enc_path, al_fnames, orig_folder, label_folder,
                args.target_layer, al_out, args.n_epochs, device
            )
            dice_al, fb_al, mae_al = evaluate_backprop(
                enc_path, al_weights, val_fnames,
                orig_folder, label_folder, args.target_layer, device
            )
            print(f"    AL:     DICE={dice_al:.3f}  Fβ={fb_al:.3f}  MAE={mae_al:.3f}")

            # ── Random: média de n_seeds ──────────────────────────────────
            rand_fbs, rand_dices, rand_maes = [], [], []
            for seed in range(args.n_seeds):
                random.seed(seed)
                rand_fnames = random.sample(fnames, min(budget, N))
                rand_out = os.path.join(
                    args.save_dir, args.markers, f"split{split}", f"budget{budget}", f"rand{seed}"
                )
                _, r_weights = train_backprop_on_subset(
                    enc_path, rand_fnames, orig_folder, label_folder,
                    args.target_layer, rand_out, args.n_epochs, device
                )
                d, f, m = evaluate_backprop(
                    enc_path, r_weights, val_fnames,
                    orig_folder, label_folder, args.target_layer, device
                )
                rand_dices.append(d); rand_fbs.append(f); rand_maes.append(m)

            dice_r = np.mean(rand_dices); fb_r = np.mean(rand_fbs); mae_r = np.mean(rand_maes)
            print(f"    Random: DICE={dice_r:.3f}  Fβ={fb_r:.3f}  MAE={mae_r:.3f}  (avg {args.n_seeds} seeds)")

            all_rows.append({
                "split": split, "budget": budget,
                "al_dice": round(dice_al,4), "al_fb": round(fb_al,4), "al_mae": round(mae_al,4),
                "rand_dice": round(dice_r,4), "rand_fb": round(fb_r,4), "rand_mae": round(mae_r,4),
                "delta_fb": round(fb_al - fb_r, 4),
            })

    if not all_rows:
        print("Nenhum resultado gerado.")
        return

    tag      = args.markers.replace("/", "-")
    csv_path = os.path.join(args.save_dir, f"{tag}_al_flim_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"CSV salvo em {csv_path}")
    print(f"{'Budget':>8} {'AL Fβ':>8} {'Rand Fβ':>8} {'ΔFβ':>8}")
    print("-"*36)
    for b in sorted(set(r["budget"] for r in all_rows)):
        sub = [r for r in all_rows if r["budget"] == b]
        al  = np.mean([r["al_fb"]   for r in sub])
        rd  = np.mean([r["rand_fb"] for r in sub])
        print(f"{b:>8} {al:>8.3f} {rd:>8.3f} {al-rd:>+8.3f}")


if __name__ == "__main__":
    main()
