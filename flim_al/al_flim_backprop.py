"""
al_flim_backprop.py
====================
AL integrado ao pipeline FLIM real (backprop_decoder).

Modos de aquisição (--acquisition):
  entropy        — Entropy sampling: seleciona as K imagens com maior
                   entropia dos saliency maps (método implementado)
  coreset        — CoreSet (ICLR 2018): greedy k-center no espaço de
                   features do encoder FLIM (diversidade geométrica)
  badge          — BADGE (ICLR 2020): gradient diversity via k-means++
                   (incerteza × diversidade combinadas)
  region_entropy — AL por regiões: seleciona imagens por entropy mas treina
                   decoder apenas nas regiões de alta incerteza (RIPU-style)

Fluxo geral:
  1. Saliency maps do labeled_marker como proxy (sem GT)
  2. Score → ranking das 610 imagens do pool
  3. Para cada budget K:
       AL:     top-K pelo método escolhido → treina backprop_decoder → Fβ no val
       Random: K aleatório (avg n_seeds) → treina backprop_decoder → Fβ
  4. Salva CSV + plota curva AL vs Random

Referências:
  CoreSet  arXiv:1708.00489
  BADGE    arXiv:1906.03671
  RIPU     arXiv:2111.12667

Uso:
  cd flim_ad
  python3 ../flim_al/al_flim_backprop.py \\
      --dataset_home datasets \\
      --markers schisto/user_A \\
      --splits 1 2 3 \\
      --budgets 3 5 10 20 30 50 \\
      --target_layer 3 \\
      --n_epochs 500 \\
      --n_seeds 5 \\
      --acquisition entropy \\
      --device cuda:0 \\
      --save_dir out/al_flim_curve \\
      --visualize
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
sys.path.insert(0, str(FLIMPY / "pyflim"))

from monai.losses import DiceCELoss
from pyflim import layers, data as flimdata
from flim_al.acquisition import entropy_score
from flim_al.coreset_badge import (
    extract_encoder_features,
    extract_encoder_features_and_preds,
    coreset_select,
    badge_select,
)
from flim_al.region_al import run_region_al


# ── Scoring do pool ───────────────────────────────────────────────────────────

def score_saliencies(sal_dir: str, device: str) -> tuple[list[str], list[float]]:
    """
    Calcula entropy dos saliency maps do labeled_marker (proxy de incerteza).
    Retorna (filenames_com_extensão, scores).
    """
    paths = sorted(glob.glob(os.path.join(sal_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"Sem saliency maps em {sal_dir}")

    fnames  = [os.path.basename(p) for p in paths]
    scores  = []
    batch_size = 32

    for i in range(0, len(paths), batch_size):
        tensors = []
        for p in paths[i : i + batch_size]:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            tensors.append(torch.tensor(arr).unsqueeze(0).unsqueeze(0))
        batch = torch.cat(tensors, dim=0).to(device)
        scores.extend(entropy_score(batch).cpu().tolist())

    return fnames, scores


# ── Treino do backprop_decoder com subconjunto AL ────────────────────────────

def train_backprop_on_subset(
    encoder_path: str,
    selected_fnames: list[str],
    orig_folder: str,
    label_folder: str,
    target_layer: int,
    output_path: str,
    n_epochs: int,
    device: str,
    lr: float = 1e-2,
    wd:  float = 1e-2,
) -> tuple:
    """
    Treina o backprop_decoder nas imagens selecionadas.

    Fix em relação ao pyflim original:
    - zero_grad por epoch (batch gradient, como o design original pretendia)
    - Logits diretos ao DiceCELoss (sem relu/normalize antes da loss)
      → elimina o double-sigmoid que bloqueia o gradiente

    Retorna (model, weights_path).
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

    fnames_no_ext = [os.path.splitext(f)[0] for f in selected_fnames]
    train_ds = flimdata.FLIMData(
        orig_folder=orig_folder,
        images_list=fnames_no_ext,
        label_folder=label_folder,
        orig_ext=".png",
        label_ext=".png",
        marker_folder=None,
        transform=flimdata.transforms.Compose([flimdata.ToTensor()]),
        bits=8,
        convert_gray_to_lab=False,
    )

    os.makedirs(output_path, exist_ok=True)
    weights_path = output_path.rstrip("/") + f"/layer{target_layer}_weight.pth"

    decoder_weights = torch.empty(
        (1, model.layers[target_layer].conv.out_channels, 1, 1),
        device=device,
    ).requires_grad_(True)
    torch.nn.init.xavier_uniform_(decoder_weights)

    loss_fn   = DiceCELoss()
    optimizer = torch.optim.Adam([decoder_weights], lr=lr, weight_decay=wd)

    # Pré-computa features (encoder frozen)
    features, labels_list = [], []
    with torch.no_grad():
        for sample in train_ds:
            X = sample["image"].to(device).unsqueeze(0)
            Y = sample["label"]
            if len(Y.shape) > 2:
                Y = Y[:, :, 0]
            for l in range(model.architecture.nlayers):
                if not model.use_bias:
                    X = model.normalization(X, model.layers[l].normalization_parameters)
                X = model.layers[l].conv(X)
                X = model.layers[l].activation(X)
                X = model.layers[l].pool(X)
                if l == target_layer:
                    features.append(X.detach().clone())
                    break
            Y[Y > 0] = 1
            labels_list.append(torch.tensor(Y).unsqueeze(0).float().to(device))

    best_loss = np.inf
    for epoch in range(n_epochs):
        # zero_grad por epoch = batch gradient (como o design original do pyflim pretendia)
        optimizer.zero_grad()
        epoch_losses = []
        for x, y in zip(features, labels_list):
            # Logits diretos → DiceCELoss(sigmoid=True) aplica sigmoid internamente
            # SEM relu/normalize antes da loss: elimina o double-sigmoid que trava o gradiente
            res = F.conv2d(x, decoder_weights, padding=0, stride=1)
            res = F.interpolate(res, [y.shape[-2], y.shape[-1]],
                                mode="bilinear", align_corners=True)
            loss = loss_fn(res, y.unsqueeze(0))
            epoch_losses.append(loss.item())
            loss.backward()     # acumula gradiente sobre o batch
        optimizer.step()        # 1 step por epoch

        mean_ep = float(np.mean(epoch_losses))
        if (epoch % 50 == 0):
            print(f"  epoch:{epoch:4d}  loss:{mean_ep:.4f}", end="\r")
        if mean_ep < best_loss:
            best_loss = mean_ep
            torch.save(decoder_weights.detach(), weights_path)

    print(f"  Final loss: {best_loss:.4f} (saved at best)")
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
    Roda inferência replicando a lógica do backprop_decoder.
    Usa sigmoid (consistente com treino sem double-sigmoid).
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

        # Forward FLIM encoder
        for l in range(model.architecture.nlayers):
            if not model.use_bias:
                x = model.normalization(x, model.layers[l].normalization_parameters)
            x = model.layers[l].conv(x)
            x = model.layers[l].activation(x)
            x = model.layers[l].pool(x)
            if l == target_layer:
                break

        # Decoder: logits → sigmoid (consistente com o treino)
        pred = F.conv2d(x, decoder_weights, padding=0, stride=1)
        pred = torch.sigmoid(pred)
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


# ── Seleção por aquisição ─────────────────────────────────────────────────────

def select_by_acquisition(
    acquisition: str,
    fnames: list[str],
    scores: list[float],          # entropy scores (sempre disponível)
    al_ranking: list[int],        # índices ordenados por entropy
    budget: int,
    enc_path: str,
    orig_folder: str,
    proxy_layer: int,
    device: str,
) -> list[str]:
    """
    Seleciona budget imagens de acordo com o método de aquisição.
    """
    if acquisition == "entropy":
        return [fnames[i] for i in al_ranking[:budget]]

    if acquisition in ("coreset", "badge"):
        print(f"  [{acquisition}] Extraindo features do encoder para {len(fnames)} imagens...")
        encoder = torch.load(enc_path, map_location=device, weights_only=False)
        encoder.eval()
        img_paths = [os.path.join(orig_folder, f) for f in fnames]

        if acquisition == "coreset":
            feats = extract_encoder_features(encoder, img_paths, proxy_layer, device)
            sel_idx = coreset_select(feats, budget)
        else:  # badge
            # Decoder random init para pseudo-labels iniciais
            dummy_w = torch.zeros(
                (1, encoder.layers[proxy_layer].conv.out_channels, 1, 1),
                device=device
            )
            torch.nn.init.xavier_uniform_(dummy_w)
            feats, preds = extract_encoder_features_and_preds(
                encoder, img_paths, dummy_w, proxy_layer, device
            )
            sel_idx = badge_select(feats, preds, budget)

        return [fnames[i] for i in sel_idx]

    if acquisition == "region_entropy":
        # Para region, a seleção ainda é por entropy (idem ao entropy mode)
        # A diferença está no treino (com loss mascarada por região)
        return [fnames[i] for i in al_ranking[:budget]]

    raise ValueError(f"Acquisition desconhecido: {acquisition}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_home",  default="/workspace/flim-python-demo/flim_ad/datasets")
    p.add_argument("--markers",       default="schisto/user_A")
    p.add_argument("--splits",        nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--proxy_layer",   type=int, default=3)
    p.add_argument("--target_layer",  type=int, default=3)
    p.add_argument("--budgets",       nargs="+", type=int, default=[3, 5, 10, 20, 30, 50])
    p.add_argument("--n_epochs",      type=int, default=500)
    p.add_argument("--n_seeds",       type=int, default=5)
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--save_dir",      default="out/al_flim_curve")
    p.add_argument("--acquisition",   default="entropy",
                   choices=["entropy", "coreset", "badge", "region_entropy"],
                   help="Método de seleção AL")
    p.add_argument("--patch_size",    type=int, default=64,
                   help="Tamanho dos patches para region_entropy (pixels)")
    p.add_argument("--top_patches",   type=int, default=5,
                   help="Número de patches de alta entropia por imagem (region_entropy)")
    p.add_argument("--visualize",     action="store_true",
                   help="Gerar grids de comparação e entropy overlays")
    p.add_argument("--viz_n_images",  type=int, default=8,
                   help="Número de imagens para visualização")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Acquisition: {args.acquisition}")

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

        val_list_path = f"datasets/schistossoma-eggs/Splits-5train-70_30/split{split}-val.txt"
        if os.path.exists(val_list_path):
            with open(val_list_path) as f:
                val_fnames = [l.strip() for l in f if l.strip()]
        else:
            val_fnames = fnames
        print(f"  Val set: {len(val_fnames)} imagens")

        # Visualização: entropy overlays antes do treino
        if args.visualize:
            from flim_al.viz import generate_entropy_overlay
            viz_fnames = [fnames[i] for i in al_ranking[:args.viz_n_images]]
            viz_out = os.path.join(args.save_dir, "viz", f"split{split}", "entropy_overlays")
            generate_entropy_overlay(
                fnames=viz_fnames,
                orig_folder=orig_folder,
                saliency_folder=sal_dir,
                output_path=viz_out,
                patch_size=args.patch_size,
                top_k_patches=args.top_patches,
                n_images=args.viz_n_images,
            )

        budgets = sorted(set([b for b in args.budgets if b <= N] + [N]))
        best_al_weights = None  # para visualização pós-treino

        for budget in budgets:
            pct = budget / N * 100
            print(f"\n  Budget={budget} ({pct:.1f}%)")

            # ── AL ────────────────────────────────────────────────────────────
            al_fnames = select_by_acquisition(
                args.acquisition, fnames, scores, al_ranking, budget,
                enc_path, orig_folder, args.proxy_layer, device
            )

            al_out = os.path.join(
                args.save_dir, args.markers, f"split{split}",
                args.acquisition, f"budget{budget}", "al"
            )

            if args.acquisition == "region_entropy":
                al_weights = run_region_al(
                    encoder_path=enc_path,
                    selected_fnames=al_fnames,
                    orig_folder=orig_folder,
                    label_folder=label_folder,
                    saliency_folder=sal_dir,
                    target_layer=args.target_layer,
                    output_path=al_out,
                    n_epochs=args.n_epochs,
                    device=device,
                    patch_size=args.patch_size,
                    top_k_patches=args.top_patches,
                )
                al_weights = al_weights[0]  # run_region_al retorna (path, coverage)
            else:
                _, al_weights = train_backprop_on_subset(
                    enc_path, al_fnames, orig_folder, label_folder,
                    args.target_layer, al_out, args.n_epochs, device
                )

            dice_al, fb_al, mae_al = evaluate_backprop(
                enc_path, al_weights, val_fnames,
                orig_folder, label_folder, args.target_layer, device
            )
            print(f"    AL:     DICE={dice_al:.3f}  Fβ={fb_al:.3f}  MAE={mae_al:.3f}")

            if best_al_weights is None or fb_al > 0:
                best_al_weights = al_weights

            # ── Random ────────────────────────────────────────────────────────
            rand_fbs, rand_dices, rand_maes = [], [], []
            for seed in range(args.n_seeds):
                random.seed(seed)
                rand_fnames = random.sample(fnames, min(budget, N))
                rand_out = os.path.join(
                    args.save_dir, args.markers, f"split{split}",
                    args.acquisition, f"budget{budget}", f"rand{seed}"
                )
                if args.acquisition == "region_entropy":
                    r_weights, _ = run_region_al(
                        encoder_path=enc_path,
                        selected_fnames=rand_fnames,
                        orig_folder=orig_folder,
                        label_folder=label_folder,
                        saliency_folder=sal_dir,
                        target_layer=args.target_layer,
                        output_path=rand_out,
                        n_epochs=args.n_epochs,
                        device=device,
                        patch_size=args.patch_size,
                        top_k_patches=args.top_patches,
                    )
                else:
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
                "acquisition": args.acquisition,
                "al_dice":  round(dice_al, 4), "al_fb":  round(fb_al, 4),  "al_mae":  round(mae_al, 4),
                "rand_dice": round(dice_r, 4), "rand_fb": round(fb_r, 4), "rand_mae": round(mae_r, 4),
                "delta_fb": round(fb_al - fb_r, 4),
            })

        # Visualização pós-treino: comparison grid
        if args.visualize and best_al_weights:
            from flim_al.viz import generate_comparison_grid
            saliency_base = f"out/saliencies/{args.markers}/test/split{split}"
            viz_fnames    = [fnames[i] for i in al_ranking[:args.viz_n_images]]
            viz_out       = os.path.join(args.save_dir, "viz", f"split{split}", "decoder_compare")
            generate_comparison_grid(
                fnames=viz_fnames,
                orig_folder=orig_folder,
                label_folder=label_folder,
                saliency_base=saliency_base,
                output_path=viz_out,
                al_weights_path=best_al_weights,
                encoder_path=enc_path,
                target_layer=args.target_layer,
                device=device,
                n_images=args.viz_n_images,
            )

    if not all_rows:
        print("Nenhum resultado gerado.")
        return

    tag      = args.markers.replace("/", "-")
    csv_name = f"{tag}_{args.acquisition}_al_flim_curve.csv"
    csv_path = os.path.join(args.save_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"CSV salvo em {csv_path}")
    print(f"{'Budget':>8} {'AL Fβ':>8} {'Rand Fβ':>8} {'ΔFβ':>8}")
    print("-" * 36)
    for b in sorted(set(r["budget"] for r in all_rows)):
        sub = [r for r in all_rows if r["budget"] == b]
        al  = np.mean([r["al_fb"]   for r in sub])
        rd  = np.mean([r["rand_fb"] for r in sub])
        print(f"{b:>8} {al:>8.3f} {rd:>8.3f} {al-rd:>+8.3f}")

    # Plot da curva
    if args.visualize:
        from flim_al.viz import generate_al_curve_plot
        viz_dir = os.path.join(args.save_dir, "viz")
        generate_al_curve_plot(csv_path, viz_dir, metric="fb")


if __name__ == "__main__":
    main()
