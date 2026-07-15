"""
al_select_train.py
==================
AL correto para FLIM: seleciona quais imagens anotar usando incerteza
dos saliency maps, depois treina o backprop_decoder nesse subconjunto.

Pergunta de pesquisa:
  Com K imagens selecionadas por AL, FLIMpb atinge Fβ próximo ao treinado
  com o dataset completo? (curva AL vs random)

Pipeline:
  1. Usa saliency maps do labeled_marker (não precisa de GT) como proxy
     de incerteza inicial
  2. Computa entropy por imagem → ranking
  3. Para cada budget K em [10, 20, 30, 50, 100, N_total]:
       a. AL:     seleciona top-K pelo entropy score
       b. Random: seleciona K aleatório
       c. Treina ConvDecoder (fine-tune sobre FLIMpb saliency) com GT masks
       d. Avalia no test set → DICE, Fβ, MAE
  4. Salva curva em al_curve.csv

Uso:
  python flim_al/al_select_train.py \
      --markers schisto/user_A \
      --split 1 \
      --budgets 5 10 20 30 50 100 \
      --n_epochs 50 \
      --device cuda:0 \
      --save_dir out/al_curve
"""

import argparse, os, sys, csv, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image
import glob

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from flim_al.acquisition import entropy_score


# ── Dataset ──────────────────────────────────────────────────────────────────

class SaliencyMaskDataset(Dataset):
    """
    Pares (saliency_map, gt_mask) com mesmo filename.
    saliency_dir : saliency maps do decoder (ex: labeled_marker/layer_3)
    mask_dir     : GT masks (schistossoma-eggs/label/)
    """
    def __init__(self, saliency_dir, mask_dir, size=(256, 256)):
        self.sal_paths = sorted(glob.glob(os.path.join(saliency_dir, "*.png")))
        self.mask_dir  = mask_dir
        self.size      = size
        if not self.sal_paths:
            raise FileNotFoundError(f"Nenhum .png em {saliency_dir}")

    def __len__(self):
        return len(self.sal_paths)

    def __getitem__(self, idx):
        sal_path = self.sal_paths[idx]
        fname    = os.path.basename(sal_path)
        mask_path = os.path.join(self.mask_dir, fname)

        sal  = Image.open(sal_path).convert("L").resize(self.size)
        sal  = torch.tensor(np.array(sal), dtype=torch.float32).unsqueeze(0) / 255.0

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L").resize(self.size)
            mask = (torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) > 127).float()
        else:
            mask = torch.zeros(1, *self.size)

        return sal, mask, idx, fname


# ── Model: refinamento sobre saliency maps ────────────────────────────────────

class SaliencyRefiner(nn.Module):
    """
    Pequena UNet-like que refina saliency maps usando GT masks.
    Entrada: saliency map 1-ch  Saída: máscara binária 1-ch
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        e = self.enc(x)
        m = self.mid(e)
        return self.dec(torch.cat([e, m], dim=1))


# ── Treino ────────────────────────────────────────────────────────────────────

def train_model(model, loader, n_epochs, device, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        for sal, mask, *_ in loader:
            sal, mask = sal.to(device), mask.to(device)
            pred = model(sal)
            loss = F.binary_cross_entropy(pred, mask)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ── Avaliação ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, fbs, maes = [], [], []
    for sal, mask, *_ in loader:
        sal, mask = sal.to(device), mask.to(device)
        pred = model(sal)
        pred_bin = (pred > 0.5).float()
        inter = (pred_bin * mask).sum(dim=[1,2,3])
        union = pred_bin.sum(dim=[1,2,3]) + mask.sum(dim=[1,2,3])
        dice  = (2 * inter / (union + 1e-8)).mean().item()
        tp = (pred_bin * mask).sum(dim=[1,2,3])
        fp = (pred_bin * (1-mask)).sum(dim=[1,2,3])
        fn = ((1-pred_bin) * mask).sum(dim=[1,2,3])
        pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
        fb = (2*pr*rc/(pr+rc+1e-8)).mean().item()
        mae = (pred - mask).abs().mean().item()
        dices.append(dice); fbs.append(fb); maes.append(mae)
    return np.mean(dices), np.mean(fbs), np.mean(maes)


# ── Scoring do pool (sem GT) ───────────────────────────────────────────────────

@torch.no_grad()
def score_pool(sal_tensors, device, batch_size=16):
    """Calcula entropy score de saliency maps diretamente (sem modelo extra)."""
    scores = []
    for i in range(0, len(sal_tensors), batch_size):
        batch = torch.stack(sal_tensors[i:i+batch_size]).to(device)
        s = entropy_score(batch)
        scores.extend(s.cpu().tolist())
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_home", default="/workspace/flim-python-demo/flim_ad/datasets/")
    p.add_argument("--markers",      default="schisto/user_A")
    p.add_argument("--splits",       nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--proxy_decoder",default="labeled_marker")
    p.add_argument("--proxy_layer",  type=int, default=3)
    p.add_argument("--budgets",      nargs="+", type=int, default=[5, 10, 20, 30, 50, 100])
    p.add_argument("--n_epochs",     type=int, default=50)
    p.add_argument("--n_seeds",      type=int, default=5,
                   help="Repetições random para intervalo de confiança")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--save_dir",     default="out/al_curve")
    p.add_argument("--image_size",   nargs=2, type=int, default=[256, 256])
    return p.parse_args()


def main():
    args  = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    size   = tuple(args.image_size)
    mask_dir = os.path.join(args.dataset_home, "schistossoma-eggs", "label")

    os.makedirs(args.save_dir, exist_ok=True)
    all_rows = []

    for split in args.splits:
        print(f"\n{'='*60}\nSplit {split}\n{'='*60}")

        proxy_dir = (
            f"out/saliencies/{args.markers}/test/split{split}"
            f"/{args.proxy_decoder}/layer_{args.proxy_layer}"
        )
        if not os.path.exists(proxy_dir):
            print(f"  proxy saliency dir not found: {proxy_dir} — skip")
            continue

        ds = SaliencyMaskDataset(proxy_dir, mask_dir, size=size)
        N  = len(ds)
        print(f"  Pool: {N} imagens")

        # ── Score todo o pool pelo entropy dos saliency maps ──────────────
        print("  Scoring pool por entropy...")
        sal_tensors = [ds[i][0] for i in range(N)]
        scores = score_pool(sal_tensors, device)
        al_ranking = sorted(range(N), key=lambda i: scores[i], reverse=True)
        print(f"  Top-5 mais incertos: {al_ranking[:5]}")

        # ── Loader de teste (todo o dataset) ─────────────────────────────
        test_loader = DataLoader(ds, batch_size=16, shuffle=False)

        budgets = [b for b in args.budgets if b <= N] + [N]
        budgets = sorted(set(budgets))

        for budget in budgets:
            print(f"\n  Budget={budget}")

            # -- AL: top-K por entropy
            al_idx = al_ranking[:budget]
            al_loader = DataLoader(Subset(ds, al_idx), batch_size=8,
                                   shuffle=True, drop_last=False)
            model_al = SaliencyRefiner().to(device)
            train_model(model_al, al_loader, args.n_epochs, device)
            dice_al, fb_al, mae_al = evaluate(model_al, test_loader, device)
            print(f"    AL:     DICE={dice_al:.3f}  Fβ={fb_al:.3f}  MAE={mae_al:.3f}")

            # -- Random: média de n_seeds runs
            rand_fbs, rand_dices, rand_maes = [], [], []
            for seed in range(args.n_seeds):
                random.seed(seed)
                rand_idx = random.sample(range(N), min(budget, N))
                r_loader = DataLoader(Subset(ds, rand_idx), batch_size=8,
                                      shuffle=True, drop_last=False)
                model_r = SaliencyRefiner().to(device)
                train_model(model_r, r_loader, args.n_epochs, device)
                d, f, m = evaluate(model_r, test_loader, device)
                rand_dices.append(d); rand_fbs.append(f); rand_maes.append(m)
            dice_r = np.mean(rand_dices); fb_r = np.mean(rand_fbs); mae_r = np.mean(rand_maes)
            print(f"    Random: DICE={dice_r:.3f}  Fβ={fb_r:.3f}  MAE={mae_r:.3f}  (avg {args.n_seeds} seeds)")

            all_rows.append({
                "split": split, "budget": budget,
                "al_dice": round(dice_al,4), "al_fb": round(fb_al,4), "al_mae": round(mae_al,4),
                "rand_dice": round(dice_r,4), "rand_fb": round(fb_r,4), "rand_mae": round(mae_r,4),
                "delta_fb": round(fb_al - fb_r, 4),
            })

    # ── Salva CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, f"{args.markers.replace('/','-')}_al_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nCurva AL salva em {csv_path}")

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Budget':>8} {'AL Fβ':>8} {'Rand Fβ':>8} {'ΔFβ':>8}")
    print(f"{'-'*36}")
    for r in all_rows:
        if r["split"] == all_rows[0]["split"]:  # mostra split 1
            print(f"{r['budget']:>8} {r['al_fb']:>8.3f} {r['rand_fb']:>8.3f} {r['delta_fb']:>+8.3f}")


if __name__ == "__main__":
    main()
