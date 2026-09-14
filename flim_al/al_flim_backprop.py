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

from pyflim import layers, data as flimdata
from flim_al.acquisition import entropy_score
from flim_al.losses import masked_dice_ce
from flim_al.coreset_badge import (
    _rgb_uint8_to_lab01,          # FIX-LAB: encoder FLIM treinado em espaço LAB
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
    init_seed: int = 0,
) -> tuple:
    """
    Treina o backprop_decoder nas imagens selecionadas.

    Fix em relação ao pyflim original:
    - zero_grad por epoch (batch gradient, como o design original pretendia)
    - Logits diretos à loss (sem relu/normalize antes)
      → elimina o double-sigmoid que bloqueia o gradiente

    FIX-SEED: `init_seed` semeia a inicialização Xavier do decoder. Sem isso,
    cada treino consumia a posição seguinte do stream global do torch, e o
    braço AL (1º treino) recebia uma inicialização diferente do braço Random
    (2º em diante). Medido no budget=N — onde os dois braços treinam com dados
    IDÊNTICOS — esse efeito sozinho movia o Fβ em até 0.109. Com o mesmo
    init_seed nos dois braços, o Δ isola a seleção de imagens.

    Retorna (n_train, weights_path) — n_train = imagens efetivamente usadas.
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

    os.makedirs(output_path, exist_ok=True)
    weights_path = output_path.rstrip("/") + f"/layer{target_layer}_weight.pth"

    decoder_weights = torch.empty(
        (1, model.layers[target_layer].conv.out_channels, 1, 1),
        device=device,
    ).requires_grad_(True)
    # FIX-SEED: inicialização reprodutível e pareada entre os braços AL/Random
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    torch.nn.init.xavier_uniform_(decoder_weights)

    # FIX-SIGMOID + FIX-MASK: masked_dice_ce(mask=None) é numericamente idêntico
    # a DiceCELoss(sigmoid=True) (ver tests/test_masked_dice_ce.py), e é a MESMA
    # função usada pelo treinador de região — assim `region_entropy` e `entropy`
    # diferem só pela máscara, não pela implementação da loss.
    optimizer = torch.optim.Adam([decoder_weights], lr=lr, weight_decay=wd)

    # Pré-computa features com encoder frozen.
    # FIX-LAB: usa LAB explícito (consistente com o treino do encoder FLIM original).
    features, labels_list = [], []
    with torch.no_grad():
        for fname in selected_fnames:
            orig_path  = os.path.join(orig_folder, fname)
            label_path = os.path.join(label_folder, fname)
            if not (os.path.exists(orig_path) and os.path.exists(label_path)):
                continue
            img_rgb = np.array(Image.open(orig_path).convert("RGB"), dtype=np.uint8)
            img_lab = _rgb_uint8_to_lab01(img_rgb)
            X = torch.tensor(img_lab.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

            mask_arr = np.array(Image.open(label_path).convert("L"), dtype=np.float32)
            Y = torch.tensor((mask_arr > 127).astype(np.float32)).unsqueeze(0).to(device)

            for l in range(model.architecture.nlayers):
                if not model.use_bias:
                    X = model.normalization(X, model.layers[l].normalization_parameters)
                X = model.layers[l].conv(X)
                X = model.layers[l].activation(X)
                X = model.layers[l].pool(X)
                if l == target_layer:
                    features.append(X.detach().clone())
                    break
            labels_list.append(Y)

    best_loss = np.inf
    for epoch in range(n_epochs):
        # zero_grad por epoch = batch gradient (como o design original do pyflim pretendia)
        optimizer.zero_grad()
        epoch_losses = []
        for x, y in zip(features, labels_list):
            # Logits diretos → masked_dice_ce aplica sigmoid internamente.
            # SEM relu/normalize antes da loss: elimina o double-sigmoid que trava o gradiente
            res = F.conv2d(x, decoder_weights, padding=0, stride=1)
            res = F.interpolate(res, [y.shape[-2], y.shape[-1]],
                                mode="bilinear", align_corners=True)
            loss = masked_dice_ce(res, y.unsqueeze(0), mask=None)
            epoch_losses.append(loss.item())
            loss.backward()     # acumula gradiente sobre o batch
        optimizer.step()        # 1 step por epoch

        mean_ep = float(np.mean(epoch_losses))
        if (epoch % 50 == 0):
            print(f"  epoch:{epoch:4d}  loss:{mean_ep:.4f}", end="\r")
        if mean_ep < best_loss:
            best_loss = mean_ep
            torch.save(decoder_weights.detach(), weights_path)

    print(f"  Final loss: {best_loss:.4f} (n_imgs={len(features)}, saved at best)")
    return len(features), weights_path


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
) -> tuple[float, float, float, float]:
    """
    Roda inferência replicando a lógica do backprop_decoder.

    FIX-LAB: usa LAB (consistente com encoder e com al_encoder_experiment.py).
    FIX-EMPTY: empty-empty → todas as métricas = 1.0 (idem ao paper).
    FIX-IOU: retorna (dice, fb, mae, iou) em vez de (dice, fb, mae).
    """
    model = torch.load(encoder_path, map_location=device, weights_only=False)
    model.device = device
    model.eval()

    decoder_weights = torch.load(weights_path, map_location=device, weights_only=True)

    dices, fbs, maes, ious = [], [], [], []
    eps = 1e-8; beta2 = 0.3

    for fname in eval_fnames:
        orig_path  = os.path.join(orig_folder,  fname)
        label_path = os.path.join(label_folder, fname)
        if not (os.path.exists(orig_path) and os.path.exists(label_path)):
            continue

        # FIX-LAB: encoder FLIM espera entrada em espaço LAB
        img_rgb = np.array(Image.open(orig_path).convert("RGB"), dtype=np.uint8)
        orig_h, orig_w = img_rgb.shape[0], img_rgb.shape[1]
        img_lab = _rgb_uint8_to_lab01(img_rgb)
        x = torch.tensor(img_lab.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

        # Forward FLIM encoder
        for l in range(model.architecture.nlayers):
            if not model.use_bias:
                x = model.normalization(x, model.layers[l].normalization_parameters)
            x = model.layers[l].conv(x)
            x = model.layers[l].activation(x)
            x = model.layers[l].pool(x)
            if l == target_layer:
                break

        # Decoder: logits → sigmoid
        pred = F.conv2d(x, decoder_weights, padding=0, stride=1)
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=True)

        mask_arr = (np.array(Image.open(label_path).convert("L"), dtype=np.float32) > 127)
        gt_bin   = mask_arr.astype(bool)
        pred_bin = (pred.squeeze().cpu().numpy() > 0.5)

        gt_sum   = int(gt_bin.sum())
        pred_sum = int(pred_bin.sum())

        # FIX-EMPTY: ambas vazias → acerto perfeito (consistente com al_encoder_experiment.py)
        if gt_sum == 0 and pred_sum == 0:
            dices.append(1.0); fbs.append(1.0); ious.append(1.0); maes.append(0.0)
            continue

        tp  = float((pred_bin & gt_bin).sum())
        fp  = float((pred_bin & ~gt_bin).sum())
        fn  = float((~pred_bin & gt_bin).sum())
        pr  = tp / (tp + fp + eps)
        rc  = tp / (tp + fn + eps)

        dice = 2 * tp / (2 * tp + fp + fn + eps)
        fb   = (1 + beta2) * pr * rc / (beta2 * pr + rc + eps)
        iou  = tp / (tp + fp + fn + eps)
        mae  = float(np.abs(pred_bin.astype(float) - gt_bin.astype(float)).mean())

        dices.append(dice); fbs.append(fb); maes.append(mae); ious.append(iou)

    if not dices:
        return 0.0, 0.0, 1.0, 0.0
    return (float(np.mean(dices)), float(np.mean(fbs)),
            float(np.mean(maes)),  float(np.mean(ious)))


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
    sal_dir: str = "",            # FIX-BADGE: pred_scores via saliency means
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
            feats   = extract_encoder_features(encoder, img_paths, proxy_layer, device)
            sel_idx = coreset_select(feats, budget)
        else:  # badge
            # FIX-BADGE: saliency mean como pred_scores (evita zero-gradient com dummy_w=0)
            # dummy_w=zeros → gradient_embedding=0 → seleção arbitrária.
            # Saliency mean ≈ probabilidade média de foreground (proxy válido).
            sal_means = []
            for f in fnames:
                sal_path = os.path.join(sal_dir, f) if sal_dir else ""
                if sal_path and os.path.exists(sal_path):
                    arr = np.array(Image.open(sal_path).convert("L"), dtype=np.float32) / 255.0
                    sal_means.append(float(arr.mean()))
                else:
                    sal_means.append(0.5)
            feats   = extract_encoder_features(encoder, img_paths, proxy_layer, device)
            sel_idx = badge_select(feats, np.array(sal_means, dtype=np.float32), budget)

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
    p.add_argument("--n_seeds",       type=int, default=5,
                   help="Repetições por braço. AL e Random usam as MESMAS "
                        "sementes de inicialização (comparação pareada).")
    p.add_argument("--skip_full_pool", action="store_true",
                   help="Não rodar a referência do pool inteiro (é ~90%% do "
                        "custo em CPU). Sem ela não há teto nem piso de ruído.")
    p.add_argument("--full_pool_seeds", type=int, default=3,
                   help="Repetições da referência do pool inteiro. Cada uma "
                        "custa ~20x um budget K=10, então 3 já dá um desvio "
                        "utilizável sem dominar o tempo total.")
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


# ── CSV por execução ──────────────────────────────────────────────────────────
# Uma linha por (split, budget, arm, seed). A agregação (média ± desvio, teste
# pareado) fica em analyze_dissertation.py — assim nenhum dado bruto se perde e
# qualquer análise posterior pode ser refeita sem re-executar nada.
_FIELDNAMES = [
    "split", "budget", "arm", "seed", "acquisition",
    "n_train_imgs", "coverage", "fb", "dice", "mae", "iou", "collapsed",
]

_COLLAPSE_FB = 0.4788   # predição toda-fundo no val set do schisto


def _is_collapsed(fb: float, dice: float, iou: float) -> int:
    """
    Predição degenerada: o decoder prevê fundo em tudo. Nesse caso Fβ, DICE e
    IoU convergem para a mesma constante (fração de imagens vazias no val set),
    porque só as imagens vazias pontuam (empty-empty → 1.0) e as demais dão 0.
    Marcar isso é essencial: uma config degenerada pode "vencer" o random sem
    que o modelo tenha aprendido nada.
    """
    return int(
        abs(fb - _COLLAPSE_FB) < 0.005
        and abs(fb - dice) < 0.01
        and abs(fb - iou) < 0.01
    )


def _load_runs(csv_path: str) -> tuple[list[dict], set]:
    rows, done = [], set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add((r["split"], r["budget"], r["arm"], r["seed"]))
        print(f"  [resume] {len(rows)} execuções já no CSV")
    return rows, done


def _append_run(csv_path: str, row: dict) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Acquisition: {args.acquisition} | "
          f"seeds/braço: {args.n_seeds}")

    orig_folder  = os.path.join(args.dataset_home, "schistossoma-eggs", "orig")
    label_folder = os.path.join(args.dataset_home, "schistossoma-eggs", "label")
    os.makedirs(args.save_dir, exist_ok=True)

    tag      = args.markers.replace("/", "-")
    csv_path = os.path.join(args.save_dir, f"{tag}_{args.acquisition}_runs.csv")
    all_rows, done_keys = _load_runs(csv_path)

    def _run_and_log(arm, split, budget, seed, fnames_sel, out_dir, sal_dir,
                     enc_path, val_fnames):
        """Treina um braço com uma semente, avalia e grava a linha do CSV."""
        key = (str(split), str(budget), arm, str(seed))
        if key in done_keys:
            print(f"    [resume] {arm} seed={seed} — skip")
            return
        # FIX-SEED: mesma semente de inicialização nos braços AL e Random do
        # mesmo índice → o Δ isola a seleção de imagens, não o sorteio do Xavier.
        init_seed = 1000 + seed
        coverage = ""

        if args.acquisition == "region_entropy":
            weights, coverage, n_imgs = run_region_al(
                encoder_path=enc_path, selected_fnames=fnames_sel,
                orig_folder=orig_folder, label_folder=label_folder,
                saliency_folder=sal_dir, target_layer=args.target_layer,
                output_path=out_dir, n_epochs=args.n_epochs, device=device,
                patch_size=args.patch_size, top_k_patches=args.top_patches,
                init_seed=init_seed,
            )
            coverage = round(coverage, 4)
        else:
            n_imgs, weights = train_backprop_on_subset(
                enc_path, fnames_sel, orig_folder, label_folder,
                args.target_layer, out_dir, args.n_epochs, device,
                init_seed=init_seed,
            )

        dice, fb, mae, iou = evaluate_backprop(
            enc_path, weights, val_fnames,
            orig_folder, label_folder, args.target_layer, device
        )
        col = _is_collapsed(fb, dice, iou)
        flag = "  [COLAPSADO]" if col else ""
        print(f"    {arm:<6} seed={seed}  Fβ={fb:.4f}  IoU={iou:.4f}  "
              f"n_imgs={n_imgs}{flag}")

        row = {
            "split": split, "budget": budget, "arm": arm, "seed": seed,
            "acquisition": args.acquisition, "n_train_imgs": n_imgs,
            "coverage": coverage,
            "fb": round(fb, 4), "dice": round(dice, 4),
            "mae": round(mae, 4), "iou": round(iou, 4), "collapsed": col,
        }
        _append_run(csv_path, row)
        all_rows.append(row)
        done_keys.add(key)

    for split in args.splits:
        print(f"\n{'='*60}\nSplit {split}\n{'='*60}")

        sal_dir = (
            f"out/saliencies/{args.markers}/test/split{split}"
            f"/labeled_marker/layer_{args.proxy_layer}"
        )
        if not os.path.exists(sal_dir):
            print(f"  Saliency dir não encontrado: {sal_dir} — skip")
            continue

        enc_path = (
            f"out/trained_models/{args.markers}/split{split}"
            f"/flim_encoder_split{split}.pth"
        )
        if not os.path.exists(enc_path):
            print(f"  Encoder não encontrado: {enc_path} — skip")
            continue

        fnames, scores = score_saliencies(sal_dir, device)

        val_list_path = f"datasets/schistossoma-eggs/Splits-5train-70_30/split{split}-val.txt"
        if os.path.exists(val_list_path):
            with open(val_list_path) as f:
                val_fnames = [l.strip() for l in f if l.strip()]
        else:
            val_fnames = list(fnames)

        # FIX-B2: remover do val as imagens com markers reais (usadas para
        # treinar o encoder). Mesmo filtro do al_encoder_experiment.py — sem
        # ele os val sets do Exp A e do Exp B diferem e as tabelas não são
        # comparáveis entre si.
        marker_dir = os.path.join("data", args.markers, f"split{split}", "markers")
        if os.path.isdir(marker_dir):
            train_fnames = {
                f.replace("-seeds.txt", ".png")
                for f in os.listdir(marker_dir) if f.endswith("-seeds.txt")
            }
            n_before = len(val_fnames)
            val_fnames = [f for f in val_fnames if f not in train_fnames]
            if len(val_fnames) < n_before:
                print(f"  [fix-B2] Removidas {n_before - len(val_fnames)} "
                      f"imgs de treino do val set")
        print(f"  Val set: {len(val_fnames)} imagens")

        # Fix C2: pool ∩ val = ∅
        _val_set  = set(val_fnames)
        _n_before = len(fnames)
        _pairs    = [(f, s) for f, s in zip(fnames, scores) if f not in _val_set]
        fnames, scores = ([], [])
        if _pairs:
            f_, s_ = zip(*_pairs)
            fnames, scores = list(f_), list(s_)
        N = len(fnames)
        al_ranking = sorted(range(N), key=lambda i: scores[i], reverse=True)
        print(f"  Pool: {_n_before} → {N} após remover overlap com val")
        print(f"  top-3 mais incertas: {[fnames[i] for i in al_ranking[:3]]}")

        budgets = sorted({b for b in args.budgets if b <= N})

        for budget in budgets:
            print(f"\n  Budget={budget} ({budget / N * 100:.1f}%)")

            al_fnames = select_by_acquisition(
                args.acquisition, fnames, scores, al_ranking, budget,
                enc_path, orig_folder, args.proxy_layer, device, sal_dir=sal_dir,
            )

            for seed in range(args.n_seeds):
                # AL: conjunto de imagens FIXO, varia só a inicialização →
                # o desvio do braço AL mede o ruído de inicialização.
                _run_and_log(
                    "al", split, budget, seed, al_fnames,
                    os.path.join(args.save_dir, args.markers, f"split{split}",
                                 args.acquisition, f"budget{budget}", f"al{seed}"),
                    sal_dir, enc_path, val_fnames,
                )
                # Random: varia o sorteio das imagens E usa a MESMA
                # inicialização do par → Δ_seed isola a seleção.
                random.seed(seed)
                rand_fnames = random.sample(fnames, min(budget, N))
                _run_and_log(
                    "rand", split, budget, seed, rand_fnames,
                    os.path.join(args.save_dir, args.markers, f"split{split}",
                                 args.acquisition, f"budget{budget}", f"rand{seed}"),
                    sal_dir, enc_path, val_fnames,
                )

        # ── Referência do pool inteiro ────────────────────────────────────────
        # Em budget=N os dois braços treinariam com o MESMO conjunto (o pool
        # todo), então o Δ seria 0 por construção. Rodamos só um braço, com
        # n_seeds inicializações: dá o TETO do decoder e, no desvio, o piso de
        # ruído de inicialização — a régua contra a qual todo Δ deve ser lido.
        if not args.skip_full_pool:
            print(f"\n  [referência] Pool inteiro (N={N}) — teto + piso de ruído "
                  f"({args.full_pool_seeds} sementes)")
            for seed in range(args.full_pool_seeds):
                _run_and_log(
                    "full", split, N, seed, fnames,
                    os.path.join(args.save_dir, args.markers, f"split{split}",
                                 args.acquisition, "full_pool", f"s{seed}"),
                    sal_dir, enc_path, val_fnames,
                )

    if not all_rows:
        print("Nenhum resultado gerado.")
        return

    print(f"\n{'='*72}")
    print(f"CSV bruto: {csv_path}  ({len(all_rows)} execuções)")
    print("Rode  python3 ../flim_al/analyze_dissertation.py  para as tabelas agregadas.")
    print(f"{'='*72}")

    # Resumo rápido (a análise completa fica no analyze_dissertation.py)
    import statistics as _st
    print(f"{'Budget':>8} {'AL Fβ':>16} {'Rand Fβ':>16} {'ΔFβ pareado':>16}")
    print("-" * 60)
    for b in sorted({int(r["budget"]) for r in all_rows}):
        al = [float(r["fb"]) for r in all_rows if int(r["budget"]) == b and r["arm"] == "al"]
        rd = [float(r["fb"]) for r in all_rows if int(r["budget"]) == b and r["arm"] == "rand"]
        fl = [float(r["fb"]) for r in all_rows if int(r["budget"]) == b and r["arm"] == "full"]
        if fl and not al:
            sd = _st.pstdev(fl) if len(fl) > 1 else 0.0
            print(f"{b:>8} {'(pool inteiro)':>16} {_st.mean(fl):>8.4f}±{sd:.4f} "
                  f"{'—':>16}")
            continue
        if not al or not rd:
            continue
        sa = _st.pstdev(al) if len(al) > 1 else 0.0
        sr = _st.pstdev(rd) if len(rd) > 1 else 0.0
        print(f"{b:>8} {_st.mean(al):>8.4f}±{sa:.4f} {_st.mean(rd):>8.4f}±{sr:.4f} "
              f"{_st.mean(al) - _st.mean(rd):>+16.4f}")


if __name__ == "__main__":
    main()
