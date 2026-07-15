"""
al_encoder_experiment.py
=========================
Experimento 2 — Contribuição real para o FLIM:
"AL seleciona quais imagens o usuário deve anotar para treinar o encoder FLIM"

Pergunta científica:
  "Com quais imagens o encoder FLIM aprende melhor,
   dado que o usuário só pode anotar K imagens?"

Pipeline:
  1. Encoder atual (3 imagens) gera saliency maps do pool
  2. AL ranqueia pool por incerteza (entropy/coreset/badge)
  3. Para cada budget K:
       AL:     seleciona K imagens mais informativas
       Random: seleciona K imagens aleatoriamente (N seeds, média)
       → Gera markers sintéticos (simula usuário desenhando seeds)
       → Re-treina encoder FLIM com 3 originais + K selecionadas
       → Roda decoder(s) no val set
       → Se --use_dt: roda iftSMansoniDelineation → Fβ pixel dos masks DT
       → Se não:      Fβ pixel dos saliency com Otsu+AF
  4. Tabela: original(3 imgs) | AL-K | Random-K por decoder

Modo com DT (reprodução do paper):
  cd flim_ad
  python3 ../flim_al/al_encoder_experiment.py \\
      --markers schisto/user_A \\
      --splits 1 \\
      --budgets 3 5 10 \\
      --acquisition entropy \\
      --n_seeds 1 \\
      --device cpu \\
      --save_dir out/al_encoder_results \\
      --use_dt \\
      --dt_bin libs/ift/bin/iftSMansoniDelineation

Modo sem DT (rápido, pixel Fβ):
  python3 ../flim_al/al_encoder_experiment.py \\
      --markers schisto/user_A \\
      --splits 1 \\
      --budgets 3 5 10 \\
      --acquisition entropy \\
      --n_seeds 1 \\
      --device cpu \\
      --save_dir out/al_encoder_results

Referências:
  CoreSet  arXiv:1708.00489
  BADGE    arXiv:1906.03671
"""

import argparse, os, sys, csv, random, glob, shutil, tempfile, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def _labf(x):
    """Função auxiliar LAB — cópia EXATA de FLIMData._labf (data.py linha 95-99)."""
    if x >= 8.85645167903563082e-3:
        return x ** (0.33333333333)
    else:
        return (841.0 / 108.0) * x + (4.0 / 29.0)


def image_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem RGB uint8 para LAB normalizado — cópia EXATA de FLIMData._image_to_lab.
    Retorna array float32 (H, W, 3) no mesmo espaço de cor usado durante o treino FLIM.
    """
    img = image.astype(np.float32) / max(image.max(), 1)  # normaliza [0,1]
    labf_v = np.vectorize(_labf)
    new_image = np.zeros_like(img, dtype=np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    X = 0.4123955889674142161*R + 0.3575834307637148171*G + 0.1804926473817015735*B
    Y = 0.2125862307855955516*R + 0.7151703037034108499*G + 0.07220049864333622685*B
    Z = 0.01929721549174694484*R + 0.1191838645808485318*G + 0.9504971251315797660*B
    # Exato: cada canal tem seu próprio white point (data.py linhas 153-155)
    X = labf_v(X / 0.950456)
    Y = labf_v(Y / 1.0)
    Z = labf_v(Z / 1.088754)
    new_image[:, :, 0] = (116 * Y - 16) / 99.998337
    new_image[:, :, 1] = (500 * (X - Y) + 86.182236) / (86.182236 + 98.258614)
    new_image[:, :, 2] = (200 * (Y - Z) + 107.867744) / (107.867744 + 94.481682)
    return new_image


def threshold_otsu(img: np.ndarray) -> float:
    """Otsu's threshold sem skimage."""
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
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

REPO   = Path(__file__).resolve().parent.parent
FLIMPY = REPO / "flim_ad" / "libs" / "flim-python"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FLIMPY))

from pyflim import layers, flim as flimlib, data as flimdata, arch as flimarch
from flim_al.acquisition import entropy_score
from flim_al.marker_generator import create_combined_marker_dir
from flim_al.coreset_badge import (
    extract_encoder_features, extract_encoder_features_and_preds,
    coreset_select, badge_select,
)


# ── Decoders que avaliamos (não precisam de treino extra) ────────────────────

EVAL_DECODERS = [
    ("labeled_marker",             3),   # (decoder_type, layer)
    ("vanilla_adaptive_decoder",   3),
    ("vanilla_adaptive_decoder_wt",3),
    ("decoder_2",                  3),
    ("decoder_3",                  3),
    ("decoder_attention",          3),
    ("hybrid_decoder",             3),
]


# ── Treino do encoder ─────────────────────────────────────────────────────────

def retrain_encoder(
    arch_file: str,
    marker_dir: str,          # diretório com seeds.txt (originais + sintéticos)
    orig_folder: str,
    label_folder: str,
    device: str,
    output_path: str,
    bits: int = 8,
) -> object:
    """
    Re-treina o encoder FLIM com as imagens cujos markers estão em marker_dir.
    Retorna o modelo treinado e salva em output_path.
    """
    images_list = [f.replace("-seeds.txt", "")
                   for f in os.listdir(marker_dir) if f.endswith("-seeds.txt")]

    print(f"  [encoder] Treinando com {len(images_list)} imagens: {images_list[:5]}...")

    # FLIMData usa concatenação de string (não os.path.join):
    # marker_folder + filename + marker_ext → precisa de trailing slash
    marker_dir_slash = marker_dir.rstrip("/") + "/"

    train_ds = flimdata.FLIMData(
        orig_folder=orig_folder,
        marker_folder=marker_dir_slash,
        images_list=images_list,
        label_folder=label_folder,
        orig_ext=".png",
        marker_ext="-seeds.txt",
        label_ext=".png",
        transform=flimdata.transforms.Compose([flimdata.ToTensor()]),
        bits=bits,
        convert_gray_to_lab=False,
    )

    architecture = flimarch.FLIMArchitecture(arch_file)
    model = flimlib.FLIMModel(
        architecture,
        device=device,
        decoder_type="vanilla_adaptive_decoder",
        adj_radius=1.5,
        multi_layer=False,
    )
    model.fit(train_ds)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model, output_path)
    print(f"  [encoder] Salvo em {output_path}")
    return model


# ── Avaliação de um decoder sobre o val set ──────────────────────────────────

# ── Dynamic Trees — helpers ──────────────────────────────────────────────────

def _ensure_dt_symlinks(dataset_folder: str) -> None:
    """
    iftSMansoniDelineation espera dataset_folder/images/ e /truelabels/.
    O dataset tem orig/ e label/ — cria symlinks se necessário.
    """
    imgs_link = os.path.join(dataset_folder, "images")
    lbl_link  = os.path.join(dataset_folder, "truelabels")
    if not os.path.exists(imgs_link):
        os.symlink(os.path.join(dataset_folder, "orig"), imgs_link)
    if not os.path.exists(lbl_link):
        os.symlink(os.path.join(dataset_folder, "label"), lbl_link)


def _run_dt(dt_bin: str, dataset_folder: str, sal_dir: str, out_dir: str,
            area_range: tuple[int, int] = (1000, 9000),
            border_dist: int = 8, seed_dil: int = 8, seed_ero: int = 8,
            thr: int = 128) -> str:
    """
    Chama iftSMansoniDelineation e retorna o diretório com as máscaras.
    O binário cria subdiretório masks/ dentro de out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f"{dt_bin} {dataset_folder} {sal_dir} 2 {out_dir} "
        f"{border_dist} {seed_dil} {seed_ero} "
        f"{area_range[0]} {area_range[1]} {thr}"
    )
    ret = os.system(cmd)
    if ret != 0:
        print(f"  [DT] aviso: binário retornou código {ret}")
    # binário cria masks/ dentro de out_dir
    masks_sub = os.path.join(out_dir, "masks")
    return masks_sub if os.path.isdir(masks_sub) else out_dir


def _fb_from_masks(masks_dir: str, val_fnames: list[str],
                   label_folder: str, beta2: float = 0.3) -> dict[str, float]:
    """
    Computa Fβ pixel-level (β²=0.3) e DICE a partir de masks binárias.
    Replica FLIMMetrics.fmeasure + dice_score de metrics.py.
    Imagens sem mask → predição vazia.
    """
    eps = 1e-8
    fbs, dices = [], []
    for fname in val_fnames:
        gt_path   = os.path.join(label_folder, fname)
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.exists(gt_path):
            continue
        gt_bin = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)
        if os.path.exists(mask_path):
            pred_bin = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
        else:
            pred_bin = np.zeros_like(gt_bin)
        # Fβ pixel
        if gt_bin.sum() == 0 and pred_bin.sum() == 0:
            fbs.append(1.0)
        else:
            tp = float(np.sum((pred_bin == 1) & (gt_bin == 1)))
            fp = float(np.sum((pred_bin == 1) & (gt_bin == 0)))
            fn = float(np.sum((pred_bin == 0) & (gt_bin == 1)))
            pr = tp / (tp + fp + eps); rc = tp / (tp + fn + eps)
            fbs.append((1 + beta2) * pr * rc / (beta2 * pr + rc + eps))
        # DICE
        gs, ps = gt_bin.sum(), pred_bin.sum()
        if gs == 0 and ps == 0:       dices.append(1.0)
        elif gs == 0 or ps == 0:      dices.append(0.0)
        else: dices.append(2.0 * float((gt_bin * pred_bin).sum()) / (gs + ps))
    return {
        "fb":   float(np.mean(fbs))   if fbs   else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "mae":  0.0,
    }


# ── Pipeline sem DT — helpers ─────────────────────────────────────────────────

def _official_filter_and_binarize(sal_uint8: np.ndarray,
                                   area_range: tuple[int, int] = (1000, 9000)) -> np.ndarray:
    """
    Replica EXATAMENTE o pipeline oficial de pós-processamento:
      1. Otsu (skimage) → binariza
      2. filter_component_by_area: mantém só componentes em [area_range]
      3. Retorna mapa binário uint8 {0,1}

    Equivalente a compute_metrics.py → filter_component_by_area(otsu=True)
    seguido de binarize_saliency no mapa filtrado.
    """
    from skimage.filters import threshold_otsu as _otsu
    from skimage import measure as _measure

    if sal_uint8.sum() == 0:
        return np.zeros_like(sal_uint8, dtype=np.uint8)

    thresh = _otsu(sal_uint8)
    bin_sal = (sal_uint8 > thresh).astype(np.uint8)  # {0,1}

    # Filtra por área de componente (pixels), igual ao filter_component_by_area
    labeled = _measure.label(bin_sal, background=0, connectivity=2)
    out = sal_uint8.copy()
    out[bin_sal == 0] = 0
    out[bin_sal > 0] = 255   # seta para 255 (igual ao oficial)
    for c in range(1, labeled.max() + 1):
        area = (labeled == c).sum()
        if area < area_range[0] or area > area_range[1]:
            out[labeled == c] = 0

    return (out > 0).astype(np.uint8)   # {0,1} final


def _official_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Cópia de FLIMMetrics.dice_score — retorna 1.0 quando ambos são vazios.
    Esta semântica é ESSENCIAL para comparar com o paper.
    """
    gt   = (gt   > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    gt_sum   = gt.sum()
    pred_sum = pred.sum()
    if gt_sum == 0 and pred_sum > 0:  return 0.0
    if gt_sum > 0 and pred_sum == 0:  return 0.0
    if gt_sum == 0 and pred_sum == 0: return 1.0   # ambos vazios → acerto perfeito
    inter = (gt * pred).sum()
    return float(2.0 * inter / (gt_sum + pred_sum))


@torch.no_grad()
def evaluate_decoder(
    encoder_path: str,
    decoder_type: str,
    target_layer: int,
    val_fnames: list[str],
    orig_folder: str,
    label_folder: str,
    device: str,
    area_range: tuple[int, int] = (1000, 9000),
    use_dt: bool = False,
    dt_bin: str | None = None,
    dataset_folder: str | None = None,
) -> dict[str, float]:
    """
    Avalia um decoder.

    Sem DT (use_dt=False):
      forward pass → saliency uint8 → Otsu+AF[area_range] → Fβ pixel (β²=0.3)

    Com DT (use_dt=True):
      forward pass → saliency uint8 → salva em temp dir →
      iftSMansoniDelineation → masks/ → Fβ pixel (β²=0.3) dos masks DT
      (reproduz pipeline paper: OT → DT → Fβ)

    Sempre carrega em CPU para evitar device mismatch no pyflim.
    """
    eval_device = "cpu"

    model = torch.load(encoder_path, map_location=eval_device, weights_only=False)
    model.device = eval_device

    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            model.layers[l].marker_labels = ml.to(eval_device)

    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1,
        adaptation_function="robust_weights",
        filter_by_size=False,
        device=eval_device,
        adj_radius=1.5,
        decoder_type=decoder_type,
        multi_layer=False,
    )

    layer_idx = target_layer - 1

    # ── Fase 1: gera saliency maps para todas as imagens ──────────────────
    sal_tmp = tempfile.mkdtemp(prefix="flim_sal_")
    try:
        for fname in val_fnames:
            orig_path = os.path.join(orig_folder, fname)
            if not os.path.exists(orig_path):
                continue

            img     = Image.open(orig_path).convert("RGB")
            orig_h, orig_w = img.size[1], img.size[0]
            arr_lab = image_to_lab(np.array(img, dtype=np.uint8))
            x       = torch.tensor(arr_lab.transpose(2, 0, 1)).unsqueeze(0)

            y_hat, _ = model.forward(x, decoder_layer=[layer_idx])
            pred = y_hat[0].float()
            if pred.dim() == 3:
                pred = pred.unsqueeze(0)

            sal_uint8 = pred.squeeze().numpy().astype(np.uint8)

            if sal_uint8.shape[0] != orig_h or sal_uint8.shape[1] != orig_w:
                sal_uint8 = np.array(
                    Image.fromarray(sal_uint8).resize((orig_w, orig_h), Image.BILINEAR),
                    dtype=np.uint8,
                )

            Image.fromarray(sal_uint8).save(os.path.join(sal_tmp, fname))

        # ── Fase 2a: com DT ───────────────────────────────────────────────
        if use_dt and dt_bin and dataset_folder and os.path.isfile(dt_bin):
            _ensure_dt_symlinks(dataset_folder)
            dt_out_tmp = tempfile.mkdtemp(prefix="flim_dt_")
            try:
                masks_dir = _run_dt(
                    dt_bin, dataset_folder, sal_tmp, dt_out_tmp, area_range
                )
                n_masks = len([f for f in os.listdir(masks_dir)
                               if f.endswith(".png")]) if os.path.isdir(masks_dir) else 0
                print(f"      [DT] {n_masks} masks geradas")
                return _fb_from_masks(masks_dir, val_fnames, label_folder)
            finally:
                shutil.rmtree(dt_out_tmp, ignore_errors=True)

        # ── Fase 2b: sem DT — Otsu + AF inline ───────────────────────────
        dices, fbs, maes = [], [], []
        eps = 1e-8; beta2 = 0.3

        for fname in val_fnames:
            sal_path   = os.path.join(sal_tmp, fname)
            label_path = os.path.join(label_folder, fname)
            if not (os.path.exists(sal_path) and os.path.exists(label_path)):
                continue

            sal_uint8 = np.array(Image.open(sal_path))
            pred_bin  = _official_filter_and_binarize(sal_uint8, area_range)
            gt_bin    = (np.array(Image.open(label_path).convert("L")) > 0).astype(np.uint8)

            dice = _official_dice(gt_bin, pred_bin)
            tp = float((pred_bin & gt_bin).sum())
            fp = float((pred_bin & ~gt_bin.astype(bool)).sum())
            fn = float((~pred_bin.astype(bool) & gt_bin).sum())
            pr = tp / (tp + fp + eps); rc = tp / (tp + fn + eps)
            fb = (1 + beta2) * pr * rc / (beta2 * pr + rc + eps)
            mae = float(np.abs(pred_bin.astype(float) - gt_bin.astype(float)).mean())

            fbs.append(fb); dices.append(dice); maes.append(mae)

        return {
            "fb":   float(np.mean(fbs))   if fbs   else 0.0,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "mae":  float(np.mean(maes))  if maes  else 1.0,
        }

    finally:
        shutil.rmtree(sal_tmp, ignore_errors=True)


# ── Avaliação de todos os decoders ───────────────────────────────────────────

def evaluate_all_decoders(
    encoder_path: str,
    val_fnames: list[str],
    orig_folder: str,
    label_folder: str,
    device: str,
    use_dt: bool = False,
    dt_bin: str | None = None,
    dataset_folder: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Avalia todos os decoders da lista EVAL_DECODERS sobre o val set.

    Com use_dt=True: avalia apenas labeled_marker (DT exige ~2 min/decoder;
    avaliamos o decoder principal do paper para manter tempo razoável).
    Retorna dict: {decoder_type: {fb, dice, mae}}
    """
    decoders_to_eval = (
        [("labeled_marker", 3)] if use_dt else EVAL_DECODERS
    )
    results = {}
    for decoder_type, layer in decoders_to_eval:
        label = "DT" if use_dt else "Otsu+AF"
        print(f"  [eval/{label}] {decoder_type} layer={layer}...", end=" ")
        try:
            m = evaluate_decoder(
                encoder_path, decoder_type, layer,
                val_fnames, orig_folder, label_folder, device,
                use_dt=use_dt, dt_bin=dt_bin, dataset_folder=dataset_folder,
            )
            results[decoder_type] = m
            print(f"Fβ={m['fb']:.3f}  DICE={m['dice']:.3f}")
        except Exception as e:
            print(f"ERRO: {e}")
            traceback.print_exc()
            results[decoder_type] = {"fb": 0.0, "dice": 0.0, "mae": 1.0}
    return results


# ── Seleção por método de aquisição ──────────────────────────────────────────

def score_saliencies(sal_dir: str, device: str):
    paths  = sorted(glob.glob(os.path.join(sal_dir, "*.png")))
    fnames = [os.path.basename(p) for p in paths]
    scores = []
    for i in range(0, len(paths), 32):
        tensors = []
        for p in paths[i:i+32]:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            tensors.append(torch.tensor(arr).unsqueeze(0).unsqueeze(0))
        batch = torch.cat(tensors, dim=0).to(device)
        scores.extend(entropy_score(batch).cpu().tolist())
    return fnames, scores


def select_images(
    acquisition: str,
    fnames: list[str],
    al_ranking: list[int],
    budget: int,
    enc_path: str,
    orig_folder: str,
    proxy_layer: int,
    device: str,
) -> list[str]:
    if acquisition == "entropy":
        return [fnames[i] for i in al_ranking[:budget]]

    encoder = torch.load(enc_path, map_location=device, weights_only=False)
    encoder.eval()
    img_paths = [os.path.join(orig_folder, f) for f in fnames]

    if acquisition == "coreset":
        feats   = extract_encoder_features(encoder, img_paths, proxy_layer, device)
        sel_idx = coreset_select(feats, budget)
    else:  # badge
        dummy_w = torch.zeros(
            (1, encoder.layers[proxy_layer].conv.out_channels, 1, 1), device=device
        )
        feats, preds = extract_encoder_features_and_preds(
            encoder, img_paths, dummy_w, proxy_layer, device
        )
        sel_idx = badge_select(feats, preds, budget)

    return [fnames[i] for i in sel_idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_home", default="datasets")
    p.add_argument("--markers",      default="schisto/user_A")
    p.add_argument("--splits",       nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--proxy_layer",  type=int, default=3)
    p.add_argument("--budgets",      nargs="+", type=int, default=[3, 5, 10, 20])
    p.add_argument("--n_seeds",      type=int, default=3,
                   help="Número de seeds aleatórias para o baseline Random")
    p.add_argument("--n_fg_markers", type=int, default=100,
                   help="Seeds de foreground por imagem sintética")
    p.add_argument("--n_bg_markers", type=int, default=300,
                   help="Seeds de background por imagem sintética")
    p.add_argument("--acquisition",  default="entropy",
                   choices=["entropy", "coreset", "badge"])
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--save_dir",     default="out/al_encoder_results")
    # Dynamic Trees
    p.add_argument("--use_dt",  action="store_true",
                   help="Usa iftSMansoniDelineation após saliency (reproduz paper)")
    p.add_argument("--dt_bin",  default="libs/ift/bin/iftSMansoniDelineation",
                   help="Caminho para o binário iftSMansoniDelineation")
    return p.parse_args()


FIELDNAMES = ["split", "budget", "method", "acquisition", "decoder", "fb", "dice", "mae"]


def _load_csv(csv_path: str) -> tuple[list[dict], set]:
    """Carrega CSV existente e retorna (rows, done_keys)."""
    rows = []
    done = set()
    if not os.path.exists(csv_path):
        return rows, done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            done.add((row["split"], row["budget"], row["method"], row["decoder"]))
    print(f"  [resume] {len(rows)} linhas já no CSV, {len(done)} combinações concluídas")
    return rows, done


def _append_csv(csv_path: str, new_rows: list[dict]) -> None:
    """Append incremental ao CSV (cria header se necessário)."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    mode   = "DT" if args.use_dt else "Otsu+AF"
    print(f"Device: {device} | Acquisition: {args.acquisition} | Eval: {mode}")

    dataset_folder = os.path.join(args.dataset_home, "schistossoma-eggs")
    orig_folder    = os.path.join(dataset_folder, "orig")
    label_folder   = os.path.join(dataset_folder, "label")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_dt:
        _ensure_dt_symlinks(dataset_folder)
        if not os.path.isfile(args.dt_bin):
            print(f"ERRO: DT binário não encontrado: {args.dt_bin}")
            sys.exit(1)
        print(f"DT binário: {args.dt_bin}")

    # CSV incremental — definido no início para suportar resume
    tag      = args.markers.replace("/", "-")
    csv_path = os.path.join(args.save_dir, f"{tag}_{args.acquisition}_encoder_al.csv")
    all_rows, done_keys = _load_csv(csv_path)

    def _done(split, budget, method, decoder):
        return (str(split), str(budget), method, decoder) in done_keys

    def _save(new_rows):
        """Salva incrementalmente e adiciona à lista global."""
        all_rows.extend(new_rows)
        _append_csv(csv_path, new_rows)
        for r in new_rows:
            done_keys.add((str(r["split"]), str(r["budget"]), r["method"], r["decoder"]))

    for split in args.splits:
        print(f"\n{'='*60}\nSplit {split}\n{'='*60}")

        sal_dir  = (f"out/saliencies/{args.markers}/test/split{split}"
                    f"/labeled_marker/layer_{args.proxy_layer}")
        arch_file = f"data/{args.markers}/split{split}/arch2D.json"
        orig_marker_dir = f"data/{args.markers}/split{split}/markers"
        enc_path = (f"out/trained_models/{args.markers}/split{split}"
                    f"/flim_encoder_split{split}.pth")

        if not all(os.path.exists(p) for p in [sal_dir, arch_file, orig_marker_dir, enc_path]):
            for p, n in [(sal_dir,"saliency"),(arch_file,"arch"),(orig_marker_dir,"markers"),(enc_path,"encoder")]:
                if not os.path.exists(p):
                    print(f"  Não encontrado: {n} ({p}) — skip split {split}")
            continue

        fnames, scores = score_saliencies(sal_dir, device)
        N = len(fnames)
        al_ranking = sorted(range(N), key=lambda i: scores[i], reverse=True)
        print(f"  Pool: {N} imagens | top-3: {[fnames[i] for i in al_ranking[:3]]}")

        val_list = f"datasets/schistossoma-eggs/Splits-5train-70_30/split{split}-val.txt"
        if os.path.exists(val_list):
            with open(val_list) as f:
                val_fnames = [l.strip() for l in f if l.strip()]
        else:
            val_fnames = fnames[:200]
        print(f"  Val: {len(val_fnames)} imagens")

        # ── Baseline: encoder original ──────────────────────────────────────
        first_dec = "labeled_marker" if args.use_dt else EVAL_DECODERS[0][0]
        if not _done(split, 3, "original_3imgs", first_dec):
            print("\n  [baseline] Encoder original (3 imagens)...")
            base_results = evaluate_all_decoders(
                enc_path, val_fnames, orig_folder, label_folder, device,
                use_dt=args.use_dt, dt_bin=args.dt_bin,
                dataset_folder=dataset_folder,
            )
            _save([{
                "split": split, "budget": 3, "method": "original_3imgs",
                "acquisition": "none", "decoder": dec,
                **{k: round(v, 4) for k, v in m.items()}
            } for dec, m in base_results.items()])
        else:
            print("  [resume] baseline já avaliado — skip")

        # ── AL e Random para cada budget ────────────────────────────────────
        budgets = sorted(set([b for b in args.budgets if b <= N]))

        for budget in budgets:
            print(f"\n  Budget = {budget}")
            work_dir = tempfile.mkdtemp(prefix=f"flim_al_split{split}_K{budget}_")

            try:
                al_method = f"al_{args.acquisition}"

                # ── AL ──────────────────────────────────────────────────────
                al_enc_path = os.path.join(
                    args.save_dir, args.markers, f"split{split}",
                    args.acquisition, f"budget{budget}", "encoder.pth"
                )
                os.makedirs(os.path.dirname(al_enc_path), exist_ok=True)

                if not _done(split, budget, al_method, first_dec):
                    al_fnames = select_images(
                        args.acquisition, fnames, al_ranking, budget,
                        enc_path, orig_folder, args.proxy_layer, device
                    )
                    print(f"    AL  selecionadas: {al_fnames[:3]}...")

                    al_marker_dir = os.path.join(work_dir, "al_markers")
                    create_combined_marker_dir(
                        orig_marker_dir, al_fnames, label_folder, al_marker_dir,
                        n_fg=args.n_fg_markers, n_bg=args.n_bg_markers,
                    )

                    # Reutiliza encoder salvo se já existir
                    if not os.path.exists(al_enc_path):
                        retrain_encoder(arch_file, al_marker_dir,
                                        orig_folder, label_folder, device, al_enc_path)
                    else:
                        print(f"    [resume] encoder AL já existe — reutilizando")

                    al_results = evaluate_all_decoders(
                        al_enc_path, val_fnames, orig_folder, label_folder, device,
                        use_dt=args.use_dt, dt_bin=args.dt_bin,
                        dataset_folder=dataset_folder,
                    )
                    al_rows = []
                    for dec, m in al_results.items():
                        print(f"    AL  {dec[:20]:20s}  Fβ={m['fb']:.3f}")
                        al_rows.append({
                            "split": split, "budget": budget, "method": al_method,
                            "acquisition": args.acquisition, "decoder": dec,
                            **{k: round(v, 4) for k, v in m.items()}
                        })
                    _save(al_rows)
                else:
                    print(f"    [resume] AL budget={budget} já avaliado — skip")

                # ── Random ──────────────────────────────────────────────────
                rand_results_acc = {dec: {"fb": [], "dice": [], "mae": []}
                                    for dec, _ in EVAL_DECODERS}
                rand_method = "random"

                for rs in range(args.n_seeds):
                    rand_enc_path = os.path.join(
                        args.save_dir, args.markers, f"split{split}",
                        args.acquisition, f"budget{budget}", f"rand{rs}", "encoder.pth"
                    )
                    os.makedirs(os.path.dirname(rand_enc_path), exist_ok=True)

                    # Chave de resume para random: usamos method=f"random_seed{rs}"
                    rand_seed_method = f"random_seed{rs}"
                    if not _done(split, budget, rand_seed_method, first_dec):
                        random.seed(rs)
                        rand_fnames = random.sample(fnames, min(budget, N))

                        rand_marker_dir = os.path.join(work_dir, f"rand{rs}_markers")
                        create_combined_marker_dir(
                            orig_marker_dir, rand_fnames, label_folder, rand_marker_dir,
                            n_fg=args.n_fg_markers, n_bg=args.n_bg_markers,
                        )

                        if not os.path.exists(rand_enc_path):
                            retrain_encoder(arch_file, rand_marker_dir,
                                            orig_folder, label_folder, device, rand_enc_path)
                        else:
                            print(f"    [resume] encoder rand{rs} já existe — reutilizando")

                        r = evaluate_all_decoders(
                            rand_enc_path, val_fnames, orig_folder, label_folder, device,
                            use_dt=args.use_dt, dt_bin=args.dt_bin,
                            dataset_folder=dataset_folder,
                        )
                        # Salva resultado individual de cada seed (para resume granular)
                        seed_rows = [{
                            "split": split, "budget": budget,
                            "method": rand_seed_method,
                            "acquisition": "random", "decoder": dec,
                            **{k: round(v, 4) for k, v in m.items()}
                        } for dec, m in r.items()]
                        _save(seed_rows)
                    else:
                        print(f"    [resume] rand{rs} budget={budget} já avaliado — skip")
                        # Carrega do CSV para acumular média
                        r = {row["decoder"]: {"fb": float(row["fb"]),
                                              "dice": float(row["dice"]),
                                              "mae": float(row["mae"])}
                             for row in all_rows
                             if (str(row["split"]) == str(split) and
                                 str(row["budget"]) == str(budget) and
                                 row["method"] == rand_seed_method)}

                    for dec, m in r.items():
                        for k, v in m.items():
                            rand_results_acc[dec][k].append(v)

                # Salva média das seeds (se ainda não foi salvo)
                if not _done(split, budget, rand_method, first_dec):
                    rand_avg_rows = []
                    for dec, acc in rand_results_acc.items():
                        if acc["fb"]:
                            m = {k: float(np.mean(v)) for k, v in acc.items()}
                            print(f"    Rand {dec[:20]:20s}  Fβ={m['fb']:.3f}  (avg {args.n_seeds} seeds)")
                            rand_avg_rows.append({
                                "split": split, "budget": budget,
                                "method": rand_method,
                                "acquisition": "random", "decoder": dec,
                                **{k: round(v, 4) for k, v in m.items()}
                            })
                    _save(rand_avg_rows)

            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nCSV salvo: {csv_path}  ({len(all_rows)} linhas)")

    # ── Tabela resumo ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Fβ médio (splits) por decoder e método")
    print(f"{'Decoder':30s} {'Original':>10} {'AL':>10} {'Random':>10} {'Δ(AL-Rand)':>12}")
    print("-" * 70)

    decoders = [d for d, _ in EVAL_DECODERS]
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        grouped[row["decoder"]][row["method"]].append(float(row["fb"]))

    for dec in decoders:
        orig = np.mean(grouped[dec].get("original_3imgs", [0]))
        al   = np.mean(grouped[dec].get(f"al_{args.acquisition}", [0]))
        rand = np.mean(grouped[dec].get("random", [0]))
        delta = al - rand
        marker = "✓" if al > rand else " "
        print(f"{dec:30s} {orig:>10.3f} {al:>10.3f} {rand:>10.3f} {delta:>+12.3f} {marker}")


if __name__ == "__main__":
    main()
