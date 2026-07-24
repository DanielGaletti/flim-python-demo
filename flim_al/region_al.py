"""
region_al.py — Region-based Active Learning para FLIM
======================================================
Em vez de selecionar IMAGENS inteiras, seleciona REGIOES (superpixels)
incertas dentro das imagens para apresentar ao especialista.

Pipeline:
  1. Saliency map do encoder atual -> incerteza por pixel
  2. SLIC superpixels na imagem original -> ~300 regioes compactas
  3. Score por regiao = entropia media dos pixels da regiao
  4. Top-K regioes = candidatas para consulta ao especialista
  5. Especialista responde: "ovo" ou "fundo" para cada regiao
     (simulado com GT masks — overlap > threshold -> ovo)
  6. Resposta -> seeds.txt FLIM (pontos fg/bg na regiao)
  7. Re-treino encoder + avaliacao com DT -> Fb e IoU

Formato seeds.txt FLIM (replicado de marker_generator.py):
  Linha 1: n_seeds H W
  Linhas seguintes: col row -1 class 0
    class = 1 -> foreground (ovo)
    class = 0 -> background (fundo)

Por que faz sentido para FLIM/Schistosoma?
  - Especialista nao precisa anotar a imagem toda, so responder K perguntas simples
  - Foca nas bordas de ovos (onde o encoder e mais incerto)
  - Gera seeds FLIM mais precisos — so nas regioes relevantes

Metricas:
  - Fb (b2=0.3): compativel com experimentos anteriores
  - IoU = TP/(TP+FP+FN): complementar ao Fb

Uso standalone:
  python3 -m flim_al.region_al \\
      --image datasets/schistossoma-eggs/orig/000002.png \\
      --saliency out/saliencies/.../000002.png \\
      --gt datasets/schistossoma-eggs/label/000002.png \\
      --n_superpixels 300 \\
      --budget 20 \\
      --out_seeds /tmp/region_seeds

Uso no loop AL (al_encoder_experiment.py):
  --acquisition region_entropy  ->  selecao por entropia com seeds de regioes
"""

from __future__ import annotations

import os
import random
import shutil
import numpy as np
from PIL import Image
from typing import Optional


# --- Dependencia: scikit-image (SLIC) ----------------------------------------

def _require_skimage():
    try:
        from skimage.segmentation import slic  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "scikit-image e necessario para region_al. "
            "Instale com: pip install scikit-image --break-system-packages"
        )


# --- Metricas ----------------------------------------------------------------

def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    IoU = TP / (TP + FP + FN).
    Retorna 1.0 quando ambos sao vazios (sem ovos na imagem).
    """
    pred = (pred_bin > 0).astype(bool)
    gt   = (gt_bin   > 0).astype(bool)

    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    return tp / (tp + fp + fn)


def fb_score(pred_bin: np.ndarray, gt_bin: np.ndarray, beta2: float = 0.3) -> float:
    """Fb pixel-level (b2=0.3, padrao FLIM-AD)."""
    pred = (pred_bin > 0).astype(bool)
    gt   = (gt_bin   > 0).astype(bool)

    if not pred.any() and not gt.any():
        return 1.0

    eps = 1e-8
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    pr = tp / (tp + fp + eps)
    rc = tp / (tp + fn + eps)
    return (1 + beta2) * pr * rc / (beta2 * pr + rc + eps)


# --- Superpixels -------------------------------------------------------------

def compute_superpixels(
    image: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    SLIC superpixels na imagem RGB.

    Returns
    -------
    labels : np.ndarray int [H, W]
    """
    from skimage.segmentation import slic

    seg = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        convert2lab=True,
    )
    return seg.astype(np.int32)


# --- Scoring de regioes ------------------------------------------------------

def score_regions_by_entropy(
    superpixels: np.ndarray,
    saliency: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """
    Para cada superpixel, calcula a entropia media dos pixels da regiao.
    Regioes com alta entropia = encoder incerto = candidatas AL.
    """
    sal = np.clip(saliency.astype(np.float64), eps, 1 - eps)
    h   = -(sal * np.log(sal) + (1 - sal) * np.log(1 - sal))

    scores = {}
    for region_id in np.unique(superpixels):
        mask = superpixels == region_id
        scores[int(region_id)] = float(h[mask].mean())

    return scores


def score_regions_by_margin(
    superpixels: np.ndarray,
    saliency: np.ndarray,
) -> dict:
    """Margin sampling por regiao: 1 - |2p - 1| medio."""
    sal = saliency.astype(np.float64)
    margin = 1.0 - np.abs(2 * sal - 1)

    scores = {}
    for region_id in np.unique(superpixels):
        mask = superpixels == region_id
        scores[int(region_id)] = float(margin[mask].mean())

    return scores


def select_top_regions(scores: dict, budget: int) -> list:
    """Retorna os top-K region IDs por score decrescente."""
    ranked = sorted(scores.keys(), key=lambda r: scores[r], reverse=True)
    return ranked[:budget]


# --- Consulta ao especialista (simulada) -------------------------------------

def simulate_expert_label(
    region_mask: np.ndarray,
    gt_mask: np.ndarray,
    fg_threshold: float = 0.15,
) -> str:
    """
    Simula resposta do especialista para uma regiao.
    Se >fg_threshold dos pixels sao ovo (GT=1), retorna "fg". Senao "bg".
    """
    if region_mask.sum() == 0:
        return "bg"
    overlap = float((region_mask & gt_mask).sum()) / float(region_mask.sum())
    return "fg" if overlap >= fg_threshold else "bg"


# --- Seeds FLIM a partir de regioes anotadas ---------------------------------

def regions_to_flim_seeds(
    superpixels: np.ndarray,
    selected_regions: list,
    region_labels: dict,
    H: int,
    W: int,
    n_seeds_per_region: int = 20,
    seed: int = 42,
) -> dict:
    """
    Converte anotacoes de regioes em seeds no formato FLIM.

    Formato interno (retornado): dict com:
      fg_seeds: np.ndarray (M, 2) — [[col, row], ...]
      bg_seeds: np.ndarray (N, 2) — [[col, row], ...]
      H, W: dimensoes da imagem

    Compativel com marker_generator.save_markers().
    """
    rng = np.random.default_rng(seed)
    fg_list = []
    bg_list = []

    for region_id in selected_regions:
        label_str = region_labels.get(region_id, "bg")
        mask = superpixels == region_id
        rows, cols = np.where(mask)

        if len(rows) == 0:
            continue

        n = min(n_seeds_per_region, len(rows))
        idx = rng.choice(len(rows), size=n, replace=False)
        pts = [[int(cols[i]), int(rows[i])] for i in idx]  # [col, row]

        if label_str == "fg":
            fg_list.extend(pts)
        else:
            bg_list.extend(pts)

    return {
        "fg_seeds": np.array(fg_list, dtype=int) if fg_list else np.empty((0, 2), dtype=int),
        "bg_seeds": np.array(bg_list, dtype=int) if bg_list else np.empty((0, 2), dtype=int),
        "H": H,
        "W": W,
    }


def save_region_seeds(seeds_dict: dict, out_path: str) -> int:
    """
    Salva seeds no formato FLIM seeds.txt.

    Formato (identico a marker_generator.save_markers):
      n_seeds H W
      col row -1 class 0
        class = 1 -> foreground
        class = 0 -> background

    Returns n_total seeds escritos.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fg = seeds_dict["fg_seeds"]
    bg = seeds_dict["bg_seeds"]
    H  = seeds_dict["H"]
    W  = seeds_dict["W"]
    n_total = len(fg) + len(bg)

    with open(out_path, "w") as f:
        f.write(f"{n_total} {H} {W}\n")
        for col, row in fg:
            f.write(f"{col} {row} -1 1 0\n")
        for col, row in bg:
            f.write(f"{col} {row} -1 0 0\n")

    return n_total


# --- Pipeline completo por imagem --------------------------------------------

def region_al_select(
    image: np.ndarray,
    saliency: np.ndarray,
    gt_mask: np.ndarray,
    budget: int = 20,
    n_superpixels: int = 300,
    method: str = "entropy",
    n_seeds_per_region: int = 15,
    fg_threshold: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Pipeline completo de Region AL para uma imagem.

    Returns
    -------
    dict com:
      superpixels     : [H,W] int — labels SLIC
      scores          : dict {region_id: score}
      selected_regions: list[int]
      region_labels   : dict {id: "fg"|"bg"}
      flim_seeds      : dict compativel com save_region_seeds()
      n_fg_regions    : int
      n_bg_regions    : int
    """
    _require_skimage()

    H, W = image.shape[:2]
    gt_bin = (gt_mask > 0)

    superpixels = compute_superpixels(image, n_segments=n_superpixels)

    if method == "entropy":
        scores = score_regions_by_entropy(superpixels, saliency)
    elif method == "margin":
        scores = score_regions_by_margin(superpixels, saliency)
    elif method == "random":
        rng = random.Random(seed)
        all_ids = [int(x) for x in np.unique(superpixels)]
        rng.shuffle(all_ids)
        scores = {r: float(i) for i, r in enumerate(all_ids)}
    else:
        raise ValueError(f"method deve ser 'entropy', 'margin' ou 'random'. Got: {method}")

    selected = select_top_regions(scores, budget)

    labels = {}
    for region_id in selected:
        region_mask = superpixels == region_id
        labels[region_id] = simulate_expert_label(region_mask, gt_bin, fg_threshold)

    n_fg = sum(1 for v in labels.values() if v == "fg")
    n_bg = sum(1 for v in labels.values() if v == "bg")

    flim_seeds = regions_to_flim_seeds(
        superpixels, selected, labels, H, W,
        n_seeds_per_region=n_seeds_per_region,
        seed=seed,
    )

    return {
        "superpixels":      superpixels,
        "scores":           scores,
        "selected_regions": selected,
        "region_labels":    labels,
        "flim_seeds":       flim_seeds,
        "n_fg_regions":     n_fg,
        "n_bg_regions":     n_bg,
    }


# --- Gerador de markers com Region AL ----------------------------------------

def create_region_marker_dir(
    orig_folder: str,
    saliency_folder: str,
    gt_folder: str,
    image_ids: list,
    out_dir: str,
    budget_per_image: int = 20,
    n_superpixels: int = 300,
    method: str = "entropy",
    n_seeds_per_region: int = 15,
    fg_threshold: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Gera seeds FLIM para uma lista de imagens usando Region AL.
    Retorna out_dir com {img_id}-seeds.txt prontos para FLIMData.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    for img_id in image_ids:
        orig_path = os.path.join(orig_folder, f"{img_id}.png")
        sal_path  = os.path.join(saliency_folder, f"{img_id}.png")
        gt_path   = os.path.join(gt_folder, f"{img_id}.png")

        if not os.path.exists(orig_path):
            print(f"  [region_al] Nao encontrado: {orig_path}")
            continue

        image   = np.array(Image.open(orig_path).convert("RGB"))
        gt_mask = (
            np.array(Image.open(gt_path).convert("L"))
            if os.path.exists(gt_path)
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )

        if os.path.exists(sal_path):
            saliency = np.array(Image.open(sal_path).convert("L"), dtype=np.float32) / 255.0
        else:
            saliency = np.random.RandomState(seed).rand(*image.shape[:2]).astype(np.float32)

        result = region_al_select(
            image, saliency, gt_mask,
            budget=budget_per_image,
            n_superpixels=n_superpixels,
            method=method,
            n_seeds_per_region=n_seeds_per_region,
            fg_threshold=fg_threshold,
            seed=seed,
        )

        seeds_path = os.path.join(out_dir, f"{img_id}-seeds.txt")
        n_written = save_region_seeds(result["flim_seeds"], seeds_path)

        summary[img_id] = {
            "n_fg_regions": result["n_fg_regions"],
            "n_bg_regions": result["n_bg_regions"],
            "n_seeds":      n_written,
        }
        print(
            f"  [region_al] {img_id}: "
            f"{result['n_fg_regions']} fg + {result['n_bg_regions']} bg regioes "
            f"-> {n_written} seeds"
        )

    return summary


def create_combined_region_marker_dir(
    original_marker_dir: str,
    selected_fnames: list,
    gt_folder: str,
    orig_folder: str,
    saliency_folder: str,
    output_dir: str,
    budget_per_image: int = 20,
    n_superpixels: int = 300,
    n_seeds_per_region: int = 15,
    fg_threshold: float = 0.15,
    method: str = "entropy",
) -> str:
    """
    Combina markers originais (copiados) com markers Region AL para imagens selecionadas.
    Equivalente a create_combined_marker_dir, mas usa regioes incertas em vez de GT random.

    Parameters
    ----------
    original_marker_dir : dir com seeds.txt das 3 imagens originais
    selected_fnames     : fnames selecionadas pelo AL (ex: ["000002.png", ...])
    gt_folder           : GT masks para consulta simulada ao especialista
    orig_folder         : imagens originais para SLIC
    saliency_folder     : saliency maps do encoder atual
    output_dir          : onde salvar tudo

    Returns
    -------
    output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copia markers originais
    for fname in os.listdir(original_marker_dir):
        src = os.path.join(original_marker_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy2(src, dst)

    # Gera markers por regiao para cada imagem selecionada
    added = 0
    for fname in selected_fnames:
        img_id = fname.replace(".png", "")

        # Nao sobrescreve originais
        dst = os.path.join(output_dir, f"{img_id}-seeds.txt")
        if os.path.exists(dst):
            continue

        orig_path = os.path.join(orig_folder, f"{img_id}.png")
        sal_path  = os.path.join(saliency_folder, f"{img_id}.png")
        gt_path   = os.path.join(gt_folder, f"{img_id}.png")

        if not os.path.exists(orig_path):
            continue

        image   = np.array(Image.open(orig_path).convert("RGB"))
        gt_mask = (
            np.array(Image.open(gt_path).convert("L"))
            if os.path.exists(gt_path)
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )
        saliency = (
            np.array(Image.open(sal_path).convert("L"), dtype=np.float32) / 255.0
            if os.path.exists(sal_path)
            else np.random.rand(*image.shape[:2]).astype(np.float32)
        )

        result = region_al_select(
            image, saliency, gt_mask,
            budget=budget_per_image,
            n_superpixels=n_superpixels,
            method=method,
            n_seeds_per_region=n_seeds_per_region,
            fg_threshold=fg_threshold,
        )

        save_region_seeds(result["flim_seeds"], dst)
        added += 1
        print(
            f"  [region_al] {img_id}: "
            f"{result['n_fg_regions']}fg + {result['n_bg_regions']}bg regioes"
        )

    print(f"  [region_al] {added} imagens com seeds de regiao adicionados")
    return output_dir


# --- Visualizacao ------------------------------------------------------------

def visualize_region_selection(
    image: np.ndarray,
    superpixels: np.ndarray,
    selected_regions: list,
    region_labels: dict,
    saliency: np.ndarray,
    out_path: str,
) -> None:
    """
    Salva visualizacao: imagem com regioes coloridas + saliency map.
    Vermelho = fg (ovo selecionado), Azul = bg (fundo selecionado).
    """
    from skimage.segmentation import mark_boundaries

    img_f = image.astype(np.float32) / 255.0
    img_b = mark_boundaries(img_f, superpixels, color=(0.5, 0.5, 0.5), outline_color=None)

    overlay = img_b.copy()
    for region_id in selected_regions:
        mask = superpixels == region_id
        lbl  = region_labels.get(region_id, "bg")
        if lbl == "fg":
            overlay[mask, 0] = np.clip(overlay[mask, 0] + 0.4, 0, 1)
            overlay[mask, 1] = np.clip(overlay[mask, 1] - 0.1, 0, 1)
            overlay[mask, 2] = np.clip(overlay[mask, 2] - 0.1, 0, 1)
        else:
            overlay[mask, 2] = np.clip(overlay[mask, 2] + 0.4, 0, 1)
            overlay[mask, 0] = np.clip(overlay[mask, 0] - 0.1, 0, 1)
            overlay[mask, 1] = np.clip(overlay[mask, 1] - 0.1, 0, 1)

    sal_rgb = np.zeros((*saliency.shape, 3), dtype=np.float32)
    sal_rgb[:, :, 0] = saliency
    sal_rgb[:, :, 2] = 1 - saliency

    H, W = image.shape[:2]
    canvas = np.ones((H, W * 2 + 10, 3), dtype=np.float32)
    canvas[:, :W]    = overlay
    canvas[:, W+10:] = sal_rgb

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8)).save(out_path)


# --- BALD por regiao (comite de encoders) ------------------------------------

def score_regions_by_bald(
    superpixels: np.ndarray,
    saliency_maps: list,
    eps: float = 1e-6,
) -> dict:
    """
    BALD por regiao usando N saliency maps de um comite de encoders.

    BALD(regiao) = H(E[p]) - E[H(p)]
      H(E[p]) = entropia da predicao media do comite
      E[H(p)] = media das entropias individuais de cada encoder

    Alto BALD = encoders discordam nessa regiao = mais informativa para anotar.

    Por que isso e melhor que entropia simples?
      Entropia simples mede incerteza de UM encoder.
      BALD mede desacordo ENTRE encoders — captura verdadeira ambiguidade
      estrutural, nao apenas instabilidade de um modelo.

    Parameters
    ----------
    superpixels  : [H, W] int — labels SLIC
    saliency_maps: list de N arrays [H, W] float32 [0,1] — um por encoder
    """
    sals = [np.clip(s.astype(np.float64), eps, 1 - eps) for s in saliency_maps]
    mean_sal = np.mean(sals, axis=0)                               # E[p]
    h_mean   = -(mean_sal * np.log(mean_sal)
                 + (1 - mean_sal) * np.log(1 - mean_sal))         # H(E[p])
    hs       = [-(s * np.log(s) + (1 - s) * np.log(1 - s))
                for s in sals]
    mean_h   = np.mean(hs, axis=0)                                 # E[H(p)]
    bald_map = np.clip(h_mean - mean_h, 0, None)                  # BALD >= 0

    scores = {}
    for region_id in np.unique(superpixels):
        mask = superpixels == region_id
        scores[int(region_id)] = float(bald_map[mask].mean())
    return scores


def region_al_select_bald(
    image: np.ndarray,
    saliency_maps: list,
    gt_mask: np.ndarray,
    budget: int = 20,
    n_superpixels: int = 300,
    n_seeds_per_region: int = 15,
    fg_threshold: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Pipeline Region AL usando BALD com comite de N encoders.

    Parameters
    ----------
    image         : [H, W, 3] uint8 RGB
    saliency_maps : list de N arrays [H, W] float32 — saliency de cada encoder
    gt_mask       : [H, W] uint8 — GT para simulacao do especialista

    Returns
    -------
    Mesmo formato que region_al_select(), mais campo 'bald_map'.
    """
    _require_skimage()
    H, W = image.shape[:2]
    gt_bin = (gt_mask > 0)

    superpixels = compute_superpixels(image, n_segments=n_superpixels)
    scores      = score_regions_by_bald(superpixels, saliency_maps)
    selected    = select_top_regions(scores, budget)

    labels = {}
    for region_id in selected:
        region_mask = superpixels == region_id
        labels[region_id] = simulate_expert_label(region_mask, gt_bin, fg_threshold)

    n_fg = sum(1 for v in labels.values() if v == "fg")
    n_bg = sum(1 for v in labels.values() if v == "bg")

    flim_seeds = regions_to_flim_seeds(
        superpixels, selected, labels, H, W,
        n_seeds_per_region=n_seeds_per_region,
        seed=seed,
    )

    return {
        "superpixels":      superpixels,
        "scores":           scores,
        "selected_regions": selected,
        "region_labels":    labels,
        "flim_seeds":       flim_seeds,
        "n_fg_regions":     n_fg,
        "n_bg_regions":     n_bg,
    }


def create_combined_region_marker_dir_bald(
    original_marker_dir: str,
    selected_fnames: list,
    gt_folder: str,
    orig_folder: str,
    saliency_folders: list,
    output_dir: str,
    budget_per_image: int = 20,
    n_superpixels: int = 300,
    n_seeds_per_region: int = 15,
    fg_threshold: float = 0.15,
) -> str:
    """
    Combina markers originais + markers Region BALD (comite) para imagens selecionadas.

    Parameters
    ----------
    saliency_folders : lista de N diretorios — um saliency dir por encoder do comite

    Returns
    -------
    output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copia markers originais
    for fname in os.listdir(original_marker_dir):
        src = os.path.join(original_marker_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy2(src, dst)

    added = 0
    for fname in selected_fnames:
        img_id = fname.replace(".png", "")
        dst = os.path.join(output_dir, f"{img_id}-seeds.txt")
        if os.path.exists(dst):
            continue

        orig_path = os.path.join(orig_folder, f"{img_id}.png")
        gt_path   = os.path.join(gt_folder, f"{img_id}.png")
        if not os.path.exists(orig_path):
            continue

        image   = np.array(Image.open(orig_path).convert("RGB"))
        gt_mask = (
            np.array(Image.open(gt_path).convert("L"))
            if os.path.exists(gt_path)
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )

        # Carrega saliency de cada encoder do comite
        saliency_maps = []
        for sal_folder in saliency_folders:
            sal_path = os.path.join(sal_folder, f"{img_id}.png")
            if os.path.exists(sal_path):
                sal = np.array(Image.open(sal_path).convert("L"),
                               dtype=np.float32) / 255.0
            else:
                sal = np.random.rand(*image.shape[:2]).astype(np.float32)
            saliency_maps.append(sal)

        result = region_al_select_bald(
            image, saliency_maps, gt_mask,
            budget=budget_per_image,
            n_superpixels=n_superpixels,
            n_seeds_per_region=n_seeds_per_region,
            fg_threshold=fg_threshold,
        )
        save_region_seeds(result["flim_seeds"], dst)
        added += 1
        print(
            f"  [region_bald] {img_id}: "
            f"{result['n_fg_regions']}fg + {result['n_bg_regions']}bg regioes (BALD)"
        )

    print(f"  [region_bald] {added} imagens com seeds BALD adicionados")
    return output_dir


# --- CLI standalone ----------------------------------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Region AL demo para uma imagem FLIM")
    p.add_argument("--image",         required=True)
    p.add_argument("--saliency",      required=True)
    p.add_argument("--gt",            required=True)
    p.add_argument("--n_superpixels", type=int,   default=300)
    p.add_argument("--budget",        type=int,   default=20)
    p.add_argument("--method",        default="entropy", choices=["entropy", "margin", "random"])
    p.add_argument("--n_seeds",       type=int,   default=15)
    p.add_argument("--fg_threshold",  type=float, default=0.15)
    p.add_argument("--out_seeds",     default="/tmp/region_seeds")
    p.add_argument("--out_viz",       default=None)
    args = p.parse_args()

    _require_skimage()

    image    = np.array(Image.open(args.image).convert("RGB"))
    saliency = np.array(Image.open(args.saliency).convert("L"), dtype=np.float32) / 255.0
    gt_mask  = np.array(Image.open(args.gt).convert("L"))

    result = region_al_select(
        image, saliency, gt_mask,
        budget=args.budget,
        n_superpixels=args.n_superpixels,
        method=args.method,
        n_seeds_per_region=args.n_seeds,
        fg_threshold=args.fg_threshold,
    )

    os.makedirs(args.out_seeds, exist_ok=True)
    seeds_path = os.path.join(
        args.out_seeds,
        os.path.basename(args.image).replace(".png", "-seeds.txt"),
    )
    n_written = save_region_seeds(result["flim_seeds"], seeds_path)

    print(f"\n=== Region AL ({args.method}) ===")
    print(f"Regioes selecionadas : {args.budget}")
    print(f"  FG (ovo)  : {result['n_fg_regions']}")
    print(f"  BG (fundo): {result['n_bg_regions']}")
    print(f"Seeds gerados: {n_written} -> {seeds_path}")

    gt_bin   = (gt_mask > 0).astype(np.uint8)
    fg_seeds = result["flim_seeds"]["fg_seeds"]
    pred_bin = np.zeros_like(gt_bin)
    H, W = pred_bin.shape
    for col, row in fg_seeds:
        if 0 <= row < H and 0 <= col < W:
            pred_bin[row, col] = 1

    print(f"\nIoU (seeds fg vs GT): {iou_score(pred_bin, gt_bin):.4f}")
    print(f"Fb  (seeds fg vs GT): {fb_score(pred_bin, gt_bin):.4f}")
    print("(Para Fb/IoU real, re-treinar encoder e rodar DT)")

    if args.out_viz:
        visualize_region_selection(
            image, result["superpixels"], result["selected_regions"],
            result["region_labels"], saliency, args.out_viz,
        )
        print(f"Visualizacao salva: {args.out_viz}")


if __name__ == "__main__":
    _cli()
