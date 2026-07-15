"""
marker_generator.py
===================
Geração automática de markers FLIM a partir de GT masks.

Simula o especialista desenhando seeds nas imagens:
  - Foreground seeds: pontos amostrados dentro dos objetos (GT=1)
  - Background seeds: pontos amostrados fora dos objetos (GT=0)

Formato seeds.txt do FLIM:
  Linha 1: n_seeds H W
  Linhas seguintes: col row label 0 0
    label = 1  → foreground (objeto de interesse)
    label = -1 → background

Uso:
    from flim_al.marker_generator import generate_markers_from_gt, save_markers

    seeds = generate_markers_from_gt("label/000002.png", n_fg=100, n_bg=200)
    save_markers(seeds, img_path="orig/000002.png",
                 out_path="temp_markers/000002-seeds.txt")
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image


def generate_markers_from_gt(
    gt_path: str,
    n_fg: int = 100,
    n_bg: int = 300,
    seed: int = 42,
    min_fg: int = 5,
) -> dict:
    """
    Gera seeds de foreground e background a partir de uma GT mask.

    Parameters
    ----------
    gt_path  : caminho para a GT mask (PNG grayscale, >127 = foreground)
    n_fg     : número de seeds de foreground a amostrar
    n_bg     : número de seeds de background a amostrar
    seed     : semente para reprodutibilidade
    min_fg   : mínimo de seeds fg necessários (se a imagem tiver poucos pixels fg)

    Returns
    -------
    dict com:
        'fg_seeds' : np.ndarray (M, 2) — [[col, row], ...] pixels foreground
        'bg_seeds' : np.ndarray (N, 2) — [[col, row], ...] pixels background
        'H'        : altura da imagem
        'W'        : largura da imagem
    """
    rng = np.random.default_rng(seed)

    gt = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8)
    H, W = gt.shape

    fg_mask = gt > 127
    bg_mask = ~fg_mask

    fg_rows, fg_cols = np.where(fg_mask)
    bg_rows, bg_cols = np.where(bg_mask)

    # Foreground seeds
    if len(fg_rows) == 0:
        fg_seeds = np.empty((0, 2), dtype=int)
    else:
        n_sample_fg = min(n_fg, len(fg_rows))
        n_sample_fg = max(n_sample_fg, min(min_fg, len(fg_rows)))
        idx = rng.choice(len(fg_rows), size=n_sample_fg, replace=False)
        fg_seeds = np.stack([fg_cols[idx], fg_rows[idx]], axis=1)  # [col, row]

    # Background seeds
    if len(bg_rows) == 0:
        bg_seeds = np.empty((0, 2), dtype=int)
    else:
        n_sample_bg = min(n_bg, len(bg_rows))
        idx = rng.choice(len(bg_rows), size=n_sample_bg, replace=False)
        bg_seeds = np.stack([bg_cols[idx], bg_rows[idx]], axis=1)

    return {
        "fg_seeds": fg_seeds,
        "bg_seeds": bg_seeds,
        "H": H,
        "W": W,
    }


def save_markers(
    seeds: dict,
    out_path: str,
) -> int:
    """
    Salva seeds no formato seeds.txt do FLIM.

    Formato REAL do FLIM (lido por data.py linha 236):
      n_seeds H W
      col row -1 class 0
        class = 0 → background  (marker_class = 1, label = 0 após -1)
        class = 1 → foreground  (marker_class = 2, label = 1 após -1)

    ATENÇÃO: campo[2] é SEMPRE -1; campo[3] é a CLASSE (0 ou 1).
    data.py lê: marker_class = int(campo[3]) + 1

    Returns
    -------
    n_total : total de seeds escritos
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fg = seeds["fg_seeds"]   # (M, 2)
    bg = seeds["bg_seeds"]   # (N, 2)
    H  = seeds["H"]
    W  = seeds["W"]

    n_total = len(fg) + len(bg)

    with open(out_path, "w") as f:
        f.write(f"{n_total} {H} {W}\n")
        for col, row in fg:
            f.write(f"{col} {row} -1 1 0\n")   # classe 1 → foreground
        for col, row in bg:
            f.write(f"{col} {row} -1 0 0\n")   # classe 0 → background

    return n_total


def generate_and_save(
    image_name: str,           # ex: "000002" (sem extensão)
    gt_folder: str,
    out_marker_folder: str,
    n_fg: int = 100,
    n_bg: int = 300,
    seed: int = 42,
) -> str:
    """
    Wrapper: gera e salva markers para uma imagem. Retorna o path salvo.
    """
    gt_path  = os.path.join(gt_folder, f"{image_name}.png")
    out_path = os.path.join(out_marker_folder, f"{image_name}-seeds.txt")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT não encontrada: {gt_path}")

    seeds = generate_markers_from_gt(gt_path, n_fg=n_fg, n_bg=n_bg, seed=seed)
    n = save_markers(seeds, out_path)
    return out_path


def create_combined_marker_dir(
    original_marker_dir: str,    # ex: data/schisto/user_A/split1/markers/
    selected_fnames: list[str],  # imagens selecionadas pelo AL (com ou sem extensão)
    gt_folder: str,
    output_dir: str,
    n_fg: int = 100,
    n_bg: int = 300,
) -> str:
    """
    Cria diretório com markers originais + markers sintéticos das imagens AL.

    Retorna o path do diretório combinado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copia markers originais
    import shutil
    for fname in os.listdir(original_marker_dir):
        src = os.path.join(original_marker_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy2(src, dst)

    original_count = len(os.listdir(original_marker_dir))

    # Gera markers sintéticos para as imagens selecionadas pelo AL
    added = 0
    for fname in selected_fnames:
        name = os.path.splitext(fname)[0]  # remove .png se presente
        out_path = os.path.join(output_dir, f"{name}-seeds.txt")

        # Não sobrescreve markers originais
        if os.path.exists(out_path):
            continue

        gt_path = os.path.join(gt_folder, f"{name}.png")
        if not os.path.exists(gt_path):
            continue

        seeds = generate_markers_from_gt(gt_path, n_fg=n_fg, n_bg=n_bg, seed=hash(name) % 2**31)
        save_markers(seeds, out_path)
        added += 1

    print(f"  [markers] {original_count} originais + {added} sintéticos = "
          f"{original_count + added} total em {output_dir}")
    return output_dir
