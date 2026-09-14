"""
Gera markers automáticos a partir do ground-truth (label erosion).
Simula um "anotador perfeito" — upper bound teórico do método.

Foreground markers: erosão da label (pixels claramente dentro do objeto)
Background markers: região exterior com margem de segurança

Saída: data/markers_oracle/<image>-seeds.txt
"""

import os
import numpy as np
from skimage import io, morphology

LABEL_FOLDER  = "data/label/"
OUTPUT_FOLDER = "data/markers_oracle/"
LABEL_EXT     = ".png"
MARKER_EXT    = "-seeds.txt"
TRAIN_LISTS   = ["split1-train.txt", "split2-train.txt", "split3-train.txt"]

# Parâmetros de erosão
FG_EROSION_RADIUS  = 5   # erosão do foreground (pixels internos ao objeto)
BG_DILATION_RADIUS = 10  # margem além da borda do objeto → fundo seguro
N_FG_SAMPLES = 300        # markers de foreground por imagem
N_BG_SAMPLES = 4500       # markers de background (ratio ~15:1 igual aos example markers)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def generate_markers(image_name):
    stem = image_name.strip().replace(".png", "")
    lpath = LABEL_FOLDER + stem + LABEL_EXT
    if not os.path.isfile(lpath):
        print(f"  [SKIP] label não encontrada: {lpath}")
        return

    lbl = io.imread(lpath).astype(np.uint8)
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    lbl_bin = (lbl > 0).astype(np.uint8)

    H, W = lbl_bin.shape

    # Foreground: erode a label para pegar pixels claramente internos
    fg_mask = morphology.binary_erosion(lbl_bin, morphology.disk(FG_EROSION_RADIUS))
    fg_pixels = np.argwhere(fg_mask)  # (row, col)

    # Background: dilate e inverte → pixels claramente externos
    dilated  = morphology.binary_dilation(lbl_bin, morphology.disk(BG_DILATION_RADIUS))
    bg_mask  = ~dilated
    bg_pixels = np.argwhere(bg_mask)

    if len(fg_pixels) == 0:
        # Objeto muito pequeno — usar a label inteira sem erosão
        fg_pixels = np.argwhere(lbl_bin)

    # Amostrar N pixels aleatoriamente
    rng = np.random.default_rng(seed=42)
    fg_idx = rng.choice(len(fg_pixels), min(N_FG_SAMPLES, len(fg_pixels)), replace=False)
    bg_idx = rng.choice(len(bg_pixels), min(N_BG_SAMPLES, len(bg_pixels)), replace=False)

    fg_pts = fg_pixels[fg_idx]  # (row, col)
    bg_pts = bg_pixels[bg_idx]

    # Formato seeds.txt: n_seeds width height
    # x y semantic_label instance_label
    # instance_label: 0 = foreground (class 1), 1 = background (class 2)
    total = len(fg_pts) + len(bg_pts)
    out_path = OUTPUT_FOLDER + stem + MARKER_EXT

    with open(out_path, "w") as f:
        f.write(f"{total} {W} {H}\n")
        for (r, c) in bg_pts:
            f.write(f"{c} {r} -1 0\n")   # background → instance 0 (classe dominante)
        for (r, c) in fg_pts:
            f.write(f"{c} {r} -1 1\n")   # foreground/eggs → instance 1

    return len(fg_pts), len(bg_pts)


if __name__ == "__main__":
    all_images = set()
    for lst in TRAIN_LISTS:
        if os.path.isfile(lst):
            with open(lst) as f:
                for line in f:
                    img = line.strip()
                    if img:
                        all_images.add(img)

    print(f"Gerando markers oracle para {len(all_images)} imagens de treino...")
    for img in sorted(all_images):
        result = generate_markers(img)
        if result:
            fg, bg = result
            print(f"  {img}: {fg} fg + {bg} bg markers")

    print(f"\nSalvo em {OUTPUT_FOLDER}")
    print(f"Para usar: edite MARKER_FOLDER em run_ab_experiment.py para '{OUTPUT_FOLDER}'")
