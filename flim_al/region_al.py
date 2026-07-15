"""
region_al.py
============
AL por regiões/pixels — inspirado em RIPU (CVPR 2022, arXiv:2111.12667).

Em vez de anotar GT mask completa de K imagens, o especialista
anota apenas os PATCHES DE ALTA INCERTEZA de cada imagem selecionada.
O decoder é treinado com loss mascarada: só computa loss nas regiões anotadas.

Fluxo:
  1. compute_entropy_mask(saliency_path)  → máscara binária (H, W)
     → identifica patches com entropia > threshold
  2. train_backprop_region(...)
     → igual ao loop normal mas com loss * mask_region
     → ignora pixels fora das regiões anotadas

Vantagem: com budget=K imagens, só P% dos pixels precisam ser anotados,
reduzindo o custo de anotação vs GT mask completa.

Referências:
  RIPU (CVPR 2022): https://arxiv.org/abs/2111.12667
  Suggestive Annotation (MICCAI 2017): https://arxiv.org/abs/1706.04737
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── Entropy mask ──────────────────────────────────────────────────────────────

def compute_entropy_mask(
    saliency_path: str,
    patch_size: int = 64,
    top_k_patches: int = 5,
    threshold_percentile: float | None = None,
) -> np.ndarray:
    """
    Gera máscara binária (H, W) marcando regiões de alta incerteza.

    Dois modos:
    - top_k_patches: seleciona exatamente os K patches com maior entropia média
    - threshold_percentile: seleciona todos os pixels com entropia > p-ésimo percentil

    Parameters
    ----------
    saliency_path        : caminho para o saliency map (PNG grayscale)
    patch_size           : tamanho dos patches quadrados (pixels)
    top_k_patches        : número de patches a selecionar
    threshold_percentile : se definido, usa percentil ao invés de top-K patches

    Returns
    -------
    mask : np.ndarray (H, W) float32, valores 0 ou 1
    """
    arr = np.array(Image.open(saliency_path).convert("L"), dtype=np.float32) / 255.0
    p = np.clip(arr, 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # Shannon binary entropy

    H, W = entropy.shape

    if threshold_percentile is not None:
        # Modo percentil: máscara pixel-a-pixel
        thresh = np.percentile(entropy, threshold_percentile)
        return (entropy >= thresh).astype(np.float32)

    # Modo top-K patches
    patches = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            mean_ent = float(entropy[y : y + patch_size, x : x + patch_size].mean())
            patches.append((mean_ent, y, x))

    if not patches:
        # Imagem menor que patch_size: usar máscara cheia
        return np.ones((H, W), dtype=np.float32)

    patches.sort(reverse=True)
    mask = np.zeros((H, W), dtype=np.float32)
    for _, y, x in patches[: top_k_patches]:
        mask[y : y + patch_size, x : x + patch_size] = 1.0

    return mask


def coverage_ratio(mask: np.ndarray) -> float:
    """Fração de pixels anotados vs total (métrica de custo de anotação)."""
    return float(mask.mean())


# ── Treino com loss mascarada por região ──────────────────────────────────────

def train_backprop_region(
    encoder,
    features_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    masks_list: list[np.ndarray],    # máscara float32 (H_orig, W_orig) por imagem
    target_layer: int,
    weights_path: str,
    n_epochs: int,
    device: str,
    lr: float = 1e-2,
    wd: float = 1e-2,
) -> tuple[torch.Tensor, float]:
    """
    Treina o backprop_decoder com loss mascarada pelas regiões incertas.

    O loss é computado APENAS nos pixels dentro da máscara (regiões anotadas).
    Pixeis fora da máscara são ignorados — o especialista não precisa anotá-los.

    Parameters
    ----------
    encoder        : modelo FLIM carregado
    features_list  : lista de tensores de features pré-computados (encoder frozen)
    labels_list    : lista de tensores GT binários (1, H, W)
    masks_list     : lista de arrays numpy (H_orig, W_orig) com regiões a anotar
    target_layer   : camada alvo para o decoder 1x1
    weights_path   : caminho para salvar os pesos
    n_epochs       : número de épocas
    device         : 'cuda:0' ou 'cpu'
    lr, wd         : learning rate e weight decay do Adam

    Returns
    -------
    decoder_weights : Tensor com os pesos treinados
    best_loss       : melhor loss alcançado
    """
    from monai.losses import DiceCELoss

    out_channels = encoder.layers[target_layer].conv.out_channels
    decoder_weights = torch.empty(
        (1, out_channels, 1, 1), device=device
    ).requires_grad_(True)
    torch.nn.init.xavier_uniform_(decoder_weights)

    loss_fn = DiceCELoss()
    optimizer = torch.optim.Adam([decoder_weights], lr=lr, weight_decay=wd)

    # Pré-converter masks para tensores
    mask_tensors = []
    for mask, y in zip(masks_list, labels_list):
        t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
        mask_tensors.append(t)

    best_loss = float("inf")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_losses = []

        for x, y, mask_t in zip(features_list, labels_list, mask_tensors):
            # Forward: logits do decoder (sem relu/normalize)
            res = F.conv2d(x, decoder_weights, padding=0, stride=1)
            # Redimensiona para tamanho do label
            res = F.interpolate(
                res, [y.shape[-2], y.shape[-1]], mode="bilinear", align_corners=True
            )
            # Mascara a região: interpolate mask para o tamanho do label
            mask_resized = F.interpolate(
                mask_t, [y.shape[-2], y.shape[-1]], mode="nearest"
            )
            # Aplica máscara: zera regiões não anotadas em pred e GT
            res_masked = res * mask_resized
            y_masked   = y.unsqueeze(0) * mask_resized

            # Loss apenas nas regiões anotadas
            loss = loss_fn(res_masked, y_masked)
            epoch_losses.append(loss.item())
            loss.backward()

        optimizer.step()
        mean_ep = float(np.mean(epoch_losses))

        if epoch % 50 == 0:
            print(f"  [region] epoch:{epoch:4d}  loss:{mean_ep:.4f}", end="\r")

        if mean_ep < best_loss:
            best_loss = mean_ep
            torch.save(decoder_weights.detach(), weights_path)

    print(f"  [region] Final loss: {best_loss:.4f} (saved at best)")
    return decoder_weights.detach(), best_loss


# ── Pipeline completo: seleciona imagens + extrai máscaras + treina ───────────

def run_region_al(
    encoder_path: str,
    selected_fnames: list[str],
    orig_folder: str,
    label_folder: str,
    saliency_folder: str,   # pasta com saliency maps do proxy (labeled_marker)
    target_layer: int,
    output_path: str,
    n_epochs: int,
    device: str,
    patch_size: int = 64,
    top_k_patches: int = 5,
    lr: float = 1e-2,
    wd: float = 1e-2,
) -> tuple[str, float]:
    """
    Pipeline completo de Region AL:
      1. Carrega encoder + pré-computa features
      2. Computa máscaras de entropia para cada imagem selecionada
      3. Treina decoder com loss mascarada
      4. Retorna path dos pesos e cobertura média de anotação

    Returns
    -------
    weights_path     : path para os pesos salvos
    mean_coverage    : fração média de pixels anotados (métrica de custo)
    """
    from pyflim import data as flimdata

    model = torch.load(encoder_path, map_location=device, weights_only=False)
    model.device = device

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
    weights_path = os.path.join(output_path, f"layer{target_layer}_weight.pth")

    # Pré-computa features (encoder frozen)
    features_list, labels_list = [], []
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
                    features_list.append(X.detach().clone())
                    break
            Y[Y > 0] = 1
            labels_list.append(torch.tensor(Y).unsqueeze(0).float().to(device))

    # Computa máscaras de entropia para cada imagem selecionada
    masks_list = []
    coverages = []
    for fname in selected_fnames:
        sal_path = os.path.join(saliency_folder, fname)
        if os.path.exists(sal_path):
            mask = compute_entropy_mask(
                sal_path, patch_size=patch_size, top_k_patches=top_k_patches
            )
        else:
            # Fallback: máscara completa (equivale ao treino normal)
            img = Image.open(os.path.join(orig_folder, fname))
            mask = np.ones((img.height, img.width), dtype=np.float32)

        masks_list.append(mask)
        coverages.append(coverage_ratio(mask))

    mean_coverage = float(np.mean(coverages))
    print(f"  [region] Cobertura média de anotação: {mean_coverage:.1%} dos pixels")

    _, best_loss = train_backprop_region(
        model,
        features_list,
        labels_list,
        masks_list,
        target_layer=target_layer,
        weights_path=weights_path,
        n_epochs=n_epochs,
        device=device,
        lr=lr,
        wd=wd,
    )

    return weights_path, mean_coverage
