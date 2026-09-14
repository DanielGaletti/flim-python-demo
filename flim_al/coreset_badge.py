"""
coreset_badge.py
================
Acquisition functions: CoreSet (ICLR 2018) e BADGE (ICLR 2020)
ambos baseados em features do encoder FLIM.

CoreSet  — greedy k-center: maximiza cobertura geométrica do espaço de features
BADGE    — gradient diversity via k-means++: combina incerteza + diversidade

Uso interno (importado por al_flim_backprop.py):
    features = extract_encoder_features(encoder, images, target_layer, device)
    selected = coreset_select(features, budget)
    selected = badge_select(features, pred_scores, budget)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def _rgb_uint8_to_lab01(img_rgb: np.ndarray) -> np.ndarray:
    """
    Converte (H,W,3) uint8 RGB → (H,W,3) float32 LAB normalizado para [0,1].

    O encoder FLIM é treinado com imagens em espaço LAB (via FLIMData).
    Features extraídas com RGB produziriam vetores em espaço errado.

    Normalização:
        L  ∈ [0, 100]   → divide por 100
        AB ∈ [-128,127] → soma 128, divide por 255
    """
    try:
        from skimage.color import rgb2lab
        lab = rgb2lab(img_rgb).astype(np.float32)
        lab[:, :, 0] /= 100.0
        lab[:, :, 1:] = (lab[:, :, 1:] + 128.0) / 255.0
    except ImportError:
        import cv2
        # cv2 LAB em uint8: L∈[0,255], AB offset em [0,255]
        lab_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = lab_cv / 255.0  # normaliza direto para [0,1]
    return lab.clip(0.0, 1.0)


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_encoder_features(
    encoder,
    image_paths: list[str],
    target_layer: int,
    device: str,
) -> np.ndarray:
    """
    Roda o encoder FLIM até target_layer e aplica Global Average Pool.
    Retorna array (N, D) com um vetor de features por imagem.

    Parameters
    ----------
    encoder     : modelo FLIM carregado (torch.load)
    image_paths : lista de caminhos absolutos das imagens PNG
    target_layer: camada do encoder a usar como representação
    device      : 'cuda:0' ou 'cpu'

    Returns
    -------
    features : np.ndarray (N, D)
    """
    from PIL import Image

    encoder.eval()
    features = []

    for path in image_paths:
        img_rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        # FIX: encoder FLIM treinado com LAB — converter antes de extrair features
        img_lab = _rgb_uint8_to_lab01(img_rgb)
        x = torch.tensor(img_lab.transpose(2, 0, 1)).unsqueeze(0).to(device)

        for l in range(encoder.architecture.nlayers):
            if not encoder.use_bias:
                x = encoder.normalization(x, encoder.layers[l].normalization_parameters)
            x = encoder.layers[l].conv(x)
            x = encoder.layers[l].activation(x)
            x = encoder.layers[l].pool(x)
            if l == target_layer:
                break

        # Global Average Pool → (D,)
        feat = x.mean(dim=[2, 3]).squeeze().cpu().numpy()
        features.append(feat)

    return np.array(features, dtype=np.float32)


@torch.no_grad()
def extract_encoder_features_and_preds(
    encoder,
    image_paths: list[str],
    decoder_weights: torch.Tensor,
    target_layer: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (features, mean_pred) onde:
    - features: (N, D) GAP das features do encoder
    - mean_pred: (N,) média da predição sigmoid por imagem (para BADGE)
    """
    from PIL import Image

    encoder.eval()
    features, preds = [], []

    for path in image_paths:
        img_rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        # FIX: LAB (mesmo espaço do treino do encoder FLIM)
        img_lab = _rgb_uint8_to_lab01(img_rgb)
        x = torch.tensor(img_lab.transpose(2, 0, 1)).unsqueeze(0).to(device)

        for l in range(encoder.architecture.nlayers):
            if not encoder.use_bias:
                x = encoder.normalization(x, encoder.layers[l].normalization_parameters)
            x = encoder.layers[l].conv(x)
            x = encoder.layers[l].activation(x)
            x = encoder.layers[l].pool(x)
            if l == target_layer:
                break

        feat = x.mean(dim=[2, 3]).squeeze().cpu().numpy()
        features.append(feat)

        # Pseudo-label: sigmoid da predição do decoder atual
        pred_logit = F.conv2d(x, decoder_weights.to(device), padding=0, stride=1)
        pred_mean = torch.sigmoid(pred_logit).mean().item()
        preds.append(pred_mean)

    return np.array(features, dtype=np.float32), np.array(preds, dtype=np.float32)


# ── CoreSet ───────────────────────────────────────────────────────────────────

def coreset_select(
    features: np.ndarray,
    budget: int,
    labeled_indices: list[int] | None = None,
    seed: int = 42,
) -> list[int]:
    """
    Greedy k-center (CoreSet — Sener & Savarese, ICLR 2018).
    arXiv:1708.00489

    Seleciona K pontos do pool que maximizam a cobertura geométrica
    do espaço de features (minimiza o maior raio de cobertura).

    Parameters
    ----------
    features        : (N, D) — embeddings de todas as imagens do pool
    budget          : K — número de imagens a selecionar
    labeled_indices : índices já selecionados (labeled set inicial)
    seed            : semente para o ponto inicial

    Returns
    -------
    selected : lista de K índices (do pool original)
    """
    np.random.seed(seed)
    N = len(features)

    if labeled_indices and len(labeled_indices) > 0:
        selected = list(labeled_indices)
        n_skip = len(labeled_indices)
    else:
        # FIX-SEEDPOINT: sem labeled set, o ponto semente do greedy k-center É
        # uma das K imagens selecionadas — o padrão do CoreSet (Sener & Savarese).
        # Antes ele era descartado (n_init=1 + `return selected[1:]`), o que
        # pedia budget+1 pontos de um pool de N e, com budget>=N, devolvia
        # DUPLICATAS (a mesma imagem entrava 2x no treino).
        selected = [int(np.random.randint(N))]
        n_skip = 0

    # Distância mínima de cada ponto ao labeled set atual
    min_dists = np.full(N, np.inf, dtype=np.float32)

    # Inicializa distâncias com o(s) ponto(s) já selecionado(s)
    for idx in selected:
        d = np.sum((features - features[idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, d)

    # FIX-OVERFLOW: nunca pedir mais pontos do que existem no pool
    target = min(n_skip + budget, N)
    while len(selected) < target:
        min_dists[selected] = -1.0   # já selecionados: excluir
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        # Atualiza distâncias
        d = np.sum((features - features[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, d)

    out = selected[n_skip:]
    assert len(out) == len(set(out)), "coreset_select devolveu duplicatas"
    return out


# ── BADGE ─────────────────────────────────────────────────────────────────────

def badge_select(
    features: np.ndarray,
    pred_scores: np.ndarray,
    budget: int,
    seed: int = 42,
) -> list[int]:
    """
    BADGE — Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds
    Ash et al., ICLR 2020 — arXiv:1906.03671

    Gradient embedding para segmentação binária:
        g_i = (sigmoid(pred_i) - 0.5) * feature_i

    k-means++ sobre os gradient embeddings: pontos com alta
    incerteza (pred ≈ 0.5) e alta diversidade (espaço de features).

    Parameters
    ----------
    features    : (N, D) embeddings do encoder
    pred_scores : (N,) predição média sigmoid por imagem (proxy)
    budget      : K — número de imagens a selecionar
    seed        : semente para k-means++

    Returns
    -------
    selected : lista de K índices
    """
    np.random.seed(seed)
    N = len(features)
    budget = min(budget, N)   # FIX-OVERFLOW: pool não tem mais que N pontos

    # Gradient embedding: escala o vetor de features pela incerteza
    uncertainty = (pred_scores - 0.5).reshape(-1, 1)   # (N, 1), |valor| é incerteza
    g = uncertainty * features                           # (N, D)

    # k-means++ initialization sobre g
    selected = [int(np.random.randint(N))]
    dists = np.sum((g - g[selected[-1]]) ** 2, axis=1).astype(np.float64)

    while len(selected) < budget:
        # Remove já selecionados do sorteio
        dists_copy = dists.copy()
        dists_copy[selected] = 0.0
        total = dists_copy.sum()
        if total < 1e-12:
            # Todos os pontos colapsos → seleção aleatória
            remaining = list(set(range(N)) - set(selected))
            selected.append(int(np.random.choice(remaining)))
        else:
            probs = dists_copy / total
            next_idx = int(np.random.choice(N, p=probs))
            selected.append(next_idx)

        # Atualiza distâncias
        new_d = np.sum((g - g[selected[-1]]) ** 2, axis=1)
        dists = np.minimum(dists, new_d)

    return selected
