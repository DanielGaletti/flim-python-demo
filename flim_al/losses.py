"""
losses.py
=========
Loss Dice+CE com suporte a máscara de região, compartilhada pelos dois
treinadores do Experimento B (imagem inteira e region AL).

Por que não usar `monai.losses.DiceCELoss` direto no caso mascarado
-------------------------------------------------------------------
A implementação anterior mascarava fazendo `pred * mask` e `gt * mask`.
Isso zera os LOGITS fora da região, não a contribuição da loss:

    logit 0  →  sigmoid(0) = 0.5

ou seja, o DiceLoss passa a enxergar "0.5 de foreground" em ~87% da imagem.
Esse termo domina o denominador do Dice e o trava em ≈1.0, neutralizando o
gradiente da região anotada. A loss observada empiricamente batia com

    dice(≈1.0) + BCE_fora(0.693 × 0.872) = 1.604      [observado: 1.54–1.61]

confirmando o diagnóstico.

Aqui a máscara entra como PESO por pixel: pixels fora da região não entram
nem no numerador nem no denominador do Dice, nem na média da BCE.

Com `mask=None` (ou toda 1) o resultado é equivalente ao
`DiceCELoss(sigmoid=True)` do MONAI — verificado numericamente em
`tests/test_masked_dice_ce.py`. Usar a MESMA função nos dois treinadores
garante que `region_entropy` e `entropy` difiram apenas pela máscara.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_dice_ce(
    logits: torch.Tensor,          # (B, 1, H, W) — logits crus do decoder
    target: torch.Tensor,          # (B, 1, H, W) — GT binário {0,1}
    mask: torch.Tensor | None = None,   # (B, 1, H, W) — 1 = anotado, 0 = ignorado
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
) -> torch.Tensor:
    """
    Dice + BCE calculados APENAS sobre os pixels com mask=1.

    Dice = 1 - (2·Σ p·y·m + smooth_nr) / (Σ p·m + Σ y·m + smooth_dr)
    BCE  = Σ bce(logit, y)·m / Σ m

    Retorna escalar (soma dos dois termos), igual à convenção do DiceCELoss.
    """
    if mask is None:
        mask = torch.ones_like(target)

    probs = torch.sigmoid(logits)

    inter = (probs * target * mask).sum()
    denom = (probs * mask).sum() + (target * mask).sum()
    dice_loss = 1.0 - (2.0 * inter + smooth_nr) / (denom + smooth_dr)

    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    n_valid = mask.sum().clamp(min=1.0)
    bce_loss = (bce_map * mask).sum() / n_valid

    return dice_loss + bce_loss
