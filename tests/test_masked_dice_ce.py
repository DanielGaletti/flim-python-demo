"""
Verifica que `masked_dice_ce(mask=None)` reproduz `DiceCELoss(sigmoid=True)`
do MONAI, e que a versão mascarada ignora de fato os pixels fora da região.

Rodar:
    cd flim-python-demo && python3 tests/test_masked_dice_ce.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from monai.losses import DiceCELoss
from flim_al.losses import masked_dice_ce

torch.manual_seed(0)


def test_equivale_ao_monai_sem_mascara():
    monai_loss = DiceCELoss(sigmoid=True)
    max_diff = 0.0
    for _ in range(20):
        logits = torch.randn(1, 1, 32, 32) * 3
        target = (torch.rand(1, 1, 32, 32) > 0.9).float()
        a = monai_loss(logits, target).item()
        b = masked_dice_ce(logits, target).item()
        max_diff = max(max_diff, abs(a - b))
    print(f"  max |monai - masked_dice_ce(mask=None)| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"divergiu do MONAI: {max_diff}"


def test_mascara_ignora_pixels_de_fora():
    """Mudar o GT/logits FORA da máscara não pode alterar a loss."""
    logits = torch.randn(1, 1, 32, 32)
    target = (torch.rand(1, 1, 32, 32) > 0.9).float()
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., :16, :16] = 1.0

    base = masked_dice_ce(logits, target, mask).item()

    logits2 = logits.clone()
    target2 = target.clone()
    logits2[..., 16:, :] = 99.0      # lixo fora da máscara
    target2[..., 16:, :] = 1.0
    alt = masked_dice_ce(logits2, target2, mask).item()

    print(f"  loss dentro-da-mascara: {base:.6f} vs {alt:.6f}")
    assert abs(base - alt) < 1e-6, "a mascara nao esta isolando a regiao"


def test_implementacao_antiga_nao_recompensa_acerto():
    """
    O defeito real: com `logits * mask`, sigmoid(0)=0.5 fora da regiao infla o
    denominador do Dice. Resultado — mesmo com predicao PERFEITA dentro da
    mascara, a loss antiga quase nao cai, entao o termo Dice nao guia o treino.
    """
    monai_loss = DiceCELoss(sigmoid=True)
    target = (torch.rand(1, 1, 64, 64) > 0.95).float()
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., :23, :] = 1.0                       # ~36% de cobertura

    # Predicao perfeita dentro da mascara
    perfeito = torch.where(target > 0.5, 10.0, -10.0)

    antiga = monai_loss(perfeito * mask, target * mask).item()
    nova = masked_dice_ce(perfeito, target, mask).item()
    print(f"  predicao PERFEITA na regiao -> antiga={antiga:.4f}  nova={nova:.6f}")

    assert nova < 0.01, f"loss nova deveria zerar com acerto perfeito, deu {nova}"
    assert antiga > 1.0, (
        f"loss antiga deveria continuar travada apesar do acerto, deu {antiga}"
    )


if __name__ == "__main__":
    for fn in [
        test_equivale_ao_monai_sem_mascara,
        test_mascara_ignora_pixels_de_fora,
        test_implementacao_antiga_nao_recompensa_acerto,
    ]:
        print(f"{fn.__name__}:")
        fn()
        print("  OK\n")
    print("todos os testes passaram")
