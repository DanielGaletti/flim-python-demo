"""
Verifica a propriedade que sustenta a comparação pareada do Experimento B:

  1. Mesmo init_seed + mesmas imagens  → pesos IDÊNTICOS (determinismo)
  2. init_seed diferente               → pesos diferentes (a semente age)

Sem (1), Δ_s = Fβ(AL,s) − Fβ(Random,s) não cancelaria o ruído de
inicialização e o teste pareado perderia o sentido.

Rodar (de dentro de flim_ad/, onde ficam out/ e datasets/):
    cd flim_ad && python3 ../tests/test_paired_seeding.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

import torch

from flim_al.al_flim_backprop import train_backprop_on_subset

ENC = "out/trained_models/schisto/user_A/split1/flim_encoder_split1.pth"
ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"
IMGS = ["000002.png", "000013.png"]
EPOCHS = 5


def _train(tag, seed):
    n, wpath = train_backprop_on_subset(
        ENC, IMGS, ORIG, LABEL, 3, f"/tmp/seedtest/{tag}",
        EPOCHS, "cpu", init_seed=seed,
    )
    return n, torch.load(wpath, map_location="cpu", weights_only=True)


def main():
    if not os.path.exists(ENC):
        print(f"SKIP: encoder não encontrado ({ENC}). "
              f"Rode este teste de dentro de flim_ad/.")
        return 0

    n_a, wa = _train("a", seed=1000)
    n_b, wb = _train("b", seed=1000)   # mesma semente
    n_c, wc = _train("c", seed=1001)   # semente diferente

    same = torch.equal(wa, wb)
    diff = not torch.equal(wa, wc)
    d_ab = (wa - wb).abs().max().item()
    d_ac = (wa - wc).abs().max().item()

    print(f"  imagens usadas: {n_a} (esperado {len(IMGS)})")
    print(f"  seed 1000 vs 1000 → max|Δw| = {d_ab:.3e}   idênticos: {same}")
    print(f"  seed 1000 vs 1001 → max|Δw| = {d_ac:.3e}   diferentes: {diff}")

    assert n_a == len(IMGS), f"n_train_imgs errado: {n_a}"
    assert same, "MESMA semente produziu pesos diferentes — pareamento quebrado"
    assert diff, "sementes diferentes deram pesos iguais — init_seed sem efeito"
    print("\nOK: comparação pareada é válida.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
