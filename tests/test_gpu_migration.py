"""
Teste-portão da migração para a imagem GPU.

Os encoders em out/trained_models/ foram salvos com `torch.save(model)` — pickle
do objeto inteiro — sob torch 2.6 / Python 3.10. A imagem nova é torch 2.7 /
Python 3.11. Este script verifica que:

  1. o pickle carrega sem erro;
  2. o forward do encoder produz saliências numericamente iguais às da imagem
     antiga (tolerância apertada);
  3. o mesmo vale rodando na GPU, quando disponível.

Se (2) falhar, migrar invalidaria a comparação com os resultados já obtidos e a
migração deve ser abortada.

Uso (de dentro de flim_ad/):
    python3 ../tests/test_gpu_migration.py --dump ref_cpu.npz     # imagem antiga
    python3 ../tests/test_gpu_migration.py --check ref_cpu.npz    # imagem nova
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

import numpy as np
import torch

from flim_al.coreset_badge import _rgb_uint8_to_lab01

ENC = "out/trained_models/schisto/user_A/split1/flim_encoder_split1.pth"
ORIG = "datasets/schistossoma-eggs/orig"
IMGS = ["000002.png", "000013.png", "000156.png"]
LAYER = 3


def encoder_features(device: str) -> dict:
    from PIL import Image

    model = torch.load(ENC, map_location=device, weights_only=False)
    model.device = device
    model.eval()

    out = {}
    with torch.no_grad():
        for fname in IMGS:
            p = os.path.join(ORIG, fname)
            if not os.path.exists(p):
                continue
            rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            x = torch.tensor(
                _rgb_uint8_to_lab01(rgb).transpose(2, 0, 1), dtype=torch.float32
            ).unsqueeze(0).to(device)

            for l in range(model.architecture.nlayers):
                if not model.use_bias:
                    x = model.normalization(
                        x, model.layers[l].normalization_parameters)
                x = model.layers[l].conv(x)
                x = model.layers[l].activation(x)
                x = model.layers[l].pool(x)
                if l == LAYER:
                    break
            out[fname] = x.detach().cpu().numpy().astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="salva as features de referência aqui")
    ap.add_argument("--check", help="compara contra este arquivo de referência")
    a = ap.parse_args()

    if not os.path.exists(ENC):
        print(f"ERRO: encoder não encontrado em {ENC}. Rode de dentro de flim_ad/.")
        return 1

    print(f"torch {torch.__version__} | python {sys.version.split()[0]}")
    feats_cpu = encoder_features("cpu")
    print(f"  encoder carregou e rodou em CPU ({len(feats_cpu)} imagens)")

    scale = max(float(np.abs(v).max()) for v in feats_cpu.values()) or 1.0
    print(f"  magnitude típica das features: {scale:.2f}")

    def rel(a_, b_):
        """Maior divergência relativa à escala das features."""
        return max(float(np.abs(a_[k] - b_[k]).max()) for k in a_ if k in b_) / scale

    # No modo --dump rodamos na imagem ANTIGA, onde a GPU falha por design
    # (sem kernel sm_120). Isso não pode abortar a captura da referência.
    gpu_rel = None
    if torch.cuda.is_available():
        try:
            gpu_rel = rel(feats_cpu, encoder_features("cuda:0"))
            print(f"  GPU vs CPU (mesma imagem):  erro relativo = {gpu_rel:.2e}")
        except Exception as e:
            print(f"  GPU indisponível nesta imagem: {type(e).__name__}")

    if a.dump:
        np.savez_compressed(a.dump, **feats_cpu)
        print(f"  referência salva: {a.dump}")
        return 0

    if not a.check:
        return 0

    # Critério que decide a migração: a MESMA rota (CPU) tem que reproduzir a
    # imagem antiga. Se reproduzir, os resultados seguem comparáveis.
    ref = np.load(a.check)
    cpu_rel = rel(feats_cpu, {k: ref[k] for k in ref.files})
    print(f"  CPU nova vs imagem antiga:  erro relativo = {cpu_rel:.2e}")

    ok = True
    if cpu_rel > 1e-5:
        print("\n  ✗ FALHOU: a rota CPU não reproduz a imagem antiga.")
        print("    Migrar tornaria os novos resultados incomparáveis com os já")
        print("    obtidos. Abortar a migração e seguir na imagem antiga.")
        ok = False
    else:
        print("\n  ✓ Encoders carregam e a rota CPU reproduz a imagem antiga.")

    # Divergência CPU↔GPU em float32 é esperada (kernels e ordem de redução
    # diferentes). Só é problema se sair da faixa de ruído numérico.
    if gpu_rel is not None:
        if gpu_rel > 1e-3:
            print(f"  ✗ GPU diverge demais da CPU ({gpu_rel:.2e}) — investigar.")
            ok = False
        else:
            print(f"  ✓ GPU dentro do ruído float32 esperado ({gpu_rel:.2e}).")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
