#!/bin/bash
# ============================================================
# run_al_k3.sh — Experimento AL mínimo: K=3, split1, 1 seed
# Rodar DENTRO do container Docker, de dentro de flim_ad/:
#   bash ../flim_al/run_al_k3.sh 2>&1 | tee ../flim_al/run_al_k3.log
# ============================================================
set -e

echo "=== [1/4] Limpando resultados antigos (markers bugados) ==="
rm -f  out/al_encoder_results/schisto-user_A_entropy_encoder_al.csv
rm -rf out/al_encoder_results/schisto/user_A/
rm -rf out/al_encoder_results/user_A/
echo "Limpo."

echo ""
echo "=== [2/4] Instalando dependências ==="
pip install "numpy<2.0" monai transformers -q
apt-get install -y liblapack3 libblas3 libfftw3-double3 2>/dev/null || true
echo "Dependências OK."

echo ""
echo "=== [3/4] Rodando AL com K=3,5,10 (split1, entropy, 1 seed) ==="
python3 ../flim_al/al_encoder_experiment.py \
    --markers schisto/user_A \
    --splits 1 \
    --budgets 3 5 10 \
    --acquisition entropy \
    --n_seeds 1 \
    --device cpu \
    --save_dir out/al_encoder_results

echo ""
echo "=== [4/4] Resultados ==="
cat out/al_encoder_results/schisto-user_A_entropy_encoder_al.csv
