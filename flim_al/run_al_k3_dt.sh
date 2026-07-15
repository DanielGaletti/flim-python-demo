#!/bin/bash
# ============================================================
# run_al_k3_dt.sh
# Experimento AL com Dynamic Trees — K=3,5,10, split1, 1 seed
#
# Rodar DENTRO do container, de dentro de flim_ad/:
#   bash ../flim_al/run_al_k3_dt.sh 2>&1 | tee ../flim_al/run_al_k3_dt.log
#
# Nota: cada avaliação roda DT (~2 min para 610 imagens).
# Total estimado: ~15-20 min (baseline + 3 budgets × AL+Random).
# ============================================================
set -e

DT_BIN="libs/ift/bin/iftSMansoniDelineation"
DATASET_HOME="datasets"
DATASET="schistossoma-eggs"

echo "=== [1/5] Verificando binário DT ==="
if [ ! -f "$DT_BIN" ]; then
    echo "ERRO: binário não encontrado: $DT_BIN"
    exit 1
fi
echo "OK: $DT_BIN"

echo ""
echo "=== [2/5] Criando symlinks para DT ==="
cd "$DATASET_HOME/$DATASET"
ln -sf orig   images     2>/dev/null || true
ln -sf label  truelabels 2>/dev/null || true
cd ../..
echo "images → orig | truelabels → label"

echo ""
echo "=== [3/5] Instalando dependências ==="
pip install "numpy<2.0" monai transformers scikit-image -q
echo "OK."

echo ""
echo "=== [4/5] Rodando experimento AL com DT ==="
# Remove resultados anteriores do modo DT para forçar re-avaliação
rm -f  out/al_encoder_results/schisto-user_A_entropy_encoder_al_dt.csv
rm -rf out/al_encoder_results/schisto/user_A/*/entropy/

python3 ../flim_al/al_encoder_experiment.py \
    --markers    schisto/user_A \
    --splits     1 \
    --budgets    3 5 10 \
    --acquisition entropy \
    --n_seeds    1 \
    --device     cpu \
    --save_dir   out/al_encoder_results_dt \
    --use_dt \
    --dt_bin     "$DT_BIN"

echo ""
echo "=== [5/5] Resultados ==="
echo ""
echo "--- Modo DT (reprodução paper) ---"
cat out/al_encoder_results_dt/schisto-user_A_entropy_encoder_al.csv 2>/dev/null || echo "(arquivo não encontrado)"
echo ""
echo "--- Referência sem DT ---"
echo "labeled_marker baseline: Fβ=0.362 (sem DT)"
echo "Paper (com DT):          Fβ≈0.860"
