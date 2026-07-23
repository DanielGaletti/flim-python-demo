#!/usr/bin/env bash
# ============================================================
# Comparação de Acquisition Functions: entropy vs margin vs least_confidence
# Roda splits 1-3, budgets 3/5/10, n_seeds=3, com Dynamic Trees
#
# Rodar DENTRO do container, de dentro de flim_ad/:
#   bash ../flim_al/run_acquisition_comparison.sh 2>&1 | tee ../flim_al/run_acq_comparison.log
# ============================================================
set -e

DT_BIN="libs/ift/bin/iftSMansoniDelineation"
DATASET="schistossoma-eggs"
SPLITS="1 2 3"
BUDGETS="3 5 10"
N_SEEDS=3
SAVE_DIR="out/al_encoder_results_acq_comparison"

# Verificar binário DT
if [ ! -f "$DT_BIN" ]; then
    echo "ERRO: binário não encontrado: $DT_BIN"
    exit 1
fi

# Criar symlinks para DT (idempotente)
cd "datasets/$DATASET"
ln -sf orig   images     2>/dev/null || true
ln -sf label  truelabels 2>/dev/null || true
cd ../..
echo "Symlinks OK: images→orig | truelabels→label"

mkdir -p "$SAVE_DIR"

for ACQ in entropy margin least_confidence; do
    echo ""
    echo "============================================================"
    echo "Acquisition: $ACQ"
    echo "============================================================"
    python3 ../flim_al/al_encoder_experiment.py \
        --markers    schisto/user_A \
        --splits     $SPLITS \
        --budgets    $BUDGETS \
        --n_seeds    $N_SEEDS \
        --acquisition "$ACQ" \
        --device     cpu \
        --save_dir   "$SAVE_DIR" \
        --use_dt \
        --dt_bin     "$DT_BIN"
done

echo ""
echo "============================================================"
echo "DONE. Resultados em: $SAVE_DIR"
ls -lh "$SAVE_DIR"/*.csv
