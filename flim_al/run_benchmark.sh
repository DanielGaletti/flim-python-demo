#!/bin/bash
# run_benchmark.sh — Benchmark multi-dataset FLIM AL
# ===================================================
# Roda todos os métodos (no_al, entropy, LC, margin, region_entropy, region_bald)
# em qualquer dataset com orig/ + label/.
#
# Uso (dentro de flim_ad/):
#   bash ../flim_al/run_benchmark.sh /path/to/dataset
#   bash ../flim_al/run_benchmark.sh /path/to/dataset "3 5 10" "no_al entropy region_entropy"
#
# Exemplo com dataset schisto:
#   bash ../flim_al/run_benchmark.sh datasets/schistossoma-eggs
#
# Com DT:
#   DT_BIN=libs/ift/bin/iftSMansoniDelineation \
#   bash ../flim_al/run_benchmark.sh datasets/schistossoma-eggs

set -e

DATASET_PATH="${1:?Uso: $0 /path/to/dataset [budgets] [methods]}"
BUDGETS="${2:-3 5 10}"
METHODS="${3:-no_al entropy least_confidence margin region_entropy region_bald}"
N_SPLITS="${4:-1}"
N_COMMITTEE="${5:-3}"
SAVE_DIR="out/benchmark"
LOG="benchmark_$(basename "$DATASET_PATH").log"
DT_BIN="${DT_BIN:-}"

echo "=== FLIM Benchmark ==="
echo "Dataset   : $DATASET_PATH"
echo "Budgets   : $BUDGETS"
echo "Métodos   : $METHODS"
echo "N Splits  : $N_SPLITS"
echo "Comitê N  : $N_COMMITTEE"
echo "Save dir  : $SAVE_DIR"
echo "Log       : $LOG"
[ -n "$DT_BIN" ] && echo "DT bin    : $DT_BIN" || echo "DT bin    : não configurado (eval Otsu+AF)"
echo ""

# Dependências
python3 -c "import skimage" 2>/dev/null || pip3 install scikit-image -q

# Monta argumentos DT
DT_ARGS=""
if [ -n "$DT_BIN" ] && [ -f "$DT_BIN" ]; then
    DT_ARGS="--dt_bin $DT_BIN"
fi

echo "[$(date +%H:%M:%S)] Iniciando benchmark..."
python3 ../flim_al/benchmark.py \
    --dataset_path "$DATASET_PATH" \
    --methods $METHODS \
    --budgets $BUDGETS \
    --n_splits "$N_SPLITS" \
    --n_committee "$N_COMMITTEE" \
    --n_rand_seeds 3 \
    --n_init 3 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    $DT_ARGS \
    2>&1 | tee "$LOG"

echo ""
echo "[$(date +%H:%M:%S)] Concluído!"
echo "Resultados: $SAVE_DIR/$(basename $DATASET_PATH)/benchmark_results.csv"
echo "Log: $LOG"
