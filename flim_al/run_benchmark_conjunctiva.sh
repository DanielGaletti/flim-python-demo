#!/bin/bash
# run_benchmark_conjunctiva.sh — Benchmark ALL AL methods on Conjunctiva dataset
# ===============================================================================
# Dataset: conjunctiva vessel segmentation (83 images, ophthalmology)
# Source:  data/conjunctiva/ (images/*.jpg, labels/*.png, markers/ with 5 seeds)
#
# Roda de dentro do container em flim_ad/:
#   bash ../flim_al/run_benchmark_conjunctiva.sh
#
# Nota: sem DT binary (iftSMansoniDelineation é especifico para schistossoma).
#       Avaliacao via Otsu + adaptive filtering (labeled_marker decoder).

set -e

BUDGETS="${1:-3 5 10}"
METHODS="${2:-no_al entropy least_confidence margin region_entropy region_bald}"
N_SPLITS="${3:-3}"
N_COMMITTEE="${4:-3}"

DATASET_PATH="$(python3 -c "import os; print(os.path.abspath('../flim_al/datasets/conjunctiva'))")"
MARKERS_SRC="$(python3 -c "import os; print(os.path.abspath('../data/conjunctiva/markers'))")"
ARCH_FILE="$(python3 -c "import os; print(os.path.abspath('../arch_conjunctiva.json'))")"
SAVE_DIR="out/benchmark_conjunctiva"
LOG="run_benchmark_conjunctiva.log"

echo "=============================="
echo "  FLIM AL — Conjunctiva"
echo "=============================="
echo "Dataset    : $DATASET_PATH"
echo "Methods    : $METHODS"
echo "Budgets    : $BUDGETS"
echo "N splits   : $N_SPLITS"
echo "Committee  : $N_COMMITTEE encoders"
echo "Save dir   : $SAVE_DIR"
echo ""

# Verificar dataset
if [ ! -d "$DATASET_PATH/orig" ] || [ ! -d "$DATASET_PATH/label" ]; then
    echo "ERRO: Dataset não encontrado em $DATASET_PATH"
    echo "Execute: python3 ../flim_al/setup_conjunctiva_dataset.py"
    exit 1
fi

N_IMGS=$(ls "$DATASET_PATH/orig"/*.png 2>/dev/null | wc -l)
echo "Imagens PNG encontradas: $N_IMGS"
if [ "$N_IMGS" -lt 10 ]; then
    echo "ERRO: Poucos arquivos PNG. Verifique a conversão JPG->PNG."
    exit 1
fi

# Instalar dependencias
python3 -c "import skimage" 2>/dev/null || pip3 install scikit-image -q --break-system-packages

echo ""
echo "[$(date +%H:%M:%S)] Iniciando benchmark..."
echo ""

python3 ../flim_al/benchmark.py \
    --dataset_path  "$DATASET_PATH" \
    --save_dir      "$SAVE_DIR" \
    --methods       $METHODS \
    --budgets       $BUDGETS \
    --n_splits      $N_SPLITS \
    --n_init        5 \
    --val_ratio     0.3 \
    --arch_file     "$ARCH_FILE" \
    --existing_markers "$MARKERS_SRC" \
    --n_committee   $N_COMMITTEE \
    --proxy_layer   3 \
    --device        cpu \
    --no_dt \
    2>&1 | tee "$LOG"

echo ""
echo "[$(date +%H:%M:%S)] Concluido! CSV em $SAVE_DIR/"
