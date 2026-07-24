#!/bin/bash
# run_region_bald.sh — Region BALD (comite de encoders)
# =====================================================
# Roda dentro do container Docker: cd flim_ad && bash ../flim_al/run_region_bald.sh
#
# O que faz:
#   1. Treina N encoders (padrao: 3) com seeds k-means diferentes
#   2. Gera N conjuntos de saliency maps do pool
#   3. Calcula BALD por pixel e por regiao (desacordo entre encoders)
#   4. Seleciona Top-K imagens por BALD image-level
#   5. Dentro de cada imagem, gera seeds nas regioes de maior BALD
#   6. Re-treina encoder + avalia com DT -> Fb e IoU

set -e

SPLITS="${1:-1 2 3}"
BUDGETS="${2:-3 5 10}"
N_COMMITTEE="${3:-3}"
MARKERS="schisto/user_A"
SAVE_DIR="out/al_bald_results"
LOG="run_region_bald.log"
DT_BIN="libs/ift/bin/iftSMansoniDelineation"

echo "=== Region BALD (Comite) ==="
echo "Splits     : $SPLITS"
echo "Budgets    : $BUDGETS"
echo "Comite N   : $N_COMMITTEE encoders"
echo "Save dir   : $SAVE_DIR"
echo "Log        : $LOG"
echo ""

# Dependencias
python3 -c "import skimage" 2>/dev/null || pip3 install scikit-image -q

echo "[$(date +%H:%M:%S)] Iniciando region_bald..."
python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition region_bald \
    --n_committee "$N_COMMITTEE" \
    --n_seeds 3 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    --use_dt \
    --dt_bin "$DT_BIN" \
    2>&1 | tee "$LOG"

echo ""
echo "[$(date +%H:%M:%S)] Concluido!"
echo ""
echo "Comparacao rapida dos CSVs:"
python3 - << 'PYEOF'
import glob, csv, os
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))
for csv_file in glob.glob("out/al_*_results/**/*.csv", recursive=True):
    acq = "?"
    if "region_bald"    in csv_file: acq = "region_bald"
    elif "region_entrop" in csv_file: acq = "region_entropy"
    elif "_entropy_"     in csv_file: acq = "entropy"
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            if row.get("method","").startswith("al_") and row.get("decoder") == "labeled_marker":
                budget = row["budget"]
                results[acq][budget].append(float(row.get("fb", 0)))

print(f"\n{'Metodo':20s} {'K=3':>8} {'K=5':>8} {'K=10':>8}")
print("-" * 50)
for acq in ["entropy", "region_entropy", "region_bald"]:
    if acq not in results: continue
    vals = results[acq]
    row = [f"{sum(vals.get(k,[0]))/max(len(vals.get(k,[1])),1):.3f}"
           for k in ["3","5","10"]]
    print(f"{acq:20s} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
PYEOF
