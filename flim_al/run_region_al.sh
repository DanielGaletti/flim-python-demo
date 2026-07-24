#!/bin/bash
# run_region_al.sh — Experimento Region-based AL com IoU
# ========================================================
# Roda dentro do container Docker: flim_ad/
# Compara: region_entropy vs entropy vs random
#
# Uso:
#   cd flim_ad
#   bash ../flim_al/run_region_al.sh [split] [budget]
#
# Exemplo:
#   bash ../flim_al/run_region_al.sh 1 3
#   bash ../flim_al/run_region_al.sh   # roda todos splits 1-3, budgets 3 5 10

set -e

SPLITS="${1:-1 2 3}"
BUDGETS="${2:-3 5 10}"
MARKERS="schisto/user_A"
SAVE_DIR="out/al_region_results"
LOG="run_region_al.log"
DT_BIN="libs/ift/bin/iftSMansoniDelineation"

echo "=== Region-based AL ==="
echo "Splits  : $SPLITS"
echo "Budgets : $BUDGETS"
echo "Log     : $LOG"
echo ""

# Instala scikit-image se necessario
python3 -c "import skimage" 2>/dev/null || \
    pip3 install scikit-image --break-system-packages -q

# ── Region Entropy ──────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Rodando region_entropy..."
python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition region_entropy \
    --n_seeds 3 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    --use_dt \
    --dt_bin "$DT_BIN" \
    2>&1 | tee -a "$LOG"

echo ""
echo "[$(date +%H:%M:%S)] region_entropy concluido."

# ── Entropy (imagem inteira, para comparacao) ────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Rodando entropy (baseline comparacao)..."
python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition entropy \
    --n_seeds 3 \
    --device cpu \
    --save_dir "$SAVE_DIR" \
    --use_dt \
    --dt_bin "$DT_BIN" \
    2>&1 | tee -a "$LOG"

echo ""
echo "[$(date +%H:%M:%S)] Todos os experimentos concluidos."
echo "Resultados em: $SAVE_DIR"
echo "Log completo : $LOG"
echo ""
echo "Para ver resumo:"
echo "  python3 -c \""
echo "    import pandas as pd, glob"
echo "    dfs = [pd.read_csv(f) for f in glob.glob('$SAVE_DIR/**/*.csv', recursive=True)]"
echo "    df = pd.concat(dfs)"
echo "    print(df.groupby(['acquisition','method'])['fb','iou'].mean().round(3))"
echo "  \""
