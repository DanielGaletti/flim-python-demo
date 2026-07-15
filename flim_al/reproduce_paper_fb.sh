#!/bin/bash
# ============================================================
# reproduce_paper_fb.sh
# Reproduz Fβ do paper usando Dynamic Trees nos saliency maps
# já existentes (labeled_marker, layer_3, splits 1-3).
#
# Rodar DENTRO do container, de dentro de flim_ad/:
#   bash ../flim_al/reproduce_paper_fb.sh 2>&1 | tee ../flim_al/reproduce_fb.log
# ============================================================
set -e

DATASET_HOME="datasets"
DATASET="schistossoma-eggs"
MARKERS="schisto/user_A"
DT_BIN="libs/ift/bin/iftSMansoniDelineation"
LAYER=3
TASK="test"

echo "=== [1/4] Criando symlinks para o DT ==="
cd "$DATASET_HOME/$DATASET"
ln -sf orig   images    2>/dev/null || true
ln -sf label  truelabels 2>/dev/null || true
ls -la images truelabels
cd ../..
echo "Symlinks OK."

echo ""
echo "=== [2/4] Rodando Dynamic Trees — labeled_marker ==="
for SPLIT in 1 2 3; do
    SAL_DIR="out/saliencies/$MARKERS/$TASK/split${SPLIT}/labeled_marker/layer_${LAYER}"
    OUT_DIR="out/saliencies_delination/$MARKERS/$TASK/split${SPLIT}/labeled_marker/layer_${LAYER}"
    mkdir -p "$OUT_DIR"

    if [ ! -d "$SAL_DIR" ]; then
        echo "  Split $SPLIT: saliency não encontrado em $SAL_DIR — skip"
        continue
    fi

    echo "  Split $SPLIT: DT em $SAL_DIR ..."
    ./$DT_BIN \
        "$DATASET_HOME/$DATASET" \
        "$SAL_DIR" \
        2 \
        "$OUT_DIR" \
        8 8 8 1000 9000 128
    echo "  Split $SPLIT: $(ls $OUT_DIR/masks/ 2>/dev/null | wc -l) masks geradas"
done

echo ""
echo "=== [3/4] Calculando Fβ (pixel-level, β²=0.3) ==="
for SPLIT in 1 2 3; do
    MASKS_DIR="out/saliencies_delination/$MARKERS/$TASK/split${SPLIT}/labeled_marker/layer_${LAYER}/masks"
    VAL_LIST="data/$MARKERS/split${SPLIT}/${TASK}${SPLIT}.csv"

    if [ ! -d "$MASKS_DIR" ]; then
        echo "  Split $SPLIT: masks não encontradas — skip"
        continue
    fi

    echo "  --- Split $SPLIT ---"
    python3 ../flim_al/eval_dt_fb.py \
        --dt_masks  "$MASKS_DIR" \
        --gt_dir    "$DATASET_HOME/$DATASET/label" \
        --val_list  "$VAL_LIST"
done

echo ""
echo "=== [4/4] Comparação: sem DT vs com DT ==="
echo "Sem DT (reprodução anterior):"
echo "  labeled_marker layer_3: Fβ=0.362, DICE=0.684"
echo ""
echo "Paper (FLIM-AD Table III, User A Parasites):"
echo "  labeled_marker:         Fβ≈0.860 (com DT)"
echo ""
echo "Resultado com DT acima (splits 1-3)."
