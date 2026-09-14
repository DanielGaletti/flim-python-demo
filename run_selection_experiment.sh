#!/usr/bin/env bash
# =============================================================================
# run_selection_experiment.sh
# E1 — Algoritmo 1 do paper (arXiv:2504.20872) com critério de seleção trocável.
#
#   oracle   usa Fβ do ground truth do pool  → TETO   (é o método do paper)
#   entropy  incerteza da saliência          → sem GT
#   coreset  diversidade no espaço de feats  → sem GT
#   badge    diversidade × incerteza         → sem GT
#   random   sorteio                          → PISO
#
# A pergunta: quanto do benefício do oráculo se recupera sem nenhum GT?
#
# USO:
#   bash run_selection_experiment.sh                      # tudo, GPU
#   bash run_selection_experiment.sh --device cpu
#   bash run_selection_experiment.sh --criteria "oracle entropy"
#   bash run_selection_experiment.sh --pool full          # markers sintéticos
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/flim_ad" || { echo "ERRO: rode de dentro de flim-python-demo/"; exit 1; }

DEVICE="cuda:0"
POOL="real"
SPLITS="1 2 3"
SEEDS=3
MAX_IMAGES=6
VAL_SUBSAMPLE=250
CRITERIA="oracle entropy coreset badge random"
SAVE_DIR="out/paper_selection"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)        DEVICE="$2"; shift 2;;
    --pool)          POOL="$2"; shift 2;;
    --splits)        SPLITS="$2"; shift 2;;
    --seeds)         SEEDS="$2"; shift 2;;
    --max-images)    MAX_IMAGES="$2"; shift 2;;
    --val-subsample) VAL_SUBSAMPLE="$2"; shift 2;;
    --criteria)      CRITERIA="$2"; shift 2;;
    --save-dir)      SAVE_DIR="$2"; shift 2;;
    *) echo "Argumento desconhecido: $1"; exit 1;;
  esac
done

# GPU: verificação com kernel real (is_available mente em arquitetura sem kernel)
if [ "$DEVICE" != "cpu" ]; then
  if ! python - <<'EOF'
import sys, torch
if not torch.cuda.is_available(): sys.exit(2)
try:
    import torch.nn.functional as F
    F.conv2d(torch.randn(1,4,32,32,device="cuda"),
             torch.randn(8,4,3,3,device="cuda"), padding=1).sum().item()
    torch.cuda.synchronize()
except Exception:
    sys.exit(3)
EOF
  then
    echo "  GPU sem kernel utilizável — seguindo em CPU."
    DEVICE="cpu"
  fi
fi

echo "============================================================"
echo "  E1 — Seleção iterativa (Algoritmo 1 com critério trocável)"
echo "  Device:      $DEVICE"
echo "  Pool:        $POOL"
echo "  Splits:      $SPLITS   Sementes: $SEEDS"
echo "  Max |T|:     $MAX_IMAGES"
echo "  Val no laço: $VAL_SUBSAMPLE  (subamostra fixa por split)"
echo "  Critérios:   $CRITERIA"
echo "  Saída:       flim_ad/$SAVE_DIR"
echo "============================================================"

for C in $CRITERIA; do
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Critério: $C"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python ../flim_al/paper_selection.py \
    --criterion      "$C" \
    --pool           "$POOL" \
    --splits         $SPLITS \
    --seeds          "$SEEDS" \
    --max_images     "$MAX_IMAGES" \
    --val_subsample  "$VAL_SUBSAMPLE" \
    --device         "$DEVICE" \
    --save_dir       "$SAVE_DIR"
done

echo ""
echo "============================================================"
echo "  Concluído. CSVs:"
for C in $CRITERIA; do
  F="${SAVE_DIR}/selection_${C}_${POOL}.csv"
  if [ -f "$F" ]; then
    echo "  ✓ $C → $F ($(($(wc -l < "$F") - 1)) rodadas)"
  else
    echo "  ✗ $C → não gerado"
  fi
done
echo "============================================================"
echo ""
echo "  Tabelas:"
echo "    python ../flim_al/analyze_selection.py --pool $POOL"
echo ""
