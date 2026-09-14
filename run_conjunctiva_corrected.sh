#!/usr/bin/env bash
# =============================================================================
# run_conjunctiva_corrected.sh
# Reexecuta o benchmark de conjuntiva com os bugs corrigidos.
#
#   benchmark.py NÃO tem pool∩val leakage (splits internos disjuntos).
#   Correções aplicadas nesta execução:
#   [A6] hash() → hashlib.md5 (seeds determinísticos entre execuções)
#   [C1] NaN em entropy (via acquisition.py corrigido)
#   [C4] CoreSet retorna K corretos (via coreset_badge.py corrigido)
#   [C5] BADGE pred_scores reais (via coreset_badge.py corrigido)
#   [C6] Features LAB (via coreset_badge.py corrigido)
#
#   Nota: o benchmark anterior (out/benchmark_conjunctiva/) rodou sem DT e
#   sem os métodos CoreSet/BADGE (eles cairam em entropy fallback).
#   Esta execução adiciona CoreSet e BADGE reais, e garante reprodutibilidade.
#
# USO:
#   cd flim_ad
#   bash ../run_conjunctiva_corrected.sh [--device cuda:0]
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/flim_ad" || { echo "ERRO: execute de dentro de flim-python-demo/"; exit 1; }

# ── Parâmetros ────────────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cuda:0}"
DATASET_PATH="datasets/conjunctiva"
SAVE_DIR="out/benchmark_conjunctiva_corrected"
BUDGETS="3 5 10"
N_SPLITS=3
N_INIT=3
N_COMMITTEE=3
DT_BIN="libs/ift/bin/iftSMansoniDelineation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    *) echo "Argumento desconhecido: $1"; exit 1;;
  esac
done

echo "============================================================"
echo "  FLIM AL — Conjunctiva Benchmark (Corrected)"
echo "  Device:  $DEVICE"
echo "  Dataset: $DATASET_PATH"
echo "  Output:  flim_ad/$SAVE_DIR"
echo "============================================================"
echo ""

if [ ! -d "$DATASET_PATH" ]; then
  echo "ERRO: Dataset não encontrado: $DATASET_PATH"
  exit 1
fi

# ── Benchmark completo: todos os métodos, sem DT ─────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Rodando benchmark corrigido (sem DT)..."
echo "  Métodos: no_al entropy least_confidence region_entropy region_bald"
echo "           coreset badge"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Exporta PYTHONHASHSEED para garantir hash() determinístico em todos os módulos
export PYTHONHASHSEED=42

python3 ../flim_al/benchmark.py \
  --dataset_path "$DATASET_PATH" \
  --save_dir "$SAVE_DIR" \
  --budgets $BUDGETS \
  --n_splits $N_SPLITS \
  --n_init $N_INIT \
  --n_committee $N_COMMITTEE \
  --device "$DEVICE" \
  --methods no_al entropy least_confidence region_entropy region_bald coreset badge

echo ""

# ── Benchmark com DT (se disponível) ─────────────────────────────────────────
if [ -f "$DT_BIN" ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Rodando benchmark corrigido (com DT)..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  export PYTHONHASHSEED=42
  python3 ../flim_al/benchmark.py \
    --dataset_path "$DATASET_PATH" \
    --save_dir "${SAVE_DIR}_dt" \
    --budgets $BUDGETS \
    --n_splits $N_SPLITS \
    --n_init $N_INIT \
    --n_committee $N_COMMITTEE \
    --device "$DEVICE" \
    --use_dt \
    --dt_bin "$DT_BIN" \
    --methods no_al entropy least_confidence region_entropy region_bald coreset badge
else
  echo "  SKIP: binário DT não encontrado em $DT_BIN"
fi

echo ""
echo "============================================================"
echo "  Benchmark concluído."
echo "  Resultados em: flim_ad/$SAVE_DIR/"
echo ""
echo "  Comparação com benchmark anterior (out/benchmark_conjunctiva/):"
echo "    - CoreSet/BADGE agora usam features LAB e K corretos"
echo "    - Seeds de markers são reproduzíveis (hashlib.md5)"
echo "    - Diferenças em entropy esperadas: ~0 (pouca saliency saturada)"
echo "    - Diferenças em CoreSet/BADGE esperadas: potencialmente grandes"
echo "============================================================"
