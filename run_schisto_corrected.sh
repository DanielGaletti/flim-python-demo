#!/usr/bin/env bash
# =============================================================================
# run_schisto_corrected.sh
# Reexecuta os experimentos schisto com todos os bugs críticos corrigidos:
#
#   CORREÇÕES APLICADAS:
#   [C1] NaN em entropy: eps=1e-6 (era 1e-8, causava underflow em float32)
#   [C2] Pool∩Val leakage: pool filtrado para excluir imagens de validação
#   [C3] LC=Margin: reconhecidos como idênticos; roda apenas entropy + LC
#   [C4] CoreSet retorna K elementos corretos (era K-1)
#   [C5] BADGE: pred_scores via saliency means (era dummy_w=zeros → grad=0)
#   [C6] CoreSet/BADGE: features em espaço LAB (era RGB/255)
#   [A1] MAE real calculado no path DT (era hardcoded 0.0)
#   [A2] empty-empty → Fβ=1.0 em ambos os paths (era 0.0 no path não-DT)
#   [A4] region_bald: seeds posicionados por BALD (era entropy)
#   [A6] hash() substituído por hashlib.md5 (seed determinístico)
#
#   LIMITAÇÕES CONHECIDAS RESTANTES:
#   [A3] AL ainda é one-shot (não iterativo) — requereria redesign completo
#   [A5] Bootstrap ainda perde multiplicidades (limitação estrutural do FLIM)
#   [L3] Budget K ainda em imagens, não seeds
#
# USO:
#   cd flim_ad
#   bash ../run_schisto_corrected.sh [--device cuda:0] [--splits 1 2 3]
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/flim_ad" || { echo "ERRO: execute de dentro de flim-python-demo/"; exit 1; }

# ── Parâmetros configuráveis ─────────────────────────────────────────────────
DEVICE="${DEVICE:-cuda:0}"
SPLITS="${SPLITS:-1 2 3}"
BUDGETS="3 5 10"
MARKERS="schisto/user_A"
PROXY_LAYER=3
N_SEEDS=3          # seeds aleatórios para baseline Random
N_FG=100           # seeds fg por imagem sintética
N_BG=300           # seeds bg por imagem sintética
N_COMMITTEE=3      # encoders no comitê (region_bald)
SAVE_BASE="out/al_corrected_results"
DT_BIN="libs/ift/bin/iftSMansoniDelineation"

# Métodos a rodar. Útil para reexecutar só o que mudou em vez do conjunto todo:
#   --methods region_bald        → só acrescenta o braço de controle random_region
#   --methods coreset            → só o método cuja seleção mudou (FIX-SEEDPOINT)
METHODS="${METHODS:-entropy entropy_dt least_confidence coreset badge region_bald}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)  DEVICE="$2"; shift 2;;
    --seeds)   N_SEEDS="$2"; shift 2;;
    --methods) METHODS="$2"; shift 2;;
    --save-dir) SAVE_BASE="$2"; shift 2;;
    --splits)  SPLITS="${@:2}"; break;;
    *) echo "Argumento desconhecido: $1"; exit 1;;
  esac
done

# Roda um método só se ele estiver na lista METHODS
_want() { [[ " $METHODS " == *" $1 "* ]]; }

echo "============================================================"
echo "  FLIM AL — Schisto Corrected Experiments"
echo "  Device: $DEVICE | Splits: $SPLITS | Budgets: $BUDGETS"
echo "  Sementes por braço aleatório: $N_SEEDS"
echo "  Métodos: $METHODS"
echo "  Resultados em: flim_ad/$SAVE_BASE"
echo "============================================================"
echo ""

# Verifica se o encoder base existe
ENC_CHECK="out/trained_models/schisto/user_A/split1/flim_encoder_split1.pth"
if [ ! -f "$ENC_CHECK" ]; then
  echo "AVISO: Encoder base não encontrado: $ENC_CHECK"
  echo "Certifique-se de ter rodado o treinamento inicial antes."
fi

if _want entropy; then
  # ── Experimento 1: Entropy AL (sem DT) ───────────────────────────────────────
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [1/6] Entropy AL (sem DT) — pool∩val corrigido, MAE real"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition entropy \
    --n_seeds $N_SEEDS \
    --n_fg_markers $N_FG \
    --n_bg_markers $N_BG \
    --device "$DEVICE" \
    --save_dir "${SAVE_BASE}/entropy"

  echo ""
fi

if _want entropy_dt; then
  # ── Experimento 2: Entropy AL com DT ─────────────────────────────────────────
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [2/6] Entropy AL (com DT) — MAE real, empty-empty correto"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  DT_OK=false
  if [ -f "$DT_BIN" ]; then
    if ldd "$DT_BIN" 2>&1 | grep -q "not found"; then
      echo "  SKIP DT: biblioteca(s) compartilhada(s) ausente(s):"
      ldd "$DT_BIN" 2>&1 | grep "not found" | sed 's/^/    /'
    else
      DT_OK=true
    fi
  else
    echo "  SKIP DT: binário não encontrado em $DT_BIN"
  fi

  if [ "$DT_OK" = true ]; then
    python3 ../flim_al/al_encoder_experiment.py \
      --markers "$MARKERS" \
      --splits $SPLITS \
      --budgets $BUDGETS \
      --acquisition entropy \
      --n_seeds $N_SEEDS \
      --n_fg_markers $N_FG \
      --n_bg_markers $N_BG \
      --device "$DEVICE" \
      --use_dt \
      --dt_bin "$DT_BIN" \
      --save_dir "${SAVE_BASE}/entropy_dt"
  fi

  echo ""
fi

if _want least_confidence; then
  # ── Experimento 3: Least Confidence (= Margin) ───────────────────────────────
  # NOTA: LC e Margin são matematicamente idênticos para segmentação binária.
  # Rodamos LC como representante único de ambos.
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [3/6] Least Confidence (≡ Margin para binário)"
  echo "  NOTA: margin_score e least_confidence são idênticos neste pipeline."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition least_confidence \
    --n_seeds $N_SEEDS \
    --n_fg_markers $N_FG \
    --n_bg_markers $N_BG \
    --device "$DEVICE" \
    --save_dir "${SAVE_BASE}/least_confidence"

  echo ""
fi

if _want coreset; then
  # ── Experimento 4: CoreSet ────────────────────────────────────────────────────
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [4/6] CoreSet — features LAB corrigidas, K items corretos (era K-1)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition coreset \
    --n_seeds $N_SEEDS \
    --n_fg_markers $N_FG \
    --n_bg_markers $N_BG \
    --device "$DEVICE" \
    --save_dir "${SAVE_BASE}/coreset"

  echo ""
fi

if _want badge; then
  # ── Experimento 5: BADGE ─────────────────────────────────────────────────────
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [5/6] BADGE — pred_scores via saliency means (era dummy_w=zeros)"
  echo "         Features LAB corrigidas"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition badge \
    --n_seeds $N_SEEDS \
    --n_fg_markers $N_FG \
    --n_bg_markers $N_BG \
    --device "$DEVICE" \
    --save_dir "${SAVE_BASE}/badge"

  echo ""
fi

if _want region_bald; then
  # ── Experimento 6: Region BALD ────────────────────────────────────────────────
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [6/6] Region BALD — region_method=bald (era entropy)"
  echo "         Seeds posicionados por score BALD de região (comitê de $N_COMMITTEE)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 ../flim_al/al_encoder_experiment.py \
    --markers "$MARKERS" \
    --splits $SPLITS \
    --budgets $BUDGETS \
    --acquisition region_bald \
    --n_committee $N_COMMITTEE \
    --n_seeds $N_SEEDS \
    --n_fg_markers $N_FG \
    --n_bg_markers $N_BG \
    --device "$DEVICE" \
    --save_dir "${SAVE_BASE}/region_bald"
fi


echo ""
echo "============================================================"
echo "  Todos os experimentos concluídos."
echo "  CSVs em: flim_ad/${SAVE_BASE}/"
echo ""
echo "  Para comparar com resultados anteriores (comprometidos):"
echo "    out/al_encoder_results_dt/         ← entropy+DT (pool∩val leak)"
echo "    out/al_encoder_results_acq_comparison/ ← acq comparison (split1 só)"
echo "    out/al_region_results/             ← region_entropy (split1 só)"
echo "    out/al_bald_results/               ← region_bald (entropy, não BALD)"
echo "============================================================"
