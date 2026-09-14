#!/usr/bin/env bash
# =============================================================================
# run_schisto_backprop.sh
# Experimento B — encoder FLIM FIXO + decoder 1×1 treinado por backprop.
#
# Protocolo (revisado)
#   Para cada (split, budget) rodamos N_SEEDS pares AL/Random. O par de índice
#   s compartilha a MESMA semente de inicialização Xavier, então
#       Δ_s = Fβ(AL,s) − Fβ(Random,s)
#   cancela o ruído de inicialização e isola a seleção de imagens.
#
#   Além disso rodamos o pool inteiro (braço `full`) com FULL_POOL_SEEDS
#   sementes: a média é o TETO do decoder e o desvio é o PISO DE RUÍDO — a
#   régua contra a qual todo Δ deve ser lido.
#
# Correções ativas
#   [C1] eps=1e-6 na entropia (acquisition.py)
#   [C2] pool ∩ val = ∅
#   [B2] val sem as imagens de treino do encoder (igual ao Exp A)
#   [LAB] features e treino em espaço LAB
#   [SEED] inicialização semeada e pareada entre braços
#   [MASK] loss de região com peso por pixel (flim_al/losses.py)
#   [CORESET] sem duplicatas quando budget ≥ N
#
# USO:
#   bash run_schisto_backprop.sh [--device cuda:0] [--epochs 500]
#                                [--seeds 5] [--full-pool-seeds 3]
#                                [--skip-full-pool]
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/flim_ad" || { echo "ERRO: execute de dentro de flim-python-demo/"; exit 1; }

# ── Parâmetros ────────────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cuda:0}"
N_EPOCHS=500
N_SEEDS=5
FULL_POOL_SEEDS=3
SKIP_FULL_POOL=""
BUDGETS="3 5 10"
SAVE_BASE="out/al_backprop_results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)           DEVICE="$2"; shift 2;;
    --epochs)           N_EPOCHS="$2"; shift 2;;
    --seeds)            N_SEEDS="$2"; shift 2;;
    --full-pool-seeds)  FULL_POOL_SEEDS="$2"; shift 2;;
    --skip-full-pool)   SKIP_FULL_POOL="--skip_full_pool"; shift;;
    *) echo "Argumento desconhecido: $1"; exit 1;;
  esac
done

# ── Checagem de GPU ───────────────────────────────────────────────────────────
# Verificação com kernel REAL: `torch.cuda.is_available()` devolve True mesmo
# quando não há kernel compilado para a arquitetura (caso da RTX 50xx/sm_120),
# e a falha só apareceria no meio do experimento.
echo "============================================================"
echo "  Verificando GPU"
echo "============================================================"
if [ "$DEVICE" != "cpu" ]; then
  if python3 - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    sys.exit(2)
try:
    import torch.nn.functional as F
    x = torch.randn(1, 4, 32, 32, device="cuda")
    w = torch.randn(8, 4, 3, 3, device="cuda")
    F.conv2d(x, w, padding=1).sum().item()
    torch.cuda.synchronize()
except Exception as e:
    print(f"  kernel falhou: {type(e).__name__}", file=sys.stderr)
    sys.exit(3)
sys.exit(0)
EOF
  then
    echo "  ✓ GPU funcional — usando $DEVICE"
  else
    echo "  ✗ GPU presente mas sem kernel utilizável (ou ausente)."
    echo "    Para habilitar a RTX 50xx: bash ../setup_gpu_blackwell.sh"
    echo "    Seguindo em CPU."
    DEVICE="cpu"
  fi
fi

python3 - <<'EOF' 2>/dev/null || true
import torch
print(f"  PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"  GPU:     {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
else:
    print("  GPU:     não disponível")
EOF

echo ""
echo "============================================================"
echo "  Experimento B — Backprop AL (encoder fixo)"
echo "  Device:          $DEVICE"
echo "  Epochs:          $N_EPOCHS"
echo "  Budgets:         $BUDGETS"
echo "  Sementes/braço:  $N_SEEDS  (AL e Random pareados)"
echo "  Pool inteiro:    ${SKIP_FULL_POOL:-$FULL_POOL_SEEDS sementes}"
echo "  Output:          flim_ad/$SAVE_BASE"
echo "============================================================"

run_method() {
  local METHOD=$1
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Método: $METHOD"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python3 ../flim_al/al_flim_backprop.py \
    --markers          schisto/user_A \
    --splits           1 2 3 \
    --budgets          $BUDGETS \
    --acquisition      "$METHOD" \
    --n_epochs         "$N_EPOCHS" \
    --n_seeds          "$N_SEEDS" \
    --full_pool_seeds  "$FULL_POOL_SEEDS" \
    --device           "$DEVICE" \
    --save_dir         "${SAVE_BASE}/${METHOD}" \
    $SKIP_FULL_POOL
}

run_method entropy
run_method coreset
run_method badge
run_method region_entropy

echo ""
echo "============================================================"
echo "  Experimento B concluído. CSVs brutos (1 linha por execução):"
for METHOD in entropy coreset badge region_entropy; do
  CSV="${SAVE_BASE}/${METHOD}/schisto-user_A_${METHOD}_runs.csv"
  if [ -f "$CSV" ]; then
    echo "  ✓ $METHOD → $CSV ($(($(wc -l < "$CSV") - 1)) execuções)"
  else
    echo "  ✗ $METHOD → não gerado"
  fi
done
echo "============================================================"
echo ""
echo "  Tabelas agregadas:"
echo "    python3 ../flim_al/analyze_dissertation.py --out ../RESULTADOS_DISSERTACAO.md"
echo ""
