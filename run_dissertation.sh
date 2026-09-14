#!/usr/bin/env bash
# =============================================================================
# run_dissertation.sh
# Experimentos da dissertação — FLIM + Active Learning (Schistosoma mansoni)
#
# FASE 1 — Experimento A: Encoder AL (encoder retreinado com markers sintéticos)
#   Para métodos region_*, roda TRÊS braços:
#     al_<metodo>    imagens do AL      + seeds de região
#     random_region  imagens aleatórias + seeds de região   ← controle novo
#     random         imagens aleatórias + seeds em pontos
#   permitindo separar o efeito da SELEÇÃO do efeito da GEOMETRIA dos seeds.
#
# FASE 2 — Experimento B: Backprop AL (encoder FIXO + decoder 1×1)
#   Pares AL/Random com a MESMA semente de inicialização → Δ pareado.
#   Braço `full` (pool inteiro) dá o teto do decoder e o piso de ruído.
#
# Ambas retomam de onde pararam (resume por configuração).
#
# USO:
#   bash run_dissertation.sh [--device cuda:0] [--epochs 500]
#                            [--seeds 5] [--full-pool-seeds 3]
#                            [--skip-full-pool] [--skip-encoder]
#                            [--methods "region_bald coreset"]
#
#   # só Experimento B (Fase 1 já concluída):
#   bash run_dissertation.sh --device cpu --skip-encoder
#
#   # rápido: Exp B sem a referência do pool inteiro (~6x mais rápido)
#   bash run_dissertation.sh --device cpu --skip-encoder --skip-full-pool
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parâmetros ────────────────────────────────────────────────────────────────
DEVICE="cuda:0"
N_EPOCHS=500
N_SEEDS=5
FULL_POOL_SEEDS=3
SKIP_ENCODER=false
SKIP_FULL_POOL=""
METHODS_A=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)          DEVICE="$2"; shift 2;;
    --epochs)          N_EPOCHS="$2"; shift 2;;
    --seeds)           N_SEEDS="$2"; shift 2;;
    --full-pool-seeds) FULL_POOL_SEEDS="$2"; shift 2;;
    --skip-full-pool)  SKIP_FULL_POOL="--skip-full-pool"; shift;;
    --skip-encoder)    SKIP_ENCODER=true; shift;;
    --methods)         METHODS_A="$2"; shift 2;;
    *) echo "Argumento desconhecido: $1"; exit 1;;
  esac
done

# ── GPU: verificação com kernel REAL ─────────────────────────────────────────
# `torch.cuda.is_available()` devolve True mesmo sem kernel para a arquitetura.
# Numa RTX 50xx (sm_120) com PyTorch <2.7 isso só estouraria no meio do
# experimento. Aqui a checagem executa uma conv2d de verdade.
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
except Exception:
    sys.exit(3)
sys.exit(0)
EOF
  then
    echo "  ✓ GPU funcional"
  else
    echo "  ✗ GPU sem kernel utilizável para esta arquitetura."
    echo "    Para habilitar a RTX 50xx (sm_120):  bash setup_gpu_blackwell.sh"
    echo "    Prosseguindo em CPU."
    DEVICE="cpu"
  fi
fi

python3 - <<'EOF' 2>/dev/null || true
import torch
print(f"  PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    # sem :02d — para cap=(12,0) o formato antigo imprimia "sm_1200"
    print(f"  GPU:     {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
else:
    print("  GPU:     não disponível — CPU mode")
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       FLIM AL — Experimentos para Dissertação                ║"
echo "║       Schistosoma mansoni eggs — Schisto dataset             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Device:            $DEVICE"
echo "║  Epochs (Exp B):    $N_EPOCHS"
echo "║  Sementes/braço:    $N_SEEDS"
echo "║  Pool inteiro:      ${SKIP_FULL_POOL:-$FULL_POOL_SEEDS sementes}"
echo "║  Pular Fase 1:      $SKIP_ENCODER"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date +%s)

# ── FASE 1 ────────────────────────────────────────────────────────────────────
if [ "$SKIP_ENCODER" = false ]; then
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  FASE 1 — Experimento A: Encoder AL                          ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
  EXTRA_A=()
  [ -n "$METHODS_A" ] && EXTRA_A+=(--methods "$METHODS_A")
  DEVICE="$DEVICE" bash "$SCRIPT_DIR/run_schisto_corrected.sh" \
    --device "$DEVICE" --seeds "$N_SEEDS" "${EXTRA_A[@]}"
  echo ""
  echo "✓ FASE 1 concluída."
  echo ""
fi

# ── FASE 2 ────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FASE 2 — Experimento B: Backprop AL (encoder FIXO)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

DEVICE="$DEVICE" bash "$SCRIPT_DIR/run_schisto_backprop.sh" \
  --device "$DEVICE" \
  --epochs "$N_EPOCHS" \
  --seeds "$N_SEEDS" \
  --full-pool-seeds "$FULL_POOL_SEEDS" \
  $SKIP_FULL_POOL

echo ""
echo "✓ FASE 2 concluída."
echo ""

# ── Análise ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ANÁLISE — tabelas com média ± desvio e teste pareado        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "$SCRIPT_DIR/flim_ad"
python3 ../flim_al/analyze_dissertation.py --out ../RESULTADOS_DISSERTACAO.md

echo ""
echo "Tempo total: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
echo ""
echo "Arquivos gerados:"
echo "  Tabelas prontas:   RESULTADOS_DISSERTACAO.md"
echo "  Exp A (por linha): flim_ad/out/al_corrected_results/{método}/*_encoder_al.csv"
echo "  Exp B (por run):   flim_ad/out/al_backprop_results/{método}/*_runs.csv"
echo ""
