#!/usr/bin/env bash
# =============================================================================
# run_all_dissertation.sh
# Pipeline completo para a janela de 48h. Todas as fases têm resume: se algo
# interromper, basta rodar de novo que ele continua de onde parou.
#
# Ordem por VALOR, não por custo — se a janela acabar no meio, o que já rodou
# é o que mais importa para a dissertação:
#
#   F1  E1 pool real     — Algoritmo 1 com 5 critérios (contribuição central)
#   F2  Exp A controle   — braço random_region do region_bald (fecha o confundidor)
#   F3  Exp A coreset    — reexecução (a seleção mudou com o fix do ponto-semente)
#   F4  E1 pool full     — análise de sensibilidade com markers sintéticos
#
# USO:
#   bash run_all_dissertation.sh                 # tudo
#   bash run_all_dissertation.sh --phases "1 2"  # só algumas fases
#   bash run_all_dissertation.sh --device cpu
# =============================================================================

set -uo pipefail    # sem -e: uma fase que falhe não pode derrubar as seguintes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="cuda:0"
PHASES="1 2 3 4"
SEEDS=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --phases) PHASES="$2"; shift 2;;
    --seeds)  SEEDS="$2"; shift 2;;
    *) echo "Argumento desconhecido: $1"; exit 1;;
  esac
done

# ── Trava contra execução duplicada ──────────────────────────────────────────
# Rodar duas instâncias em paralelo faz as duas escreverem no MESMO CSV, o que
# intercala linhas e duplica configurações. Com `docker exec -d` não há saída
# no terminal avisando que já havia uma rodando, então a trava é a única
# proteção. O lock é liberado no fim, inclusive se o script morrer.
LOCK="/tmp/flim_dissertation.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER=$(cat "$LOCK/pid" 2>/dev/null || echo "?")
  if [ "$OWNER" != "?" ] && kill -0 "$OWNER" 2>/dev/null; then
    echo "ERRO: já existe uma execução em andamento (PID $OWNER)."
    echo "      Acompanhe com: tail -f run_all_48h.log"
    echo "      Para forçar:   kill $OWNER && rm -rf $LOCK"
    exit 1
  fi
  echo "  (lock órfão de um processo morto — reaproveitando)"
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM

_want() { [[ " $PHASES " == *" $1 "* ]]; }
_stamp() { date +"%Y-%m-%d %H:%M:%S"; }
_banner() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  printf "║  %-60s║\n" "$1"
  printf "║  início: %-52s║\n" "$(_stamp)"
  echo "╚══════════════════════════════════════════════════════════════╝"
}

START=$(date +%s)
echo "============================================================"
echo "  Pipeline completo — dissertação FLIM + AL"
echo "  Device: $DEVICE | Fases: $PHASES | Sementes: $SEEDS"
echo "  Início: $(_stamp)"
echo "============================================================"

# ── F1 — E1 pool real ────────────────────────────────────────────────────────
if _want 1; then
  _banner "F1 — Algoritmo 1 com critério trocável (pool real, 31 imagens)"
  bash "$SCRIPT_DIR/run_selection_experiment.sh" \
    --device "$DEVICE" --pool real --seeds "$SEEDS" \
    --criteria "oracle entropy coreset badge random"
  echo "✓ F1 concluída em $(_stamp)"
fi

# ── F2 — braço de controle do Exp A ──────────────────────────────────────────
if _want 2; then
  _banner "F2 — Exp A: braço de controle random_region (region_bald)"
  echo "  O resume pula AL e random já existentes; calcula só o braço novo."
  DEVICE="$DEVICE" bash "$SCRIPT_DIR/run_schisto_corrected.sh" \
    --device "$DEVICE" --seeds "$SEEDS" --methods region_bald
  echo "✓ F2 concluída em $(_stamp)"
fi

# ── F3 — CoreSet do Exp A (seleção mudou) ────────────────────────────────────
if _want 3; then
  _banner "F3 — Exp A: reexecução do CoreSet"
  CS="$SCRIPT_DIR/flim_ad/out/al_corrected_results/coreset/schisto-user_A_coreset_encoder_al.csv"
  if [ -f "$CS" ]; then
    # Renomeia (não apaga) — os resultados antigos continuam disponíveis para
    # comparação, mas saem do caminho do resume.
    MV="${CS%.csv}_OBSOLETO_pre_fix_seedpoint.csv"
    if [ ! -f "$MV" ]; then
      mv "$CS" "$MV"
      echo "  CSV antigo preservado em: $(basename "$MV")"
    fi
  fi
  DEVICE="$DEVICE" bash "$SCRIPT_DIR/run_schisto_corrected.sh" \
    --device "$DEVICE" --seeds "$SEEDS" --methods coreset
  echo "✓ F3 concluída em $(_stamp)"
fi

# ── F4 — E1 pool completo (sensibilidade) ────────────────────────────────────
if _want 4; then
  _banner "F4 — Algoritmo 1 no pool completo (markers sintéticos)"
  echo "  Mais caro: gera saliências de centenas de imagens por rodada."
  bash "$SCRIPT_DIR/run_selection_experiment.sh" \
    --device "$DEVICE" --pool full --seeds 2 --max-images 5 \
    --criteria "oracle entropy coreset"
  echo "✓ F4 concluída em $(_stamp)"
fi

# ── Análise final ────────────────────────────────────────────────────────────
_banner "ANÁLISE — tabelas de todas as fases"
cd "$SCRIPT_DIR/flim_ad" || exit 1

python ../flim_al/analyze_selection.py --pool real \
  --out ../RESULTADOS_SELECAO.md 2>/dev/null || echo "  (E1 real sem dados)"
python ../flim_al/analyze_selection.py --pool full \
  --out ../RESULTADOS_SELECAO_FULL.md 2>/dev/null || echo "  (E1 full sem dados)"
python ../flim_al/analyze_dissertation.py \
  --out ../RESULTADOS_DISSERTACAO.md 2>/dev/null || echo "  (Exp A/B sem dados)"

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "============================================================"
echo "  Tempo total: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
echo "  Fim: $(_stamp)"
echo ""
echo "  Documentos gerados:"
echo "    RESULTADOS_SELECAO.md        E1 pool real"
echo "    RESULTADOS_SELECAO_FULL.md   E1 pool completo"
echo "    RESULTADOS_DISSERTACAO.md    Exp A e Exp B"
echo "============================================================"
