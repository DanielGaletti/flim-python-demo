#!/usr/bin/env bash
# =============================================================================
# arquivar_resultados.sh — copia os CSV/JSON de resultado para dentro do git
#
# Por que existe
#   `flim_ad/out/` tem 230 mil arquivos (228 mil PNG de predição, 1.400
#   checkpoints .pth) e 3,4 GB. Versionar aquilo é inviável, e ignorá-lo por
#   inteiro deixa a EVIDÊNCIA experimental — 476 CSVs, 3,9 MB — fora do
#   histórico, numa máquina só, sem backup.
#
#   Este script copia só os arquivos tabulares para `evidencia/bruto/`,
#   preservando a estrutura de diretórios, de modo que o caminho de origem
#   continue legível. `flim_ad/out/` segue ignorado e os scripts de experimento
#   continuam escrevendo lá — nada muda de lugar.
#
#   A duplicação é transitória: a partir da Fase B o runner grava direto no
#   formato canônico em `evidencia/execucoes/`.
#
# USO
#   bash scripts/arquivar_resultados.sh           # copia o que mudou
#   bash scripts/arquivar_resultados.sh --dry-run # só lista
# =============================================================================
set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGEM="$RAIZ/flim_ad/out"
DESTINO="$RAIZ/evidencia/bruto"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

if [[ ! -d "$ORIGEM" ]]; then
  echo "ERRO: $ORIGEM não existe — nada a arquivar." >&2
  exit 1
fi

copiados=0
inalterados=0
bytes=0

while IFS= read -r -d '' src; do
  rel="${src#"$ORIGEM/"}"
  dst="$DESTINO/$rel"
  if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
    inalterados=$((inalterados + 1))
    continue
  fi
  if [[ $DRY -eq 1 ]]; then
    echo "  copiaria: $rel"
  else
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
  fi
  copiados=$((copiados + 1))
  bytes=$((bytes + $(stat -c%s "$src")))
done < <(find "$ORIGEM" \( -name '*.csv' -o -name '*.json' \) -type f -print0)

printf '%s: %d arquivo(s), %.1f MB · %d já iguais\n' \
  "$([[ $DRY -eq 1 ]] && echo 'copiaria' || echo 'arquivados')" \
  "$copiados" "$(echo "$bytes" | awk '{print $1/1048576}')" "$inalterados"

if [[ $DRY -eq 0 && $copiados -gt 0 ]]; then
  echo
  echo "Revise e commite:"
  echo "  git add evidencia/bruto && git commit -m 'evidência: novos resultados'"
fi
