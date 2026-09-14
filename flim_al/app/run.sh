#!/bin/bash
# run.sh — Inicia o app FLIM Local
# ==================================
# Uso (de dentro de flim_al/app/):
#   bash run.sh
#
# Ou da raiz do projeto:
#   bash flim_al/app/run.sh

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

# Instala dependências se necessário
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Instalando dependências..."
    pip3 install -r requirements.txt -q
fi

echo "=============================="
echo "  FLIM Local Training Tool"
echo "  http://localhost:8000"
echo "=============================="
echo ""

# Abre browser após 1.5s
(sleep 1.5 && python3 -c "import webbrowser; webbrowser.open('http://localhost:8000')") &

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
