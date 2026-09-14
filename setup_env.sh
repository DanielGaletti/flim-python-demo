#!/bin/bash
# ============================================================
# setup_env.sh — Instala todas as dependências no container
# Rodar UMA VEZ ao entrar num container limpo:
#   bash /workspace/flim-python-demo/setup_env.sh
# ============================================================
set -e

echo "=== [1/3] Downgrade numpy para <2.0 (compatibilidade faiss/scipy/skimage) ==="
pip install "numpy<2.0" -q
echo "numpy: $(python -c 'import numpy; print(numpy.__version__)')"

echo ""
echo "=== [2/3] Instalando monai e transformers (requeridos pelo pyflim libs) ==="
pip install monai transformers -q
echo "monai: $(python -c 'import monai; print(monai.__version__)')"

echo ""
echo "=== [3/3] Verificando dependências principais ==="
python -c "
import torch
import numpy as np
import PIL
import sklearn
print(f'  torch:   {torch.__version__}')
print(f'  numpy:   {np.__version__}')
print(f'  PIL:     {PIL.__version__}')
print(f'  sklearn: {sklearn.__version__}')
try:
    import monai
    print(f'  monai:   {monai.__version__}')
except: print('  monai:   FALTA')
try:
    import transformers
    print(f'  transformers: {transformers.__version__}')
except: print('  transformers: FALTA')
try:
    import faiss
    print(f'  faiss:   OK')
except: print('  faiss:   indisponivel (fallback sklearn OK)')
"

echo ""
echo "=== Setup concluido! Agora rode: ==="
echo "  cd /workspace/flim-python-demo/flim_ad"
echo "  bash ../flim_al/run_al_k3.sh"
