#!/bin/bash
# Baixa o dataset completo via Python (urllib) e prepara as pastas
set -e

WORKSPACE="/workspace"
ZIP_URL="https://github.com/LIDS-Datasets/schistossoma-eggs/archive/refs/heads/main.zip"
TMP_ZIP="/tmp/schistossoma.zip"
TMP_DIR="/tmp/schistossoma-eggs-main"

echo "=== [1/4] Baixando dataset via Python (~500MB, pode demorar) ==="
if [ ! -f "$TMP_ZIP" ]; then
    python3 - <<PYEOF
import urllib.request, sys

url = "$ZIP_URL"
dest = "$TMP_ZIP"

def progress(count, block, total):
    pct = min(count * block * 100 // total, 100)
    print(f"\r  {pct}% ({count*block//1024//1024}MB / {total//1024//1024}MB)", end="", flush=True)

print("  Iniciando download...")
urllib.request.urlretrieve(url, dest, reporthook=progress)
print("\n  Download completo.")
PYEOF
else
    echo "  ZIP já existe, pulando download."
fi

echo "=== [2/4] Extraindo ZIP ==="
if [ ! -d "$TMP_DIR" ]; then
    python3 -c "import zipfile; zipfile.ZipFile('$TMP_ZIP').extractall('/tmp/')"
    echo "  Extraído em $TMP_DIR"
else
    echo "  Já extraído, pulando."
fi

echo "=== [3/4] Copiando imagens orig ==="
cp -u "$TMP_DIR"/orig/*.png "$WORKSPACE/data/orig/"
echo "  data/orig/ agora tem: $(ls $WORKSPACE/data/orig/ | wc -l) imagens"

echo "=== [4/4] Copiando labels ==="
cp -u "$TMP_DIR"/label/*.png "$WORKSPACE/data/label/"
echo "  data/label/ agora tem: $(ls $WORKSPACE/data/label/ | wc -l) labels"

SPLITS_SRC="$TMP_DIR/Splits-5train-70_30"
if [ -d "$SPLITS_SRC" ]; then
    echo "=== Copiando split files oficiais ==="
    find "$SPLITS_SRC" -name "*.txt" -exec cp -u {} "$WORKSPACE/" \;
    echo "  Splits copiados: $(find $SPLITS_SRC -name '*.txt' | wc -l) arquivos"
fi

echo ""
echo "=== PRONTO ==="
echo "Imagens orig: $(ls $WORKSPACE/data/orig/ | wc -l)"
echo "Labels:       $(ls $WORKSPACE/data/label/ | wc -l)"
