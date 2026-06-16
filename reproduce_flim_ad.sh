#!/bin/bash
# ============================================================
# REPRODUÇÃO COMPLETA — FLIM-AD (arXiv 2504.20872)
# Uso: bash reproduce_flim_ad.sh [--step N] [--from N] [--no-brats] [--cpu]
#
# Steps:
#   0  — instalar deps sistema
#   1  — clonar repo flim_ad
#   2  — instalar pyflim + fix faiss
#   3  — compilar IFT
#   4  — preparar dataset Parasites
#   5  — patch paths hardcoded nos scripts
#   6  — treinar encoders FLIM (Parasites)
#   7  — treinar decoder backprop (Parasites)
#   8  — treinar U-NetFLIM (Parasites)
#   9  — validação decoders (Parasites)
#   10 — delineação val (Parasites)
#   11 — selecionar melhor camada (Parasites)
#   12 — teste final (Parasites)
#   13 — pipeline BraTS completo
#   14 — tabela paper vs reprodução
# ============================================================
set -e

SKIP_BRATS=0
DEVICE="cuda:0"
ONLY_STEP=""
FROM_STEP=0

for arg in "$@"; do
    case $arg in
        --no-brats)   SKIP_BRATS=1 ;;
        --cpu)        DEVICE="cpu" ;;
        --step=*)     ONLY_STEP="${arg#*=}" ;;
        --from=*)     FROM_STEP="${arg#*=}" ;;
    esac
done

# REPO_DIR dentro do volume montado → persiste entre containers
REPO_DIR="/workspace/flim-python-demo/flim_ad"
DATASET_HOME="$REPO_DIR/datasets"

should_run() {
    local n=$1
    [ -n "$ONLY_STEP" ] && { [ "$ONLY_STEP" = "$n" ] && return 0 || return 1; }
    [ "$n" -ge "$FROM_STEP" ] && return 0 || return 1
}

# ─── Preamble: deps sistema + venv (sempre — container é efêmero) ────
echo "▶ [pre] Instalando dependências do sistema..."
apt-get update -qq && apt-get install -y -qq \
    git wget unzip make gcc g++ \
    libatlas-base-dev libblas-dev liblapack-dev libomp-dev 2>&1 | tail -2
echo "  OK"

VENV="/workspace/flim-python-demo/flim_venv"
export PIP_CACHE_DIR="/workspace/flim-python-demo/.pip_cache"
export NEWIFT_DIR="$REPO_DIR/libs/ift"
mkdir -p "$PIP_CACHE_DIR"

# Versão python do container (pode ser 3.10, 3.11, 3.12)
SYS_PY=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")

# Se venv existe mas foi criado com python diferente → deleta
if [ -f "$VENV/bin/python" ]; then
    VENV_PY=$("$VENV/bin/python" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    if [ "$VENV_PY" != "$SYS_PY" ]; then
        echo "  venv desatualizado ($VENV_PY vs $SYS_PY) — recriando..."
        rm -rf "$VENV"
    fi
fi

if [ ! -f "$VENV/bin/python" ]; then
    echo "▶ [pre] Criando venv ($SYS_PY)..."
    python3 -m venv "$VENV" --system-site-packages
fi

# torch: usa o do sistema se >= 2.4 (pytorch image já tem 2.7+cu128)
TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ "$TORCH_VER" == "none" ]] || [[ "$TORCH_VER" < "2.4" ]]; then
    echo "  torch no venv: '$TORCH_VER' — tentando PyPI..."
    "$VENV/bin/pip" install -q --force-reinstall torch torchvision 2>&1 | tail -3
fi
echo "  torch: $("$VENV/bin/python" -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Deps: instala pyflim requirements + extras do pipeline
# --no-deps só para monai (evita sobrescrever torch cu128)
NEED_DEPS=$("$VENV/bin/python" -c "import transformers, monai, faiss, sklearn, pandas, cv2" 2>/dev/null && echo "ok" || echo "missing")
if [ "$NEED_DEPS" = "missing" ]; then
    echo "  Instalando dependências do pyflim + pipeline..."
    # pyflim deps (sem torch/torchvision — já no sistema)
    "$VENV/bin/pip" install -q \
        "numpy==1.26.4" \
        "pillow>=10.2.0" \
        "scikit-image>=0.25.2" \
        "scikit-learn>=1.6.1" \
        "faiss-cpu>=1.10.0" \
        "matplotlib" \
        "random2" 2>&1 | tail -2
    # monai sem deps para não reinstalar torch
    "$VENV/bin/pip" install -q "monai>=1.4.0" --no-deps 2>&1 | tail -1
    # deps extras do pipeline flim_ad
    "$VENV/bin/pip" install -q \
        "transformers" \
        "pandas" \
        "opencv-python-headless" \
        "tqdm" \
        "einops" \
        "nibabel" \
        "scipy" 2>&1 | tail -2
fi

# pyflim → site-packages (path dinâmico)
PYFLIM_SRC="/workspace/flim-python-demo/flim_ad/libs/flim-python/pyflim"
SITE_PKG="$VENV/lib/$SYS_PY/site-packages"
if [ -d "$PYFLIM_SRC" ] && [ -d "$SITE_PKG" ]; then
    cp -rf "$PYFLIM_SRC" "$SITE_PKG/"
    echo "  pyflim → $SITE_PKG"
fi

# Ativa venv: PATH modificado é herdado por subshells (bash scripts/...)
set +e  # desativa set -e temporariamente para source não matar o script
source "$VENV/bin/activate"
ACTIVATE_EXIT=$?
set -e
if [ $ACTIVATE_EXIT -ne 0 ]; then
    echo "  WARN: source activate falhou, usando python do venv diretamente"
    export PATH="$VENV/bin:$PATH"
fi
export PYTHONDONTWRITEBYTECODE=1
echo "▶ [pre] numpy: $($VENV/bin/python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'N/A')"

# Patch de device sempre no preamble — substitui QUALQUER device anterior pelo atual
if [ -d "$REPO_DIR/scripts" ]; then
    GILSON_PATH="/home/gilson/Documents/datasets"
    for f in "$REPO_DIR"/scripts/schisto/*.sh "$REPO_DIR"/scripts/brats/*.sh; do
        [ -f "$f" ] || continue
        sed -i 's|#!/bash/bin|#!/bin/bash|g'               "$f" 2>/dev/null || true
        sed -i "s|$GILSON_PATH|$DATASET_HOME|g"             "$f" 2>/dev/null || true
        # Substitui qualquer device (cpu ou cuda:X) pelo device atual
        sed -i "s|--device cpu|--device $DEVICE|g"          "$f" 2>/dev/null || true
        sed -i "s|--device cuda:[0-9]|--device $DEVICE|g"   "$f" 2>/dev/null || true
        sed -i "s|--gpu cpu|--gpu $DEVICE|g"                "$f" 2>/dev/null || true
        sed -i "s|--gpu cuda:[0-9]|--gpu $DEVICE|g"         "$f" 2>/dev/null || true
    done
fi

# ─── Step 1: clone repo ──────────────────────────────────────
if should_run 1; then
    echo "▶ [1] Clonando repositório flim_ad..."
    if [ -d "$REPO_DIR/.git" ]; then
        echo "  já existe, pulando clone"
    else
        git clone https://github.com/LIDS-UNICAMP/flim_ad "$REPO_DIR"
    fi
    echo "  OK"
fi

# cd só se o repo existir (steps 2+ dependem disso)
[ -d "$REPO_DIR" ] && cd "$REPO_DIR"

# ─── Step 2: instalar pyflim + fix faiss (primeira vez) ──────
if should_run 2; then
    echo "▶ [2] Instalando pyflim (versão do paper) + fix faiss..."
    pip install -e "$REPO_DIR/libs/flim-python/" -q
    pip install faiss-cpu --force-reinstall -q 2>&1 | tail -1
    echo "  OK — pyflim $(pip show pyflim 2>/dev/null | grep Version | awk '{print $2}')"
fi

# ─── Step 3: compilar IFT ────────────────────────────────────
if should_run 3; then
    echo "▶ [3] Compilando biblioteca IFT..."
    export NEWIFT_DIR="$REPO_DIR/libs/ift"

    # Fix bug Makefile: '$@.c: $@.c' → '%: %.c'
    AUX_MK="$NEWIFT_DIR/demo/FLIM/auxiliary_operations/Makefile"
    if grep -q '^\$@\.c:' "$AUX_MK" 2>/dev/null; then
        sed -i 's/^\$@\.c: \$@\.c/%: %.c/' "$AUX_MK"
        echo "  Makefile patched"
    fi

    mkdir -p "$NEWIFT_DIR/bin"
    make -C "$NEWIFT_DIR" -j4 -s 2>&1 | tail -3

    cd "$NEWIFT_DIR/demo/FLIM/auxiliary_operations"
    make IFT_GPU=0 iftSMansoniDelineation 2>&1 | tail -5
    cd "$REPO_DIR"
    echo "  OK"
fi

# ─── Step 4: dataset Parasites ───────────────────────────────
if should_run 4; then
    echo "▶ [4] Preparando dataset Parasites (Schistossoma)..."
    SCHISTO_DST="$DATASET_HOME/schisto"
    # Clone persistente dentro do volume montado
    SCHISTO_RAW="$DATASET_HOME/schistossoma-eggs"

    if [ ! -d "$SCHISTO_RAW/orig" ]; then
        echo "  Clonando schistossoma-eggs (persistente)..."
        git clone https://github.com/LIDS-Datasets/schistossoma-eggs "$SCHISTO_RAW"
    else
        echo "  Dataset já clonado: $SCHISTO_RAW"
    fi

    mkdir -p "$SCHISTO_DST"

    # O código espera datasets/schisto/images/ e truelabels/
    # Substitui os diretórios parciais (17 imgs) por symlinks para o dataset completo
    rm -rf "$SCHISTO_DST/images" "$SCHISTO_DST/truelabels" \
           "$SCHISTO_DST/orig"   "$SCHISTO_DST/label"
    ln -sf "$SCHISTO_RAW/orig"  "$SCHISTO_DST/images"
    ln -sf "$SCHISTO_RAW/label" "$SCHISTO_DST/truelabels"
    ln -sf "$SCHISTO_RAW/orig"  "$SCHISTO_DST/orig"
    ln -sf "$SCHISTO_RAW/label" "$SCHISTO_DST/label"

    echo "  OK — $(ls "$SCHISTO_DST/images/" | wc -l) imagens"
fi

# ─── Step 5: patch paths hardcoded ───────────────────────────
if should_run 5; then
    echo "▶ [5] Corrigindo paths hardcoded nos scripts..."
    GILSON_PATH="/home/gilson/Documents/datasets"
    for f in scripts/schisto/*.sh scripts/brats/*.sh; do
        sed -i 's|#!/bash/bin|#!/bin/bash|g'            "$f" 2>/dev/null || true
        sed -i "s|$GILSON_PATH|$DATASET_HOME|g"          "$f" 2>/dev/null || true
        sed -i "s|cuda:0|$DEVICE|g"                      "$f" 2>/dev/null || true
    done
    sed -i "s|--gpu cpu|--gpu $DEVICE|g"  scripts/schisto/train_flim_unet.sh 2>/dev/null || true
    sed -i "s|--gpu cpu|--gpu $DEVICE|g"  scripts/brats/train_flim_unet.sh   2>/dev/null || true
    sed -i "s|--epochs 1|--epochs 100|g"  scripts/schisto/train_flim_unet.sh 2>/dev/null || true
    sed -i "s|--epochs 1|--epochs 100|g"  scripts/brats/train_flim_unet.sh   2>/dev/null || true
    echo "  OK"
fi

# ─── Step 6–12: Pipeline Parasites ───────────────────────────
if should_run 6;  then echo "▶ [6]  Treinando encoders FLIM (Parasites)..."; bash scripts/schisto/train_flim_encoders.sh;   echo "  OK"; fi
if should_run 7;  then echo "▶ [7]  Treinando decoder backprop (FLIMpb)..."; bash scripts/schisto/train_backprop_decoder.sh; echo "  OK"; fi
if should_run 8;  then echo "▶ [8]  Treinando U-NetFLIM...";                  bash scripts/schisto/train_flim_unet.sh;         echo "  OK"; fi
if should_run 9;  then echo "▶ [9]  Validação decoders (Parasites)...";       bash scripts/schisto/val_run_flim_decoders.sh;   echo "  OK"; fi
if should_run 10; then echo "▶ [10] Delineação val (Dynamic Trees)...";        bash scripts/schisto/val_run_delineation.sh;     echo "  OK"; fi
if should_run 11; then echo "▶ [11] Computando métricas val...";               bash scripts/schisto/val_compute_metrics.sh;     echo "  OK"; fi
if should_run 12; then echo "▶ [12] Selecionando melhor camada...";            bash scripts/schisto/val_get_best_layers.sh;     echo "  OK"; fi
if should_run 13; then echo "▶ [13] Teste final Parasites...";                 bash scripts/schisto/test_flim_decoders.sh;      echo "  OK"; fi
if should_run 14; then echo "▶ [14] Métricas de teste (Parasites)...";         bash scripts/schisto/test_compute_metrics.sh;    echo "  OK"; fi

# ─── Step 15: Pipeline BraTS ─────────────────────────────────
if should_run 15 && [ $SKIP_BRATS -eq 0 ]; then
    BRATS_DST="$DATASET_HOME/brats"
    if [ ! -d "$BRATS_DST/images" ]; then
        echo "  ⚠  BraTS não encontrado em $BRATS_DST/images/ — use --no-brats ou coloque o dataset primeiro"
    else
        echo "▶ [13] Pipeline BraTS..."
        bash scripts/brats/train_flim_encoders.sh
        bash scripts/brats/train_backprop_decoder.sh
        bash scripts/brats/train_flim_unet.sh
        bash scripts/brats/val_run_flim_decoders.sh
        bash scripts/brats/val_get_best_layers.sh
        bash scripts/brats/test_flim_decoders.sh
        echo "  OK"
    fi
fi

# ─── Step 14: Tabela comparativa ─────────────────────────────
if should_run 14; then
    echo "▶ [14] Tabela paper vs reprodução..."
    python3 - << 'PYEOF'
import json, glob

PAPER = {
    "FLIMts":  {"Parasites": {"A":(0.010,0.747), "B":(0.013,0.687)}, "BraTS": {"A":(0.017,0.709), "B":(0.020,0.697)}},
    "FLIMat":  {"Parasites": {"A":(0.011,0.740), "B":(0.014,0.660)}, "BraTS": {"A":(0.022,0.679), "B":(0.024,0.694)}},
    "FLIMpb":  {"Parasites": {"A":(0.006,0.857), "B":(0.006,0.847)}, "BraTS": {"A":(0.019,0.703), "B":(0.022,0.691)}},
}

results = {}
for f in glob.glob("output/**/metrics*.json", recursive=True) + glob.glob("out/**/metrics*.json", recursive=True):
    try:
        with open(f) as fp: results[f] = json.load(fp)
    except: pass

print("\n" + "="*70)
print(f"  {'Modelo':<10} {'Dataset':<10} {'User':<6} {'MAE paper':>10} {'MAE real':>10} {'Fβ paper':>9} {'Fβ real':>9}")
print("-"*70)
for model, datasets in PAPER.items():
    for ds, users in datasets.items():
        for user, (pmae, pfb) in users.items():
            print(f"  {model:<10} {ds:<10} {user:<6} {pmae:>10.3f} {'N/A':>10} {pfb:>9.3f} {'N/A':>9}")
if not results:
    print("\n  [INFO] Sem resultados reais ainda — rode steps 6-12 primeiro.")
print("="*70)
PYEOF
fi

echo ""
echo "✓ Concluído. Resultados em: $REPO_DIR/out/"
