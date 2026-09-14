#!/usr/bin/env bash
# =============================================================================
# setup_gpu_blackwell.sh
# Habilita a RTX 50xx (sm_120 / Blackwell) para os experimentos FLIM AL.
#
# O problema
#   PyTorch 2.6 + CUDA 12.4 não traz kernels compilados para sm_120. Note que
#   `torch.cuda.is_available()` retorna True mesmo assim — a falha só aparece
#   no primeiro kernel, como:
#       "CUDA error: no kernel image is available for execution on the device"
#   Por isso este script valida com uma conv2d de verdade, não com is_available().
#
#   Suporte a sm_120 entra no PyTorch 2.7 com CUDA 12.8.
#
# USO (de dentro do container):
#   bash setup_gpu_blackwell.sh            # verifica e instala se precisar
#   bash setup_gpu_blackwell.sh --check    # só diagnostica, não instala
#
# IMPORTANTE: mudanças dentro do container somem quando ele é recriado.
# Depois que funcionar, congele a imagem a partir do HOST:
#   docker commit <container_id> flim-ad-env:gpu
# =============================================================================

set -uo pipefail
CHECK_ONLY=false
[ "${1:-}" = "--check" ] && CHECK_ONLY=true

# ── Teste real de GPU (não confia em is_available) ───────────────────────────
gpu_works() {
  python3 - <<'EOF' 2>/dev/null
import sys, torch
if not torch.cuda.is_available():
    sys.exit(2)
try:
    import torch.nn.functional as F
    x = torch.randn(1, 4, 32, 32, device="cuda")
    w = torch.randn(8, 4, 3, 3, device="cuda")
    y = F.conv2d(x, w, padding=1)
    z = (y.sum() + torch.randn(64, 64, device="cuda").mm(
         torch.randn(64, 64, device="cuda")).sum())
    torch.cuda.synchronize()
    float(z.item())
except Exception as e:
    print(f"kernel falhou: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(3)
sys.exit(0)
EOF
}

echo "============================================================"
echo "  Diagnóstico GPU"
echo "============================================================"
python3 - <<'EOF'
import torch
print(f"  PyTorch:  {torch.__version__}")
print(f"  CUDA build: {torch.version.cuda}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"  GPU:      {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
    archs = torch.cuda.get_arch_list()
    print(f"  Kernels compilados: {archs}")
    print(f"  sm_{cap[0]}{cap[1]} presente: {f'sm_{cap[0]}{cap[1]}' in archs}")
else:
    print("  GPU:      não visível para o PyTorch")
EOF

if gpu_works; then
    echo ""
    echo "  ✓ GPU FUNCIONAL — conv2d e matmul executaram."
    echo "    Rode os experimentos com --device cuda:0"
    exit 0
fi

echo ""
echo "  ✗ GPU não utilizável com o PyTorch atual."

if [ "$CHECK_ONLY" = true ]; then
    echo "    (--check: nada foi instalado)"
    exit 1
fi

# ── Pré-requisito: glibc do sistema ──────────────────────────────────────────
# Os wheels PyTorch cu128 (2.7+) são publicados com a tag manylinux_2_28, ou
# seja, exigem glibc >= 2.28. Se o sistema for mais antigo, o pip recusa os
# wheels e reporta "from versions: none" — que parece "não existe versão", mas
# na verdade é "nenhuma compatível com este sistema". Sem esta checagem, a
# instalação abaixo termina com exit 0 sem trocar nada, o que já produziu um
# falso "instalado com sucesso".
echo ""
echo "============================================================"
echo "  Pré-requisito: glibc"
echo "============================================================"
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "0")
GLIBC_MAJOR=${GLIBC%%.*}
GLIBC_MINOR=${GLIBC##*.}
echo "  glibc do sistema:      $GLIBC"
echo "  exigido pelos wheels:  2.28  (tag manylinux_2_28_x86_64)"

if [ "$GLIBC_MAJOR" -lt 2 ] || { [ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -lt 28 ]; }; then
    cat <<EOF

  ✗ INCOMPATÍVEL — glibc $GLIBC < 2.28.

  Nenhuma combinação de flags do pip resolve isso: os wheels simplesmente não
  rodam neste sistema. $(cat /etc/os-release 2>/dev/null | grep '^PRETTY_NAME' | cut -d'"' -f2)

  Opções, em ordem de esforço:

    1. CONTINUAR EM CPU  (recomendado se o prazo aperta)
         bash run_dissertation.sh --device cpu --skip-encoder --skip-full-pool
       Os resultados são idênticos aos da GPU, só mais lentos.

    2. RECONSTRUIR A IMAGEM sobre uma base com glibc >= 2.28
       (Ubuntu 22.04 = 2.35, Debian 12 = 2.36). Exige reinstalar pyflim,
       monai, scikit-image e as libs IFT — meio dia de trabalho e algum risco.

    3. USAR A IMAGEM NGC DA NVIDIA, que já traz suporte a Blackwell:
         docker pull nvcr.io/nvidia/pytorch:25.01-py3
       Também exige reinstalar as dependências do projeto por cima.

EOF
    exit 1
fi

echo "  ✓ glibc compatível"

# ── Instalação ───────────────────────────────────────────────────────────────
CUR_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "?")
echo ""
echo "============================================================"
echo "  Instalando PyTorch cu128 (suporta sm_120)"
echo "  Versão atual: $CUR_TORCH  →  será substituída"
echo "============================================================"
echo ""
echo "  Para reverter, se algo quebrar:"
echo "    pip install torch==${CUR_TORCH%%+*} torchvision --index-url https://download.pytorch.org/whl/cu124"
echo ""

# Pin explícito: sem ele, `--upgrade` num índice sem candidato compatível
# apenas reporta "Requirement already satisfied" e sai com 0.
pip install --upgrade --no-cache-dir \
    "torch>=2.7" torchvision --index-url https://download.pytorch.org/whl/cu128 \
  || pip install --upgrade --no-cache-dir --pre \
    "torch>=2.7" torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 \
  || true

# Sucesso é a versão ter MUDADO — não o exit code do pip.
NEW_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "?")
if [ "$NEW_TORCH" = "$CUR_TORCH" ]; then
    echo ""
    echo "  ✗ A versão do PyTorch não mudou ($NEW_TORCH)."
    echo "    O pip não encontrou wheel compatível com este sistema."
    echo "    Diagnostique com:"
    echo "      pip install --dry-run 'torch==999' --index-url https://download.pytorch.org/whl/cu128"
    echo "    Siga em CPU (--device cpu)."
    exit 1
fi
echo "  → instalado: $CUR_TORCH → $NEW_TORCH"

echo ""
echo "============================================================"
echo "  Revalidando com kernel real"
echo "============================================================"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"

if gpu_works; then
    echo "  ✓ GPU FUNCIONAL. Rode com --device cuda:0"
    echo ""
    echo "  Verifique que as dependências continuam de pé:"
    python3 - <<'EOF'
for mod in ("monai", "skimage", "sklearn", "numpy"):
    try:
        m = __import__(mod)
        print(f"    {mod}: OK ({getattr(m, '__version__', '?')})")
    except Exception as e:
        print(f"    {mod}: QUEBROU — {type(e).__name__}: {e}")
EOF
    echo ""
    echo "  Congele a imagem a partir do HOST para não perder isto:"
    echo "    docker commit \$(hostname) flim-ad-env:gpu"
    exit 0
fi

echo "  ✗ Ainda não funciona. Continue em CPU (--device cpu)."
exit 1
