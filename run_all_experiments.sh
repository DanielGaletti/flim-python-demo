#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script master: roda FLIM + baselines supervisionados e gera tabela completa
# Executar no container: bash run_all_experiments.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd /workspace

# Instalar torchvision se necessário
pip install torchvision --break-system-packages -q 2>/dev/null || true

echo "============================================================"
echo "  1. FLIM: FLIMts + FLIMat + FLIMpb × Parasites + BraTS"
echo "============================================================"
python3 run_full_comparison.py

echo ""
echo "============================================================"
echo "  2. BASELINES: SAMNet + MSCNet + MEANet × Parasites + BraTS"
echo "============================================================"
# Rodar um modelo por vez para economizar memória
python3 -m baselines.train_baselines --models SAMNet    --epochs 100
python3 -m baselines.train_baselines --models MSCNet    --epochs 100
python3 -m baselines.train_baselines --models MEANet    --epochs 100
python3 -m baselines.train_baselines --models UNet      --epochs 100
python3 -m baselines.train_baselines --models UNetFLIM  --epochs 100

echo ""
echo "============================================================"
echo "  3. GERAR TABELA COMPARATIVA COMPLETA"
echo "============================================================"
python3 gen_full_table.py

echo ""
echo "DONE → results/full_comparison_table.html"
