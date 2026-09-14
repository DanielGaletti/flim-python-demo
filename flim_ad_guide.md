# Guia de Reprodução — flim_ad (repositório correto)
**Repo:** https://github.com/LIDS-UNICAMP/flim_ad  
**Artigo:** FLIM-based Salient Object Detection Networks with Adaptive Decoders (arXiv 2504.20872)

> ⚠️ O `flim-python-demo` NÃO é o repo correto. Confirmado pelo autor via e-mail.

---

## Diferenças críticas vs flim-python-demo

| Item | flim-python-demo (errado) | flim_ad (correto) |
|---|---|---|
| Biblioteca | pip pyflim (desatualizada) | `libs/flim-python/` local |
| Split Parasites | 70-30 | 50-50 |
| Marker files | ausentes | `data/schisto/user_A/split{1-3}/markers/` |
| IFT (delineação) | ausente | `libs/ift/` — necessário compilar |
| Pipeline | script único | 5 etapas sequenciais |
| Seleção de camada | fixa | `val_get_best_layers.sh` otimiza na val |

---

## Setup no Docker

```bash
# 1. Clonar
git clone https://github.com/LIDS-UNICAMP/flim_ad /workspace/flim_ad
cd /workspace/flim_ad

# 2. Instalar biblioteca FLIM correta
pip install -e libs/flim-python/

# 3. Compilar IFT (necessário para Parasites)
export NEWIFT_DIR=libs/ift
make -C libs/ift/
bash libs/ift/demo/FLIM/auxiliary_operations/compile.sh 0
```

---

## Datasets

### Parasites (Schistossoma)
```bash
# Clonar dataset
git clone https://github.com/LIDS-Datasets/schistossoma-eggs datasets/schisto_raw

# Estrutura esperada após organizar:
# datasets/schisto/orig/    ← imagens (de schistossoma-eggs/orig/)
# datasets/schisto/label/   ← máscaras (de schistossoma-eggs/label/)
# OBS: usar 50-50 split, NÃO o split do repo (70-30)
# Os splits corretos já estão em data/schisto/user_A/split{1,2,3}/train{1,2,3}.csv
```

### BraTS
```bash
# Baixar de: https://www.synapse.org/Synapse:syn51156910/wiki/622351
# Processar 3D → 2D (slices axiais)
# Selecionar slices corretos usando data/brats/user_A/split{1,2,3}/train{1,2,3}.csv
# Nome do arquivo: BraTS2021_00000_a1_s0072.png → usar slice s0072
```

---

## Pipeline (5 etapas obrigatórias)

### Etapa 1 — Treinar encoders FLIM
```bash
cd scripts/schisto/   # ou scripts/brats/
bash train_flim_encoders.sh
# Treina para user_A e user_B, splits 1-3
# GPU: cuda:0 por padrão — editar script se necessário
```

### Etapa 1.1 — Treinar decoder backprop (FLIMpb)
```bash
bash train_backprop_decoder.sh
```

### Etapa 1.2 — Treinar U-NetFLIM
```bash
bash train_flim_unet.sh
bash test_flim_unet.sh
# Modelos → out/models/
# Saliências → out/saliencies_unet/
```

### Etapa 2 — Rodar FLIM na validação
```bash
bash val_run_flim_decoders.sh
# Roda todos os decoders adaptativos (FLIMts, FLIMat, FLIMpb)
# Saída → out/saliencies/
```

### Etapa 3 — Delineação (SOMENTE Parasites)
```bash
bash val_run_delineation.sh
# Dynamic Trees algorithm via IFT
# Pós-processamento das saliency maps
```

### Etapa 4 — Selecionar melhor arquitetura
```bash
bash val_get_best_layers.sh
# Gera: output/best_layers_X.json
# Contém: melhor camada por user, decoder, split
```

### Etapa 5 — Teste final
```bash
bash test_flim_decoders.sh
# Pipeline completo no test set com a melhor camada
# Resultados finais aqui
```

---

## Comparação esperada (Paper Table III)

| Modelo | Dataset | MAE (A/B) | Fβ (A/B) |
|---|---|---|---|
| FLIMts | Parasites | 0.010/0.013 | 0.747/0.687 |
| FLIMat | Parasites | 0.011/0.014 | 0.740/0.660 |
| FLIMpb | Parasites | 0.006/0.006 | 0.857/0.847 |
| FLIMts | BraTS | 0.017/0.020 | 0.709/0.697 |
| FLIMat | BraTS | 0.022/0.024 | 0.679/0.694 |
| FLIMpb | BraTS | 0.019/0.022 | 0.703/0.691 |

---

## Comando Docker completo

```bash
# Na pasta Flim2/ (onde está flim_ad/)
docker run -it --gpus all \
  -v %cd%:/workspace \
  -w /workspace/flim_ad \
  jleomelo/flim-python:v1 \
  bash
# Dentro do container:
# pip install -e libs/flim-python/
# export NEWIFT_DIR=libs/ift && make -C libs/ift/
# bash libs/ift/demo/FLIM/auxiliary_operations/compile.sh 0
# cd scripts/schisto && bash train_flim_encoders.sh
```
