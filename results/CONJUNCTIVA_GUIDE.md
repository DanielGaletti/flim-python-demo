# Conjunctiva Experiment Guide

Dataset: Conjunctiva Segmentation (Kaggle - mdraselsarker)
Split: 57 train / 18 val / 8 test
Modelos: SAMNet, MSCNet, MEANet, UNet, UNetFLIM, FLIMts, FLIMat, FLIMpb
Métricas: MAE, Fβ (β²=0.3), IoU

## Rodar no container

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  jleomelo/flim-python:v1 \
  python run_conjunctiva.py
```

## Resultados salvos em

`results/conjunctiva/conjunctiva_results.json`

## Estrutura de dados criada

```
data/conjunctiva/
├── images/         # 83 imagens .jpg
├── labels/         # 83 máscaras .png (binário 0/255)
├── splits/
│   ├── train.txt   # 57 imagens
│   ├── valid.txt   # 18 imagens
│   └── test.txt    # 8 imagens
└── markers/        # 5 arquivos -seeds.txt para FLIM
```

## Notas

- Markers gerados automaticamente via centroide da máscara (FG) + cantos (BG)
- FLIM treinado nas 5 primeiras imagens de train (imagens com marcadores)
- Supervised baselines usam TODAS as 57 imagens de train + REPEAT=8
- Avaliação feita no test set (8 imagens) com threshold Otsu
- IoU calculado sobre a máscara binarizada (mesmo threshold Otsu)

## Resultados esperados (após rodar)

Ver `results/conjunctiva/conjunctiva_results.json`
