# FLIM — Resultados e Análise do Experimento

## Setup

- **Método:** FLIMpb (`decoder_type="decoder_3"`)
- **Dataset:** Schistossoma Eggs — 1219 imagens, 366 no test set
- **Splits:** 3 splits com 5 imagens de treino cada
- **Arquitetura:** `arch_best.json` — selecionada por architecture search no split1-val (100 imgs)

```json
dilations=[1, 2, 5, 9], channels=[16, 12, 8, 6], kpm=15
```

---

## Resultados no Test Set

| Experimento | MAE | Fβ | DICE |
|---|---|---|---|
| User_A_split1 (example markers) | 0.0205 | 0.3582 | 0.3985 |
| User_A_split2 (example markers) | 0.0214 | 0.2703 | 0.3320 |
| User_B_split3 (example markers) | 0.0177 | 0.3058 | 0.3473 |
| **Paper TABLE IV — User A (FLIMpb)** | **0.005** | **0.857** | — |
| **Paper TABLE IV — User B (FLIMpb)** | **0.005** | **0.843** | — |

---

## Análise do Gap vs Paper

### 1. Architecture Search Parcial
A busca de arquitetura foi feita apenas no split1-val e aplicada globalmente.  
No paper, cada split tem sua própria arquitetura otimizada no val set correspondente.  
Resultado no val: Fβ=0.4809 (split1). No test: Fβ=0.3582 — overfitting ao val.

### 2. Example Markers ≠ Anotações Reais
O repositório fornece apenas `example-markers` (aproximações). O paper utilizou anotações reais de dois usuários especialistas (User A e User B).

**Teste com Oracle Markers (ground-truth erosion):**

| Oracle | MAE | Fβ | DICE |
|---|---|---|---|
| Oracle_split1 | 0.0129 | 0.0000 | 0.4945 |

**Causa:** Markers derivados de erosão são muito homogêneos. O k-means da última camada (C=6 canais) colapsa todos os canais para uma única classe → pesos do decoder ficam zerados → saliência constante.

**Conclusão:** A diversidade visual dos markers reais (bordas, texturas variadas) é essencial para o k-means do FLIM criar kernels discriminativos.

### 3. Post-processing não implementado
O paper aplica Dynamic Trees após o FLIMpb para refinar a segmentação. Este passo não está no pipeline atual.

---

## Bugs Corrigidos no Código

| Arquivo | Linha | Bug | Fix |
|---|---|---|---|
| `pyflim/flim.py` | 446 | `forward()` com 3 args posicionais (aceita 2) | Usar keyword arg `decoder_layer=decoder_layer` |
| `pyflim/flim.py` | 464 | `decoder_layer=-1` não tratado → NoneType error | Tratar `-1` igual a `None` |
| `pyflim/layers.py` | 609 | `torch.from_numpy(y)` quando `y` já é Tensor | Remover `from_numpy()` |
| `pyflim/layers.py` | 731 | `mean_0` referenciado antes de atribuição quando `foreground_weights=0` | Guard `if foreground_weights != 0 and background_weights != 0` |
| `pyflim/metrics.py` | 689 | `saliency` UnboundLocalError quando arquivo não encontrado | Adicionar `continue` |
| `pyflim/metrics.py` | 705 | MAE com uint8 → underflow (`np.uint8(0)-1 = 255`) | `.astype(float)` antes do MAE |

---

## Scripts Gerados

| Script | Função |
|---|---|
| `run_ab_experiment.py` | Experimento principal: 3 splits × 2 marker types no test set |
| `arch_search.py` | Architecture search round 1 (30 configs) |
| `arch_search2.py` | Architecture search round 2 (30 configs, foco em padrões promissores) |
| `gen_oracle_markers.py` | Geração de markers automáticos via erosão do ground-truth |
| `setup_full_dataset.sh` | Download e setup do dataset completo (1219 imagens) |
| `GUIA_EXECUCAO.md` | Guia de execução no container Docker |

---

## Reprodução

```bash
# 1. Subir container
docker run -it -v "<path>/flim-python-demo:/workspace" jleomelo/flim-python:v1 bash

# 2. Setup
cd /workspace && pip install -e . -q

# 3. Rodar experimento
python3 run_ab_experiment.py
```
