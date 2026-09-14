# Contexto da Dissertação — FLIM + Active Learning (Schisto)

> Documento de contexto para sessão Claude Cowork. Contém tudo necessário para analisar
> os resultados e dar continuidade ao projeto.

> ⚠ **As seções 6 e 7 (tabelas de resultados) estão OBSOLETAS.**
> Os números foram produzidos com um protocolo que continha dois problemas
> confirmados: (a) o ruído de inicialização do decoder não era controlado e
> chegava a ±0.109 de Fβ, maior que qualquer Δ reportado; (b) nos métodos
> `region_*`, os braços AL e Random usavam geradores de seeds diferentes e
> treinavam com números diferentes de imagens. Além disso, ~50% das avaliações
> do decoder `labeled_marker` eram predições degeneradas (toda-fundo, Fβ=0.4788)
> contadas como desempenho.
>
> O diagnóstico completo e o protocolo corrigido estão em
> [PROTOCOLO_REVISADO.md](PROTOCOLO_REVISADO.md). Os resultados válidos, com
> média ± desvio e teste pareado, são gerados em `RESULTADOS_DISSERTACAO.md`
> por `flim_al/analyze_dissertation.py`.

---

## 1. Visão Geral do Projeto

**Tema:** Active Learning para seleção de imagens de treinamento do encoder FLIM em segmentação de ovos de *Schistosoma mansoni* (microscopia).

**Objetivo da dissertação:** Demonstrar que AL pode selecionar quais imagens o usuário deve anotar com markers (seeds), melhorando o encoder FLIM com menos esforço humano.

**Dataset:** Schisto (`data/schisto/user_A/`) — imagens de ovos de esquistossomose.
- ~1200 imagens totais; splits 5-fold (train/val/test); usamos splits 1, 2 e 3
- Val: ~847 imagens por split (imagens sem anotação manual, avaliadas com GT masks)
- Pool de seleção AL: ~610 imagens de test/ com saliency maps pré-computados

---

## 2. Arquitetura FLIM

**FLIM (Filter Learning with Image Markers):** encoder convolucional cujos filtros são aprendidos por k-means a partir de patches extraídos nas posições de markers (seeds.txt) anotados pelo usuário em espaço de cor LAB.

**Pipeline:**
```
Imagens RGB → [conversão LAB] → Encoder FLIM → Feature maps (4 camadas)
                                                    ↓
                                           Decoders (7 tipos) → Segmentação binária
                                                    ↓
                                           Métricas: Fβ (β²=0.3), DICE, IoU, MAE
```

**Decoders avaliados:**
- `labeled_marker` — decoder padrão do paper original
- `vanilla_adaptive_decoder` — decoder adaptativo
- `vanilla_adaptive_decoder_wt` — versão ponderada
- `decoder_2` — decoder com skip connections
- `decoder_3` — decoder mais profundo
- `decoder_attention` — decoder com atenção
- `hybrid_decoder` — híbrido

**Métrica principal:** Fβ com β²=0.3 → `Fβ = (1+0.3)*PR*RC / (0.3*PR + RC + ε)`
- Penaliza mais falso-positivo (RC tem menos peso)
- empty-empty (GT vazia + predição vazia) → Fβ=1.0 (correto)

---

## 3. Dois Experimentos da Dissertação

### Experimento A — Encoder AL (encoder retreinado com sintéticos)

**Hipótese:** AL seleciona imagens mais informativas → encoder retreinado com essas imagens (+ markers sintéticos gerados automaticamente) supera seleção aleatória.

**Problema fundamental identificado:** Retreinar o encoder com markers *sintéticos* degrada o desempenho porque:
- Markers sintéticos são posicionados por Otsu/watershed (imperfeitos)
- O encoder perde a qualidade dos markers manuais originais
- Todos os métodos ficam abaixo do baseline (encoder original com 3 imagens manuais)

**Resultado esperado no Exp A:** AL > Random (mesmo que ambos abaixo do baseline), mostrando que a *seleção de imagens* importa.

### Experimento B — Backprop AL (encoder FIXO + decoder via backprop) ← PRINCIPAL

**Hipótese principal da dissertação:** Manter o encoder FLIM fixo e treinar apenas um decoder linear (1×1 conv) por backprop com DiceCELoss + Adam, usando as imagens selecionadas pelo AL.

**Por que é o experimento principal:**
- Isola o efeito do AL na *seleção de imagens* sem a confusão dos markers sintéticos
- O encoder original (3 imagens manuais) é preservado
- Mede diretamente: "quais imagens são mais valiosas para treinar um classificador linear sobre as features FLIM?"

**Status:** Ainda não executado — aguarda Fase 1 (Exp A) terminar.

---

## 4. Bugs Críticos Corrigidos

### Bugs no Pipeline AL (anteriores à sessão atual)

| ID | Arquivo | Bug | Fix |
|----|---------|-----|-----|
| C1 | `acquisition.py` | NaN em entropy: eps=1e-8 causa underflow float32 | eps=1e-6 |
| C2 | `al_encoder_experiment.py` | Pool∩Val leakage: imagens de val no pool de seleção → ~245 overlap | Filtra pool para excluir val |
| C3 | `al_encoder_experiment.py` | LC = Margin matematicamente idênticos para binário | Roda só LC como representante |
| C4 | `coreset_badge.py` | CoreSet retornava K-1 elementos | Loop `while len < n_init+budget` |
| C5 | `coreset_badge.py` | BADGE: pred_scores=zeros → gradientes zero | Usa saliency means como pred_scores |
| C6 | `coreset_badge.py` | Features CoreSet/BADGE em RGB → erradas para encoder LAB | Usa `_rgb_uint8_to_lab01()` |
| A1 | `al_encoder_experiment.py` | MAE hardcoded 0.0 no path não-DT | MAE real calculado |
| A2 | `al_encoder_experiment.py` | empty-empty → 0.0 no path não-DT | empty-empty → 1.0 |
| A4 | `region_al.py` | region_bald usava entropy em vez de BALD para scores | BALD (variância do comitê) |
| A6 | `marker_generator.py` | hash() Python não-reproduzível entre runs | hashlib.md5 |

### Bugs corrigidos nesta sessão

| ID | Arquivo | Bug | Fix |
|----|---------|-----|-----|
| B1 | `al_encoder_experiment.py` | Resume: iou faltando no dict → KeyError no resume de configs com IoU | Adiciona `iou` ao dict de resume |
| B2 | `al_encoder_experiment.py` | Val set incluía imagens de treino (000479.png com markers em todos splits) | Filtra val_fnames removendo imagens com markers no dir `split{N}/markers/` |
| FIX-CUDA | `run_dissertation.sh`, `run_schisto_backprop.sh` | RTX 5070 (sm_120/Blackwell) incompatível com PyTorch 2.6+cu124 | Detecta sm_120 e instala PyTorch cu128; fallback conda nightly |
| FIX-DT | `run_schisto_corrected.sh` | Binary `iftSMansoniDelineation` existe mas `liblapack.so.3` ausente → DT retorna Fβ=0.4788 silenciosamente | `ldd "$DT_BIN"` verifica deps antes de rodar; skip se not found |

### Fixes no Experimento B (al_flim_backprop.py) — já aplicados

| ID | Fix |
|----|-----|
| A1 | Import `_rgb_uint8_to_lab01` de coreset_badge |
| A2 | `train_backprop_on_subset`: loop explícito com LAB, não FLIMData (que usa RGB) |
| A3 | `evaluate_backprop`: retorna tuple 4 (fb, dice, mae, iou); empty-empty → 1.0; IoU real |
| A4 | `select_by_acquisition`: BADGE usa saliency means como pred_scores |
| A5 | `main()`: `_FIELDNAMES` com iou; resume via done_keys; append incremental ao CSV |

---

## 5. Estrutura de Arquivos

```
flim-python-demo/
├── run_dissertation.sh          # Script principal: Fase1 + Fase2
├── run_schisto_corrected.sh     # Fase1: Encoder AL (5 métodos)
├── run_schisto_backprop.sh      # Fase2: Backprop AL (4 métodos)
├── CONTEXTO_DISSERTACAO.md      # Este arquivo
│
├── flim_al/
│   ├── al_encoder_experiment.py # Loop Exp A (encoder retreinado)
│   ├── al_flim_backprop.py      # Loop Exp B (encoder fixo + backprop)
│   ├── acquisition.py           # entropy_score, least_confidence, etc.
│   ├── coreset_badge.py         # CoreSet, BADGE, _rgb_uint8_to_lab01
│   ├── region_al.py             # Region BALD (comitê de encoders)
│   └── marker_generator.py      # Gera seeds.txt sintéticos de GT masks
│
└── flim_ad/
    ├── out/al_corrected_results/    # CSVs do Exp A
    │   ├── entropy/                 # ✅ 336 linhas (completo)
    │   ├── least_confidence/        # ✅ 336 linhas (completo)
    │   ├── coreset/                 # ✅ 336 linhas (completo)
    │   ├── badge/                   # ✅ 336 linhas (completo)
    │   └── region_bald/             # 🔄 em progresso (~155/336)
    │
    └── out/al_backprop_results/     # CSVs do Exp B (ainda não rodou)
        ├── entropy/
        ├── coreset/
        ├── badge/
        └── region_entropy/
```

**Formato CSV (Exp A):**
`split, budget, method, decoder, fb, dice, mae, iou`
- `method`: `original_3imgs` | `al_entropy` | `rand0` | `rand1` | `rand2`
- 336 linhas = 1 header + 3 splits × (1 baseline + 3 budgets × 5 configs) × 7 decoders = 336

**Formato CSV (Exp B):**
`split, budget, method, n_epochs, al_fb, al_dice, al_mae, al_iou, rand_fb, rand_dice, rand_mae, rand_iou, delta_fb, delta_dice`

---

## 6. Resultados Experimento A — Encoder AL (COMPLETOS para 4/5 métodos)

### Baseline (encoder original, 3 imagens manuais)
| Split | Fβ labeled_marker |
|-------|------------------|
| Split 1 | 0.744 |
| Split 2 | 0.731 |
| Split 3 | 0.757 |
| **Média** | **0.744** |

> **Todos os métodos AL ficam abaixo do baseline.** Isso é esperado e explicável: markers sintéticos degradam o encoder. O que importa é AL > Random.

### Δ(AL − Random) por método e decoder — Fβ médio (3 splits × 3 budgets)

#### Entropy
| Decoder | AL | Random | Δ |
|---------|-----|--------|---|
| decoder_3 | 0.400 | 0.277 | **+0.124** ✓ |
| decoder_attention | 0.247 | 0.163 | **+0.085** ✓ |
| decoder_2 | 0.510 | 0.441 | **+0.069** ✓ |
| vanilla_adaptive_decoder_wt | 0.210 | 0.161 | +0.049 ✓ |
| vanilla_adaptive_decoder | 0.453 | 0.409 | +0.044 ✓ |
| hybrid_decoder | 0.371 | 0.377 | -0.006 |
| labeled_marker | 0.502 | 0.516 | -0.014 |

#### Least Confidence (≡ Margin para binário)
| Decoder | AL | Random | Δ |
|---------|-----|--------|---|
| decoder_3 | 0.420 | 0.275 | **+0.145** ✓ |
| decoder_2 | 0.529 | 0.439 | **+0.091** ✓ |
| decoder_attention | 0.250 | 0.162 | **+0.089** ✓ |
| vanilla_adaptive_decoder_wt | 0.214 | 0.161 | +0.053 ✓ |
| vanilla_adaptive_decoder | 0.456 | 0.408 | +0.047 ✓ |
| hybrid_decoder | 0.375 | 0.378 | -0.002 |
| labeled_marker | 0.498 | 0.518 | -0.020 |

#### CoreSet
| Decoder | AL | Random | Δ |
|---------|-----|--------|---|
| decoder_2 | 0.549 | 0.442 | **+0.108** ✓ |
| decoder_3 | 0.367 | 0.275 | **+0.091** ✓ |
| decoder_attention | 0.162 | 0.162 | -0.000 |
| labeled_marker | 0.504 | 0.517 | -0.013 |
| vanilla_adaptive_decoder | 0.392 | 0.409 | -0.017 |
| vanilla_adaptive_decoder_wt | 0.139 | 0.161 | -0.021 |
| hybrid_decoder | 0.367 | 0.377 | -0.010 |

#### BADGE
| Decoder | AL | Random | Δ |
|---------|-----|--------|---|
| vanilla_adaptive_decoder | 0.415 | 0.408 | +0.006 ✓ |
| hybrid_decoder | 0.379 | 0.377 | +0.002 ✓ |
| labeled_marker | 0.518 | 0.518 | -0.000 |
| decoder_2 | 0.434 | 0.439 | -0.005 |
| decoder_attention | 0.132 | 0.162 | -0.029 |
| vanilla_adaptive_decoder_wt | 0.128 | 0.161 | -0.033 |
| decoder_3 | 0.225 | 0.274 | **-0.049** ✗ |

> **BADGE é o único método que claramente perde para random.** Resultado relevante para dissertação.

#### Region BALD (parcial — Splits 1+2, Split 3 em progresso)
| Decoder | AL | Random | Δ |
|---------|-----|--------|---|
| decoder_3 | 0.582 | 0.302 | **+0.280** ✓ |
| decoder_2 | 0.598 | 0.460 | **+0.138** ✓ |
| decoder_attention | 0.305 | 0.174 | **+0.131** ✓ |
| hybrid_decoder | 0.492 | 0.398 | +0.093 ✓ |
| vanilla_adaptive_decoder_wt | 0.283 | 0.191 | +0.092 ✓ |
| labeled_marker | 0.590 | 0.545 | **+0.045** ✓ |
| vanilla_adaptive_decoder | 0.459 | 0.433 | +0.026 ✓ |

> **Region BALD é o método mais forte — único onde labeled_marker também melhora com AL (+0.045).** Usa comitê de 3 encoders bootstrap + BALD para posicionar seeds por regiões de alta incerteza.

### Ranking dos métodos (Exp A, decoder_3 como referência):
1. **Region BALD** — Δ=+0.280 (parcial, só S1+S2)
2. **Least Confidence** — Δ=+0.145
3. **Entropy** — Δ=+0.124
4. **CoreSet** — Δ=+0.091
5. **BADGE** — Δ=-0.049 (pior que random!)

---

## 7. Experimento B — Backprop AL (NÃO EXECUTADO AINDA)

**Script:** `run_schisto_backprop.sh --device cpu --epochs 500`

**Métodos:** entropy, coreset, badge, region_entropy

**O que roda:**
```python
# Para cada método, split, budget:
# 1. AL seleciona K imagens do pool usando saliency maps do encoder ORIGINAL
# 2. Treina decoder 1×1 conv (Linear) por 500 épocas com DiceCELoss + Adam
# 3. Avalia Fβ, DICE, MAE, IoU no val set
# 4. Compara com Random (mesma quantidade de imagens aleatórias)
```

**Decoder do Exp B:** Uma 1×1 convolução linear (simples) — não os 7 decoders do Exp A. O encoder fica completamente congelado.

**Resultado esperado:** AL > Random de forma mais limpa (sem ruído dos markers sintéticos).

---

## 8. Estado Atual da Execução

```bash
# Rodando agora (dentro do Docker):
bash run_dissertation.sh --device cpu --epochs 500
```

**Status:**
- ✅ [1/6] Entropy encoder AL — 336 linhas (resume total)
- ✅ [2/6] Entropy+DT — SKIPPED (liblapack.so.3 e libblas.so.3 ausentes no Docker)
- ✅ [3/6] Least Confidence — 336 linhas
- ✅ [4/6] CoreSet — 336 linhas
- ✅ [5/6] BADGE — 336 linhas
- 🔄 [6/6] Region BALD — ~155/336 linhas (Split 2 Budget 5 em andamento)
- ⏳ Fase 2 (Backprop AL) — aguardando Fase 1 terminar

**Ambiente Docker:** `flim-ad-env:latest`
- Python 3.10, PyTorch 2.6.0+cu124
- RTX 5070 (sm_120/Blackwell) — PyTorch NÃO suporta → rodando em CPU
- Experimentos em CPU: Exp A ~2-4h total; Exp B ~1-2h total (decoder trivial)

---

## 9. Problemas Conhecidos / Limitações

### Limitações do Experimento A (encoder AL)
1. **AL é one-shot** (não iterativo) — ranking calculado uma vez do encoder original, não atualizado após cada seleção. Limita o ganho teórico do AL.
2. **Budget em imagens** (não seeds) — K=3,5,10 imagens, não seeds individuais.
3. **Markers sintéticos imperfeitos** — degradam o encoder (todos ficam abaixo do baseline).
4. **Bootstrap perde multiplicidades** — comitê region_bald sofre de pouca diversidade quando pool de treino é pequeno (3 imgs → 2 únicas em cada bootstrap).

### DT (Dynamic Trees) desativado
- Binário `libs/ift/bin/iftSMansoniDelineation` existe mas `liblapack.so.3` não instalado no Docker
- Todos os experimentos rodam sem DT (path Otsu+Adaptive Filter)
- MAE retorna 0.0 sem DT (não há segmentação pixel-level além do Fβ)

### CUDA não disponível para RTX 5070
- PyTorch 2.6+cu124 não compila kernels para sm_120 (Blackwell)
- PyTorch 2.7+ com cu128 não disponível via pip/conda neste Docker
- Todos os experimentos rodando em CPU → resultados idênticos, só mais lentos

---

## 10. Próximos Passos

1. **Aguardar Fase 1 terminar** — Region BALD Split 3 ainda rodando
2. **Fase 2 inicia automaticamente** — `run_schisto_backprop.sh --device cpu --epochs 500`
3. **Após Fase 2:** analisar CSVs em `flim_ad/out/al_backprop_results/`
4. **Tabelas para dissertação:** comparar Exp A vs Exp B, mostrar que Exp B (encoder fixo) é mais limpo

---

## 11. Comandos Úteis

```bash
# Verificar progresso
wc -l flim_ad/out/al_corrected_results/region_bald/*.csv
wc -l flim_ad/out/al_backprop_results/*/*.csv

# Ler resultados parciais
python3 -c "
import csv
for row in csv.DictReader(open('flim_ad/out/al_corrected_results/region_bald/schisto-user_A_region_bald_encoder_al.csv')):
    if row['decoder'] == 'labeled_marker':
        print(row['split'], row['budget'], row['method'], row['fb'])
"

# Rodar só Fase 2 (se Fase 1 já terminou)
bash run_schisto_backprop.sh --device cpu --epochs 500

# Rodar tudo do zero (com resume)
bash run_dissertation.sh --device cpu --epochs 500
```

---

## 12. Perguntas de Pesquisa da Dissertação

1. **AL melhora o encoder FLIM quando usamos markers sintéticos?**
   - Resposta preliminar: Sim para a maioria dos métodos (exceto BADGE), mas com gains modestos (Δ=+0.09 a +0.28 dependendo do decoder/método)

2. **Qual método de aquisição é mais efetivo para FLIM?**
   - Resposta preliminar: Region BALD >> LC ≈ Entropy > CoreSet >> BADGE

3. **Manter o encoder fixo e treinar apenas o decoder melhora os resultados do AL?**
   - **A ser respondido pelo Experimento B**

4. **Qual o budget mínimo de anotação para obter resultados satisfatórios?**
   - K=3 já mostra melhora em vários decoders; K=10 melhora ainda mais para region_bald
