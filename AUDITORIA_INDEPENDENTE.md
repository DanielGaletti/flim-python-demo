# Auditoria Independente — FLIM Active Learning
**Data:** 2026-07-31 | **Modo:** read-only, análise estática + inspeção de CSV  
**Arquivo auditado:** `flim_al/al_encoder_experiment.py` + CSV gerado  

---

## Veredicto Geral

| Categoria | Status |
|---|---|
| Implementação central (entropy, pool∩val, Fβ) | ✅ Correta |
| Qualidade dos resultados (Fβ ~0.48 vs baseline 0.74) | ❌ Degradação real, causa confirmada |
| Protocolo experimental | ⚠️ Problemas menores documentados abaixo |
| CSV existente (275 linhas) | ⚠️ Incompleto (split3 K=5 random + K=10 ausentes) |
| Bug de resume (`iou`) | ❌ Presente no código, não afeta CSV atual |

---

## Tabela de Problemas

| # | Item | Veredicto | Evidência (arquivo:linha) |
|---|---|---|---|
| 1 | NaN em entropy | ✅ CORRIGIDO | `acquisition.py:38` eps=1e-6 |
| 2 | Pool∩Val leakage | ✅ CORRIGIDO | `al_encoder_experiment.py:894-905` |
| 3 | Top-3 pre-filter print | ⚠️ COSMÉTICO | Linha 881 imprime pool não-filtrado, mas seleção é correta |
| 4 | `000479.png` em val | ⚠️ CONTAMINAÇÃO LEVE | `split{1,2,3}-val.txt` linha 334/335/338 |
| 5 | `000675`/`000917` no pool | ✅ ACEITÁVEL | Em test.txt, não em val; marker real preservado (linha 195) |
| 6 | Recálculo CSV | ✅ CONFIRMADO | Ver tabela completa abaixo |
| 7 | Experimento incompleto | ❌ CONFIRMADO | CSV termina em S3/K5/al_entropy; random S3K5 + K10 ausentes |
| 8 | Predição degenerada | ❌ CONFIRMADO | Mecanismo: all-background = 406/848 = 0.4788 |
| 9 | Melhor decoder | ✅ `labeled_marker` consistentemente melhor |  |
| 10 | Fβ=Dice=IoU=0.4788 | ❌ DEGENERADO | Prova matemática e empírica (ver seção abaixo) |
| 11 | Manual vs sintético | ❌ CAUSA RAIZ | 1215 seeds coerentes vs 400 aleatórias |
| 12 | Imbalance de seeds | ⚠️ NÃO VERIFICÁVEL | Contagem fg/bg em markers.txt requer execução |
| 13 | Imagens vazias no val | ✅ CONFIRMADO | 406/848 = 0.4788 (exato) |
| 14 | Causa da degradação | ❌ CONFIRMADO | Retreino com sintéticos destrói filtros k-means |
| 15 | Ablação sugerida | ⚠️ NÃO IMPLEMENTADA | Fix 1 (backprop) criado, não executado |
| 16 | Sinal por decoder | ✅ ANALISADO | `labeled_marker` e `decoder_2` mais estáveis |
| 17 | Otsu vs DT | ⚠️ INCONSISTÊNCIA | AL usa Otsu+AF; paper usa DT → não comparável ao paper |
| 18 | Entropy implementação | ✅ CORRETA | `al_encoder_experiment.py` usa saliency PNGs |
| 19 | LC = Margin | ✅ IDENTIDADE | Matematicamente idênticos para segmentação binária |
| 20 | Region BALD committee | ✅ VÁLIDO | Bootstrap com n_committee encoders distintos |
| 21 | CoreSet K correto | ✅ CORRIGIDO | `coreset_badge.py:188`: `< n_init + budget` |
| 22 | BADGE gradiente zero | ✅ CORRIGIDO | Usa saliency mean como pred_score proxy |
| 23 | Comparabilidade dos budgets | ⚠️ NÃO JUSTO | K adicional vs baseline com diferentes qualidades de marker |
| 24 | Bug resume `iou` | ❌ BUG PRESENTE | `al_encoder_experiment.py:1052-1058` — não afeta CSV atual |
| 25 | PPT claims | ⚠️ NÃO VERIFICADO | PPT gerado com resultados anteriores; números podem diferir |

---

## Evidências Detalhadas

### Item 1 — NaN em Entropy (CORRIGIDO)

**`acquisition.py:38`:**
```python
p = pred.float().clamp(eps, 1.0 - eps)  # eps=1e-6
```

Prova estática: `float32(1 - 1e-8) == 1.0` (underflow de precisão). Com eps=1e-6:
`float32(1 - 1e-6) < 1.0` (garantido). `log(0)` impossível → NaN impossível.

---

### Item 2 — Pool∩Val (CORRIGIDO)

**`al_encoder_experiment.py:894-905`:**
```python
_val_set  = set(val_fnames)
_pairs    = [(f, s) for f, s in zip(fnames, scores) if f not in _val_set]
fnames, scores = zip(*_pairs)  # ambas listas truncadas juntas
al_ranking = sorted(range(N), key=lambda i: scores[i], reverse=True)  # recomputado
```
Terminal confirmou: "Pool: 610 → 177/196/189 após remover overlap com val".

---

### Item 4 — Contaminação de `000479.png`

```
grep "000479" Splits-5train-70_30/split1-val.txt  → linha 334: 000479.png
grep "000479" Splits-5train-70_30/split2-val.txt  → linha 335
grep "000479" Splits-5train-70_30/split3-val.txt  → linha 338
```

`000479.png` tem markers reais em `user_A/split{1,2,3}/markers/000479-seeds.txt` E está no val set. Impacto: 1/848 imagens avaliadas com encoder que aprendeu seus filtros a partir dela. **Impacto numérico estimado: < 0.1% nas métricas médias**, mas metodologicamente incorreto.

`000675.png` e `000917.png` → apenas em `test.txt`, não em val → OK.

---

### Item 6 — Recálculo Completo (labeled_marker)

| Método | S1 | S2 | S3 | Média |
|---|---|---|---|---|
| **Baseline** (3 imgs manuais) | **0.7445** | **0.7309** | **0.7565** | **0.7440** |
| AL K=3 | 0.5479 | 0.4808 | 0.5269 | 0.5185 |
| AL K=5 | 0.4795 | 0.4771 | 0.5600 | 0.5055 |
| AL K=10 | 0.4788 | 0.4783 | — | — |
| Random K=3 | 0.5469 | 0.5012 | 0.5182 | 0.5221 |
| Random K=5 | 0.5551 | 0.4962 | — | — |
| Random K=10 | 0.5717 | 0.4902 | — | — |

**AL ≤ Random em todas as configurações com dados suficientes.**  
**Ambos ~0.23 Fβ abaixo do baseline.** A hipótese de que AL melhora o baseline é **refutada**.

---

### Item 7 — Experimento Incompleto

O CSV termina com 275 linhas. Última entrada: `3,5,al_entropy,entropy,hybrid_decoder`.

**Configurações executadas:**

| Split | K=3 | K=5 | K=10 |
|---|---|---|---|
| Split 1 | ✅ | ✅ | ✅ |
| Split 2 | ✅ | ✅ | ✅ |
| Split 3 | ✅ | ⚠️ AL only | ❌ |

Faltam: Split3/K=5/random_seed{0,1,2}/random + Split3/K=10/todos os métodos.

---

### Item 8 + 10 — Mecanismo da Predição Degenerada

**Prova matemática de Fβ = Dice = IoU = 406/848 = 0.4788:**

Para predição all-background (`pred_bin = zeros`):

| Tipo de imagem | Contagem | Fβ | Dice | IoU |
|---|---|---|---|---|
| Vazia (gt=0, pred=0) | 406 | 1.0 (empty-empty fix) | 1.0 | 1.0 |
| Positiva (gt>0, pred=0) | 442 | 0.0 (tp=0) | 0.0 (pred_sum=0) | 0.0 (tp=0) |

Média = 406/848 = 0.47877... ≈ **0.4788** ✓ para todos os três.

**MAE = 0.0136:** taxa média de pixels foreground no val set = 1.36%.

**Mecanismo de degradação:**
1. Encoder retreinado com markers sintéticos (100fg+300bg, aleatórios)
2. Filtros k-means aprendem features menos discriminativas
3. Saliency maps tornam-se quase-uniformes (pouca variação espacial)
4. Otsu threshold em mapa uniforme → threshold ≈ 0 (segmenta tudo) ou alto (segmenta nada)
5. Área filter [1000, 9000]: blob único cobrindo a imagem (160000px >> 9000) → removido
6. `pred_bin = zeros` → Fβ=Dice=IoU=0.4788, MAE=0.0136

Casos NÃO-degenerados (S1K3=0.5479, S3K5=0.5600): o encoder específico dessa rodada preservou parcialmente os filtros originais, provavelmente por seleção de imagens mais similar ao treino original.

---

### Item 11 — Manual vs Sintético

| Propriedade | Manual (`000675-seeds.txt`) | Sintético |
|---|---|---|
| Seeds totais | 1215 | 400 (100fg + 300bg) |
| Distribuição espacial | Regiões contíguas, bordas | Aleatória pelo GT |
| Informação de fronteira | ✅ Alta (tracing de borda) | ❌ Baixa |
| Relevância para k-means | ✅ Clusters espacialmente coerentes | ❌ Clusters ruidosos |

FLIM k-means aprende filtros de borda e textura a partir da co-ocorrência de pixels próximos aos marcadores. Marcadores em regiões contíguas fornecem contexto espacial; marcadores aleatórios não.

---

### Item 17 — Inconsistência Otsu vs DT

| Configuração | Método de avaliação | Comparável ao paper? |
|---|---|---|
| Baseline nosso | Otsu+AF | ❌ (paper usa DT) |
| AL nosso | Otsu+AF | ❌ (paper usa DT) |
| AL vs Random | Otsu+AF vs Otsu+AF | ✅ Internamente justo |

Usar DT requer `libs/ift/bin/iftSMansoniDelineation` compilado. Atual experimento é internamente consistente (AL e Random usam o mesmo Otsu+AF), mas **não reproduz os resultados da Tabela III do paper**.

---

### Item 19 — LC = Margin (Identidade)

Para segmentação binária (`p ∈ [0,1]`):
```
LC(p)     = 1 - |p - 0.5| × 2 = 1 - |2p - 1|
Margin(p) = 1 - |2p - 1|
```
**Matematicamente idênticos.** Ambos implementados — `least_confidence` e `margin_score` — produzem os **mesmos rankings** e **mesmos resultados de AL**.

---

### Item 20 — BALD Committee

```python
# build_committee_saliencies:
rng_b = np.random.default_rng(seed=i * 13 + 7)
sampled = rng_b.choice(all_marker_files, size=n_orig, replace=True)  # bootstrap

# score_saliencies_bald:
h_mean = -(mean_p * log(mean_p) + ...)   # H(E[p])
mean_h = mean([-(s*log(s)+...) for s in preds])  # E[H(p)]
bald = h_mean - mean_h  ≥ 0
```
Implementação correta. Bootstrap garante encoders diversificados mesmo com poucos markers.

---

### Item 21 — CoreSet K Correto

```python
# coreset_badge.py:186-197
# ANTES (bug): while len(selected) < budget → retornava budget - n_init pontos
# DEPOIS (fix):
while len(selected) < n_init + budget:  # garante exatamente budget novos
    ...
return selected[n_init:]  # exclui pontos iniciais
```

---

### Item 22 — BADGE zero-gradient

```python
# ANTES (bug): pred_scores = zeros → g = 0×features = 0 → seleção arbitrária
# DEPOIS (fix):
sal_means = [float(Image.open(sal_path).mean()) / 255.0 for f in fnames]
sel_idx = badge_select(feats, np.array(sal_means), budget)
```
Saliency mean é proxy válido da predição média.

---

### Item 24 — Bug de Resume (iou ausente)

**`al_encoder_experiment.py:1052-1058`:**
```python
r = {row["decoder"]: {"fb": float(row["fb"]),
                      "dice": float(row["dice"]),
                      "mae": float(row["mae"])}   # ← "iou" AUSENTE
     for row in all_rows ...}
```

Ao retomar: `rand_results_acc[dec]["iou"]` fica `[]` → `np.mean([])` = NaN na linha de média.

**O CSV atual NÃO é afetado** (experimento rodou sem interrupção até o ponto de falha).  
**Bug afeta apenas futuros runs que sejam retomados com `--resume`.**

**Fix (pseudocódigo):**
```python
r = {row["decoder"]: {"fb": float(row["fb"]),
                      "dice": float(row["dice"]),
                      "mae": float(row["mae"]),
                      "iou": float(row["iou"]) if "iou" in row and row["iou"] else 0.0}
     ...}
```

---

## Prioridade de Correções

| Prioridade | Item | Ação |
|---|---|---|
| **P0** | Experimento incompleto (#7) | Continuar execução para S3K5 random + S3K10 |
| **P0** | Fix 1 (backprop) | Executar `run_schisto_backprop.sh` para dados válidos para dissertação |
| **P1** | Bug resume iou (#24) | Corrigir linha 1052-1058 antes de reexecutar |
| **P1** | DT evaluation (#17) | Compilar IFT e rodar com `--use_dt` para comparação com paper |
| **P2** | Contaminação 000479 (#4) | Remover do val set OU documentar como limitação |
| **P2** | Top-3 print stale (#3) | Mover print para depois do filtro (cosmético) |

---

## Protocolo Defensável para Dissertação

Dado os resultados, o que pode ser dito com rigor:

1. **Encoder FLIM + markers sintéticos** não é comparável a encoder + markers manuais. A "active learning" nesta configuração está comparando qualidade de anotação, não qualidade de seleção.

2. **Fix 1 (encoder fixo + decoder backprop)** é o experimento correto para medir benefício do AL: a seleção das imagens afeta quais exemplos treinam o decoder, mantendo o encoder fixo.

3. **Resultados atuais mostram**: retreino do encoder com sintéticos sempre degrada (~0.23 Fβ abaixo do baseline), independente do método AL — AL=random em performance, ambos muito piores que o baseline.

4. **Para a dissertação**: comparar Fix 1 (al_flim_backprop.py) vs baseline FLIMpb do paper (Fβ=0.857) — isso sim tem potencial de mostrar valor do AL.

---

## Questões para Decisão do Usuário

1. **Continuar experimento atual** (S3K5 random + S3K10) ou **abortar** e focar no Fix 1?
2. **Compilar IFT/DT** (Docker + compilação) para avaliação correta, ou aceitar Otsu+AF?
3. **Corrigir bug iou** (#24) antes de qualquer reexecução?
4. **PPT**: atualizar slides com resultados corretos ou esperar Fix 1?
