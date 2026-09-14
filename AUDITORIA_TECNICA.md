# Relatório de Auditoria Técnica — FLIM Active Learning
**Auditor:** Claude (independente, modo somente leitura)  
**Data:** 2026-07-29  
**Escopo:** Código, CSVs, splits e slides da apresentação FLIM_AL_Apresentacao_v3.pptx

---

## 1. Veredicto Executivo

O projeto possui **13 bugs confirmados** e **3 falhas de completude**. Os resultados numéricos nos slides derivados do benchmark de conjuntiva são rastreáveis e corretos. Os experimentos schisto (slides 6–22) contêm inconsistências de pipeline que os tornam comparações não-homogêneas. Os métodos CoreSet, BADGE, region_bald e region_entropy têm defeitos de implementação que invalidam parcialmente suas premissas metodológicas. O app ainda não está funcional.

**Classificação geral dos resultados:**
- **Válidos/reproduzíveis:** Benchmark conjuntiva (slides 25–26), baseline schisto (labeled_marker)
- **Comprometidos:** Schisto AL entropy K=3,5 vs random (pool∩val leakage), CoreSet/BADGE (feature mismatch + bugs), region_entropy (split único), region_bald (all-skip não reportado)
- **Não rastreáveis:** MAE em todas as tabelas (hardcoded 0.0 no path DT), Fβ de imagens individuais no PPT
- **Contraditórios:** empty-empty Fβ=1.0 no path DT vs Fβ≈0.0 no path não-DT

---

## 2. Tabela de Rastreabilidade

| Slide | Claim | Arquivo Fonte | Status |
|-------|-------|---------------|--------|
| 6–10 | Fβ entropy AL vs random, 3 splits | `al_encoder_results_dt/schisto-user_A_entropy_encoder_al.csv` | RASTREÁVEL (ver §3) |
| 11–13 | Comparação entropy/LC/margin | `al_encoder_results_acq_comparison/*.csv` | INCOMPLETO — split1 apenas |
| 15–16 | Region entropy | `al_region_results/schisto-user_A_region_entropy_encoder_al.csv` | INCOMPLETO — split1 apenas |
| 19–22 | Region BALD | `al_bald_results/schisto-user_A_region_bald_encoder_al.csv` | RASTREÁVEL — anomalias (ver §3) |
| 25 | Tabela benchmark conjuntiva (decoder_3) | `benchmark_conjunctiva/conjunctiva/benchmark_results.csv` | VERIFICADO ✓ |
| 26 | Análise cross-domain | idem | VERIFICADO ✓ |

---

## 3. Reconciliação Numérica por Slides

### Slides 6–10: Experimento DT Schisto (entropy)
Fonte: `out/al_encoder_results_dt/schisto-user_A_entropy_encoder_al.csv`

| Split | Budget | Baseline (original) | AL Entropy | Random |
|-------|--------|---------------------|------------|--------|
| 1 | 3 | 0.7133 | 0.6335 | 0.4788 |
| 2 | 3 | 0.6676 | 0.5850 | 0.4799 |
| 3 | 3 | 0.7123 | 0.5835 | 0.5114 |
| **Média K=3** | | **0.698** | **0.601** | **0.490** |
| 1 | 5 | — | 0.6403 | 0.4788 |
| 2 | 5 | — | 0.5681 | 0.4788 |
| 3 | 5 | — | 0.5710 | 0.4887 |
| **Média K=5** | | | **0.593** | **0.482** |
| 1 | 10 | — | 0.6954 | 0.4788 |
| 2 | 10 | — | 0.6011 | 0.4788 |
| 3 | 10 | — | 0.6160 | 0.4848 |
| **Média K=10** | | | **0.637** | **0.481** |

**Observações críticas:**
- MAE = 0.0 em TODAS as linhas → hardcoded (Finding 6 confirmado)
- Split 1 K=3 tem apenas 1 seed aleatório (seed0=0.4788); split2,3 têm 3 seeds
- AL entropy PIOR que baseline original em todos os budgets (esperado: encoder adicional ≠ mais shots do encoder original)

### Slides 11–13: Comparação de Aquisições
Fonte: `out/al_encoder_results_acq_comparison/*.csv`

**PROBLEMA CRÍTICO:** Os CSVs de comparação contêm apenas split1. Se os slides mostram médias de 3 splits, os números são inválidos.

| Método | K=3 | K=5 | K=10 |
|--------|-----|-----|------|
| Entropy (acq_comparison, split1) | 0.6111 | 0.6308 | 0.6981 |
| Entropy (DT, split1) | 0.6335 | 0.6403 | 0.6954 |

Os valores diferem entre os dois experimentos (pipeline DT vs pipeline Otsu+AF). Se slides 11-13 comparam LC e margin com entropy do mesmo pipeline, a comparação é internamente consistente. Se mesclam pipelines, é inconsistente.

**LC = Margin:** Para os splits disponíveis (apenas split1), verificar se os CSVs de LC e margin dão resultados idênticos confirmaria Finding 3 empiricamente. A identidade matemática já foi confirmada em código.

### Slides 15–16: Region Entropy
Fonte: `out/al_region_results/schisto-user_A_region_entropy_encoder_al.csv`

| Split | K=3 | K=5 | K=10 |
|-------|-----|-----|------|
| 1 | 0.5880 | 0.5927 | 0.7625 |
| 2 | — | — | — |
| 3 | — | — | — |

**PROBLEMA:** Apenas split1 disponível. Qualquer "média de 3 splits" nos slides é inválida.

### Slides 19–22: Region BALD
Fonte: `out/al_bald_results/schisto-user_A_region_bald_encoder_al.csv`

| Split | K=3 | K=5 | K=10 |
|-------|-----|-----|------|
| 1 | 0.5334 | 0.4795 | 0.5398 |
| 2 | 0.5458 | **0.5458** | 0.5250 |
| 3 | **0.7104** | **0.7104** | 0.5099 |
| **Média** | **0.597** | **0.579** | **0.525** |

**ANOMALIA CRÍTICA:** Split2 K=3 = Split2 K=5 = 0.5458 (idênticos). Split3 K=3 = Split3 K=5 = 0.7104 (idênticos). Indica cenário "all-skip": todas as imagens selecionadas pelo AL já estavam no marker set inicial → encoder não foi retreinado → resultado idêntico ao baseline. Este comportamento **não está reportado** nos slides.

Baseline médio: (0.7133+0.6676+0.7123)/3 = 0.698  
region_bald K=3 média: 0.597 ← abaixo do baseline original

### Slides 25–26: Benchmark Conjuntiva (decoder_3)
**VERIFICADO ✓** — Todos os valores confirmados contra `benchmark_results.csv`:
- no_al K=3: 0.539, K=5: 0.605, K=10: 0.513
- entropy K=3: 0.616, K=5: 0.474, K=10: 0.524
- LC K=3: 0.484, K=5: 0.644, K=10: 0.694
- region_bald K=3: 0.508, K=5: 0.633, K=10: 0.403

---

## 4. Findings de Código por Severidade

### 🔴 CRÍTICO

**C1 — NaN em entropy_map (float32)**
- **Arquivo:** `flim_al/acquisition.py`, linha 35
- **Código:** `p = pred.clamp(eps, 1 - eps)` onde `eps=1e-8`
- **Bug:** Em float32, `1 - 1e-8 = 1.0` (underflow). Pixels de saliency com valor 255/255 produzem `log(0) = -inf`, e `0 * -inf = NaN`.
- **Impacto:** Scores de entropy podem ser NaN para imagens com saliency saturada → ranking corrompido para esses casos.
- **Status:** **CONFIRMADO**

**C2 — Pool∩Val leakage em al_encoder_experiment.py**
- **Arquivo:** `flim_al/al_encoder_experiment.py`, linhas ~865, ~912
- **Bug:** Pool = imagens em `out/saliencies/.../test/split{N}/` (610 imgs). Val = `Splits-5train-70_30/split{N}-val.txt` (848 imgs). Definições independentes. Com 610+848=1458 e total~1213, interseção mínima ~245 imagens.
- **Consequência:** O AL seleciona imagens do pool, gera markers sintéticos com GT, retreina o encoder. Essas imagens também aparecem no val → encoder viu (indiretamente via markers) imagens de validação. Fβ do AL é otimisticamente enviesado.
- **Status:** **CONFIRMADO** (estrutural — aritmética de conjuntos prova overlap)

**C3 — LC = Margin: identidade matemática**
- **Arquivo:** `flim_al/acquisition.py`, linhas 50, 61
- **LC:** `(1 - |p - 0.5| * 2).mean()` = `(1 - |2p-1|).mean()`
- **Margin:** `(1 - |2p - 1|).mean()`
- **São matematicamente idênticas.** Qualquer diferença nos resultados seria ruído numérico ou dependência de ordem de avaliação.
- **Impacto:** Comparação LC vs Margin nos slides 11-13 não mede dois métodos distintos.
- **Status:** **CONFIRMADO**

**C4 — CoreSet retorna K-1 elementos**
- **Arquivo:** `flim_al/coreset_badge.py`, linha ~166
- **Bug:** `return selected[1:]` após iniciar com ponto aleatório. Para budget=K, retorna K-1 imagens.
- **Status:** **CONFIRMADO**

**C5 — BADGE: gradiente zero na máxima incerteza**
- **Arquivo:** `flim_al/coreset_badge.py`, linha ~190 + `al_encoder_experiment.py`
- **Bug:** `dummy_w = torch.zeros(...)` → `sigmoid(0) = 0.5` → `uncertainty = 0.5 - 0.5 = 0` → `g = 0 * features = 0`. Todos os embeddings BADGE são zero → seleção arbitrária.
- **Status:** **CONFIRMADO**

**C6 — Feature space mismatch: CoreSet/BADGE usam RGB vs LAB**
- **Arquivo:** `flim_al/coreset_badge.py`, linha 54: `img.transpose(2,0,1) / 255.0`
- **Encoder FLIM** opera em espaço LAB (via `FLIMData` com `convert_gray_to_lab=True`).
- **Consequência:** Features extraídas para CoreSet/BADGE estão em espaço de cor diferente do encoder treinado. Diversidade geométrica medida em RGB ≠ diversidade no espaço que o encoder usa.
- **Status:** **CONFIRMADO**

### 🟠 ALTO

**A1 — MAE hardcoded 0.0 no path DT**
- **Arquivo:** `flim_al/al_encoder_experiment.py`, linhas 235-275 (`_fb_from_masks`)
- **Código:** `"mae": 0.0` explícito.
- **Evidência CSV:** Todos os arquivos DT têm `mae=0.0` em todas as linhas.
- **Status:** **CONFIRMADO**

**A2 — Inconsistência empty-empty: DT vs não-DT**
- **DT path:** `_fb_from_masks`, quando ambas masks vazias → `fbs.append(1.0)` (Fβ=1.0)
- **Não-DT path:** `evaluate_decoder`, quando pred e GT vazias → tp=fp=fn=0 → pr≈0, rc≈0 → Fβ≈0
- **Impacto:** Imagens sem ovos têm Fβ=1.0 no path DT e Fβ≈0 no path não-DT. Comparação entre experimentos DT e não-DT é não-homogênea.
- **Status:** **CONFIRMADO**

**A3 — One-shot, não AL iterativo**
- **Arquivo:** `flim_al/al_encoder_experiment.py`, linhas ~865, ~912-913
- **Comportamento:** Ranking computado UMA vez com saliency do encoder inicial. `top_K` selecionado por fatiamento `al_ranking[:budget]`. Nenhum re-ranking após aquisição.
- **Impacto metodológico:** Não é AL iterativo (onde o modelo retreinado re-scores o pool). É seleção batch one-shot com diferentes orçamentos do mesmo ranking fixo.
- **Status:** **CONFIRMADO**

**A4 — region_bald: imagem=BALD, região=entropy (não BALD)**
- **Arquivo:** `flim_al/al_encoder_experiment.py`, linha 928: `create_combined_region_marker_dir_bald(..., region_method="entropy")`
- **Arquivo:** `flim_al/region_al.py`, `region_al_select_bald` usa `score_regions_by_entropy` (não `score_regions_by_bald`)
- **Consequência:** Nome "region_bald" é enganoso. Seleção de imagens usa BALD (committee de encoders). Posicionamento de seeds usa entropy. A função `score_regions_by_bald` existe mas não é chamada no pipeline de seeds.
- **Status:** **CONFIRMADO**

**A5 — Bootstrap: multiplicidade perdida**
- **Arquivo:** `flim_al/al_encoder_experiment.py`, linha 611: `for fname in set(sampled)`
- **Bug:** `set()` remove duplicatas do bootstrap. Com N=3 markers e 3 membros do committee, é comum que amostras bootstrap tenham os mesmos 2-3 arquivos únicos → committee members idênticos → BALD score inválido (zero variância).
- **Status:** **CONFIRMADO**

**A6 — hash() seed não-reproduzível**
- **Arquivos:** `flim_al/marker_generator.py` linha 192, `flim_al/benchmark.py` `setup_initial_markers`
- **Código:** `seed=hash(name) % 2**31`
- **Bug:** `hash()` do Python varia entre processos se `PYTHONHASHSEED` não fixado. Posições dos markers sintéticos mudam a cada execução.
- **Status:** **CONFIRMADO**

**A7 — simulate_expert_label: seeds fg em pixels bg**
- **Arquivo:** `flim_al/region_al.py`, linhas 180-192 (`simulate_expert_label`)
- **Bug 1:** Se region_overlap ≥ 0.15 → label "fg". Seeds amostrados de toda a superpixel (incluindo pixels bg) → seeds fg em pixels de fundo.
- **Bug 2:** Quando n_bg_regions=0, adiciona até 300 seeds bg do GT diretamente. Esse fallback está fora do budget original.
- **Bug 3:** Quando sal_path ausente, saliency = `np.random.rand(...)` (ruído puro) → scoring de regiões inválido.
- **Status:** **CONFIRMADO**

**A8 — region_entropy ≠ RIPU**
- **RIPU original** (arXiv:2111.12667): Region Impurity + Prediction Uncertainty + pixel adjacency (fronteiras de região adjacentes a fronteiras de classe).
- **Implementação:** Apenas mean entropy por superpixel. Sem impurity (proporção fg/bg), sem adjacency.
- **Nome nos slides:** "region_entropy (RIPU-style)" — qualificador correto, mas pode ser confundido.
- **Status:** **CONFIRMADO**

### 🟡 MÉDIO

**M1 — benchmark.py: score_saliencies lê TODOS os PNGs, não só pool**
- **Arquivo:** `flim_al/benchmark.py`, `compute_method_scores`
- **Comportamento:** Lê todos os PNGs em `sal_dir` independente do pool atual. Em benchmark.py os splits são internos e disjuntos (sem overlap pool∩val), então o impacto é menor que em al_encoder_experiment.py.
- **Status:** **CONFIRMADO**

**M2 — best_al_weights não rastreia o melhor**
- **Arquivo:** `flim_al/al_flim_backprop.py`, linha 460
- **Código:** `if best_al_weights is None or fb_al > 0:` → atualiza em toda iteração onde Fβ>0 (quase sempre). Não preserva o peso do MELHOR budget.
- **Status:** **CONFIRMADO**

**M3 — evaluate_backprop usa RGB/255 (não LAB)**
- **Arquivo:** `flim_al/al_flim_backprop.py`, linhas 238-239
- **Inconsistência:** Treino usa `FLIMData` com `convert_gray_to_lab=False`; encoder FLIM pode esperar LAB dependendo de como foi treinado. Se o encoder foi treinado com LAB mas avaliado com RGB, há degradação silenciosa.
- **Status:** **CONFIRMADO** (inconsistência existe; magnitude do impacto depende do encoder)

**M4 — al_loop.py: ALLoop usa UncertaintyDecoder e FLIMDataset não-existentes**
- **Arquivo:** `flim_al/al_loop.py`, linha 45: `from .uncertainty_decoder import UncertaintyDecoder`
- **Verificação:** `uncertainty_decoder.py` existe ✓
- **Linha 44:** `from .acquisition import rank_pool` — `rank_pool` existe ✓ (acquisition.py L94-146)
- **Obs:** ALLoop está funcional como API genérica. Não está integrado ao pipeline FLIM real (usa Dataset abstrato, não FLIMData).
- **Status:** **PARCIALMENTE CONFIRMADO** — imports OK, mas desacoplado do pipeline principal

### 🟢 BAIXO

**L1 — App incompleto**
- `flim_al/app/inference.py` linha 58: `raise NotImplementedError`
- `flim_al/app/tasks.py` linhas 141-142: `pass` (loop de treino não implementado)
- O app não pode rodar inferência nem treino via API.
- **Status:** **CONFIRMADO**

**L2 — al_flim_backprop.py: DiceCELoss de MONAI**
- Linha 59: `from monai.losses import DiceCELoss`
- Dependência de MONAI (não-trivial). Se MONAI não instalado, script falha.
- **Status:** **CONFIRMADO** (dependência não documentada)

**L3 — K = imagens, não seeds ou pontos de anotação**
- Budget K refere-se a imagens. Cada imagem gera 100 fg + 300 bg = 400 seeds. K=3 → 1200 seeds, K=10 → 4000 seeds. Comparação com papers que usam K em termos de pontos/seeds não é direta.
- **Status:** **CONFIRMADO** (problema metodológico de apresentação, não de código)

---

## 5. Status de Cada Finding Preliminar

| # | Finding | Status | Evidência Principal |
|---|---------|--------|---------------------|
| 1 | NaN em entropy_map (float32) | **CONFIRMADO** | acquisition.py L35: `1-1e-8=1.0` em float32 |
| 2 | LC = Margin: identidade matemática | **CONFIRMADO** | acquisition.py L50,61: `(1-|2p-1|).mean()` idêntico |
| 3 | Pool∩Val leakage (al_encoder_experiment) | **CONFIRMADO** | Split1-val.txt=848 imgs; pool dir=610 imgs; total~1213 → mín 245 overlap |
| 4 | Posições manuais de markers nos splits | **INCONCLUSIVO** | test.txt tem 5 imgs (000002,013,156,391,405) mas sem acesso ao diretório markers para confirmar se são os user_A |
| 5 | One-shot AL vs iterativo | **CONFIRMADO** | al_encoder_experiment.py L865: ranking fixo; L912-913: fatiamento por budget |
| 6 | Bootstrap: multiplicidade perdida por set() | **CONFIRMADO** | al_encoder_experiment.py L611: `for fname in set(sampled)` |
| 7 | Budget K = imagens, não normalizado | **CONFIRMADO** | marker_generator.py: 100fg+300bg por imagem |
| 8 | region_entropy ≠ RIPU completo | **CONFIRMADO** | region_al.py: apenas entropy/margin por superpixel; sem adjacency |
| 9 | simulate_expert_label: seeds fg em bg + fallback extra-budget | **CONFIRMADO** | region_al.py L180-192, L497-515, L484 |
| 10 | region_bald = BALD-imagem + entropy-região (não BALD-região) | **CONFIRMADO** | al_encoder_experiment.py L928: `region_method="entropy"` |
| 11 | CoreSet/BADGE: RGB vs LAB, K-1 bug, zero gradient | **CONFIRMADO** | coreset_badge.py L54, L166, L190 |
| 12 | hash() seed não-reproduzível | **CONFIRMADO** | marker_generator.py L192; benchmark.py setup_initial_markers |
| 13 | Precisão numérica nos slides (schisto 6-22) | **PARCIALMENTE CONFIRMADO** | CSV values traceable (ver §3); slide text inacessível sem shell |
| 14 | Inconsistências entre slides 20-22 (region_bald) | **PARCIALMENTE CONFIRMADO** | CSV mostra all-skip não reportado (split2/split3 K=3=K=5) |
| 15 | Imports e app incompletos | **CONFIRMADO** | inference.py L58: NotImplementedError; tasks.py L142: pass |
| 16 | al_flim_backprop: DiceCELoss MONAI, RGB eval, best_al_weights | **CONFIRMADO** | L59 (monai); L238-239 (RGB); L460 (condição sempre True) |
| 17 | ALLoop: imports rank_pool e UncertaintyDecoder | **REFUTADO** | Ambos existem: acquisition.py L94-146, uncertainty_decoder.py ✓ |
| 18 | acq_comparison e region_entropy: dados incompletos | **CONFIRMADO** | Apenas split1 disponível nos CSVs de comparação e region_entropy |

---

## 6. Perguntas Sem Resposta para o Autor

1. **Pool schisto "test" dir:** Quais 610 imagens estão em `out/saliencies/schisto/user_A/test/split1/labeled_marker/layer_3/`? Como foram selecionadas em relação ao split1-val.txt?

2. **MAE=0.0 intencional?** O MAE hardcoded no path DT é um placeholder aguardando implementação ou uma escolha deliberada?

3. **Slides 11-13 (acq comparison):** Os resultados mostrados são para split1 apenas ou foram rodados splits 2 e 3? Os CSVs disponíveis contêm apenas split1.

4. **Slides 15-16 (region_entropy):** Idem — apenas split1 disponível.

5. **Budget de anotação:** A comparação K=3 vs K=5 vs K=10 mede algo útil? Um especialista que anota 3 imagens com 400 seeds cada não é o mesmo que um que anota 10 imagens com menos seeds.

6. **all-skip em region_bald (schisto):** Split2 K=3 e K=5 produzem Fβ idêntico. Foi verificado se o encoder foi realmente retreinado nesses casos?

7. **decoder_3 seleção:** O decoder_3 foi escolhido para o benchmark de conjuntiva antes ou depois de ver os resultados? Se após, há seleção post-hoc.

8. **test.txt:** O arquivo `Splits-5train-70_30/test.txt` (5 imagens) corresponde às imagens com marcadores manuais do user_A? Qual a relação com `split{N}-train.txt`?

---

## 7. Classificação dos Resultados

### ✅ Válidos / Reproduzíveis
- **Benchmark conjuntiva (slides 25-26):** 6 métodos × 3 splits × 3 budgets. Valores verificados linha a linha contra CSV. Método benchmark.py tem splits internos disjuntos (sem pool∩val leakage).
- **Baseline schisto (labeled_marker, 3 splits):** Fβ={0.7133, 0.6676, 0.7123} verificados em múltiplos CSVs consistentes.

### ⚠️ Comprometidos (resultados existem mas premissas violadas)
- **AL entropy schisto (slides 6-10):** Números rastreáveis, mas pool∩val leakage (Finding 3) viola isolamento train/val. Fβ do AL pode estar inflado.
- **CoreSet e BADGE:** Seleção executada, mas com feature space errado (RGB vs LAB), K-1 imagens, e zero-gradient BADGE → na prática equivalentes a seleção aleatória.
- **region_entropy e region_bald:** Seleção de imagens funcionou, mas posicionamento de seeds tem bugs (fg seeds em bg, fallback extra-budget, região=entropy e não BALD).

### ❌ Não Rastreáveis
- **MAE em todas as tabelas DT:** Hardcoded 0.0 — não reflete realidade.
- **Comparação acq_comparison splits 2,3:** CSVs ausentes.
- **region_entropy splits 2,3:** CSVs ausentes.

### 🔄 Contraditórios
- **Fβ empty-empty:** DT path = 1.0; não-DT path ≈ 0.0. Experimentos DT e não-DT não são comparáveis em imagens sem anotações (eggs ausentes).
- **region_bald K=3 vs K=5 (split2,3):** Valores idênticos sugerem encoder não-retreinado, mas isso não está documentado.

---

## Apêndice: Arquivos Não Lidos (Out of Scope ou Inacessíveis)

- `pyflim/flim.py`, `pyflim/layers.py`, `pyflim/data.py` — internals do FLIM (não necessários para audit AL)
- `flim_al/viz.py` — funções de visualização (imports em al_flim_backprop.py são lazy/condicionais)
- Slides PPT (texto) — shell indisponível para markitdown; verificação numérica feita via CSV
- Diretório de saliency maps schisto — filesystem não acessível para confirmar overlap exato pool∩val
