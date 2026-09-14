# Paisagem de Pesquisa — MSc Daniel Galetti
**Framework Interativo e Adaptativo: AL + IL + CL para Segmentação Semântica**

---

## 1. Contexto e Lacuna

Seu trabalho fecha o ciclo: **interação humana orientada por incerteza (AL) → refinamento contínuo (CL) → menor custo de anotação**. A literatura ainda não integra os três pilares de forma online e sinérgica. Essa é a contribuição central.

---

## 2. Métodos Relevantes na Literatura (2021–2025)

### 2.1 Segmentação Interativa por Cliques (IL)

| Método | Ano | Base | Destaques | NoC@90 |
|---|---|---|---|---|
| **RITM** (Sofiiuk et al.) | 2022 | HRNet/ViT | Iterative training com mask guidance; benchmark padrão | ~4.0 |
| **SimpleClick** (Liu et al.) | 2023 | ViT-H | Plain ViT, sem arquitetura especial; SOTA eficiente | ~3.5 |
| **FocalClick** (Chen et al.) | 2022 | SegFormer | Refina foco progressivo; baixo custo por clique | ~4.2 |
| **SAM** (Kirillov et al.) | 2023 | ViT-H (1B params) | Zero-shot, point/box/mask prompts; SA-1B dataset | ~5.0* |
| **SAM 2** (Meta) | 2024 | ViT-L | Adiciona suporte a vídeo; melhor zero-shot | ~4.5* |
| **Grounded SAM** | 2024 | SAM+DINO | Text + point prompts; open-vocabulary | - |
| **GPCIS** | 2023 | Diffusion | Generative prior para interactive seg | ~3.8 |

*SAM é zero-shot — NoC estimado, não treinado para cliques interativos padrão.

**→ Para comparação no mestrado:** RITM e SimpleClick são os benchmarks honestos. SAM é o baseline VLM.

---

### 2.2 Active Learning para Segmentação

| Método | Ano | Estratégia | Nota |
|---|---|---|---|
| **RIPU** (Xie et al., 2022) | 2022 | Region Impurity + Prediction Uncertainty | já citado; forte baseline |
| **Adaptive Superpixel AL** (Kim et al., 2023) | 2023 | Superpixel adaptativo por incerteza | alinha direto com sua proposta |
| **HALO** (Franco et al., 2023) | 2023 | Hardware-aware AL | já citado |
| **BADGE** | 2020 | Gradient embedding diversity | já listado nas siglas |
| **CEREALS** (Mackowiak et al., 2018) | 2018 | Region-based cost-effective AL | já citado |
| **Active Continual Learning** (Vu et al., 2023) | 2023 | Balanço entre retention e learnability | **crítico — fecha o gap AL+CL** |
| **DIAL** (Karamcheti et al., 2021) | 2021 | Deep Interactive + Active Learning, sensoriamento remoto | já citado |
| **S4AL** (Rangnekar et al., 2023) | 2023 | Semi-supervised + superpixel AL | próximo ao seu framework |

---

### 2.3 Continual Learning para Segmentação

| Método | Ano | Estratégia | Nota |
|---|---|---|---|
| **MiB** (Cermelli et al., 2020) | 2020 | Modeling the Background — KD sem old classes | baseline forte para CL sem replay |
| **PLOP** (Douillard et al., 2021) | 2021 | Pseudo-Label + distilação de features intermediárias | referência para CL on-the-fly |
| **DKD** (Decomposed KD, 2022) | 2022 | Separa KD por task vs. class | mais preciso que KD simples |
| **EWC** (Kirkpatrick et al., 2017) | 2017 | Elastic Weight Consolidation | já citado |
| **iCaRL** | 2017 | Replay exemplar | já citado |
| **Zheng et al. (AAAI 2021)** | 2021 | IL + CL integrados | sua base — reimplementado em PyTorch |

---

### 2.4 Métodos Baseados em Marcadores (onde entra o FLIM)

| Método | Paradigma | Anotação | Nota |
|---|---|---|---|
| **FLIM** (Bragantini et al.) | k-means CNN, sem backprop | 3–5 imagens + marcadores | baseline leve; ya testado |
| **GrabCut** (Rother et al., 2004) | Graph cuts | BG/FG scribbles | clássico, referência histórica |
| **ScribbleSup** (Lin et al., 2016) | CNN + CRF | Scribbles | benchmark de anotação esparsa |
| **DEXTR** (Maninis et al., 2018) | CNN + extreme points | 4 cliques extremos | muito eficiente em domínio médico |
| **BIFSeg** | Bayesian + scribbles | Scribbles | bom para imagens médicas |

---

## 3. Datasets Interessantes Não Citados na Qualificação

### 3.1 Benchmarks Padrão para Interactive Segmentation
| Dataset | Tamanho | Uso |
|---|---|---|
| **GrabCut** (Rother et al.) | 50 imgs | Clássico — obrigatório reportar |
| **Berkeley** (McGuinness & O'Connor) | 100 imgs | Benchmark IL padrão |
| **SBD** (Hariharan et al.) | 8.5k imgs | Benchmark NoC padrão |
| **DAVIS** | 10.4k frames | Video object segmentation |
| **SA-1B** (Meta, 2023) | 1B masks / 11M imgs | Treinamento SAM — enorme |

### 3.2 Domínio Médico (complementar ao seu plano)
| Dataset | Especialidade | Diferencial |
|---|---|---|
| **PolypPVT** (2022) | Pólipo | Imagens difíceis, bordas irregulares |
| **CholecSeg8k** (2023) | Cirurgia laparoscópica | Temporal, multi-estrutura |
| **IDRID** (2018) | Retinopatia diabética | Vasos finos, lesões pequenas |
| **Ham10000** (ISIC ext.) | Dermatologia | 10k imagens, 7 classes |
| **TotalSegmentator** (2023) | CT full-body | 104 estruturas, segmentação volumétrica |

### 3.3 Benchmarks para AL em Segmentação
| Dataset | Por quê é interessante |
|---|---|
| **CamVid** | Sequência temporal — AL pode priorizar frames mais informativos |
| **ADE20K** | 150 classes — desbalanceamento extremo, ideal para AL |
| **LVIS** | 1.2k classes com cauda longa — natural para AL adaptativo |

---

## 4. Contribuições Possíveis

### 4.1 Contribuições Primárias (alinhadas à qualificação)

**C1 — Online Active-Continual Loop com Superpixels adaptativos**
- Combinar Kim et al. (2023) + Zheng et al. (2021) + sua KD
- Métrica principal: NoC@85/90 + BWT
- Diferencial: o ciclo é totalmente online, sem batch offline

**C2 — Estratégia de click guiado por incerteza regional (RL-based)**
- Agente RL aprende qual região (superpixel) clicar para maximizar delta-IoU
- Comparar com simulação determinística atual (centroide do maior erro)
- Potencial: redução de 20–40% no NoC reportado na literatura

**C3 — Benchmark público de sequências de interação**
- Dataset com sequências temporais de cliques + máscaras intermediárias
- Permite comparação sistemática de frameworks IL
- Contribuição prática de alto impacto para a comunidade

### 4.2 Contribuições Secundárias / Análise Comparativa

**C4 — FLIM como baseline de anotação mínima**
- Posicionar FLIM no espectro: sem backprop vs. framework completo
- Mostra custo-benefício: FLIM (3–5 imgs) vs. seu framework (10–50 cliques)
- Axis: esforço de anotação × qualidade de segmentação

**C5 — SAM como teto zero-shot**
- SAM/SAM2 como upper bound sem fine-tuning de domínio
- Seu framework deve superar SAM em domínios específicos (médico) com < 50 cliques
- Narrativa forte: "superamos VLM 1B params com framework leve + poucos cliques"

---

## 5. Método de Interactive Learning para Comparação Direta

### Candidato Principal: **RITM** (Reviving Iterative Training with Mask Guidance)

**Por quê RITM:**
- Código aberto, bem mantido, reprodutível
- Benchmark padrão da área (todos os papers novos comparam com ele)
- Treina com a mesma simulação de cliques que você já implementou
- Disponível para todos os datasets do seu plano (SBD, GrabCut, Berkeley)

**Como comparar:**
```
Métrica          RITM (baseline)   SimpleClick   SAM (zero-shot)   SEU FRAMEWORK
────────────────────────────────────────────────────────────────────────────────
NoC@85 (SBD)       X.X               X.X           X.X               X.X
NoC@90 (SBD)       X.X               X.X           X.X               X.X
NoC@90 (GrabCut)   X.X               X.X           X.X               X.X
IoU após 5 clicks  X.X               X.X           X.X               X.X
BWT (CL)            -                 -              -                X.X
```

**Repositório RITM:** https://github.com/saic-vul/ritm_interactive_segmentation

---

## 6. Roadmap Sugerido

```
Fase 1 (feito): DeepLabV3 + IL + KD → PascalVOC2012
Fase 2 (próxima): Integrar RITM/SimpleClick como baseline de comparação
Fase 3: Implementar AL strategy (superpixel-based incerteza) no loop online
Fase 4: Experimentos médicos (Kvasir-SEG, BUSI, REFUGE2)
Fase 5: RL para seleção de cliques + benchmark público
```

---

## 7. Referências Adicionais Críticas

- Sofiiuk et al. (2022) — **RITM**: https://arxiv.org/abs/2102.06583
- Liu et al. (2023) — **SimpleClick**: https://arxiv.org/abs/2210.11006
- Kirillov et al. (2023) — **SAM**: https://arxiv.org/abs/2304.02643
- Ravi et al. (2024) — **SAM 2**: https://arxiv.org/abs/2408.00714
- Vu et al. (2023) — **Active Continual Learning**: https://arxiv.org/abs/2305.03923
- Cermelli et al. (2020) — **MiB**: https://arxiv.org/abs/2002.00718
- Douillard et al. (2021) — **PLOP**: https://arxiv.org/abs/2011.11390
