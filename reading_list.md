# Lista de Leitura — Mestrado Daniel Galetti
**Prioridade de leitura por pilar do framework**

---

## PILAR 1 — Interactive Segmentation (IL) — Leia primeiro

### 1. RITM — Reviving Iterative Training with Mask Guidance
**Sofiiuk et al., 2021**
- **PDF:** https://arxiv.org/pdf/2102.06583
- **Código:** https://github.com/saic-vul/ritm_interactive_segmentation
- **Por que ler:** É o benchmark padrão da área. Todo paper de IL compara com ele. Usa HRNet/ViT com mask guidance do passo anterior — exatamente o mecanismo que você vai precisar entender para implementar sua simulação de cliques corretamente. Treinado em COCO+LVIS.
- **Foco:** Seção 3 (mask guidance input), Seção 4 (iterative training), Tabela 1 (NoC@90 no SBD/GrabCut).

---

### 2. SimpleClick — Plain ViT for Interactive Segmentation
**Liu et al., 2023**
- **PDF:** https://arxiv.org/pdf/2210.11006
- **Código:** https://github.com/uncbiag/SimpleClick
- **Por que ler:** SOTA atual. Usa ViT puro (sem hierarquia) com symmetric patch embedding para cliques. Alcança 4.15 NoC@90 no SBD — melhor que RITM em 21.8%. Avaliado em imagens médicas também. Mostra que não precisa de arquitetura complexa.
- **Foco:** Seção 3 (symmetric patch embedding), Seção 5 (experimentos médicos — direto relevante para você).

---

### 3. SAM — Segment Anything
**Kirillov et al., Meta AI, 2023**
- **PDF:** https://arxiv.org/pdf/2304.02643
- **Site/Código:** https://segment-anything.com
- **Por que ler:** Foundation model que você usará como baseline zero-shot. Entender o prompt encoder (point, box, mask) é essencial. Dataset SA-1B com 1B masks.
- **Foco:** Seção 3 (model), Seção 5 (zero-shot experiments) — especialmente Tabela 7 (interactive evaluation vs RITM).

---

### 4. SAM 2 — Segment Anything in Images and Videos
**Ravi et al., Meta AI, 2024**
- **PDF:** https://arxiv.org/pdf/2408.00714
- **Código:** https://github.com/facebookresearch/sam2
- **Por que ler:** Versão melhorada: 6x mais rápido que SAM, melhor zero-shot, suporte a vídeo. Usa streaming memory transformer. Se você for usar SAM como baseline, use SAM 2.
- **Foco:** Seção 3 (model architecture), Seção 5 (image segmentation comparison).

---

## PILAR 2 — Active Learning para Segmentação

### 5. RIPU — Region Impurity and Prediction Uncertainty
**Xie et al., 2022**
- **PDF:** https://arxiv.org/pdf/2111.12940
- **Código:** https://github.com/BIT-DA/RIPU
- **Por que ler:** Um dos melhores AL para segmentação. Combina impureza de região (estrutura espacial) com incerteza de predição. Relevante para seu mecanismo de seleção de regiões a anotar. Já citado na sua qualificação.
- **Foco:** Seção 3 (acquisition strategy RIPU), Seção 4 (comparação com pixel-based vs region-based).

---

### 6. Adaptive Superpixel for Active Learning
**Kim et al., 2023**
- **PDF:** https://arxiv.org/pdf/2303.16817
- **Por que ler:** Combina superpixels adaptativos (baseados em features aprendidas, não RGB fixo) com AL para segmentação. Inclui mecanismo de sieving para ruído de anotação. Alinha diretamente com seu objetivo de usar superpixels + AL. Cityscapes + PascalVOC.
- **Foco:** Seção 3.1 (adaptive superpixel), Seção 3.2 (sieving mechanism), Tabela 1.

---

## PILAR 3 — Continual Learning para Segmentação

### 7. MiB — Modeling the Background for Incremental Learning
**Cermelli et al., CVPR 2020**
- **PDF:** https://arxiv.org/pdf/2002.00718
- **Código:** https://github.com/fcdl94/MiB
- **Por que ler:** Resolve o problema de background shift em CL para segmentação — classes antigas viram "background" e o modelo esquece. KD revisitado para este cenário. PascalVOC2012 + ADE20K.
- **Foco:** Seção 3 (background shift), Seção 4 (distillation loss), Tabela 1 (Pascal disjoint/overlapped).

---

### 8. PLOP — Pseudo-Labels for Online Plasticity
**Douillard et al., 2021**
- **PDF:** https://arxiv.org/pdf/2011.11390
- **Por que ler:** Multi-scale pooling distillation (Local POD) que preserva relações espaciais de curto e longo alcance. Pseudo-labeling do background baseado em entropia. Diretamente relevante para sua KD adaptativa.
- **Foco:** Seção 3 (Local POD distillation), Seção 3.3 (pseudo-labeling de background), comparação com MiB.

---

### 9. Active Continual Learning
**Vu et al., 2023**
- **PDF:** https://arxiv.org/pdf/2305.03923
- **Por que ler:** O único paper que estuda diretamente AL + CL juntos — seu gap exato. Analisa o trade-off entre reter conhecimento antigo (CL) e aprender novo rápido (AL). Domain, class e task-incremental.
- **Foco:** Seção 3 (ACL framework), Seção 4 (forgetting-learning profile — esse conceito é chave para sua contribuição).

---

### 10. Zheng et al. — Continual Learning for Interactive Segmentation
**AAAI 2021** *(sua base — reimplementada)*
- **PDF:** https://ojs.aaai.org/index.php/AAAI/article/view/16752
- **Por que ler:** Já é sua base. Reler depois dos outros para identificar exatamente o que você melhorou.

---

## Ordem de Leitura Recomendada

```
Semana 1:  RITM → SimpleClick          (entender IL e benchmarks)
Semana 2:  SAM → SAM 2                 (entender o teto zero-shot)
Semana 3:  MiB → PLOP                  (entender CL para segmentação)
Semana 4:  Kim et al. → RIPU           (entender AL com superpixels)
Semana 5:  Active CL → reler Zheng     (fechar o gap e definir contribuição)
```

---

## Dataset do Kaggle: BUSI (Breast Ultrasound Images)

**Está citado na sua própria qualificação (referência Al-Dhabyani et al., 2020).**

- **Kaggle:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **Tamanho:** 780 imagens (~250 MB)
- **Classes:** Normal (133), Benigno (437), Maligno (210)
- **Resolução:** ~500×500 px, ultrassom
- **Máscaras:** GT pixel-level disponíveis

**Por que é ideal para um experimento adicional:**
1. Já está no seu planejamento (Tabela 6 da qualificação)
2. Está no Kaggle — download imediato com a API
3. Domínio de ultrassom = ruído speckle intenso → desafio real para IL
4. Tamanho gerenciável (780 imgs) para experimento rápido
5. 3 classes com forte desbalanceamento → bom para testar AL

**Comando para baixar:**
```bash
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
```

**Estrutura esperada após extração:**
```
Dataset_BUSI_with_GT/
├── benign/          # imagem + _mask.png
├── malignant/       # imagem + _mask.png
└── normal/          # imagem + _mask.png
```
