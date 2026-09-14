# Referências e Direções de Pesquisa — FLIM-AD + Active Learning

> Feedback da reunião com a professora — organizado com links verificados.

---

## 1. Artigos base do projeto

### Paper reproduzido (FLIM-AD)
**FLIM-Based Salient Object Detection Networks with Adaptive Decoders**  
Soares, Cerqueira, Gomes, Najman, Guimarães, Falcão — UNICAMP/LIDS, 2025  
- arXiv: https://arxiv.org/abs/2504.20872  
- Código: https://github.com/LIDS-UNICAMP/FLIM-AD  
- Lab: https://lids.ic.unicamp.br

### Paper anterior (Flyweight CNNs — precursor direto)
**Adaptive Decoders for FLIM-based Salient Object Detection Networks**  
Soares et al., SIBGRAPI 2024  
- IEEE: https://ieeexplore.ieee.org/document/10716332  
*(Dissertação de mestrado de Gilson Junior Soares — UNICAMP, orientador: Prof. Alexandre Xavier Falcão)*

### Paper original do FLIM (conceito de markers + k-means)
O conceito de treinar CNNs com marcadores e k-means (sem backprop) vem dos trabalhos do grupo LIDS:
- Lab LIDS-UNICAMP (Alexandre X. Falcão): https://lids.ic.unicamp.br/researches/1
- Buscar por: `"Feature Learning from Image Markers" Falcão UNICAMP`
- Biblioteca pyflim: https://github.com/LIDS-UNICAMP/flim-python

---

## 2. Datasets relevantes para ampliar experimentos

### Salient Object Detection (SOD) — benchmark padrão

| Dataset | Imagens | Característica | Link |
|---------|---------|----------------|------|
| **DUTS** | 15.572 | Benchmark SOD padrão; DUTS-TR (10k treino) + DUTS-TE (5k teste) | http://saliencydetection.net/duts/ |
| **ECSSD** | 1.000 | Objetos complexos em backgrounds confusos — alto desafio | https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/ |
| **DUT-OMRON** | 5.168 | Múltiplos objetos, backgrounds estruturados, alta complexidade | http://saliencydetection.net/dut-omron/ |
| **PASCAL-S** | 850 | Derivado do PASCAL VOC; objetos de classes diversas | http://cbi.gatech.edu/salobj/ |
| **SOD** | 300 | Dataset pioneiro; objetos simples — bom como baseline fácil | (clássico, buscar em paperswithcode) |
| **HKU-IS** | 4.500 | Múltiplos objetos salientados por imagem | https://i.cs.hku.hk/~gbli/deep_saliency.html |

### Casos extremos / Camuflagem

| Dataset | Imagens | Característica | Link |
|---------|---------|----------------|------|
| **MoCA** | 87 vídeos | Objetos camuflados em movimento — caso extremo de alta entropia | https://www.robots.ox.ac.uk/~vgg/data/MoCA/ |
| **COD10K** | 10.000 | Camouflaged Object Detection — foreground = background | https://github.com/DengPingFan/SINet |

### Médico / domínio específico

| Dataset | Imagens | Característica | Link |
|---------|---------|----------------|------|
| **Schistossoma** (atual) | 1.219 | Ovos de parasita em microscopia fecal — LIDS-UNICAMP | https://github.com/LIDS-Datasets/schistossoma-eggs |
| **ISIC Skin Lesion** | ~33k | Lesões de pele — domínio médico com anotação cara | https://challenge.isic-archive.com |
| **Kvasir-SEG** | 1.000 | Pólipos em colonoscopia — objetos pequenos e escassos | https://datasets.simula.no/kvasir-seg/ |

### Critério de curadoria sugerido pela professora
- **Alta saliência**: DUTS, HKU-IS — modelo deve funcionar bem
- **Baixa saliência / camuflagem**: COD10K, MoCA — caso difícil; alta entropia → bom para AL
- **Médico**: Schistossoma (já temos), Kvasir, ISIC — custo real de anotação justifica AL
- **Diversidade de domínio**: natural + médico + camuflagem para validar generalização

---

## 3. Estado da arte em Active Learning

### 3.1 Uncertainty-based (incerteza — o que já implementamos)

**Entropy Sampling** — baseline clássico  
- Seleciona amostras com maior entropia na predição  
- Nossa implementação: entropy dos saliency maps do `labeled_marker`  
- Referência survey: https://arxiv.org/abs/2009.00236 *(A Survey of Deep Active Learning)*

**BALD — Bayesian Active Learning by Disagreement**  
- Seleciona por mutual information entre predição e parâmetros do modelo (requer dropout)  
- Gal et al., 2017: https://arxiv.org/abs/1703.02910

### 3.2 Diversity-based (diversidade)

**CoreSet (ICLR 2018)**  
Sener & Savarese — seleciona K pontos que maximizam cobertura geométrica do espaço de features  
- arXiv: https://arxiv.org/abs/1708.00489  
- Ideia: minimizar o maior raio de cobertura entre pontos selecionados e não-selecionados

### 3.3 Hybrid: uncertainty + diversity

**BADGE (ICLR 2020)**  
*Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds*  
- Ash et al. — gradientes de alta magnitude + diversidade via k-means++  
- arXiv: https://arxiv.org/abs/1906.03671  
- **Relevância**: combina os dois critérios sem hiperparâmetros manuais

**BatchBALD (NeurIPS 2019)**  
*BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning*  
- Kirsch, van Amersfoort, Gal — mutual information para batch inteiro (não ponto a ponto)  
- arXiv: https://arxiv.org/abs/1906.08158  
- **Relevância**: evita selecionar pontos redundantes no mesmo batch

### 3.4 AL por regiões/pixels (direção sugerida pela professora)

**RIPU (CVPR 2022)**  
*Region Impurity and Prediction Uncertainty for Active Learning in Semantic Segmentation*  
- Seleciona **superpixels** com alta impureza de região × incerteza de predição  
- arXiv: https://arxiv.org/abs/2111.12667  
- **Por que importa**: em vez de anotar imagem inteira, anota apenas regiões incertas  
- Código: https://github.com/BIT-DA/RIPU

**PixelAL / Suggestive Annotation**  
*Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation*  
- Yang et al., MICCAI 2017 — seleciona **pixels** ou regiões para anotação em imagens médicas  
- arXiv: https://arxiv.org/abs/1706.04737  
- **Relevância direta**: mesmo domínio (médico) + mesma granularidade pixel

**ViewAL (CVPR 2020)**  
*ViewAL: Active Learning with Viewpoint Information for Semantic Segmentation*  
- Siddiqui et al. — incerteza por superpixel com consistência multi-view  
- arXiv: https://arxiv.org/abs/1911.11789

### 3.5 AL com interação do especialista (Interactive AL)

**DBAL com feedback de especialista**  
- Conceito: especialista não anota GT mask completa; apenas corrige regiões erradas  
- Integração natural com o paradigma FLIM (seeds/markers já são interativos)  
- Referência: *Interactive Active Learning* — buscar por "human-in-the-loop segmentation"

---

## 4. Direções propostas pela professora — como implementar

### 4.1 Visualização antes vs depois
```
Gerar grid visual para cada decoder + resultado AL:
  - Col 1: imagem original
  - Col 2–9: saliency de cada decoder (FLIMts, FLIMat, FLIMpb, FLIMad2, FLIMad3, etc.)
  - Col 10: resultado AL (backprop_decoder treinado com K selecionado por entropia)
  - Col 11: GT mask
```
Código: script Python com matplotlib → salvar grid como PNG por imagem

### 4.2 Por que 3–5 imagens para treinar o encoder FLIM?

FLIM usa **k-means por camada** — não backprop. Com 3–5 imagens:
- Cada imagem contribui com patches de objeto + fundo (via markers desenhados pelo usuário)
- k-means encontra centroides representativos → esses viram os kernels convolucionais
- Mais imagens não ajudam muito: k-means satura a diversidade de patches a partir de certo ponto
- É análogo a aprendizado por exemplares ("exemplar-based learning")
- Número mínimo é empírico: suficiente para cobrir variabilidade visual do domínio

### 4.3 AL por pixels ao invés de imagens inteiras
Em vez de selecionar imagens inteiras para anotação GT mask:
1. Calcular **mapa de entropia por pixel** sobre o saliency map
2. Identificar **regiões de alta incerteza** (patches 64×64 ao redor dos pixels com entropia > threshold)
3. Pedir ao especialista para anotar **apenas esses patches** (muito mais rápido que GT mask completa)
4. Treinar backprop_decoder apenas nos patches anotados (atenção ao forward pass parcial)

Paper de referência: RIPU (https://arxiv.org/abs/2111.12667)

### 4.4 Intervenção do especialista em imagens de alta confusão
Conceito: **Interactive Active Learning integrado ao FLIM**
1. Ranquear imagens por entropia (já implementado)
2. Especialista vê imagem + saliency map atual (com regiões de erro destacadas)
3. Em vez de GT mask completa → adiciona **novos seed markers** nas regiões erradas
4. FLIM re-gera saliency com os markers adicionais → sem retreino do encoder
5. Loop: se saliency melhorar, marcar como "corrigida"; senão, pedir GT mask

Vantagem: mantém o paradigma original do FLIM (seeds leves) sem custo de anotação completa.

---

## 5. Recursos do lab LIDS-UNICAMP

- Site: https://lids.ic.unicamp.br
- Repositório pyflim: https://github.com/LIDS-UNICAMP/flim-python
- Repositório FLIM-AD: https://github.com/LIDS-UNICAMP/FLIM-AD
- Dataset Schistossoma: https://github.com/LIDS-Datasets/schistossoma-eggs
- Contato: afalcao at ic dot unicamp dot br

---

## 6. Próximos passos sugeridos

- [ ] Implementar AL por região (superpixel entropy) baseado em RIPU
- [ ] Levantar DUTS + ECSSD para testar generalização do FLIM-AD
- [ ] Gerar visualizações antes/depois para apresentação
- [ ] Comparar BADGE vs Entropy sampling no experimento backprop_decoder
- [ ] Explorar Interactive AL: markers em regiões de alta entropia ao invés de GT masks
