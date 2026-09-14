# Estado atual — dissertação FLIM + Active Learning

Documento único de contexto. Substitui a leitura de `CONTEXTO_DISSERTACAO.md`
(cujas tabelas estão obsoletas), `PROTOCOLO_REVISADO.md` e
`PLANO_EXPERIMENTOS.md`, que ficam como histórico.

Defesa 14/11/2026 · depósito 05–16/10/2026.

---

## 1. A pergunta da dissertação, reformulada

O paper reproduzido (Soares et al., arXiv:2504.20872) **já faz active
learning** no Algoritmo 1: T começa com uma imagem aleatória, treina, avalia
em Z₁\T, escolhe a próxima entre as de **pior Fβ**, com backtracking. Mas o
artigo declara: *"We adopted a **supervised** approach to select a few
representative images"* — o passo 7 exige ground truth de todo o pool (848
imagens) para escolher 3.

Isso derrota o propósito do FLIM. A pergunta bem-posta passa a ser:

> Quanto do benefício do Algoritmo 1 se recupera com um critério que não usa
> ground truth nenhum?

| braço | usa GT do pool? | papel |
|---|---|---|
| Algoritmo 1 (paper) | sim | teto |
| entropy / coreset / badge / BALD | não | proposta |
| aleatório | não | piso |

Isso também responde ao "por que 3 a 5 imagens?" do orientador: o algoritmo
para quando acrescentar imagem deixa de melhorar o Fβ, e satura em 3–4.

---

## 2. O que está estabelecido

**Experimento B** (encoder fixo + decoder 1×1, 396 execuções, comparação
pareada, piso de ruído ±0.023):

- **CoreSet é o único método que supera o aleatório**: ΔFβ +0.072,
  IC95% [+0.038, +0.106], p<0.001.
- **Com K=10, CoreSet e BADGE igualam o pool completo** (~190 imagens):
  0.575 e 0.563 contra 0.567. Diferença dentro do ruído → 19× menos imagens
  anotadas sem perda mensurável.
- **Incerteza pura é pior que o acaso em orçamento pequeno**: entropy K=3 dá
  −0.076 (p=0.012); region_entropy K=3 dá −0.153 (p<0.001, vence em 1 de 15).
- **Teto baixo**: o decoder 1×1 com o pool inteiro chega a 0.567, contra 0.744
  do `labeled_marker` com 3 markers. O Exp. B mede seleção dentro desse teto.

**Experimento A** (encoder retreinado) — decomposição com o braço de controle
`random_region`:

- **A geometria dos seeds pesa ~2× a seleção de imagens** (média +0.108 vs
  +0.050). O ganho aparente do region_bald vem sobretudo de COMO anotar.

**Algoritmo 1 reimplementado** (`paper_selection.py`):

- Sobe monotonicamente até Fβ=0.7445 com |T|=4, no patamar do encoder
  publicado.
- **O critério "pior Fβ" seleciona o insegmentável, não o informativo** — as
  imagens de Fβ=0 são as que o filtro de área zera.
- **O algoritmo, como publicado, pode entrar em ciclo**: sem lista de
  rejeitados, a imagem removida pelo backtracking volta a ser a de menor Fβ.
- Desperdiça 2.0 anotações por execução (33% das rodadas).

**Ferramenta de anotação caracterizada** (`analyze_real_markers.py`, 31
arquivos reais): pincel circular de raio 5 (81 px), ~4 toques de foreground a
~12 px dentro da borda, ~9 de background a ~121 px fora.

---

## 3. Hipóteses testadas e refutadas

Registradas porque cada uma custou execução e todas foram descartadas por
medição:

1. *"O colapso do encoder vem dos markers pontilhados."* Refutada: com imagens
   que têm foreground, o gerador de pontos funciona bem (Fβ 0.54–0.64).
2. *"Dispersão alta da entropia = critério com sinal."* **Invertida**: o
   encoder bom (Fβ=0.744) tem dispersão baixa (0.005–0.007); alta é sintoma de
   modelo ruim.
3. *"Conjunto de treino pequeno causa o colapso."* Refutada: com 1 imagem o
   Fβ é 0.54.
4. *"Markers realistas corrigem o Experimento A."* Refutada: 6 de 6
   configurações pioram (−0.11 a −0.32). Com 12 pinceladas o Fβ se recupera
   ao nível do gerador antigo, mas não o supera.

**O que sobrou e se sustenta:** 49% do pool não tem foreground nenhum, e o
critério do paper prioriza justamente essas imagens.

---

## 4. Restrições

- **Markers reais só para 31 imagens** do Schisto. AL de imagem sobre o pool
  completo exige markers sintéticos.
- **49% do pool sem foreground** → encoder degenerado por construção.
- **Filtro de área [1000, 9000] px** é prior rígido de tamanho: ovos fora
  dessa faixa são insegmentáveis, independentemente do encoder.
- **Avaliação com Otsu+AF**, não Dynamic Trees (faltam `liblapack`/`libblas`)
  → não comparável à Tabela III do paper.
- **GPU**: só na imagem `flim-ad-env:gpu` (Ubuntu 22.04). A `flim-ad-env:latest`
  é Ubuntu 18.04 (glibc 2.27) e não roda os wheels cu128.

---

## 5. Datasets disponíveis

| dataset | imagens | resolução | markers | splits |
|---|---|---|---|---|
| Schisto (Parasites) | 1220 | 400×400 RGB | 31 reais | 5train-70_30 |
| **BraTS** | **3753** | 240×240 L | **3753** | 50_50 (4 train / 1872 val / 1877 test) |
| Conjunctiva | 83 | 1079×863 RGB | 5 | 57 / 18 / 8 |

BraTS e Conjunctiva estão em `data/`, não em `flim_ad/datasets/` (que tem
cópias parciais).

---

## 6. Artefatos

| arquivo | conteúdo |
|---|---|
| `TABELA_IMAGEM_VS_REGIAO.md` | AL por imagem vs por região + decomposição |
| `RESULTADOS_DISSERTACAO.md` | Exp. A e B com IC e teste pareado |
| `RESULTADOS_SELECAO.md` | Algoritmo 1, pool real |
| `figs/` | 7 figuras de processo |
| `FLIM_AL_Apresentacao_v4.pptx` | 17 slides |

Scripts: `run_all_dissertation.sh` (pipeline, com lock contra execução dupla),
`run_selection_experiment.sh`, `run_schisto_backprop.sh`,
`run_schisto_corrected.sh`. Testes em `tests/`.
