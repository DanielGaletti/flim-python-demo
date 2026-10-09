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
representative images for T from the validation set Z₁\T"* — o passo 7 exige
ground truth de todo o Z₁\T para escolher 3.

> **Correção de 2026-10-09.** Este parágrafo dizia "848 imagens". O número
> está errado como descrição do artigo: 848 é o tamanho do pool da **nossa**
> reprodução (70% de 1211 no split `5train-70_30`), não do conjunto sobre o
> qual o Algoritmo 1 opera. O artigo divide Z em Z₁ e Z₂ com 50% cada, sobre
> 1219 imagens, de modo que **Z₁\T tem cerca de 606**. Verificado no PDF:
> seções "Experimental setup" e "E. Representative image selection".
> Corrigido também em `1_introducao.tex` e `3_metodologia.tex`.

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

  > **ESCOPO, e superado em 2026-10 — ver §2.1.** Este número vale para
  > **Schisto apenas**, com **encoder fixo** e decoder 1×1 cujo teto é 0.567.
  > Não se reproduz quando o encoder é retreinado, nem nos outros datasets.
  > Dois motivos concretos: o CoreSet **nunca rodou no BraTS** até 2026-09
  > (`extract_encoder_features` assumia 3 bandas e o BraTS é grayscale), e a
  > conjuntivite nunca rodou com filtro de área válido. No confirmatório de
  > 10 sementes × 3 datasets × 2 decoders, o CoreSet fica **−0.0606 ABAIXO**
  > do sorteio na média (p=0.0020).
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

## 2.1 Estado em 2026-10 — o que mudou, e por quê

Esta seção supera parte da §2. Ela existe porque três bugs de montagem
invalidaram números anteriores, e porque duas campanhas confirmatórias
fecharam.

### Os bugs que invalidaram resultado

| bug | efeito | onde |
|---|---|---|
| `extract_encoder_features` assumia 3 bandas | **CoreSet nunca rodou no BraTS** (grayscale → `IndexError`) | `flim_al/coreset_badge.py` |
| `orig_ext=".png"` fixo | **Conjuntivite nunca rodou** (imagens `.jpg`) | `scripts/tabela_k_por_modelo.py` |
| filtro de área `[1000,9000]` copiado do Schisto | **0 de 39** objetos da conjuntivite cabiam; Fβ 0.050 → 0.573 ao corrigir para `[5000,300000]` | `flim_app/datasets.py` |
| top-K em lote com encoder de 1 imagem | não era Active Learning: faltava o laço iterativo com retreino | `scripts/tabela_k_por_modelo.py` |
| `rng.integers()` avançava por arquivo | marcadores dependiam da ordem de leitura do diretório | idem |

120 execuções da conjuntivite foram postas em quarentena
(`evidencia/execucoes/descartados_filtro_conjuntivite.csv`). Qualquer número
anterior a essas correções que envolva BraTS ou conjuntivite **não vale**.

### Nenhuma vantagem do Active Learning sobre o sorteio

Testado em 3 datasets × 4 orçamentos (K ∈ {2,3,5,8}) × 7 decoders, com
pré-registro, teste pareado e correção de Benjamini-Hochberg:

- **Seleção de imagem**: nenhum critério se destaca. 96 testes, 0 sobrevivem a
  BH. Sinal do CoreSet: 14 de 24 células, p=0.541.
- **O oracle do próprio artigo** (passo 7 do Algoritmo 1, que lê o gabarito do
  pool) **perde do sorteio em 9 de 24** células.
- **AL de região** contra clique aleatório dentro do objeto: vence 4 de 12,
  Δ médio −0.0215, 0 sobrevivem a BH.
- **Tabela III reproduzida**, cliques realocados: AL−artigo dá 7/7 negativo
  significativo, mas **aleatório−artigo também dá 7/7**, e **AL−aleatório dá
  0 de 7**. O dano vem de tirar o clique da borda, não da escolha do AL.
- **Dispersão** (confirmatório `confirmatorio_estabilidade_2026-10-01`,
  10 sementes inéditas): entropy contra sorteio, teste primário com **um
  conjunto de teste por célula, n=16**: Δ desvio −0.0186, **p=0.5674 →
  FALSEADO**. Na média, CoreSet fica −0.0606 abaixo do sorteio (p=0.0020).

  > **Errata de 2026-10-05**, registrada em `RESULTADO/ERRATA_2026-10-05` do
  > JSON da campanha. A versão anterior deste parágrafo dizia *24 células* e
  > *p=0.8765*; as duas estavam erradas. 4 das 12 células do teste primário
  > misturavam dois conjuntos de teste, o que infla o desvio por troca de
  > conjunto e não por efeito do critério. Nos três cortes limpos o p é 0.83
  > (schisto+brats, 8 células), 0.24 (conjuntivite, 4 células) e 0.5674
  > (tudo, 16 células). **O erro afetou o número, não o veredito.**

**Formulação correta:** não há evidência de superioridade nas condições
avaliadas. Ausência de significância **não** é prova de equivalência.

### Dois falsos positivos capturados pelo pré-registro

Vale mais que um ganho pequeno, e é argumento de método:

1. Ganho da conjuntivite na densidade: p=0.0016 na exploratória, **p=0.586**
   no confirmatório.
2. Entropia menos dispersa: p=0.082 post-hoc com 3 sementes, **p=0.5674** no
   teste primário com 10 sementes inéditas (ver a errata acima; o p=0.8765
   que este item trazia antes vinha de células contaminadas).

### O diagnóstico do ganho marginal — onde está o mecanismo

Campanha `ganho_marginal` (pré-registro + adendo em `evidencia/campanhas/`):
fixa partição, imagens, marcadores base, encoder inicial e validação, e varia
só **qual região recebe os ~300 px seguintes**.

Firme nas 6 células (3 datasets × 2 decoders), nos dois sentidos:

- **melhor região** supera o sorteio: +0.133 a +0.303, p ≤ 0.0037 em 6 de 6
- **pior região** fica abaixo: −0.199 a −0.502, p ≤ 0.0015 em 6 de 6
- **uma anotação extra em lugar sorteado tem Δ médio negativo**
- o **sorteio uniforme reponderado sem GT** fica a p ≥ 0.16 do estratificado:
  o viés da estratificação por gabarito era desprezível

Ou seja: **há oportunidade real na escolha da região, e há risco grande.** O
que falha é o critério, não o regime.

O escore de entropia bate o sorteio em 2 de 6 células (p=0.017 e p=0.0067),
**ambas no decoder de que o escore foi extraído** (`labeled_marker`, ver
`ganho_marginal.py:315`). Nas 3 células não alinhadas: ~0. Com BH sobre as 6,
1 sobrevive. **O escore prevê ganho para o decoder de que saiu — não é "AL
funciona".**

E o Δ próprio da entropia nunca se distingue de zero (IC95% [−0.058,+0.226],
[−0.347,+0.120], [−0.073,+0.151]). Ela só bate o sorteio porque **o sorteio é
nocivo**. A afirmação defensável é *evita o dano*, não *melhora o modelo*.

### Capacidade do encoder NÃO explica o efeito

Hipótese anterior deste projeto — mais pixels, mais kernels, encoder melhor —
**contradita em duas frentes**:

- Schisto: ρ(Δkernels, ΔFβ) = **−0.314** (p=0.0008). Mais kernels, **pior**.
- BraTS (`noutput_channels: 8`) e conjuntivite (16/12/8/6): kernels
  **constantes**, e o Δ ainda varia de −0.78 a +0.76.

Compatível com **deslocamento**: sem backpropagação os filtros *são* os
patches anotados, e anotação nova injeta agrupamentos que diluem os que
funcionavam. **Hipótese, não mecanismo demonstrado** — a correlação com
kernels não estabelece causalidade. A ablação que fecharia isso varia
`noutput_channels` no mesmo dataset com os mesmos marcadores, e depende de
verificar se é válida nesta variante do FLIM, onde o número de kernels é
consequência dos patches disponíveis e não hiperparâmetro livre.

A `fração de foreground` da região prevê ganho em todas as células medidas
(ρ +0.28 / +0.12 / +0.28) — sinal mais simples que a entropia.

### Determinismo: o limite medido

Determinismo vale **dentro de um processo**, a partir dos mesmos arquivos de
marcador (3 repetições → Fβ idêntico a 10 casas). **Entre** execuções há
ruído: na reexecução completa do Schisto, 538 de 540 registros deram Fβ
idêntico; os 2 restantes divergiram em no máximo **0.0034** — duas ordens de
grandeza abaixo dos efeitos reportados.

### Lacunas declaradas

- **Não há controle direto borda vs interior.** O que existe é borda (cliques
  reais do especialista, ~12 px dentro da borda) contra uniforme **dentro do
  objeto**, que inclui posições perto da borda. O "Fβ 0.665 vs 0.000" que
  circulava vem de `flim_al/fig_tutorial.py`, um gerador de figura sobre
  **3 imagens, sem sementes e sem teste pareado** — é ilustração, não medida.
- As 10 sementes do diagnóstico **compartilham pool e validação**
  (`semente_particao=0`). O IC95% vale para *aquela* validação sob sorteio das
  imagens de treino, e não generaliza para o dataset.
- A oportunidade medida é entre os **22 candidatos examinados** de ~380
  superpixels. Não é teto absoluto.
- A imagem nova inteira gasta **milhares** de pixels contra 300 da região: é
  controle diagnóstico de mecanismo, **não** comparação de orçamento igual.
  O braço de orçamento igual está pendente.
- Anotação simulada do gabarito. **Nada aqui autoriza afirmar redução de tempo
  de especialista** — isso exigiria avaliação humana.

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
