# Os modelos do FLIM e o que foi aplicado sobre eles

Documento de referência da dissertação. Explica, um a um, o que é cada modelo
da Tabela III/IV de Soares et al. (arXiv:2504.20872), como cada um está
implementado neste repositório, e o que o active learning acrescenta.

Companheiro de `ESTADO_ATUAL.md` (resultados) e `TABELA_IMAGEM_VS_REGIAO.md`
(AL por imagem vs por região).

---

## 1. O problema

Detecção de objetos salientes (SOD) em imagens de microscopia de fezes, para
encontrar ovos de *Schistosoma mansoni*. Duas restrições dominam o projeto:

1. **Anotação é cara.** Cada imagem anotada exige um especialista. O artigo
   inteiro é construído em torno de treinar com **3 a 5 imagens**.
2. **O modelo roda em campo.** Sem GPU, sem nuvem. Daí a insistência em
   arquiteturas de centenas de milhares de parâmetros, não milhões.

A métrica é o **Fβ com β² = 0.3** — pesa precisão mais que revocação, porque um
falso positivo manda o técnico examinar uma lâmina limpa.

---

## 2. O que é um encoder FLIM

FLIM = *Feature Learning from Image Markers*. É a peça que todos os modelos
"FLIM_*" compartilham. A ideia cabe em um parágrafo:

> O especialista rabisca alguns traços sobre o objeto e sobre o fundo. Cada
> traço vira um conjunto de posições. Em cada posição, recorta-se o *patch*
> 3×3 ao redor. Roda-se k-means sobre esses patches. **Os centróides viram os
> kernels da convolução.** Repete-se por bloco, propagando os marcadores para
> a resolução seguinte.

Consequências que importam para a dissertação:

- **Não há backpropagation no encoder.** Nenhum gradiente, nenhuma época,
  nenhum learning rate. Treinar leva ~5 segundos.
- **Cada kernel carrega o rótulo do marcador que o gerou** — λ(kernel) ∈
  {objeto, fundo}. Essa etiqueta é o que os decoders `lm`, `lt`, `pb` e `mb`
  consomem.
- **A arquitetura é pequena por construção**: 334 K parâmetros em média contra
  1.33 M do SAMNet e 3.27 M do MEANet (Tabela II do artigo).
- O espaço de cor é **LAB**, não RGB. Confirmado na implementação
  (`image_to_lab` em `flim_al/al_encoder_experiment.py:69`); trocar por RGB
  degrada tudo.

O encoder produz um mapa de ativações I_L com m′ canais. Cada canal é uma
resposta a um protótipo de textura. **Quem transforma m′ canais em um mapa de
saliência é o decoder** — e é aí que os oito modelos diferem.

---

## 3. Os decoders adaptativos, um a um

Um decoder adaptativo é uma convolução 1×1 cujos pesos **não são treinados**:
são recalculados para cada imagem de entrada por uma função heurística. Um
neurônio por pixel, seguido de ReLU:

    S(p) = Φ( Σᵢ αᵢ · I_Lⁱ(p) )

A diferença entre os modelos está inteiramente em **como se estima α**.

### FLIM_lm — *label-based fixed-weight* (código: `labeled_marker`)

O mais simples e, na prática, o melhor. O peso do canal é literalmente o
rótulo do marcador que gerou o kernel:

    αᵢ = λ(I_Lⁱ)     ∈ {+1 objeto, −1 fundo}

Nada é estimado a partir da imagem. A hipótese embutida: *um kernel derivado
de um marcador de objeto produz um canal que se acende no objeto*. Que ele
funcione tão bem (Fβ = 0.857 no teste do artigo) é o principal achado empírico
do trabalho — valida a etiquetagem de kernels como sinal de supervisão.

**É o decoder de referência de toda a dissertação.**

### FLIM_ts — *tri-state* (código: `vanilla_adaptive_decoder`)

Assume que o objeto ocupa pouco da imagem. Classifica cada canal por dois
testes simultâneos: a média de ativação do canal contra o limiar de Otsu τ da
distribuição de médias, e a fração tᵢ de pixels acima de Otsu **dentro** do
canal.

    αᵢ = −1   se μᵢ ≥ τ+σ  e  tᵢ > 0.2      (canal de fundo)
    αᵢ = +1   se μᵢ ≤ τ−σ  e  tᵢ < 0.1      (canal de objeto)
    αᵢ =  0   caso contrário                 (canal descartado)

Os limiares 0.2 e 0.1 são empíricos — calibrados nestes datasets. É o ponto
frágil do modelo: um prior de tamanho de objeto disfarçado de heurística.

### FLIM_lt — *label-based tri-state* (código: `hybrid_decoder`)

Combina os dois anteriores. Roda o tri-state, depois **zera todo canal cujo
kernel veio de marcador de fundo**:

    αᵢ = 0     se λ(I_Lⁱ) = fundo
    αᵢ = αᵢ(ts) caso contrário

Elimina o caso em que um kernel de fundo produz ativação de objeto e o
tri-state o interpreta mal. Na implementação (`layers.py:1116`) é exatamente
a tabela de gate entre `weights` e `adapted_weights`.

### FLIM_at — *attention-based* (código: `decoder_attention`)

Atenção de canal sem parâmetro treinável. Constrói uma atenção espacial `a`
somando max-pooling e average-pooling ao longo dos canais (ambos
renormalizados para [0,1]), e mede a importância de cada canal pelo **cosseno**
entre o canal e essa atenção:

    cᵢ = ⟨a, bᵢ⟩ / (‖a‖·‖bᵢ‖)

    αᵢ = +1  se cᵢ < μ_c − σ_c/2
    αᵢ = −1  se cᵢ > μ_c + σ_c/2
    αᵢ =  0  caso contrário

O sinal invertido não é engano: canais de objeto são os **menos** correlacionados
com a atenção média, porque o objeto é pequeno.

### FLIM_pb — *probability-based* (código: `decoder_2`)

O primeiro decoder que estima **um peso por pixel**, não por canal. Para cada
pixel p, olha uma vizinhança circular A(p) e calcula média μⱼ(p) e variância
σ²ⱼ(p) das ativações nos canais de rótulo j ∈ {objeto, fundo}. Daí uma
verossimilhança gaussiana:

    φᵢ,ⱼ(p) = exp( −(I_Lⁱ(p) − μⱼ(p))² / 2σ²ⱼ(p) )

    αᵢ(p) = +1  se λᵢ = objeto e φᵢ,objeto > φᵢ,fundo
    αᵢ(p) = −1  se λᵢ = fundo  e φᵢ,objeto < φᵢ,fundo
    αᵢ(p) =  0  caso contrário

Ou seja: o canal só vota se **concordar localmente com o próprio rótulo**. É o
decoder mais caro (janelas deslizantes por pixel) e, junto com `lm`, o mais
preciso no artigo.

### FLIM_mb — *mean-based* (código: `decoder_3`)

Simplificação do `pb` por eficiência: descarta as variâncias e compara só as
médias.

    αᵢ(p) = +1  se λᵢ = objeto e μ_objeto(p) > μ_fundo(p)
    αᵢ(p) = −1  se λᵢ = fundo  e μ_objeto(p) < μ_fundo(p)
    αᵢ(p) =  0  caso contrário

No artigo entrega quase o mesmo Fβ do `pb` por uma fração do custo.

### `vanilla_adaptive_decoder_wt` — variante sem contraparte no artigo

Existe no código, é o tri-state **sem a regra de proporção tᵢ**: decide só pela
média do canal (`robust_adaptation_weights_wt`, `layers.py:605`). Não aparece
na Tabela III. Nos nossos números é sistematicamente o pior de todos — o que é
uma evidência útil: **a regra de proporção de Otsu é o que faz o tri-state
funcionar**, não o teste de média. Reportamos como `FLIM_ts*` e tratamos como
ablação, não como modelo do artigo.

---

## 4. As linhas de base

### FLIM_bp — *backpropagation-based fixed-weight* (código: `backprop_decoder`)

Mesmo encoder FLIM, **congelado**. O vetor α da convolução 1×1 é inicializado
por Xavier e otimizado por backpropagation com a ground truth de T: Adam,
lr = 0.01, 100 épocas, perda = média de Dice e BCE.

Serve de teto do que a informação do encoder permite quando se abandona a
heurística e se usa gradiente. No artigo fica em 0.757 — **abaixo** do `lm`
(0.857), que não usa gradiente nenhum. É o resultado mais contraintuitivo da
tabela e a melhor defesa do método.

Este é o decoder do **Experimento B** da dissertação (396 execuções).

### U-Net_FLIM

Encoder FLIM congelado + decoder em U com skip connections de cada bloco do
encoder para o bloco correspondente do decoder, treinado por backprop em T.
2.29 M parâmetros — 7× o FLIM puro. Fβ = 0.777.

### SAMNet, MSCNet, MEANet

Três modelos leves de SOD do estado da arte, treinados por backpropagation nas
mesmas 3–5 imagens. Não usam markers. Estão na tabela para responder à
pergunta óbvia: *e se eu simplesmente usar uma rede pronta?* Resposta: com 3
imagens, SAMNet chega a Fβ = 0.422 com desvio de 0.241 — instável a ponto de
ser inutilizável. Entre 1.33 M e 3.27 M parâmetros.

---

## 5. Pós-processamento (Tabela I do artigo)

| dataset | modelo | pós-processamento |
|---|---|---|
| Parasites | leves | Otsu + filtro de área [1000, 9000] |
| Parasites | **FLIM** | Otsu + filtro de área [1000, 9000] **+ Dynamic Trees** |
| Parasites | U-Net_FLIM | Otsu + filtro de área [1000, 9000] |
| BraTS | todos | Otsu + filtro de área [100, 20000] |

Mais a remoção de componentes conectadas à borda da imagem, no Parasites.

> **Restrição desta reprodução.** O binário `iftSMansoniDelineation`, que
> implementa Dynamic Trees, depende de `liblapack`/`libblas` ausentes na
> imagem Docker. Todos os números reproduzidos usam **apenas Otsu + filtro de
> área**. É a explicação principal da diferença para o artigo, e por isso as
> comparações válidas são sempre **dentro** das nossas tabelas.

---

## 6. Como o artigo escolhe as imagens de treino

Algoritmo 1 do artigo, verbatim:

```
Entrada: Z₁, T ← {}
1  sorteia z ∈ Z₁, T ← T ∪ {z}
2  x_prev ← 0, z_prev ← z
3  enquanto o usuário não estiver satisfeito:
4      treina uma CNN FLIM com decoder adaptativo em T
5      avalia o modelo em Z₁\T
6      x ← Fβ médio em Z₁\T
7      entre as imagens de MENOR Fβ, seleciona z ∈ Z₁\T
8      se x < x_prev:
9          T ← T \ {z_prev}            (backtracking)
10     senão:
11         T ← T ∪ {z}
12     x_prev ← x, z_prev ← z
```

Isso **já é active learning**. Mas o passo 7 lê o Fβ de todas as 848 imagens do
pool — ou seja, **exige ground truth de todo o pool para escolher 3 imagens**.
O artigo é explícito: adota-se uma abordagem *supervisionada*. Isso anula a
premissa do FLIM, que é justamente não ter anotação.

**A pergunta da dissertação nasce daqui:**

> Quanto do benefício do Algoritmo 1 se recupera com um critério que não usa
> ground truth nenhum?

---

## 7. O que foi aplicado — os critérios de aquisição

Cada critério ordena o pool não anotado e devolve as K imagens a anotar.

| critério | usa GT do pool? | ideia |
|---|---|---|
| **oracle** (Algoritmo 1) | **sim** | menor Fβ real — teto |
| **entropy** | não | maior entropia binária média do mapa de saliência |
| **least confidence** | não | menor distância de 0.5 (para binário ≡ margin) |
| **CoreSet** | não | k-center guloso no espaço de features do encoder — cobertura, não incerteza |
| **BADGE** | não | k-means++ sobre embeddings de gradiente — magnitude do gradiente × diversidade |
| **BALD** | não | discordância entre um comitê de encoders |
| **random** | não | piso |

Dois **níveis de granularidade**, e essa distinção é a contribuição central:

- **AL por imagem** — *quais imagens o especialista deve anotar?*
  Sai uma lista de nomes de arquivo. É o que o Algoritmo 1 faz.
- **AL por região** — *dentro da imagem, onde vale a pena anotar?*
  Sai um mapa de regiões. Usa a incerteza espacialmente resolvida para
  posicionar os marcadores. Corresponde ao pedido da orientadora de "explorar
  AL sobre pixels em vez de imagens".

---

## 8. O braço de controle que mudou a conclusão

Os métodos de região pareciam vencer com folga. Mas eles mudam **duas coisas
ao mesmo tempo**: quais imagens entram em T *e* como os seeds são desenhados
dentro delas. Sem separar, o ganho da geometria é creditado à seleção.

Introduzimos o braço `random_region`: imagens **sorteadas**, anotadas com o
**mesmo gerador de seeds** do AL. Ele decompõe o ganho:

    seleção    = AL_região − random_região      quais imagens
    geometria  = random_região − random         como anotar

Resultado, média sobre os sete decoders:

| componente | ΔFβ |
|---|---|
| seleção de imagens | +0.050 |
| **geometria dos seeds** | **+0.108** |

**A geometria pesa 2.1× a seleção.** Ou seja: para o FLIM, *onde* o
especialista desenha importa mais que *qual imagem* ele abre. Isso reorienta a
dissertação — e é um achado que só apareceu porque o controle foi construído.

---

## 9. O que sabemos que está quebrado

Registro honesto, porque cada item custou execução:

1. **49% do pool não tem foreground nenhum** (181 de 371 imagens na fase F4).
   O critério "pior Fβ" do artigo prioriza exatamente essas — elas têm Fβ = 0
   porque o filtro de área as zera, não porque sejam informativas. **O
   critério publicado seleciona o insegmentável, não o informativo.**
2. **O Algoritmo 1 pode entrar em ciclo.** Sem lista de rejeitados, a imagem
   removida pelo backtracking volta a ser a de menor Fβ e é re-selecionada
   indefinidamente. Nossa reimplementação acrescenta uma lista tabu.
3. **O algoritmo desperdiça 2.0 anotações por execução** (33% das rodadas):
   imagens que o especialista anotou, que pioraram o modelo e foram
   descartadas.
4. **Predição colapsada** tem assinatura numérica exata: Fβ = DICE = IoU ≈
   0.4788, que é 406/848 — a fração de imagens vazias. Toda tabela marca essas
   execuções com ⚠, porque o "ganho" delas não é aprendizado.

Quatro hipóteses foram testadas e **refutadas** por medição: markers
pontilhados não causam o colapso; conjunto pequeno não causa o colapso (Fβ =
0.54 com 1 imagem); markers realistas não corrigem o Experimento A (6 de 6
configurações pioram); e "dispersão alta de entropia = sinal" está
**invertida** — o encoder bom tem dispersão *baixa* (0.005–0.007).

---

## 10. Mapa nome-do-artigo ↔ nome-no-código

Estabelecido lendo as equações da Seção IV contra `pyflim/layers.py`.

| artigo | código | onde | peso por |
|---|---|---|---|
| FLIM_lm | `labeled_marker` | `layers.py:688` | canal (rótulo do marcador) |
| FLIM_bp | `backprop_decoder` | `layers.py` | canal (backprop) |
| FLIM_ts | `vanilla_adaptive_decoder` | `layers.py:575` | canal (tri-state) |
| FLIM_lt | `hybrid_decoder` | `layers.py:1116` | canal (tri-state × rótulo) |
| FLIM_at | `decoder_attention` | `layers.py:744` | canal (cosseno com atenção) |
| FLIM_pb | `decoder_2` | `layers.py:~795` | **pixel** (gaussiana) |
| FLIM_mb | `decoder_3` | `layers.py:~990` | **pixel** (comparação de médias) |
| — (ablação) | `vanilla_adaptive_decoder_wt` | `layers.py:605` | canal (tri-state sem tᵢ) |

---

## 11. A reprodução, lado a lado com o artigo

Teste Z₂, usuário A, média ± desvio sobre os três splits. Otsu + filtro de
área, **sem** Dynamic Trees. O bloco de cada decoder é escolhido pelo melhor Fβ
na validação Z₁\T, como o artigo especifica na Seção VI-A.

| modelo | MAE | Fβ reproduzido | Fβ artigo | Δ | posto (nosso → artigo) |
|---|---|---|---|---|---|
| FLIM_mb | 0.009 | **0.790±0.011** | 0.847 | −0.057 | 1 → 3 |
| FLIM_lm | 0.009 | 0.788±0.009 | 0.857 | −0.069 | 2 → 1 |
| FLIM_pb | 0.009 | 0.786±0.005 | 0.857 | −0.071 | 3 → 1 |
| FLIM_lt | 0.010 | 0.731±0.015 | 0.803 | −0.072 | 4 → 4 |
| FLIM_ts | 0.011 | 0.660±0.032 | 0.747 | −0.087 | 5 → 5 |
| FLIM_at | 0.014 | 0.611±0.067 | 0.733 | −0.122 | 6 → 6 |
| FLIM_ts* | 0.015 | 0.585±0.035 | — | — | sem par |

**Correlação de postos com a Tabela IV: ρ de Spearman = 0.81 (p = 0.05).** As
três primeiras posições permutam entre si dentro de 0.004 de Fβ — abaixo do
desvio entre splits, portanto indistinguíveis. Da quarta posição para baixo, a
ordem do artigo é reproduzida **exatamente**.

O deslocamento é uniforme, Δ médio = −0.080, e tem causa conhecida: a ausência
de Dynamic Trees. Não é ruído distribuído; é um viés de sinal único que atinge
todos os modelos.

### A escolha do bloco vale 0.08 de Fβ

Existe uma segunda reprodução no repositório, gerada pelo pipeline original com
o bloco **fixo** (`flim_ad/out/metrics/`). Ela cobre os dois usuários e inclui
o `FLIM_bp`, mas fica bem pior:

| variante | Δ médio | ρ de Spearman | cobertura |
|---|---|---|---|
| bloco escolhido na validação | **−0.080** | **0.81** (p = 0.05) | usuário A, 7 decoders |
| bloco fixo | −0.160 | 0.60 (p = 0.16) | usuários A e B, 8 decoders |

A diferença de 0.08 de Fβ entre as duas variantes é inteiramente atribuível ao
passo de seleção de bloco. Isso não é detalhe de implementação: **metade da
lacuna que parecia ser "falta de Dynamic Trees" era, na verdade, avaliar todos
os decoders numa profundidade que não era a melhor para eles.** Os números da
variante de bloco fixo, para registro:

| modelo | Fβ (A) | Fβ (B) |
|---|---|---|
| FLIM_mb | 0.699±0.014 | 0.599±0.138 |
| FLIM_bp | 0.694±0.020 | 0.671±0.010 |
| FLIM_lm | 0.691±0.010 | 0.681±0.012 |
| FLIM_pb | 0.685±0.013 | 0.590±0.089 |
| FLIM_lt | 0.635±0.016 | 0.524±0.107 |
| FLIM_ts | 0.576±0.055 | 0.461±0.121 |
| FLIM_at | 0.529±0.070 | 0.360±0.212 |
| FLIM_ts* | 0.473±0.022 | 0.359±0.213 |

O desvio entre usuários é grande para os decoders fracos: `at` vai de 0.529 (A)
a 0.360 (B), com desvio de 0.212 entre splits. Isso reproduz o comportamento do
artigo, onde `at` também é o mais instável.

---

## 12. O active learning, medido por modelo

As tabelas do Capítulo 7 comparam **critérios de aquisição** entre si, fixando o
decoder. Esta compara os **decoders** entre si, fixando o critério — é a coluna
que se encaixa ao lado da Tabela IV do artigo.

Dois braços, mesma tubulação, mesmo protocolo de seleção de bloco:

- **T do artigo** — as 3 imagens que os usuários anotaram rodando o Algoritmo 1,
  cujo passo 7 lê a ground truth de todo o pool.
- **T do AL** — as imagens que o critério de entropia escolheu, **sem ver
  ground truth nenhuma**, com os markers reais das mesmas imagens.

Teste Z₂ (366 imagens), usuário A, média ± desvio sobre 3 splits. `p` é o teste
t pareado por split.

| modelo | Fβ · T do artigo | Fβ · T do AL | ΔFβ | p | IoU (AL) |
|---|---|---|---|---|---|
| FLIM_lm | 0.788±0.009 | 0.790±0.018 | +0.002 | 0.889 | 0.663 |
| FLIM_pb | 0.786±0.005 | 0.783±0.020 | −0.003 | 0.796 | 0.657 |
| FLIM_mb | 0.790±0.011 | 0.792±0.018 | +0.002 | 0.909 | 0.666 |
| FLIM_lt | 0.731±0.015 | 0.698±0.070 | −0.032 | 0.569 | 0.585 |
| FLIM_ts | 0.660±0.032 | 0.669±0.100 | +0.010 | 0.858 | 0.556 |
| **FLIM_at** | 0.611±0.067 | **0.712±0.045** | **+0.101** | **0.015** | 0.591 |
| FLIM_ts* | 0.585±0.035 | 0.618±0.089 | +0.033 | 0.410 | 0.508 |

### O resultado é o empate, não o ganho

Nos três decoders fortes — `lm`, `pb`, `mb` — o Δ é de ±0.003 com p > 0.79.
Isso não é "o AL não funcionou". É a afirmação central:

> Um critério que **nunca vê ground truth** iguala a seleção supervisionada do
> artigo, anotando o mesmo tanto (3.0 contra 3.3 imagens em média).

O passo 7 do Algoritmo 1 exige Fβ de 848 imagens anotadas para escolher 3. Se a
entropia do próprio modelo produz um T de qualidade estatisticamente
indistinguível, esse custo é dispensável — e a premissa do FLIM, de treinar sem
anotação em massa, deixa de ser contrariada pelo próprio método de seleção.

### A exceção que é real

O `FLIM_at` ganha **+0.101 (p = 0.015)**, e o ganho é consistente nos três
splits: +0.117, +0.111, +0.076. É o único efeito fora do ruído na tabela.

Uma leitura plausível: o decoder de atenção não usa os rótulos dos kernels
(Eq. 7–8 dependem só das ativações), então é o mais sensível à qualidade do
encoder — e portanto o que mais tem a ganhar com um T melhor. Os decoders que
consomem λ(kernel) já têm uma âncora de supervisão que os protege de um T ruim.
É uma hipótese coerente com a construção dos modelos, **não uma medição**:
testá-la exigiria variar a qualidade do encoder de forma controlada.

### O que NÃO se sustenta

Com os splits 1 e 3 apenas, os quatro decoders fracos pareciam ganhar de forma
sistemática (+0.117/+0.076, +0.098, +0.055/+0.076, +0.048). O split 2 reverte
`ts` (−0.062), `lt` (−0.118) e `ts*` (−0.030), e a correlação entre a força do
decoder e o ganho fica em ρ = −0.64 com **p = 0.12** — sugestiva, não
significativa. Registrado aqui porque é exatamente o tipo de padrão que n = 2
faz parecer sólido.

Note também que no split 2 o AL anotou **4** imagens contra 3 do artigo e mesmo
assim perdeu na maioria dos decoders. Mais anotação não compra desempenho
quando as imagens escolhidas são piores.

---

## 13. BraTS — a reprodução sem etapa faltando

O Parasites deixa uma dúvida que nenhuma análise interna resolve: o Δ de −0.080
contra a Tabela IV é mesmo dos Dynamic Trees, ou é erro de reprodução com uma
desculpa conveniente? A Tabela I do artigo dá o experimento que decide.

| dataset | modelo | pós-processamento |
|---|---|---|
| Parasites | FLIM | Otsu + área [1000, 9000] **+ Dynamic Trees** |
| BraTS | FLIM | Otsu + área [100, 20000] — **sem DT** |

No BraTS não falta nada. Se a explicação estiver certa, o Δ ali tem de
desaparecer.

### Resultado

Teste Z₂ completo (1877 imagens), 4 imagens de treino por split, encoder de 3
blocos, bloco escolhido na validação, média ± desvio sobre os 3 splits.

| modelo | Fβ reproduzido | Fβ artigo | Δ |
|---|---|---|---|
| FLIM_lm | 0.697±0.007 | 0.678 | +0.019 |
| FLIM_pb | 0.680±0.012 | 0.711 | −0.031 |
| FLIM_mb | **0.739±0.016** | 0.712 | +0.027 |
| FLIM_lt | 0.726±0.022 | 0.731 | −0.005 |
| FLIM_ts | 0.699±0.008 | 0.720 | −0.021 |
| FLIM_at | 0.708±0.014 | 0.688 | +0.020 |
| FLIM_ts* | 0.698±0.016 | — | — |

### O contraste é o resultado

| | Δ médio | negativos | p (média = 0) | p (teste de sinal) |
|---|---|---|---|---|
| **BraTS** | **+0.0015** | 3/6 | 0.881 | 1.000 |
| **Parasites** | −0.0797 | **6/6** | **0.0004** | **0.031** |

No Parasites o desvio é significativamente diferente de zero **e**
unidirecional: os seis decoders com par no artigo erram todos para baixo, o que
sozinho tem p = 0.031. No BraTS o desvio médio é de quinze décimos de milésimo,
os sinais se dividem meio a meio, e nenhum decoder passa de |Δ| = 0.031.

Isso é a assinatura de um componente ausente no pipeline, não de erro de
reprodução. Um erro de implementação produziria desvio em direções
imprevisíveis nos dois datasets, ou nos dois de forma sistemática — não
sistemático em exatamente aquele que recebe uma etapa a mais.

**Consequência para a defesa.** A afirmação deixa de ser "reproduzimos a ordem
dos modelos mas não os valores" e passa a ser: *a reprodução é exata onde o
pipeline está completo; onde falta um componente identificado, o desvio é
uniforme, unidirecional e do tamanho esperado.*

### Por que o BraTS também é o melhor terreno para o active learning

Duas fraquezas do experimento principal desaparecem aqui:

| | Schisto | BraTS |
|---|---|---|
| markers reais | 31 para 1220 imagens | **3753 para 3753** |
| imagens do val com foreground | 51% | **100%** |

A primeira significa que o AL por imagem sobre o pool completo do Schisto só
existia com markers sintéticos — uma muleta documentada. No BraTS o critério
escolhe entre candidatos que têm anotação humana de verdade.

A segunda é mais importante. A crítica central ao passo 7 do Algoritmo 1 — que
o critério "pior Fβ" prioriza imagens insegmentáveis — depende de existirem
imagens sem objeto, e no Schisto 49% do pool é assim. No BraTS não há nenhuma.
Portanto o BraTS testa se as conclusões sobre AL sobrevivem **justamente quando
a patologia que as gera desaparece**. Se o padrão se repetir, ele não era
artefato do dataset; se não se repetir, isso delimita o escopo da conclusão — e
os dois resultados são publicáveis.

### Duas armadilhas do pyflim que custaram execução

Ambas falham em silêncio ou com erro deslocado da causa, e ambas só aparecem
fora do layout do Schisto:

1. **Grayscale.** Com `convert_gray_to_lab=False`, o `FLIMData` devolve para
   uma imagem 2-D **um canal cru, 0–255, sem LAB e sem normalizar**
   (`data.py`, ramo `else` do `__getitem__`). O `evaluate_decoder` fazia
   `.convert("RGB")` seguido de LAB, ou seja três canais normalizados. Treinar
   com um canal e avaliar com três não gera exceção — gera um encoder avaliado
   fora da distribuição em que foi estimado. Corrigido com um carregador único,
   `load_input`, verificado bit-a-bit contra o comportamento antigo no Schisto.

2. **A extensão do arquivo.** O `FLIMData` decide se acrescenta `.png` com
   `len(image_path.split(".")) == 1`. Um caminho relativo com `..` introduz
   dois pontos, o teste falha, a extensão não é acrescentada e o `imread`
   quebra reportando um caminho absoluto sem extensão — longe da causa. Por
   isso a configuração do BraTS usa caminhos **absolutos**.

---

## 14. Como reexecutar

Tudo roda dentro do container `flim-gpu` (`flim-ad-env:gpu`), com o repositório
montado em `/workspace/flim-python-demo`, e a partir de `flim_ad/`. A avaliação
dos decoders é feita em CPU por exigência do pyflim; o treino do encoder usa a
GPU.

**1. A tabela reproduzida e a de AL por modelo** (~3 h com os 3 splits em
paralelo; cada split leva ~1 h por braço):

```bash
for s in 1 2 3; do docker exec -d flim-gpu bash -lc "cd /workspace/flim-python-demo/flim_ad && OMP_NUM_THREADS=4 python -u ../flim_al/final_model_comparison.py --split $s --n_val 120 --blocks 2 3 --examples > out/final_cmp_s$s.log 2>&1"; done
```

**2. Escolher os exemplos que discriminam e regerar as máscaras** (~5 min). Sem
este passo as figuras usam as primeiras imagens do teste, que são as fáceis —
os sete decoders produzem máscaras indistinguíveis e a figura não compara nada:

```bash
python ../flim_al/pick_examples.py --split 1 --n_cand 45
```

**3. As figuras de segmentação** (`fig8_seg_por_modelo.png`,
`fig9a_al_melhorou.png`, `fig9b_al_piorou.png`):

```bash
python ../flim_al/make_seg_figure.py --split 1 --out ../figs
```

**4. A apresentação** (19 slides; lê os CSVs e as figuras produzidos acima):

```bash
python ../flim_al/build_deck.py --out ../FLIM_AL_Apresentacao_v5.pptx
```

A variante de bloco fixo, se for preciso regerá-la, sai de
`bash ../run_schisto_corrected.sh`.
