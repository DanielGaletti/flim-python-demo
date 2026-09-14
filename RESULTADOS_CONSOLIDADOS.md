# Resultados consolidados — AL sobre FLIM

Estado em 26/08/2026. Só o que está **fechado e verificado**; o que ainda roda
está listado no fim.

---

## 1. A pergunta

O Algoritmo 1 do artigo (Soares et al., arXiv:2504.20872) escolhe quais 3 a 5
imagens o especialista deve anotar. Mas o passo 7 lê o Fβ de **todas as 848
imagens do pool** para escolher 3 — precisa da ground truth que o FLIM existe
para evitar. O artigo é explícito: *"We adopted a supervised approach"*.

> **Quanto do benefício se recupera com um critério que não usa ground truth?**

Para responder é preciso régua com os dois extremos: o **teto** (a seleção
supervisionada do artigo) e o **piso** (escolha aleatória, mesmo orçamento).

---

## 2. A resposta

Teste Z₂ comum (364 imagens), usuário A, **9 execuções independentes** do
Algoritmo 1 por split, média sobre 3 splits.

| modelo | teto (supervisionado) | entropia (sem GT) | piso (aleatório) | ganho sobre o acaso | p | recuperação |
|---|---|---|---|---|---|---|
| FLIM_lm | 0.788 | 0.735 | 0.728 | +0.008 | 0.309 | **13%** |
| FLIM_pb | 0.786 | 0.734 | 0.728 | +0.007 | 0.492 | 11% |
| FLIM_mb | 0.790 | 0.705 | 0.699 | +0.006 | 0.571 | 7% |
| FLIM_lt | 0.730 | 0.700 | 0.691 | +0.010 | 0.392 | 25% |
| FLIM_ts | 0.660 | 0.672 | 0.650 | +0.023 | 0.026 | — |
| FLIM_at | 0.607 | 0.654 | 0.594 | **+0.060** | **0.016** | — |
| FLIM_ts* | 0.584 | 0.589 | 0.587 | +0.002 | 0.926 | — |

**Sobre os 21 pontos: entropia − aleatório = +0.0165, p = 0.0056.**

Duas afirmações verdadeiras ao mesmo tempo:

1. **A entropia funciona** — supera o acaso de forma significativa, não é ruído.
2. **Recupera pouco.** Nos três decoders fortes a supervisão vale 0.060 de Fβ e
   a entropia compra de volta 0.008: **7 a 13%**. O valor está na ground truth,
   não no critério.

O artigo afirma precisar do passo supervisionado e nunca mede quanto isso vale.
Agora está medido.

### A exceção com sentido

O `FLIM_at` tem o maior ganho (+0.060, p = 0.016) e é o **único decoder que não
usa os rótulos dos markers** — as Eq. 7–8 dependem só das ativações. Sem essa
âncora de supervisão, ele é o mais sensível à qualidade das imagens escolhidas.
É uma leitura coerente com a construção dos modelos, não uma medição direta.

---

## 3. A reprodução, e por que ela é confiável

| dataset | Δ médio vs artigo | negativos | p (média = 0) | p (sinal) |
|---|---|---|---|---|
| **BraTS** | **+0.0015** | 3/6 | 0.881 | 1.000 |
| Parasites | −0.0797 | **6/6** | **0.0004** | **0.031** |

O Parasites fica 0.08 abaixo do publicado, **sistematicamente**, nos seis
decoders com par no artigo. O BraTS bate quase exato, com erros nos dois
sentidos.

A diferença entre os dois casos é uma linha da Tabela I do artigo: os modelos
FLIM do Parasites recebem **Dynamic Trees** no pós-processamento, e os do BraTS
não. O binário `iftSMansoniDelineation` depende de `liblapack`/`libblas`,
ausentes na imagem Docker.

Isso converte a maior fraqueza da reprodução em evidência: **onde o pipeline
está completo, a reprodução é exata; onde falta um componente identificado, o
desvio é uniforme, unidirecional e do tamanho esperado.** Um erro de
implementação produziria desvio nos dois datasets, ou em direções
imprevisíveis — não sistemático em exatamente aquele que recebe uma etapa a
mais.

### A escolha do bloco vale 0.08

| variante | Δ médio | ρ de Spearman |
|---|---|---|
| bloco escolhido na validação (protocolo do artigo) | −0.080 | **0.81** (p = 0.05) |
| bloco fixo | −0.160 | 0.60 (p = 0.16) |

Metade da lacuna que parecia ser "falta de Dynamic Trees" era, na verdade,
avaliar cada decoder numa profundidade que não era a melhor para ele.

---

## 4. Três defeitos no Algoritmo 1, medidos

### 4.1 O critério prioriza o insegmentável

49% do pool do Schisto não tem objeto nenhum. Essas imagens têm Fβ = 0 porque o
filtro de área zera a predição — não porque sejam informativas. O passo 7
escolhe "entre as de menor Fβ", ou seja, justamente elas.

### 4.2 O backtracking pode entrar em ciclo

O pseudocódigo devolve a imagem rejeitada ao conjunto de candidatos. Como o
critério é determinístico, ela volta a ser a de menor Fβ na rodada seguinte e o
laço trava. Observado com `000013`. A reimplementação usa lista tabu.

### 4.3 A parada confunde ruído com saturação

O passo 8 retrocede na **primeira** piora. No BraTS o algoritmo encerra com
|T| = 1 ou 2 imagens, contra as 4 que os usuários anotaram.

---

## 5. O achado mais forte: a variância vem do sorteio

| fonte de variação | desvio de Fβ |
|---|---|
| **imagem inicial sorteada** (passo 1) | **0.10 a 0.22** |
| partição dos dados (split) | 0.009 |

A imagem inicial explica uma ordem de grandeza mais variância que os dados. O
artigo roda uma execução e reporta desvio sobre splits — mede a fonte menor e
ignora a maior.

Consequência prática: **qualquer conclusão baseada numa execução do Algoritmo 1
é uma amostra de uma distribuição larga.** Foi por isso que o "empate" apareceu
e desapareceu três vezes nesta investigação, conforme o número de execuções
subiu de 1 para 3 e depois para 9.

---

## 6. O mapa nome-do-artigo ↔ nome-no-código

Não estava documentado. Estabelecido lendo as equações da Seção IV contra
`pyflim/layers.py`.

| artigo | código | equação |
|---|---|---|
| FLIM_lm | `labeled_marker` | α = λ(kernel) |
| FLIM_bp | `backprop_decoder` | 1×1 por Adam |
| FLIM_ts | `vanilla_adaptive_decoder` | Eq. 6 |
| FLIM_lt | `hybrid_decoder` | Eq. 9 |
| FLIM_at | `decoder_attention` | Eq. 7–8 |
| FLIM_pb | `decoder_2` | Eq. 12–13 |
| FLIM_mb | `decoder_3` | Eq. 14 |

`vanilla_adaptive_decoder_wt` **não tem contraparte no artigo** — é o tri-state
sem a regra de proporção de Otsu. Reportado como `FLIM_ts*`, tratado como
ablação.

---

## 7. Armadilhas do pyflim que custaram execução

Ambas falham em silêncio ou com erro deslocado da causa, e só aparecem fora do
layout do Schisto:

1. **Grayscale.** Com `convert_gray_to_lab=False` o `FLIMData` devolve para uma
   imagem 2-D **um canal cru, sem LAB e sem normalizar**. A avaliação fazia
   `.convert("RGB")` + LAB — três canais normalizados. Treinar com um canal e
   avaliar com três não gera exceção.
2. **A extensão do arquivo.** O `FLIMData` decide se acrescenta `.png` com
   `len(caminho.split(".")) == 1`. Um caminho relativo com `..` tem dois pontos,
   o teste falha e o `imread` quebra reportando um caminho absoluto — longe da
   causa. Por isso a configuração do BraTS usa caminhos absolutos.

---

## 8. Vazamento de conjunto de teste, encontrado e corrigido

| dataset | vazamento | correção |
|---|---|---|
| BraTS | 1877 dos 3753 markers estão no teste | pool restrito a Z₁ |
| Schisto | `000675` e `000917` | removidas do **teste**, não do pool |

A distinção importa: vazamento é problema de **avaliação**, não de seleção.
Excluir as imagens do pool proibiria o critério de escolher o que o usuário
escolheu — o T do artigo no split 1 é `{000479, 000675, 000917}` — e a
comparação passaria a medir o handicap, não o critério.

---

## 9. Aplicação interativa

`http://localhost:8000` (container `flim-app`). Desenha markers, treina o
encoder em ~3 s, mostra a segmentação, e o critério sugere a próxima imagem.

Só é viável porque o FLIM não usa backpropagation no encoder: estimar os
kernels é k-means sobre os patches. A demonstração é, ela própria, um argumento
sobre o método.

---

## 10. Ainda em execução

| experimento | o que responde |
|---|---|
| CoreSet no protocolo v3 | recupera mais que os 13% da entropia? |
| Paciência = 2 | a parada por ruído custa quanto? |
| Orçamento em pinceladas | poucas imagens densas ou muitas esparsas? |
| Ablação (medoide, filtro, proposto) | qual correção pesa? |
| Usuário B | a conclusão vale para outro anotador? |

A régua já existe: cada um responde **quanto dos 0.060 recupera**.
