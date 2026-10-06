# Prompt para outro chat: reescrever as tabelas e o texto da dissertação

Copie tudo a partir da linha `---` até o fim e cole no outro chat.
O repositório precisa estar aberto nele, porque há caminhos a abrir.

---

Você vai ajustar a apresentação das tabelas e o texto de uma dissertação de
mestrado em LaTeX. O conteúdo científico já está medido e registrado. Sua
tarefa é de **apresentação e redação**, não de produzir resultado novo.

# 0. A regra que não cede

**Nenhum número pode ser inventado, estimado, arredondado para ficar bonito,
ou alterado à mão.** Todos os valores desta dissertação saem de um registro
de execuções versionado. Se um número parecer errado, inconsistente ou
inconveniente, a conduta é apontar isso em texto e perguntar, nunca corrigir
o dígito.

Vale também para referências bibliográficas: **toda citação precisa de DOI,
arXiv ID ou URL verificável**. Se não for possível verificar, escreva
`UNKNOWN` no campo em vez de um palpite plausível. Referência fabricada
reprova a defesa.

Se o autor pedir dado fictício "só para ilustrar" ou "só para o slide",
recuse essa parte, diga em uma frase por quê, e siga com o resto.

Três consequências práticas:

1. Ao mudar uma tabela, mude **a forma**, não os dígitos. Larguras, colunas,
   quebra em duas tabelas, `booktabs`, notas movidas para o corpo do texto,
   tudo isso é livre. O valor dentro da célula não é.
2. A tabela ideal não é editada no `.tex`: é editada no **gerador**,
   `flim_al/tabelas.py`, e regerada. Os `.tex` trazem no cabeçalho
   `% NAO EDITE A MAO: a proxima geracao sobrescreve`. Se você editar só o
   `.tex`, a próxima geração desfaz, e pior, o número editado deixa de bater
   com a evidência bruta.
3. Há distinção entre quatro níveis que não devem colapsar:

| nível | exemplo |
|---|---|
| dado bruto | Fβ de cada semente, em `evidencia/` |
| observação | "a média subiu de 0,688 para 0,697" |
| resultado agregado | "média ± desvio, IC95%, t pareado p = 0,33" |
| interpretação | "CoreSet parece mais eficiente em amostra" |
| afirmação (claim) | "sob o dataset D e orçamento B, CoreSet melhorou a eficiência de amostra em relação ao sorteio" |

"Melhorou" e "melhorou significativamente" são afirmações diferentes. Sem
teste pareado e sem n, a segunda não se escreve. `p` alto significa **não
decidido**, nunca "igual": ausência de evidência não é evidência de
equivalência.

# 1. A dissertação, em uma página

**Programa:** PPGCC UFSCar, mestrado. Defesa em 14/11/2026, depósito entre
05/10/2026 e 16/10/2026.

**Objeto:** FLIM (Feature Learning from Image Markers). É uma forma de
construir uma rede convolucional de segmentação **sem retropropagação**: o
especialista desenha alguns traços na imagem, o método recorta os patches
sob esses traços, roda k-means em dois níveis, e os centróides **são** os
filtros da camada. Treinar custa segundos, não horas, e funciona com três a
cinco imagens anotadas.

**A pergunta original e por que ela mudou.** O trabalho de referência
(Soares et al., arXiv:2504.20872) já faz seleção ativa no seu Algoritmo 1:
começa com uma imagem, treina, avalia no pool, escolhe a próxima entre as de
**pior Fβ**, com backtracking. Só que o próprio artigo declara que a
abordagem de seleção é *supervisionada*: o passo 7 exige ground truth de
todo o pool (848 imagens) para escolher 3. Isso derrota o propósito do FLIM,
que é justamente não precisar de anotação em massa. A pergunta bem-posta
passou a ser:

> Quanto do benefício do Algoritmo 1 se recupera com um critério que **não
> usa ground truth nenhum**?

| braço | usa GT do pool? | papel |
|---|---|---|
| Algoritmo 1 do artigo | sim | teto |
| entropy, least confidence, CoreSet, BADGE, BALD | não | proposta |
| sorteio aleatório | não | piso |

**O que a investigação encontrou, e que é o eixo da dissertação.** Nenhum
critério de Active Learning supera o sorteio aleatório quando aplicado ao
FLIM, em cerca de 142 comparações. Isso **não** é um resultado vazio, porque
a investigação seguiu e mostrou **por que** não transfere, numa cadeia em que
cada elo foi medido:

1. **A margem existe.** Com orçamento fixo de pixels, existe uma região cuja
   anotação vale de **+0,135 a +0,309 de Fβ** acima do candidato sorteado, em
   6 de 6 células (3 datasets × 2 decodificadores), todas com p ≤ 0,0031.
   Esse é **o custo do Active Learning**: o que se deixa na mesa quando o
   critério erra o lugar. É o número que o autor quer no centro do trabalho.
2. **Os critérios encontram o objeto, mas erram o lugar dentro dele.** Eles
   caem no interior do objeto, onde o filtro já acerta, e não na borda, onde
   erra.
3. **O gerador de candidatos é o gargalo, não o escore.** Só 2% a 12% dos
   superpixels SLIC cruzam a borda do objeto, porque o SLIC segue justamente
   as bordas. O critério não pode escolher o que não está na lista.
4. **Intervir no gerador funciona.** Trocar superpixel por faixa de contorno
   previsto dá **+0,0772 de Fβ** no dataset de parasitas, IC95%
   [+0,0393, +0,1151], **10 de 10 sementes**, p = 0,0013, sobrevivendo a
   correção de Benjamini-Hochberg. O nulo do BraTS é **previsto pelo
   mecanismo** e confirma ele.

**Também entram no trabalho:** Continual Learning sem retropropagação
(interpolação do banco de filtros, parâmetro α), Interactive Learning com
usuário simulado e a métrica NoC, uma plataforma localhost de demonstração, e
cinco modelos de comparação treinados com retropropagação.

**Volume alvo:** mais de 120 páginas. Hoje o texto tem cerca de 22 mil
palavras distribuídas em 5 capítulos, o que fica abaixo disso. Expandir é
parte da tarefa, usando as tabelas ainda não citadas, os modelos de
comparação, e as capturas da plataforma.

# 2. Onde está cada coisa

## 2.1 Leia primeiro, nesta ordem

| arquivo | o que é |
|---|---|
| `CLAUDE.md` | como trabalhar no repositório: as regras de integridade e as armadilhas. **Leitura obrigatória.** |
| `ESTADO_ATUAL.md` | fonte de verdade sobre o estado **científico**: o que está estabelecido, o que foi refutado, quais lacunas são declaradas |
| `DISSERTACAO/LEIAME.md` | como a pasta do Overleaf está organizada |

## 2.2 O projeto LaTeX

```
DISSERTACAO/
  main.tex              documento principal (capa, resumo, sumário, ordem dos capítulos)
  ufscar.cls            classe do programa, NÃO alterar
  capitulos/
    1_introducao.tex      problema, motivação, objetivos, questões e hipóteses
    2_fundamentos.tex     ML, segmentação, superpixels, IL, CL, FLIM, relacionados
    3_metodologia.tex     FLIM em detalhe, métricas, AL, CL, IL, plataforma, protocolo
    4_resultados.tex      as medições
    5_conclusao.tex       respostas, contribuições, limitações, futuros
  pretextual/abreviaturas.tex
  referencias/referencias.bib
  tabelas/              14 arquivos: 7 tabelas + 7 notas. GERADO, não editar à mão.
  figuras/              9 em uso; `nao_usadas/` guarda 56 da qualificação
  notas/Reuniao.tex     anotação, fora do texto
```

## 2.3 A evidência

| caminho | o que tem |
|---|---|
| `evidencia/execucoes/execucoes.csv` | **o registro canônico**: 14.193 execuções, uma linha por (dataset, critério, braço, decoder, semente, orçamento) com Fβ, Dice, IoU, MAE, tempos, commit do git e hash de config |
| `evidencia/tabelas/` | **24 tabelas geradas**, cada uma em 5 formatos: `.csv`, `.md` (legível), `.tex` (um `tabular` puro), `.nota.tex` e `.proveniencia.json` |
| `evidencia/bruto/` | CSV e JSON arquivados das campanhas |
| `evidencia/campanhas/` | **os 18 pré-registros**, cada um com hipótese, desfecho primário, falsificador, n planejado, cláusula `se_falhar`, o `RESULTADO` e, quando houve, a `ERRATA`. É a fonte para tudo que a dissertação diz sobre método |
| `evidencia/conjuntos/` | partições |
| `results/baselines/baseline_results.json` | os cinco modelos com retropropagação |

**Comece pelos `.md` de `evidencia/tabelas/`.** São a mesma coisa que os
`.tex`, legíveis, com a nota no fim. Só 7 das 24 estão citadas na
dissertação; as outras 17 são material para expandir os capítulos.

## 2.4 O código

| caminho | o que faz |
|---|---|
| `flim_al/evidencia.py` | o registro: `registrar()`, `carregar()`, `resumo()`. Tem trava entre processos e escrita atômica, porque uma condição de corrida já perdeu 293 registros |
| `flim_al/tabelas.py` | **gera as tabelas**. É aqui que se muda a apresentação |
| `flim_al/region_al.py` | AL por região, features, CoreSet, medoide, faixa de contorno |
| `flim_al/cl_flim.py` | Continual Learning: a interpolação do banco de filtros |
| `flim_al/il_flim.py` | Interactive Learning: simulação de clique e o laço NoC |
| `flim_al/determinismo.py` | fixa threads via `threadpool_limits` |
| `scripts/gerar_tabelas.py` | catálogo das 24 tabelas; regera todas |
| `scripts/conferir_dissertacao.py` | **rode antes de subir ao Overleaf** |
| `flim_app/` | a plataforma localhost |

# 3. Como rodar

```bash
# regerar todas as tabelas a partir do registro
python scripts/gerar_tabelas.py

# conferir o LaTeX antes de subir
python scripts/conferir_dissertacao.py

# testes
python -m pytest
```

`scripts/conferir_dissertacao.py` pega o que revisão humana cansa de não
pegar: citação sem entrada no `.bib`, entrada no `.bib` nunca citada, `\ref`
para label inexistente, sigla usada sem declarar, lista de autores com
vírgula que faz o BibTeX parar, arquivo `.tex` que ninguém inclui, travessão
usado como pontuação, e **arquivo referenciado com a caixa trocada** (o
Overleaf compila em Linux, que diferencia maiúscula de minúscula; o Windows
não).

# 4. O problema a resolver: as tabelas estão ruins

O autor considera a apresentação atual insatisfatória. Diagnóstico concreto:

1. **Largura.** `al_vs_flim` tem 10 colunas, `k_por_modelo` tem 14, `onde_marcar`
   tem 9, `noc_noc` tem 11. Em A4 com margem ABNT isso estoura a caixa de
   texto. Hoje não há `\resizebox`, `\small`, `landscape`, nem quebra em duas.
2. **Regras horizontais.** O gerador emite `tabular` com `\hline` no topo,
   após o cabeçalho e no fim. Padrão editorial de tese é `booktabs`
   (`\toprule`, `\midrule`, `\bottomrule`), sem linha vertical.
3. **Números desalinhados.** As colunas numéricas são `str` em coluna `c` ou
   `l`. Deveriam alinhar no separador decimal (`siunitx`, coluna `S`), e usar
   vírgula decimal, que é o padrão em português. Hoje saem com ponto.
4. **Notas gigantes.** Cada `.nota.tex` é um parágrafo de 100 a 250 palavras
   em `\footnotesize`, colado depois do float. O conteúdo é necessário, as
   ressalvas é que dão sentido aos números, mas o lugar talvez não seja ali.
   Avalie mover a maior parte para o corpo do texto ou para um apêndice de
   protocolo, deixando na nota só o que o leitor precisa para não ler a
   tabela errado.
5. **Linhas vazias.** `comparacao_criterios` tem 41 linhas, e 24 delas são
   `sem pares suficientes` com todas as células vazias. Isso é informação
   real (a comparação é **impossível** com os dados existentes, não ausente
   por descuido), mas não precisa ocupar 24 linhas de tabela.
6. **Falta `\caption`, `\label` e float.** Os `.tex` são `tabular` puro, de
   propósito, para caber em qualquer float. Quem põe `\begin{table}`,
   `\caption` e `\label` é o capítulo. Confira se todos têm legenda ABNT, com
   fonte indicada.
7. **Uma tabela gerada não está citada:** `tabelas/ganho_mecanismo_schisto_FLIM_lm_funcional`
   existe na pasta e nenhum capítulo a inclui.

Restrições de compilação que já quebraram o documento uma vez:

- `ufscar.cls` carrega `[utf8]{inputenc}` com `[T1]{fontenc}`. Isso resolve
  acentuação latina mas **não** letra grega. Um `α`, `β`, `ρ`, `Δ`, `≠`, `−`
  ou `₀` em linha de código LaTeX quebra o `pdflatex`. O gerador já converte
  esses símbolos em `flim_al/tabelas._SIMBOLOS_TEX`; se você acrescentar
  símbolo novo, acrescente a conversão.
- O `.bib` já parou o BibTeX uma vez com `Too many commas in name`: lista de
  autores separada por vírgula em vez de ` and `.
- **Travessão e meia-risca como pontuação não entram no corpo do texto.**
  Use vírgula, ponto, dois-pontos ou parênteses. Hífen comum fica, porque é
  ortografia. O verificador acusa.

Estilo da redação, pedido pelo autor:

- Português natural de estudante de mestrado, não prosa de modelo de
  linguagem. Frases que uma pessoa escreveria.
- Detalhamento suficiente para quem tem **pouca familiaridade técnica**
  acompanhar: explicar o que é um filtro convolucional, como o clique é
  simulado, o que significa cada métrica.
- Fórmulas matemáticas explícitas.
- Baixa similaridade em CopySpider: texto próprio, não paráfrase colada.
- Estrutura ABNT do PPGCC UFSCar.

# 5. Métodos, para você entender o que as tabelas dizem

## 5.1 FLIM

Para cada camada: o especialista marca traços; o método recorta um patch em
volta de cada pixel marcado; roda k-means **em dois níveis** (primeiro por
marcador, depois sobre os centróides resultantes); os centróides finais são
os kernels da convolução. Sem retropropagação, sem função de perda.

**Dois regimes no segundo agrupamento**, e essa distinção explica vários
resultados:

- se o número de candidatos `n` é **menor ou igual** ao número de kernels
  pedido, a função devolve os patches **sem alterar**. O acréscimo é
  **aditivo**: o que já existia continua lá.
- se `n` é **maior**, roda k-means de verdade, e todos os centróides se
  movem. O acréscimo é **deslocante**: anotar mais pode piorar.

Candidatos por vaga, medido: parasitas cerca de 150 para 200 vagas (abaixo do
teto, regime aditivo), BraTS cerca de 250 para 8 vagas, conjuntivite cerca de
750 para 16. Daí vem a previsão de que a intervenção no gerador não pode
funcionar no BraTS: lá os dois braços acrescentam exatamente 24 kernels, e o
deslocamento que a intervenção evita não tem como ocorrer.

**O encoder encolhe com poucos marcadores.** A arquitetura pede 200 kernels
por camada, mas o k-means só produz tantos quantos os patches permitem.
Medido: 54/51/48/48 com uma imagem contra 200/200/200/200 com oito. Com pouca
anotação a rede não é só menos treinada, **é menor**. Qualquer curva de
orçamento mistura supervisão com capacidade, e isso precisa estar dito.

**Decodificador adaptativo:** convolução 1×1 com pesos em {−1, +1},
recalculados por imagem. Pós-processamento: Otsu mais filtro de área.

## 5.2 Métricas

- **Fβ com β² = 0,3**, que pesa precisão mais que revocação. É a métrica
  principal, a mesma do artigo de referência.
- **IoU** (interseção sobre união).
- **Acurácia é `1 − MAE`**, a fração de pixels certos. Passa de 0,97 em quase
  tudo porque o objeto ocupa uma fração mínima da imagem e acertar o fundo já
  garante isso. **Não use acurácia para separar braços**, quem separa é Fβ e
  IoU.
- **NoC@τ**: número de cliques até a IoU passar do alvo, com teto de 12. É o
  eixo de RITM (arXiv:2102.06583) e SimpleClick (arXiv:2210.11006), e **não**
  se compara com o NoC publicado deles, que usa outros conjuntos e outra
  definição de alvo. Imagem que não atinge o alvo entra com o teto, e por isso
  a **taxa de sucesso** tem que aparecer em toda linha: um braço pode mostrar
  NoC baixo por desistir mais cedo.

## 5.3 Simulação de clique: não é centróide

Pergunta que o autor fez explicitamente. O clique **não** vai no centróide da
região de erro. Vai no **máximo da transformada de distância dentro do maior
componente conexo de erro**, pelo protocolo de robot user de Xu et al.
(2016). O motivo é concreto: o centróide de uma região concava pode cair
**fora** dela, e aí o clique rotularia o pixel errado. O máximo da
transformada de distância é, por construção, o ponto mais interior.

O usuário simulado **acerta sempre**, porque o rótulo vem do ground truth.
Então a métrica mede **número de interações, não tempo de especialista**, e
o ganho reportado é um **limite superior**: é o que se obtém quando o clique
é perfeito.

## 5.4 Active Learning: o que foi testado

Critérios de **incerteza**: entropia da predição, least confidence, margin,
BALD. Critérios de **diversidade**: CoreSet (k-center greedy sobre features
do encoder), medoide, BADGE. E `oracle`, que usa o ground truth para escolher
a imagem de pior Fβ, replicando o passo 7 do Algoritmo 1. **Oracle não é
método, é referência**, e o fato de ele também não se destacar é o achado
mais informativo da tabela geral.

Dois níveis, que não se misturam:

- **AL de imagem:** escolhe quais imagens anotar.
- **AL de região:** usa as **mesmas imagens** do sorteio e muda só **onde o
  traço cai**, com a mesma contagem de pixels. O Δ dele mede **posição de
  anotação**, não escolha de imagem.

Comparar os dois Δ como se medissem a mesma coisa é erro. E para técnicas de
região existe um controle de nível (`random_region`): o Δ contra `random`
soma seleção e geometria, e a geometria pesa cerca de 2× a seleção.

**Gerador de candidatos.** A intervenção central do trabalho. Em vez de
superpixels SLIC, os candidatos são discos de raio ρ sobre o **contorno
previsto**, `∂Ŷ = Ŷ \ erosão(Ŷ)`. Os dois braços sorteiam o candidato e
gastam os mesmos cerca de 300 px: a **única** diferença é o gerador da lista.
Por isso a comparação isola o gerador e **não diz nada sobre qual escore
usar**.

## 5.5 Continual Learning sem retropropagação

Não há gradiente para regularizar, então EWC e afins não se aplicam. O
mecanismo é **interpolação do banco de filtros**:

`k̃ = (1 − α) · anc(k) + α · k`

onde `anc(k)` é o filtro antigo mais próximo do novo. A âncora é tomada **por
centróide novo**, e a direção importa: ancorar ao contrário forçaria o
tamanho do banco antigo e quebraria a contabilidade de `noutput_channels` e
da normalização.

`α = 1` é o FLIM puro, que refaz o banco a cada clique. `α = 0` congela o
banco, e o clique chega ao modelo só pelos rótulos que o decodificador
`labeled_marker` lê.

## 5.6 Protocolo estatístico

- **A unidade de análise é a semente.** Os 7 decodificadores de uma semente
  **compartilham o encoder** e não são observações independentes: entram
  colapsados por média. Tratá-los como independentes já inflou um n de 5 para
  10 neste projeto.
- **t pareado bicaudal**, com correção de **Benjamini-Hochberg** quando há
  família de testes. **McNemar exato** para taxa de sucesso.
- **Testes separados por dataset.** Agregar parasitas e BraTS, com 7% e 30%
  de sucesso, inflou um achado exploratório a p = 0,0265 que separado não
  passa de p = 0,10.
- **Pré-registro:** hipótese, desfecho primário, falsificador, n, e "o que
  isto não vai autorizar", declarados **antes** de rodar. Quatro falsos
  positivos foram capturados assim, e eles entram na dissertação como
  resultado de método.
- **O conjunto de teste comum de 364 imagens é tocado uma vez por
  configuração final.** As campanhas de diagnóstico medem em **validação**.

# 6. Os resultados

Tudo abaixo sai de `evidencia/tabelas/`. O nome do arquivo está em cada
cabeçalho. Confira contra o `.md` antes de escrever: este documento é um
resumo, o arquivo é a fonte.

## 6.1 O resultado principal: a margem existe, e é grande

`ganho_marginal_{dataset}_{decoder}_funcional`

Dentro de cada semente a partição, as imagens iniciais, os marcadores base, o
encoder inicial e o conjunto de validação são **idênticos**. A única variável
é **qual região recebeu os cerca de 300 px adicionais**.

Δ Fβ de **melhor região amostrada** menos **candidato sorteado**:

| dataset | decoder | n | Δ vs sorteado | p |
|---|---|---|---|---|
| Parasitas | FLIM_lm | 10 | **+0,3089** | 0,0000 |
| BraTS | FLIM_lm | 7 | **+0,2766** | 0,0026 |
| Conjuntivite | FLIM_lm | 10 | **+0,1997** | 0,0001 |
| Parasitas | FLIM_pb | 10 | **+0,1351** | 0,0000 |
| BraTS | FLIM_pb | 8 | **+0,2384** | 0,0031 |
| Conjuntivite | FLIM_pb | 10 | **+0,1436** | 0,0001 |

**6 de 6 células positivas, todas com p ≤ 0,0031, faixa de +0,135 a +0,309.**
É este o "custo do Active Learning" que o autor quer no centro da
dissertação: o que se perde quando o critério erra o lugar do clique.

Ressalvas que **precisam** acompanhar o número:

- `melhor região` escolhe pelo Fβ da **validação** e **não é uma
  estratégia**. Mede o que existe para capturar, **entre os 22 candidatos
  examinados** de cerca de 380 superpixels. **Não é teto absoluto**, é teto
  amostrado.
- As sementes compartilham o pool e a validação: o IC95% vale para **esta**
  validação sob sorteio das imagens de treino, e **não generaliza para o
  dataset**.
- A separação por regime (`funcional` contra `colapsada`) é post-hoc e
  forçada pela aritmética: Δ a partir de Fβ = 0 exato não pode ser negativo,
  então as duas populações não têm média comparável.
- `melhor região` e `pior região` consultam o ground truth na estratificação.
  Isso tem que estar declarado.

Tabela completa de um dataset, para você ver a forma (Parasitas, FLIM_lm,
n = 10 sementes):

| o que recebeu o marcador | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 0,2191 | [0,0851; 0,3530] | 0,0300 | 0,5777 | +0,3089 | 0,0000 |
| argmax entropia (o AL) | 0,0843 | [−0,0577; 0,2264] | −0,1196 | 0,4676 | +0,1741 | 0,0157 |
| argmax least confidence | 0,1078 | [−0,0267; 0,2423] | −0,1168 | 0,4647 | +0,1976 | 0,0071 |
| candidato sorteado (média) | −0,0898 | [−0,1697; −0,0099] | −0,2672 | 0,1461 | referência | — |
| pior região | −0,3490 | [−0,4764; −0,2216] | −0,6091 | −0,0433 | −0,2592 | 0,0000 |
| imagem nova inteira (orçamento MAIOR) | 0,0506 | [−0,0402; 0,1414] | −0,1427 | 0,3172 | +0,1404 | 0,0002 |
| sorteio uniforme (reponderado, sem GT) | −0,1050 | [−0,1789; −0,0311] | −0,3038 | 0,1034 | −0,0152 | 0,1930 |

Note que **o AL de verdade** (`argmax entropia`, +0,1741) captura parte da
margem, mas fica bem abaixo do teto amostrado (+0,3089). A diferença entre
esses dois é a lacuna que o resto da dissertação investiga.

Os outros cinco arquivos (`brats_lm`, `conjunctiva_lm`, `schisto_pb`,
`brats_pb`, `conjunctiva_pb`) têm a mesma estrutura e estão em
`evidencia/tabelas/`.

## 6.2 Nenhum critério de seleção de imagem supera o sorteio

`geral_al_lm` e `geral_al_pb`

21 linhas (critério × K), colunas por dataset, com Δ médio contra o sorteio,
p pareado e contagem de células vencidas de 9.

| critério | K | Parasitas | BraTS | Conjuntivite | Δ médio vs sorteio | p | vence em |
|---|---|---|---|---|---|---|---|
| CoreSet (imagem) | 2 | 0,543 | 0,000 | 0,642 | −0,061 | 0,5568 | 3 de 9 |
| CoreSet (imagem) | 3 | 0,538 | 0,033 | 0,587 | +0,023 | 0,6074 | 3 de 9 |
| CoreSet (imagem) | 5 | 0,640 | 0,489 | 0,748 | +0,112 | 0,3323 | 7 de 9 |
| CoreSet (imagem) | 8 | 0,690 | 0,230 | 0,636 | −0,067 | 0,4144 | 3 de 9 |
| Entropia (imagem) | 2 | 0,457 | 0,001 | 0,635 | −0,092 | 0,3677 | 3 de 9 |
| Entropia (imagem) | 3 | 0,474 | 0,000 | 0,591 | −0,008 | 0,9104 | 2 de 9 |
| Entropia (imagem) | 5 | 0,599 | 0,498 | 0,667 | +0,074 | 0,5274 | 6 de 9 |
| Entropia (imagem) | 8 | 0,681 | 0,000 | 0,604 | **−0,157** | 0,0427 | 2 de 9 |
| Least confidence | 2 | 0,420 | 0,001 | 0,635 | −0,105 | 0,3115 | 3 de 9 |
| Least confidence | 3 | 0,451 | 0,000 | 0,591 | −0,016 | 0,8178 | 2 de 9 |
| Least confidence | 5 | 0,583 | 0,498 | 0,667 | +0,069 | 0,5531 | 6 de 9 |
| Least confidence | 8 | 0,684 | 0,000 | 0,441 | **−0,210** | 0,0327 | 2 de 9 |
| AL de REGIÃO | 2 | 0,142 | 0,252 | 0,442 | −0,178 | 0,2461 | 1 de 9 |
| AL de REGIÃO | 3 | 0,280 | 0,257 | 0,443 | −0,036 | 0,7332 | 3 de 9 |
| AL de REGIÃO | 5 | 0,039 | 0,499 | 0,261 | −0,247 | 0,1570 | 2 de 9 |
| AL de REGIÃO | 8 | 0,412 | 0,525 | 0,308 | −0,170 | 0,1903 | 2 de 9 |
| Oracle (usa GT) | 2 | 0,093 | 0,196 | 0,274 | −0,269 | 0,0921 | 1 de 9 |
| Oracle (usa GT) | 3 | 0,420 | 0,630 | 0,312 | +0,091 | 0,5347 | 5 de 9 |
| Oracle (usa GT) | 5 | 0,618 | 0,445 | 0,572 | +0,031 | 0,7792 | 6 de 9 |
| Oracle (usa GT) | 8 | 0,518 | 0,008 | 0,535 | **−0,232** | 0,0107 | 0 de 9 |

Leitura: as três diferenças com p < 0,05 são todas **negativas**. O oracle,
que usa o gabarito, perde em 0 de 9 células com K = 8. Isso é o achado mais
informativo: o problema não é o escore ser ruim, é a seleção de imagem não
ser a alavanca.

Complementos: `comparacao_criterios` (41 linhas, todas as técnicas que
rodaram, inclusive as que perderam), `al_vs_flim`, `comparacao_final`,
`dissertacao_lm` e `dissertacao_pb` (72 linhas com Fβ, acurácia, IoU e tempo
por dataset × K × braço), `regiao_vs_imagem` (por decodificador).

## 6.3 A reprodução do artigo, e por que não bate com o publicado

`artigo_vs_regiao`

Mesmas imagens (as que os especialistas A e B anotaram), mesmo número de
cliques, mesmo traço de fundo **copiado e não regerado** nos três braços. Só
muda para onde vão os cliques de objeto. n = 6 pares (usuário, split).

| decoder | Fβ artigo | Fβ AL região | Fβ aleatório | Δ AL−artigo | p | Δ aleat−artigo | p | Δ AL−aleat | p | Fβ publicado (A) |
|---|---|---|---|---|---|---|---|---|---|---|
| FLIM_lm | 0,744 | 0,622 | 0,669 | −0,122 | 0,0370 | −0,075 | 0,0480 | −0,047 | 0,3088 | 0,860 |
| FLIM_pb | 0,720 | 0,494 | 0,537 | −0,225 | 0,0140 | −0,182 | 0,0240 | −0,043 | 0,4223 | 0,857 |
| FLIM_mb | 0,718 | 0,502 | 0,544 | −0,217 | 0,0106 | −0,174 | 0,0121 | −0,042 | 0,3997 | 0,843 |
| FLIM_at | 0,518 | 0,303 | 0,354 | −0,214 | 0,0051 | −0,164 | 0,0015 | −0,050 | 0,1178 | 0,740 |
| FLIM_ts | 0,591 | 0,466 | 0,491 | −0,124 | 0,0031 | −0,099 | 0,0000 | −0,025 | 0,2922 | 0,747 |
| FLIM_lt | 0,649 | 0,549 | 0,575 | −0,100 | 0,0043 | −0,075 | 0,0238 | −0,026 | 0,2467 | 0,810 |
| FLIM_ts* | 0,506 | 0,349 | 0,391 | −0,158 | 0,0029 | −0,115 | 0,0005 | −0,042 | 0,1427 | — |

A decomposição em três diferenças é essencial: `aleat−artigo` isola **o ato
de tirar o clique da borda**, e `AL−aleat` isola **a escolha do Active
Learning**. Sem a terceira, o Δ negativo da primeira seria creditado ao AL
quando pode ser todo da segunda. E de fato: `AL−aleat` não é significativo em
nenhuma linha, enquanto `aleat−artigo` é em todas. **O efeito está na
geometria, não na escolha.**

**A coluna `Fβ publicado (A)` vem da Tabela III do artigo e NÃO é comparável
com a coluna reproduzida.** O artigo aplica Dynamic Trees, cujo binário é um
ELF de Linux e não executa nesta máquina; a avaliação aqui usa Otsu mais
filtro de área. Isso tem que estar dito em toda menção à Tabela III.

Mapeamento dos nomes, que o artigo não explicita:
`decoder_2` = pb, `decoder_3` = mb, `hybrid_decoder` = lt. O
`vanilla_adaptive_decoder_wt` não aparece no artigo.

## 6.4 Para onde os critérios apontam

`ganho_mecanismo_{dataset}_{decoder}_funcional`

ρ de Spearman calculado **dentro** de cada semente, sobre os candidatos
daquela semente; são os ρ por semente que entram no teste, com n = sementes.
Agregar candidatos de sementes diferentes trataria observação correlacionada
como independente.

Parasitas, FLIM_lm, n = 10:

| variável | ρ com Δ Fβ | IC95% | p |
|---|---|---|---|
| entropia da região | 0,427 | [0,292; 0,562] | 0,0001 |
| least confidence | 0,427 | [0,292; 0,562] | 0,0001 |
| fração de foreground | 0,308 | [0,221; 0,395] | 0,0000 |
| Δ kernels do encoder | −0,299 | [−0,429; −0,170] | 0,0005 |
| Δ Fβ médio ao anotar região de **objeto** | −0,0740 | [−0,1661; 0,0181] | — |
| Δ Fβ médio ao anotar região de **fundo** | −0,1077 | [−0,1816; −0,0339] | — |

Dois pontos:

- A entropia **ordena** as regiões por utilidade no FLIM_lm (ρ = 0,427,
  p = 0,0001), mas no FLIM_pb **não** (ρ = −0,045, p = 0,6400). O escore não
  é inútil, é **dependente do decodificador**.
- `Δ kernels` tem ρ **negativo** (−0,299, p = 0,0005): acrescentar mais
  filtros ao encoder **piora**. Isso mede capacidade, não posição, e é o
  mecanismo que o pré-registro da intervenção no gerador usou como predição.
  **Correlação de kernels não é causalidade**, e o texto não deve afirmar o
  contrário.

BraTS: ρ = 0,167 (p = 0,0501), no limite. Conjuntivite: ρ = 0,209
(p = 0,0009). `Δ kernels` sai como **constante** e **não estimável** em BraTS
e conjuntivite, porque lá o banco está travado no teto.

**A hipótese de borda contra interior é hipótese, não medição direta.** Não
há um controle borda contra interior isolado neste conjunto. O que há é o
`regiao_borda` de `onde_marcar`, que usa o gabarito e é teto, não método.

## 6.5 A intervenção que funciona: trocar o gerador de candidatos

`gerador_contorno`

| dataset | n (sementes) | Δ médio | IC95% | vitórias | p | Δ pior caso | p | Δ dispersão | p |
|---|---|---|---|---|---|---|---|---|---|
| Parasitas | 10 | **+0,0772** | [+0,0393; +0,1151] | **10 de 10** | **0,0013** | +0,2153 | 0,0001 | −0,1091 | 0,0000 |
| BraTS | 3 | −0,0089 | [−0,3263; +0,3084] | 1 de 3 | 0,9145 | −0,1063 | 0,2703 | +0,0464 | 0,1376 |
| Conjuntivite | 5 | +0,0452 | [−0,0452; +0,1357] | 4 de 5 | 0,2374 | +0,1759 | 0,0480 | −0,0577 | 0,1492 |

**Leia a análise de sensibilidade antes de escrever esta tabela.** Está em
`RESULTADO/SENSIBILIDADE_2026-10-06` dentro de
`evidencia/campanhas/gerador_de_contorno_2026-10-06.json`.

O desenho declarou 8 candidatos por braço, mas o gerador de contorno entregou
**menos** em três sementes: 3 no Schisto (semente 2), 3 no BraTS (semente 1) e
**1** na conjuntivite (semente 4). O braço de superpixel entregou 8 em todas.
Isso não afeta o desfecho primário, porque a média é estimador não viesado
qualquer que seja n. Mas **afeta os dois secundários**, porque `pior caso` é um
**mínimo**, e o mínimo de 8 sorteios é por construção mais extremo que o mínimo
de 1, mesmo que a distribuição seja a mesma. O viés empurra na direção que a
tabela reporta.

Refazendo com o braço de superpixel subamostrado ao mesmo n (sem reposição,
B = 2000, mesmo t pareado por semente):

| dataset | desfecho | como está na tabela | com n casado | veredito |
|---|---|---|---|---|
| Parasitas | pior caso | +0,2153 (p = 0,0001) | +0,2085 (p = 0,0002) | sobrevive |
| Parasitas | dispersão | −0,1091 (p = 0,0000) | −0,1078 (p = 0,0000) | sobrevive |
| BraTS | pior caso | −0,1063 (p = 0,2703) | −0,1281 (p = 0,2013) | nulo nos dois |
| Conjuntivite | pior caso | +0,1759 (p = **0,0480**) | +0,1369 (p = **0,1032**) | **NÃO sobrevive** |

Consequência para o texto: o resultado do Schisto fica como está, inclusive os
secundários. **O pior caso da conjuntivite não pode ser apresentado como
significativo.** E entra uma limitação do método que ainda não estava escrita:
o gerador de contorno pode entregar **menos** candidatos que o pedido, não só
zero, e qualquer estatística de extremo ou de dispersão sobre os candidatos
precisa casar o n antes de comparar.

Isso é **análise de sensibilidade de um desfecho pré-registrado**, não um
desfecho novo. Rotule assim onde reportar.

### O critério ainda ajuda depois de trocar o gerador (EXPLORATÓRIO)

Campanha `criterio`, Schisto, 10 sementes completas, decodificadores
colapsados por semente. Dados brutos em
`evidencia/bruto/criterio_sobre_contorno_schisto.csv`, agregados em
`evidencia/campanhas/criterio_sobre_contorno_2026-10-06.json`, reproduzíveis
com `python scripts/criterio_sobre_contorno.py`. **Não há pré-registro para
estas comparações**, então isto é exploratório: orienta o próximo passo, não
autoriza afirmação.

| comparação | n | Δ | IC95% | vitórias | p |
|---|---|---|---|---|---|
| critério sobre contorno vs sorteio no contorno | 10 | +0,0262 | [+0,0055; +0,0468] | 9 de 10 | 0,0185 |
| critério sobre superpixel vs sorteio no superpixel | 10 | +0,0987 | [+0,0204; +0,1771] | 8 de 10 | 0,0191 |
| sorteio no contorno vs sorteio no superpixel | 10 | +0,0772 | [+0,0393; +0,1151] | 10 de 10 | 0,0013 |
| **critério sobre contorno vs sorteio no superpixel (efeito total)** | 10 | **+0,1033** | [+0,0657; +0,1409] | **10 de 10** | **0,0002** |

Duas leituras, nessa ordem de importância:

1. **A terceira linha reproduz exatamente o +0,0772 publicado**, com o mesmo
   p e as mesmas 10 vitórias. É verificação interna: o cálculo novo, sobre a
   campanha nova, devolve o número antigo.
2. **Os dois efeitos são praticamente aditivos.** 0,0772 do gerador mais
   0,0262 do critério dá 0,1034, contra 0,1033 medidos no efeito total. Isso
   sugere que gerador e escore operam por caminhos diferentes, o que é
   coerente com o diagnóstico: o gerador resolve *o que está na lista* e o
   escore resolve *qual item da lista*.

Ressalvas que precisam acompanhar: um único dataset; sem pré-registro; os
quatro p não receberam correção para múltiplas comparações (com
Benjamini-Hochberg sobre os quatro, todos ainda passariam, mas isso é
verificação post-hoc). E a segunda linha **não contradiz** o resultado de que
nenhum critério de AL supera o sorteio: aquele é sobre **seleção de imagem com
treino completo, medida no teste**, e este é sobre **uma anotação marginal,
medida na validação**. São perguntas diferentes, e o texto precisa manter a
distinção.

Este é o único resultado positivo robusto do trabalho, e vale escrever com
cuidado:

- Sobrevive a correção de Benjamini-Hochberg.
- **Pior caso e dispersão foram declarados no pré-registro**, com o mecanismo
  escrito **antes** de rodar: o candidato de contorno é pequeno e local,
  acrescenta menos filtros, e o ganho marginal já media correlação negativa
  entre filtros acrescentados e ganho. A predição se confirmou nas três
  colunas.
- **O nulo do BraTS é previsto e confirma o mecanismo.** A arquitetura dele
  fixa o banco em 8 filtros por camada, os dois braços acrescentam
  exatamente 24, e o deslocamento que a intervenção evita não pode ocorrer.
  Um nulo previsto é evidência a favor, não contra.
- Orçamento verificado idêntico: mediana de 300 px nos dois braços.
- A comparação **isola o gerador** e não diz nada sobre qual escore usar.
- Semente cuja base degenerou (Fβ = 0) sai, porque Δ a partir de zero não
  pode ser negativo.

## 6.6 Onde anotar, com orçamento fixo em pixels

`onde_marcar`

24 linhas (braço × px/imagem × decoder), n = 5 sementes cada. Orçamento em
**pixels anotados por imagem**, não em imagens, porque o que limita o FLIM é
a quantidade de pixels marcados.

O braço `uniforme_balanceado` é o controle que separa *onde se anotou* de
*quanto de cada classe se anotou*: escolher região incerta escolhe junto
muito mais foreground (cerca de 73% contra 16%, medido **antes** de rodar).

Resultado resumido: `regiao_incerteza` perde do uniforme em 6 de 6 células,
com p entre 0,019 e 0,264; `regiao_borda` (que usa gabarito, logo teto) fica
dentro do ruído do uniforme. O que **separa** é o balanceamento de classe,
não a posição.

## 6.7 Esforço de interação e Continual Learning

`noc_noc` e `noc_confirmatorio`

| dataset | α | n | NoC médio | NoC mediano | taxa de sucesso | IoU final | Δ NoC vs α=1 | p | McNemar |
|---|---|---|---|---|---|---|---|---|---|
| Parasitas | 0,00 | 38 | 11,03 | 12,0 | 13% | 0,5267 | −0,53 | 0,1031 | 3/0 (p=0,250) |
| Parasitas | 0,50 | 22 | 11,09 | 12,0 | 14% | 0,5490 | −0,55 | 0,2876 | 2/0 (p=0,500) |
| Parasitas | 1,00 | 38 | 11,55 | 12,0 | 5% | 0,5501 | referência | — | — |
| BraTS | 0,00 | 40 | 7,42 | 12,0 | 48% | 0,4096 | −0,35 | 0,5829 | 4/3 (p=1,000) |
| BraTS | 0,50 | 20 | 6,00 | 3,0 | 60% | 0,5273 | +0,05 | 0,7715 | 1/1 (p=1,000) |
| BraTS | 1,00 | 40 | 7,78 | 12,0 | 45% | 0,4064 | referência | — | — |
| Conjuntivite | 0,00 | 1 | 12,00 | 12,0 | 0% | 0,0000 | — | — | 0/1 (p=1,000) |
| Conjuntivite | 0,50 | 1 | 6,00 | 6,0 | 100% | 0,7617 | — | — | — |
| Conjuntivite | 1,00 | 1 | 6,00 | 6,0 | 100% | 0,7764 | referência | — | — |

**Nada aqui é significativo.** Congelar o banco (α = 0) não reduz o número de
cliques de forma detectável. A conjuntivite tem **n = 1** e não autoriza
leitura nenhuma: escreva isso explicitamente, ou corte as três linhas.

Este resultado tem uma história que **entra na dissertação como método**: a
versão agregando parasitas e BraTS dava p = 0,0265 e parecia um achado.
Separado por dataset, não passa de p = 0,10. O `noc_confirmatorio` é a
replicação, e confirma o nulo.

## 6.8 Curva de orçamento, e a resposta ao "por que 3 a 5 imagens?"

`curva_orcamento`

| imagens | Fβ médio | desvio | IC95% | n execuções | sementes |
|---|---|---|---|---|---|
| 1 | 0,602 | 0,084 | [0,580; 0,625] | 54 | 9 |
| 2 | 0,576 | 0,143 | [0,538; 0,615] | 54 | 9 |
| 3 | 0,645 | 0,074 | [0,625; 0,664] | 57 | 9 |
| 4 | 0,670 | 0,077 | [0,650; 0,691] | 54 | 9 |
| 5 | 0,665 | 0,093 | [0,642; 0,688] | 63 | 9 |
| 6 | 0,645 | 0,104 | [0,619; 0,672] | 58 | 9 |
| 7 | 0,680 | 0,080 | [0,659; 0,701] | 54 | 9 |
| 8 | 0,676 | 0,085 | [0,656; 0,697] | 65 | 9 |
| 9 | 0,686 | 0,063 | [0,671; 0,702] | 64 | 9 |
| 10 | 0,599 | 0,111 | [0,554; 0,644] | 26 | 5 |
| 12 | 0,767 | 0,000 | [0,767; 0,767] | 3 | **0** |
| 16 | 0,781 | 0,002 | [0,778; 0,785] | 3 | **0** |
| 20 | 0,772 | 0,001 | [0,771; 0,773] | 3 | **0** |
| 25 | 0,759 | 0,006 | [0,743; 0,775] | 3 | **0** |
| 31 | 0,711 | 0,000 | [0,711; 0,711] | 3 | **0** |

**Cuidado com as cinco últimas linhas:** `sementes = 0` significa que o
registro não guardou semente distinta ali. Três execuções de uma semente só
**não são três repetições**, e o desvio 0,000 é artefato disso, não
estabilidade. Ou marque essas linhas, ou separe-as, ou corte.

A resposta ao orientador: a curva satura entre 3 e 4 imagens, e a partir dali
o ganho cabe dentro do IC. É isso que justifica o "3 a 5 imagens" do artigo.
Mas some a ressalva da §5.1: parte do ganho inicial é **capacidade do
encoder**, não supervisão.

## 6.9 Custo computacional

`k_por_modelo`

36 linhas (critério × K) com Fβ nos sete decodificadores, Δ contra o artigo,
e **tempo de treino e de teste medidos**. Exemplo da ordem de grandeza:
treino de 0,8 s com K = 1 a 48,1 s com K = 8; teste de 83,6 s a 266,5 s.

O treino é o que o FLIM promete ser barato (estimar kernels por k-means), e o
teste é rodar os sete decodificadores no conjunto de avaliação. O teste cresce
com K **porque o encoder cresce**, pelo motivo da §5.1.

Todas as linhas vêm de **uma única campanha, com uma semente** (`n = 1`).
Serve para ordem de grandeza de custo, **não** para comparar critérios.

## 6.10 Modelos de comparação com retropropagação

`results/baselines/baseline_results.json`, 3 splits cada:

| modelo | dataset | Fβ médio | desvio | MAE médio | desvio |
|---|---|---|---|---|---|
| SAMNet | Parasitas | 0,4353 | 0,0051 | 0,00590 | 0,00044 |
| MSCNet | Parasitas | 0,4450 | 0,0090 | 0,00497 | 0,00112 |
| MEANet | Parasitas | 0,4359 | 0,0100 | 0,00612 | 0,00163 |
| UNet | Parasitas | 0,4262 | 0,0108 | 0,00739 | 0,00182 |
| UNetFLIM | Parasitas | 0,3763 | 0,0098 | 0,00995 | 0,00181 |
| SAMNet | BraTS | 0,8226 | 0,0250 | 0,01369 | 0,00069 |
| MSCNet | BraTS | 0,8511 | 0,0046 | 0,01195 | 0,00057 |
| MEANet | BraTS | 0,8049 | 0,0304 | 0,01600 | 0,00095 |
| UNet | BraTS | 0,8208 | 0,0287 | 0,01499 | 0,00114 |
| UNetFLIM | BraTS | 0,7421 | 0,0273 | 0,02891 | 0,00592 |

**Advertência obrigatória:** estes números **não estão no registro
canônico**, e `results/baselines/` **não está versionado no git**. Antes de
entrar na dissertação precisam ser migrados para `evidencia/execucoes/` com
proveniência, ou a tabela tem que declarar que a proveniência é mais fraca
que a do resto. Não apresente numa mesma tabela, sem ressalva, número do
registro e número de fora dele.

## 6.11 Resultados de natureza metodológica

Estes são resultado de pesquisa, não constrangimento. O capítulo 4 já tem uma
seção para eles, e vale expandir.

**Determinismo, medido.** O treino é determinístico **dentro de um processo**,
a partir dos mesmos arquivos de marcador: três repetições dão Fβ idêntico até
a décima casa. **Entre** execuções há ruído: a reexecução completa do schisto
deu Fβ idêntico em **538 de 540** registros, e nos 2 restantes o desvio
máximo foi **0,0034**, duas ordens de grandeza abaixo dos efeitos reportados.
Determinismo dentro do processo **não** é isolamento de posição; são coisas
diferentes e o texto não deve confundi-las.

**Quatro falsos positivos capturados pelo pré-registro:**

| achado aparente | depois | fonte |
|---|---|---|
| densidade na conjuntivite, p = 0,0016 | p = 0,586 | `evidencia/campanhas/confirmatorio2_densidade_2026-09-30.json` |
| dispersão da entropia, p = 0,082 | p = 0,5674 | `evidencia/campanhas/confirmatorio_estabilidade_2026-10-01.json`, campo `RESULTADO/teste_primario` |
| NoC com α = 0, p = 0,0265 | p = 0,1385 ao separar por dataset | `evidencia/campanhas/confirmatorio_retencao_2026-10-05.json` |
| mecanismo de α = 0 | falseado pelo terciário pré-declarado | idem |

Atenção ao segundo: ele próprio teve uma **errata**. A primeira versão
reportava 24 células e p = 0,8765, e as duas coisas estavam erradas (4 das 12
células do teste primário misturavam dois conjuntos de teste, o que infla o
desvio por troca de conjunto e não por efeito do critério). O teste primário
limpo tem **n = 16 células**, Δ desvio = −0,0186, **p = 0,5674**. Os três
cortes limpos dão p = 0,83, 0,24 e 0,5674. **O erro afetou o número, não o
veredito.** Isso está registrado em `RESULTADO/ERRATA_2026-10-05` dentro do
JSON da campanha, e vale a pena escrever na dissertação: uma errata
registrada é argumento de método.

**Quatro hipóteses refutadas**, cada uma custou execução
(`ESTADO_ATUAL.md` §3):

1. "O colapso do encoder vem dos markers pontilhados." Refutada: com imagens
   que têm foreground, o gerador de pontos vai bem (Fβ 0,54 a 0,64).
2. "Dispersão alta da entropia é critério com sinal." **Invertida**: o encoder
   bom (Fβ = 0,744) tem dispersão **baixa** (0,005 a 0,007); alta é sintoma de
   modelo ruim.
3. "Conjunto de treino pequeno causa o colapso." Refutada: com 1 imagem o Fβ
   é 0,54.
4. "Markers realistas corrigem o Experimento A." Refutada: 6 de 6
   configurações pioram (−0,11 a −0,32).

**Defeitos que invalidaram medição anterior:** uma condição de corrida no
registro perdeu **293 registros** (sem trava em `registrar`; corrigido com
lockfile `O_EXCL` e escrita atômica via `os.replace`); `extract_encoder_features`
assumia 3 bandas, então **CoreSet nunca rodou no BraTS** até 2026-09; a
conjuntivite nunca rodou com filtro de área válido. Resultados anteriores a
essas correções não valem.

**Uma versão descartada antes de medir:** o primeiro desenho do gerador de
contorno era `faixa ∩ superpixel`. Um teste sintético mostrou **0% de
candidatos cruzando a borda**, porque as arestas do SLIC seguem justamente as
bordas do objeto. Foi descartado **antes** de qualquer medição, e isso entra
no texto.

**Qualidade da proveniência**, `proveniencia_registro`:

| família | execuções | sem seed | sem commit |
|---|---|---|---|
| ganho_marginal | 2448 | 0% | 0% |
| al_corrected_results | 2316 | 42% | 100% |
| tabela_k_por_modelo | 1840 | 0% | 0% |
| final_comparison_v3 | 1029 | 4% | 100% |
| benchmark_conjunctiva | 588 | 68% | 100% |
| sel_cs_grande | 540 | 0% | 100% |

As famílias com `sem commit = 100%` são anteriores ao registro de
proveniência. Isso **limita** o que se pode afirmar a partir delas, e a
dissertação declara a limitação em vez de esconder.

# 7. Os datasets

| dataset | imagens | resolução | markers | partições |
|---|---|---|---|---|
| Schisto (Parasitas) | 1220 | 400×400 RGB | **31 reais** | 5train-70_30 |
| BraTS | 3753 | 240×240 L | 3753 | 50_50 (4 treino / 1872 val / 1877 teste) |
| Conjunctiva | 83 | 1079×863 RGB | 5 | 57 / 18 / 8 |

Restrições que afetam a leitura de quase toda tabela:

- **Marker real só existe para 31 imagens** do Schisto. AL de imagem sobre o
  pool completo **exige marker sintético**. Por isso o braço do artigo e os
  braços de AL sobre o pool completo **não são comparáveis** diretamente:
  `marker_origem` entra no pareamento justamente para que esse par não se
  forme, porque ele mediria **quem desenhou o traço** e apresentaria como
  efeito da seleção. Onde a coluna aparece vazia, a comparação é
  **impossível** com os dados existentes, não ausente por descuido.
- **49% do pool do Schisto não tem foreground nenhum**, e o critério do artigo
  prioriza justamente essas imagens. Esse é o achado que sobreviveu de toda a
  investigação inicial.
- **Filtro de área [1000, 9000] px** é prior rígido de tamanho: ovo fora dessa
  faixa é insegmentável, independentemente do encoder.
- **Dynamic Trees não executa** (binário ELF de Linux, faltam
  `liblapack`/`libblas`). Avaliação com Otsu mais filtro de área, logo **não
  comparável à Tabela III** do artigo.

# 8. Armadilhas deste repositório

Cada uma já custou tempo ou resultado errado. Estão em `CLAUDE.md` §2.

1. **Existem DUAS cópias do `pyflim`.** `pyflim/` na raiz **não é a que
   roda**; `flim_ad/libs/flim-python/pyflim/` é. Os scripts inserem a segunda
   primeiro no `sys.path`. Patch só na primeira não tem efeito nenhum.
2. **faiss muda os resultados silenciosamente.** `cluster_patches_faiss` usa
   `faiss.Kmeans` quando o import funciona e cai no `sklearn.KMeans` quando
   não. Centróides diferentes, kernels diferentes, Fβ diferente. **Todos os
   resultados da dissertação vieram do fallback do sklearn.** Um
   `pip install -r requirements.txt` ingênuo instala faiss e a partir daí os
   números deixam de ser comparáveis **sem nenhum aviso**. Use
   `requirements-dissertacao.txt`. `tests/test_ambiente_experimental.py` falha
   se faiss estiver presente.
3. **`flim_ad/` é um repositório git aninhado** (clone de
   `LIDS-UNICAMP/flim_ad`). O git daqui não desce nele; alterações locais ali
   são invisíveis ao histórico. Por isso existe
   `vendor/flim_ad/local-changes.patch`, que precisa ser regerado ao mexer lá.
4. **`flim_ad/out/` tem 230 mil arquivos e é ignorado.** A evidência tabular é
   arquivada em `evidencia/bruto/` por `scripts/arquivar_resultados.sh`, e é
   essa cópia que entra no git.
5. **`requirements.txt` não é deste projeto**, é o da biblioteca upstream:
   incompleto (falta `monai`, `transformers`, `scipy`) e pede faiss.
6. **A aplicação interativa não produz evidência científica.** `flim_app/` é
   demonstração. Os números que ela mostra vêm de execução única com K
   pequeno, onde a amplitude do braço aleatório (0,10 a 0,22) é maior que
   qualquer efeito observável. **Nunca cite número da aplicação na
   dissertação.** Capturas de tela da interface, sim; números dela, não.
7. Os scripts de experimento assumem **cwd = `flim_ad/`**. Rodar da raiz
   quebra os caminhos relativos.
8. **Antes de apagar qualquer coisa**, registre em `vendor/REMOVIDOS.txt` o
   que era e por quê.

# 9. Uma campanha está rodando agora. Não use.

A campanha `argmax_ent_contorno` (critério aplicado sobre candidatos de
contorno) está em execução e tem **4 de 10 sementes** do Schisto. **Não
escreva número dela.**

Isto não é cautela genérica: já aconteceu três vezes neste projeto que um
resultado parcial foi relatado como final. O "empate" apareceu e sumiu
conforme as execuções subiram de 1 para 3 e para 9; o CoreSet mostrou 55% com
7 de 9 sementes e fechou em 16%. Se precisar mencionar antes do fim, diga
explicitamente **quantas de quantas** rodaram.

# 10. O que entregar

1. **Diagnóstico das tabelas**, uma por uma, dizendo o que está ruim e o que
   você propõe. Antes de mexer.
2. **A mudança feita no gerador** (`flim_al/tabelas.py`) e as tabelas
   regeradas com `python scripts/gerar_tabelas.py`, para que a próxima
   geração não desfaça. Se alguma mudança só fizer sentido no capítulo (float,
   `\caption`, `landscape`), faça no capítulo e diga qual é qual.
3. **Decisão sobre as notas:** o que fica na nota, o que vai para o corpo, o
   que vai para apêndice. Nenhuma ressalva pode simplesmente desaparecer.
4. **As 17 tabelas ainda não citadas**, avaliadas: quais entram em qual
   capítulo e por quê. O texto precisa crescer para passar de 120 páginas, e
   elas são o material.
5. **`python scripts/conferir_dissertacao.py` passando**, e
   `python -m pytest` passando.
6. **Uma lista do que você NÃO fez** e por quê. Escopo reduzido é decisão do
   autor, não sua.

Se em qualquer ponto a única forma de fechar uma tabela for inventar,
estimar, ou alterar um número, **pare e diga**. É sempre a resposta certa
aqui.
