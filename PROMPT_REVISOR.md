# Prompt para outro chat: revisar e reescrever o texto da dissertação

Copie tudo a partir da linha `---` e cole no outro chat, com o repositório
aberto nele.

Para a **apresentação das tabelas** existe um prompt separado,
`PROMPT_TABELAS.md`. Este aqui é sobre **o texto**.

---

Você é revisor de uma dissertação de mestrado em LaTeX. O trabalho
experimental está concluído e registrado; os números existem e não se
discutem. Sua tarefa é **ler a dissertação inteira e reescrever o texto**
para que ele diga, com precisão e na voz certa, aquilo que os dados
sustentam.

Comece lendo. Não reescreva nada antes de ter lido os cinco capítulos e os
dois documentos de contexto listados na seção 3.

# 1. A regra que não cede

**Nenhum número pode ser inventado, estimado, arredondado para ficar redondo,
ou digitado de memória.** Todo valor vem de arquivo de resultado. Se um
número do texto não bater com o arquivo, o arquivo está certo e o texto está
errado: corrija o texto. Se não houver arquivo para um número, diga isso em
vez de deixá-lo passar.

**Toda referência precisa de DOI, arXiv ID ou URL verificável.** Sem
verificação, escreva `UNKNOWN`. Referência fabricada reprova a defesa.

Quatro níveis que não devem colapsar um no outro:

| nível | exemplo |
|---|---|
| dado bruto | Fβ de cada semente, em `evidencia/` |
| observação | "a média subiu de 0,688 para 0,697" |
| resultado agregado | "média ± desvio, IC95%, t pareado p = 0,33" |
| interpretação | "o CoreSet parece mais eficiente em amostra" |
| afirmação | "sob o dataset D e orçamento B, o CoreSet melhorou a eficiência de amostra em relação ao sorteio" |

"Melhorou" e "melhorou significativamente" são frases diferentes. Sem teste
pareado e sem n, a segunda não se escreve. **`p` alto significa não decidido,
nunca igual**: ausência de evidência não é evidência de equivalência.

Se em algum ponto a única forma de fechar um parágrafo for inventar,
arredondar ou suavizar, pare e diga. É sempre a resposta certa aqui.

# 2. A contribuição que deve ficar no centro, e como enunciá-la

O autor quer que a dissertação seja organizada em torno de **quanto vale o
clique no lugar certo**. Esse é o número, e ele está medido:

> Com orçamento de anotação fixo em pixels, nas mesmas imagens e com a mesma
> base, **anotar a melhor região examinada em vez de uma região sorteada vale
> de +0,135 a +0,309 de Fβ**, em 6 de 6 combinações de conjunto de dados e
> decodificador, todas com p ≤ 0,0031.

Fonte: `evidencia/campanhas/custo_do_al_2026-10-06.json` e as seis tabelas
`evidencia/tabelas/ganho_marginal_*_funcional.md`.

| conjunto | decodificador | n | Δ vs sorteio | p |
|---|---|---|---|---|
| Parasitas | FLIM_lm | 10 | **+0,3089** | 0,0000 |
| BraTS | FLIM_lm | 7 | **+0,2766** | 0,0026 |
| Conjuntivite | FLIM_lm | 10 | **+0,1997** | 0,0001 |
| Parasitas | FLIM_pb | 10 | **+0,1351** | 0,0000 |
| BraTS | FLIM_pb | 8 | **+0,2384** | 0,0031 |
| Conjuntivite | FLIM_pb | 10 | **+0,1436** | 0,0001 |

## 2.1 A frase que NÃO pode ser escrita

**"O Active Learning melhorou o Fβ em 0,1 a 0,3" é falso**, e escrever isso
derruba o trabalho na banca em uma pergunta.

O que vale de 0,135 a 0,309 é o **teto amostrado**: a melhor região entre as
examinadas, escolhida **pelo Fβ da validação**. Isso usa o resultado para
escolher, logo **não é estratégia e não é método**. Mede a margem que
**existe**.

O que o Active Learning de fato entrega é bem menos, e a diferença entre as
duas coisas **é o resultado**. Chame isso do **custo do Active Learning**: o
que se deixa na mesa porque o critério erra o lugar do clique.

| conjunto | decoder | teto (clique perfeito) | argmax entropia (o AL) | p do AL | **lacuna** | p da lacuna | captura |
|---|---|---|---|---|---|---|---|
| Parasitas | FLIM_lm | +0,3089 | +0,1741 | 0,0157 | **+0,1347** | 0,0230 | 56% |
| Parasitas | FLIM_pb | +0,1351 | +0,0096 | 0,7313 | **+0,1255** | 0,0006 | 7% |
| BraTS | FLIM_lm | +0,2766 | +0,0884 | 0,2847 | **+0,1882** | 0,1233 | 32% |
| BraTS | FLIM_pb | +0,2384 | −0,0283 | 0,7698 | **+0,2667** | 0,0495 | −12% |
| Conjuntivite | FLIM_lm | +0,1997 | +0,1121 | 0,0042 | **+0,0876** | 0,0038 | 56% |
| Conjuntivite | FLIM_pb | +0,1436 | +0,0598 | 0,0762 | **+0,0839** | 0,0117 | 42% |

**A lacuna vai de +0,0839 a +0,2667 e é significativa em 4 das 6 células.** É
esse o número que organiza a dissertação.

As três faixas que o texto afirma em prosa estão no campo
`FAIXAS_PARA_O_TEXTO` do mesmo JSON, computadas sobre as 6 células, para que
você não precise montar min e max à mão:

| faixa | valor | células |
|---|---|---|
| margem do clique perfeito | 0,1351 a 0,3089 | 6 |
| custo de uma anotação mal posicionada | 0,2075 a 0,4954 | 6 |
| lacuna do argmax entropia | 0,0839 a 0,2667 | 6 |

`python scripts/custo_do_al.py` regera o JSON do registro canônico. O script
reproduz as seis tabelas `ganho_marginal_*` célula por célula, inclusive os
`p`, o que serve de verificação interna: se ele divergir delas, uma das duas
está errada.

Honestidade obrigatória, porque a banca vai olhar: **há uma célula em que um
critério vai bem.** `argmax least confidence` no BraTS com FLIM_lm captura 84%
do teto (+0,2320, p = 0,0012). Isso tem de aparecer no texto, não ser
omitido. Enfraquece a generalização e fortalece a credibilidade, e a leitura
correta é que o desempenho do escore **depende do decodificador e do
domínio**, não que o escore seja inútil.

Cuidado com a coluna `captura`: é uma razão entre duas médias e fica
instável, ou absurda, quando o denominador é pequeno. Use-a como ilustração,
nunca como estatística. Prefira reportar a lacuna absoluta, que tem teste.

## 2.2 A formulação defensável, em uma frase

> Sob orçamento de anotação fixo, existe uma margem de 0,135 a 0,309 de Fβ
> entre a melhor região examinada e uma região sorteada. Os critérios de
> Active Learning capturam parte dela, e a parte não capturada, de 0,084 a
> 0,267, é o custo de o critério escolher o lugar errado. O gargalo não está
> no escore, e sim na lista de candidatos que ele recebe.

Note o encadeamento: **a margem existe**, logo o problema não é o FLIM ser
insensível à anotação; **o critério não a captura**, logo o problema é de
seleção; **a lista é pobre**, logo a intervenção correta é no gerador de
candidatos, e não no escore. É essa cadeia que transforma um resultado
negativo em contribuição.

## 2.3 As ressalvas que precisam acompanhar o número, sempre

Nenhuma delas é opcional. Omitir qualquer uma torna a afirmação indefensável:

- **Teto amostrado, não absoluto.** A escolha é entre os **22 candidatos
  examinados** de cerca de 380 superpixels.
- **Mede-se na validação, não no teste.** O conjunto de teste comum de 364
  imagens é tocado uma vez por configuração final, e esta campanha não o
  tocou.
- **As sementes compartilham pool e validação.** O IC95% vale para **esta**
  validação sob sorteio das imagens de treino e **não generaliza para o
  dataset**.
- **O teto e a pior região consultam o ground truth** na estratificação. Isso
  tem de estar declarado.
- **A estratificação por regime da base é post-hoc** e forçada pela
  aritmética: Δ a partir de Fβ = 0 exato não pode ser negativo, então as duas
  populações não têm média comparável. Todas as células acima são de **base
  funcional**.
- **O clique simulado acerta sempre**, porque o rótulo vem do ground truth.
  Logo o número é **limite superior**: é o que se obtém quando o clique é
  perfeito, e mede **número de interações, não tempo de especialista**.

# 3. Leia estes arquivos, nesta ordem

| arquivo | por quê |
|---|---|
| `CLAUDE.md` | as regras de integridade e as armadilhas do repositório. **Obrigatório.** |
| `ESTADO_ATUAL.md` | fonte de verdade sobre o estado **científico**: o estabelecido, o refutado, as lacunas declaradas |
| `DISSERTACAO/capitulos/1_introducao.tex` | **o maior problema do texto está aqui.** Ver a seção 5.1 |
| `DISSERTACAO/capitulos/2_fundamentos.tex` | o mais longo, cerca de 11,8 mil palavras, parte herdada da qualificação |
| `DISSERTACAO/capitulos/3_metodologia.tex` | reescrito para o FLIM, com as fórmulas |
| `DISSERTACAO/capitulos/4_resultados.tex` | as medições |
| `DISSERTACAO/capitulos/5_conclusao.tex` | **o melhor capítulo hoje.** Já tem o enquadramento certo. Use a voz dele como referência |
| `DISSERTACAO/LEIAME.md` | como a pasta está organizada |

# 4. O projeto LaTeX

```
DISSERTACAO/
  main.tex              documento principal: capa, resumo, sumário, ordem dos capítulos
  ufscar.cls            classe do PPGCC UFSCar, NÃO alterar
  capitulos/            1_introducao .. 5_conclusao
  pretextual/abreviaturas.tex     siglas, usadas por \listasiglas e \ac{}
  referencias/referencias.bib     75 entradas
  tabelas/              GERADO por scripts/gerar_tabelas.py. Não editar à mão
  figuras/              9 em uso; `nao_usadas/` guarda 56 da qualificação
  notas/Reuniao.tex     anotação, fora do texto
```

Tamanho atual, por `wc -w`:

| capítulo | palavras |
|---|---|
| 1 introdução | 2186 |
| 2 fundamentos | 11772 |
| 3 metodologia | 3762 |
| 4 resultados | 3085 |
| 5 conclusão | 1607 |
| **total** | **22412** |

**A meta é passar de 120 páginas.** O capítulo 2 tem metade do texto e o
capítulo 4, que carrega os resultados, tem um quarto do tamanho dele. A
proporção está invertida para uma dissertação cuja contribuição é
experimental.

## 4.1 Restrições de compilação, que já quebraram o documento

- `ufscar.cls` carrega `[utf8]{inputenc}` com `[T1]{fontenc}`: resolve
  acentuação latina mas **não letra grega**. Um `α`, `β`, `ρ`, `Δ`, `≠`, `−`
  ou `₀` solto em linha de LaTeX quebra o `pdflatex`. Em modo matemático
  (`$\beta$`, `$\Delta$`) funciona. O gerador de tabelas já converte esses
  símbolos; se você introduzir símbolo novo, use modo matemático.
- O `.bib` já parou o BibTeX com `Too many commas in name`: lista de autores
  separada por vírgula em vez de ` and `.
- **Travessão e meia-risca como pontuação não entram no corpo do texto.** Use
  vírgula, ponto, dois-pontos ou parênteses. Hífen comum fica, porque é
  ortografia.
- O Overleaf compila em Linux, que diferencia maiúscula de minúscula. Uma
  figura `Oracle_Superpixel.PNG` citada como `.png` compila no Windows e falha
  lá.

Rode isto antes de considerar qualquer trecho pronto:

```bash
python scripts/conferir_dissertacao.py
```

Ele pega citação sem entrada no `.bib`, entrada nunca citada, `\ref` para
label inexistente, sigla usada sem declarar, lista de autores que o BibTeX
recusa, arquivo `.tex` que ninguém inclui, travessão como pontuação, e
arquivo referenciado com a caixa trocada.

## 4.2 Estilo pedido pelo autor

- **Português natural de estudante de mestrado**, não prosa de modelo de
  linguagem. Frases que uma pessoa escreveria. Evite "é importante ressaltar
  que", "vale destacar", "no presente trabalho buscou-se", e a enumeração
  mecânica de "primeiramente, em segundo lugar, por fim".
- **Detalhamento para quem tem pouca familiaridade técnica.** O leitor deve
  sair entendendo o que é um filtro convolucional, como o clique é simulado,
  o que cada métrica mede e por que a acurácia engana aqui.
- **Fórmulas explícitas**, numeradas e referenciadas.
- **Baixa similaridade em CopySpider**: texto próprio, não paráfrase colada.
- Estrutura ABNT do PPGCC UFSCar.
- O capítulo 5 já está nessa voz. Leia-o primeiro e escreva como ele.

# 5. O que precisa ser consertado no texto

## 5.1 A introdução é de outra dissertação. Este é o item 1.

`capitulos/1_introducao.tex` ficou da qualificação, quando o trabalho usava
DeepLabV3-ResNet50 com Destilação de Conhecimento sobre o PascalVOC2012. Ela
não descreve o trabalho que os capítulos 3, 4 e 5 relatam. Especificamente:

- O **objetivo geral** promete "um framework interativo para segmentação
  semântica capaz de reduzir o custo e o tempo de anotação manual", integrando
  AL, CL, superpixels **e Aprendizado por Reforço**. **Não há Aprendizado por
  Reforço no trabalho.**
- **H1** afirma que o framework "reduzirá significativamente o tempo e o custo
  de anotação, preservando a qualidade". **Nada no trabalho mede tempo de
  anotação nem custo em horas de especialista.** A métrica de esforço é NoC,
  número de cliques, com usuário simulado que acerta sempre. Esta hipótese,
  como está escrita, não é sustentada por nenhum resultado.
- **Q2 e H2** falam de AL "aliado ao RL". Não existe.
- **Q4 e H4** falam de Destilação de Conhecimento contra esquecimento
  catastrófico. O trabalho mostra que o esquecimento no FLIM tem forma própria,
  **sem gradiente**, e que os três grupos de métodos consolidados, Destilação
  entre eles, **não se aplicam**. A hipótese está invertida em relação ao
  achado.
- **Q3** fala de imagens 4K multiclasse. Os três conjuntos usados são
  400×400 RGB, 240×240 em tom de cinza e 1079×863 RGB, todos **binários**.

A introdução precisa ser reescrita de ponta a ponta, derivada do capítulo 5,
que já tem a pergunta e as respostas corretas. Objetivos e hipóteses têm de
ser os que o trabalho de fato testou. **Uma hipótese que o trabalho não testou
não pode constar**, e uma que foi refutada deve constar **como refutada**,
porque hipótese refutada é resultado de pesquisa e este trabalho tem quatro
delas.

Em `Contribuições Esperadas`, substitua expectativa por contribuição medida,
na ordem da seção 2 deste prompt.

## 5.2 Dois números do texto estão defasados. Confira todos.

Encontrados por conferência mecânica contra a evidência:

| onde | o texto diz | a evidência diz |
|---|---|---|
| `4_resultados.tex:51` e `5_conclusao.tex:11` | margem "entre 0,133 e 0,303" | **0,1351 a 0,3089** |
| `4_resultados.tex:51` | pior região "entre 0,199 e 0,502" abaixo do sorteio | **0,2075 a 0,4954** |

São de uma versão anterior das campanhas, de quando havia menos sementes.
Ambos aparecem em mais de um lugar, então corrija em todos.

**Isso é um alerta, não uma lista completa.** Trate cada número do texto como
suspeito até conferi-lo contra `evidencia/tabelas/*.md` ou
`evidencia/campanhas/*.json`. Em particular, confira estes, que eu não
rastreei até a fonte:

- "valor entre 0,126 e 0,166 de Fβ" para regiões que atravessam a fronteira,
  com marcadores reais (`5_conclusao.tex`)
- "divergência de até 0,134 nos pesos" do codificador
- "elevou o Fβ mediano de 0,024 para 0,594" na correção do filtro de área
- "leva um codificador de Fβ 0,78 para zero" (`4_resultados.tex:51`)

Onde o número puder ser lido de uma tabela gerada, **cite a tabela em vez de
repetir o número no corpo**. É a regra do repositório: número copiado à mão
para o texto é número que vai se desatualizar.

## 5.3 O capítulo 2 está desproporcional e parte dele é herança

11.772 palavras, metade da dissertação, com seções de Aprendizado de Máquina,
Segmentação Semântica e Classificação dos Algoritmos que vêm da qualificação
e não servem ao argumento atual. As seções de FLIM, Interactive Learning e
Continual Learning, que servem, são as mais curtas.

Reequilibre: encurte o que é livro-texto, expanda o que sustenta o método.
Não apague sem registrar: o repositório exige anotar em
`vendor/REMOVIDOS.txt` o que saiu e por quê.

## 5.4 O capítulo 4 precisa crescer, e há material

3.085 palavras para a parte que carrega a contribuição. **17 das 24 tabelas
geradas não estão citadas em nenhum capítulo.** Estão em `evidencia/tabelas/`,
em `.md` legível, e a seção 6 deste prompt resume o que cada uma diz.

## 5.5 Coisas que o texto precisa deixar de confundir

Cada uma destas confusões já esteve no texto ou está:

- **Determinismo dentro do processo não é isolamento de posição.** O treino é
  determinístico **dentro de um processo**; entre execuções há ruído medido de
  até 0,0034. São afirmações diferentes.
- **AL de imagem e AL de região medem coisas diferentes.** O primeiro escolhe
  quais imagens anotar; o segundo usa as mesmas imagens e muda só **onde o
  traço cai**, com a mesma contagem de pixels. Comparar os dois Δ como se
  fossem a mesma grandeza é erro.
- **Fβ reproduzido e Fβ publicado não se comparam.** O artigo aplica Dynamic
  Trees, cujo binário é um ELF de Linux e não executa nesta máquina; aqui a
  avaliação usa Otsu mais filtro de área. Toda menção à Tabela III precisa
  dizer isso.
- **Acurácia é `1 − MAE`** e passa de 0,97 em quase tudo porque o objeto
  ocupa fração mínima da imagem. Não use acurácia para separar braços.
- **Correlação de kernels não é causalidade.** `Δ kernels` tem ρ negativo com
  o ganho, o que é compatível com o mecanismo proposto, mas não o demonstra.
- **Borda contra interior é hipótese**, não medição direta. Não existe neste
  conjunto um controle isolado de borda contra interior.
- **O nulo do BraTS é previsto**, e previsto **antes** de rodar: a arquitetura
  fixa o banco em 8 filtros por camada, os dois braços acrescentam exatamente
  24, e o deslocamento que a intervenção evita não pode ocorrer. Um nulo
  previsto é evidência a favor do mecanismo, e o texto deve apresentá-lo
  assim, não como fracasso parcial.
- **Números da aplicação `flim_app/` nunca entram na dissertação.** Ela é
  demonstração, com execução única e K pequeno, onde a amplitude do braço
  aleatório (0,10 a 0,22) é maior que qualquer efeito observável. Capturas de
  tela, sim; números, não.

# 6. Os resultados, para você saber o que existe

Todos saem de `evidencia/`. O nome do arquivo está em cada cabeçalho. **Leia o
arquivo antes de escrever o parágrafo**: o que está aqui é resumo.

## 6.1 Nenhum critério de seleção de IMAGEM supera o sorteio

`geral_al_lm.md`, `geral_al_pb.md`, `comparacao_criterios.md`,
`dissertacao_lm.md`, `dissertacao_pb.md`, `regiao_vs_imagem.md`,
`comparacao_final.md`, `al_vs_flim.md`.

Em 3 conjuntos × 4 orçamentos (K ∈ {2,3,5,8}) × 7 decodificadores, com
pré-registro e correção de Benjamini-Hochberg: 96 testes, **0 sobrevivem a
BH**. Das 20 linhas de `geral_al_lm`, as três com p < 0,05 são todas
**negativas**. O **oracle**, que lê o gabarito e representa o teto do
procedimento do artigo, perde do sorteio em **0 de 9 células** com K = 8
(Δ = −0,232, p = 0,0107).

A formulação correta é "não há evidência de superioridade nas condições
avaliadas", e o texto deve dizer explicitamente que isso não é prova de
equivalência.

## 6.2 A margem existe: a contribuição central

`ganho_marginal_{schisto,brats,conjunctiva}_{FLIM_lm,FLIM_pb}_funcional.md` e
`custo_do_al_2026-10-06.json`. Ver a seção 2 deste prompt, que é onde está o
número.

Tabela completa de uma célula, para você ver a forma (Parasitas, FLIM_lm,
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

Duas linhas merecem parágrafo próprio no texto:

- **`pior região`**: uma anotação adicional **mal posicionada** custa de
  0,2075 a 0,4954 de Fβ, nas 6 células, faixa também em `FAIXAS_PARA_O_TEXTO`.
  O risco é da mesma ordem do prêmio, e isso é parte do argumento: no FLIM a
  anotação **não é monotonicamente boa**, e é esse fato que dá sentido a
  escolher onde anotar.
- **`imagem nova inteira`** gasta **mais** orçamento que uma região e rende
  menos que a melhor região. Sustenta a mudança de granularidade, de imagem
  para região.

## 6.3 Para onde os critérios apontam, e por que erram

`ganho_mecanismo_*.md`

ρ de Spearman calculado **dentro** de cada semente, com n = sementes.
Parasitas, FLIM_lm:

| variável | ρ com Δ Fβ | IC95% | p |
|---|---|---|---|
| entropia da região | 0,427 | [0,292; 0,562] | 0,0001 |
| least confidence | 0,427 | [0,292; 0,562] | 0,0001 |
| fração de foreground | 0,308 | [0,221; 0,395] | 0,0000 |
| Δ kernels do encoder | −0,299 | [−0,429; −0,170] | 0,0005 |

Mas no FLIM_pb o mesmo escore dá ρ = −0,045 (p = 0,6400). **O escore não é
inútil, é dependente do decodificador**, e o texto tem de dizer isso em vez de
generalizar. BraTS: ρ = 0,167 (p = 0,0501), no limite. Conjuntivite:
ρ = 0,209 (p = 0,0009). `Δ kernels` sai **constante e não estimável** no BraTS
e na conjuntivite, porque lá o banco está travado no teto.

## 6.4 O gargalo é a lista, não o escore

Só 2% a 12% dos superpixels SLIC atravessam a fronteira do objeto, porque o
SLIC é construído para que suas bordas coincidam com as bordas da imagem. Como
um critério apenas **ordena a lista que recebe**, nenhuma escolha de escore
resolve isso. É o elo que leva à intervenção da seção 6.5.

Ver também `onde_marcar.md`: com orçamento fixo em pixels, o que separa os
braços é o **balanceamento de classe**, não a posição. Escolher região incerta
escolhe junto muito mais foreground, cerca de 73% contra 16%, medido **antes**
de rodar, e o braço `uniforme_balanceado` existe para separar as duas coisas.

## 6.5 A intervenção que funciona

`gerador_contorno.md` e `evidencia/campanhas/gerador_de_contorno_2026-10-06.json`

Trocar superpixels por discos de raio ρ centrados no **contorno previsto**,
`∂Ŷ = Ŷ \ erosão(Ŷ)`, com escore e orçamento fixos:

| conjunto | n | Δ médio | IC95% | vitórias | p |
|---|---|---|---|---|---|
| Parasitas | 10 | **+0,0772** | [+0,0393; +0,1151] | **10 de 10** | **0,0013** |
| BraTS | 3 | −0,0089 | [−0,3263; +0,3084] | 1 de 3 | 0,9145 |
| Conjuntivite | 5 | +0,0452 | [−0,0452; +0,1357] | 4 de 5 | 0,2374 |

Sobrevive a Benjamini-Hochberg. Pior caso e dispersão foram **declarados no
pré-registro com o mecanismo escrito antes de rodar**.

**Leia `RESULTADO/SENSIBILIDADE_2026-10-06` no JSON da campanha antes de
escrever sobre os secundários.** O gerador de contorno entregou **menos**
candidatos que os 8 pedidos em três sementes (3 no Schisto, 3 no BraTS, 1 na
conjuntivite). A média não se afeta, mas `pior caso` é um **mínimo**, e o
mínimo de 8 sorteios é por construção mais extremo que o mínimo de 1. Casando
o n:

| conjunto | desfecho | na tabela | com n casado | veredito |
|---|---|---|---|---|
| Parasitas | pior caso | +0,2153 (p = 0,0001) | +0,2085 (p = 0,0002) | sobrevive |
| Parasitas | dispersão | −0,1091 (p = 0,0000) | −0,1078 (p = 0,0000) | sobrevive |
| Conjuntivite | pior caso | +0,1759 (p = 0,0480) | +0,1369 (p = 0,1032) | **não sobrevive** |

**O pior caso da conjuntivite não pode ser apresentado como significativo.** E
entra uma limitação do método: o gerador pode entregar **menos** candidatos
que o pedido, não só zero.

### E o critério ainda ajuda depois de trocar o gerador (EXPLORATÓRIO)

`evidencia/campanhas/criterio_sobre_contorno_2026-10-06.json`, Schisto, 10
sementes. **Sem pré-registro**, logo exploratório:

| comparação | Δ | vitórias | p |
|---|---|---|---|
| critério sobre contorno vs sorteio no contorno | +0,0262 | 9 de 10 | 0,0185 |
| sorteio no contorno vs sorteio no superpixel | +0,0772 | 10 de 10 | 0,0013 |
| **efeito total** | **+0,1033** | **10 de 10** | **0,0002** |

Os dois efeitos são praticamente aditivos: 0,0772 + 0,0262 = 0,1034 contra
0,1033 medidos. Coerente com o diagnóstico, porque o gerador resolve **o que
está na lista** e o escore resolve **qual item da lista**. A linha do meio
reproduz exatamente o +0,0772 publicado, o que é verificação interna.

## 6.6 Esforço de interação e aprendizado contínuo

`noc_noc.md`, `noc_confirmatorio.md`

**Nada aqui é significativo.** Congelar o banco de filtros (α = 0) não reduz o
número de cliques de forma detectável. A conjuntivite tem **n = 1** e não
autoriza leitura nenhuma.

O mecanismo de retenção é `k̃ = (1 − α) · anc(k) + α · k`, com a âncora tomada
**por centróide novo**; a direção importa, porque ancorar ao contrário forçaria
o tamanho do banco antigo e quebraria a contabilidade de `noutput_channels`.

Este resultado tem uma história que **entra no texto como método**: a versão
agregando Parasitas e BraTS dava p = 0,0265 e parecia achado; separado por
conjunto, não passa de p = 0,10. O `noc_confirmatorio` é a replicação e
confirma o nulo.

## 6.7 A reprodução do artigo

`artigo_vs_regiao.md`. Mesmas imagens, mesmo número de cliques, mesmo traço de
fundo copiado nos três braços. n = 6 pares (usuário, split).

A decomposição em três diferenças é o ponto: `aleat−artigo` isola **tirar o
clique da borda** e `AL−aleat` isola **a escolha do Active Learning**.
`AL−aleat` não é significativo em nenhuma das 7 linhas, enquanto
`aleat−artigo` é em todas. **O dano vem da geometria, não da escolha.** Sem a
terceira diferença, o Δ da primeira seria creditado ao AL.

Mapeamento dos nomes, que o artigo não explicita: `decoder_2` é o pb,
`decoder_3` é o mb, `hybrid_decoder` é o lt. O `vanilla_adaptive_decoder_wt`
não aparece no artigo.

## 6.8 Curva de orçamento, e o "por que 3 a 5 imagens?"

`curva_orcamento.md`. A curva satura entre 3 e 4 imagens e o ganho posterior
cabe no IC. **Cuidado com as cinco últimas linhas** (12, 16, 20, 25, 31
imagens): `sementes = 0` significa que o registro não guardou semente distinta
ali, e três execuções de uma semente só **não são três repetições**. O desvio
0,000 é artefato disso.

Ressalva que o texto precisa fazer: **parte do ganho inicial é capacidade do
encoder, não supervisão.** A arquitetura pede 200 kernels por camada, mas o
k-means só produz tantos quantos os patches permitem. Medido: 54/51/48/48 com
uma imagem contra 200/200/200/200 com oito. Com pouca anotação a rede não é só
menos treinada, **é menor**.

## 6.9 Custo computacional e modelos de comparação

`k_por_modelo.md` traz tempo de treino e de teste medidos: treino de 0,8 s com
K = 1 a 48,1 s com K = 8; teste de 83,6 s a 266,5 s. Todas as linhas vêm de
**uma única campanha com uma semente** (n = 1): serve para ordem de grandeza,
não para comparar critérios.

`results/baselines/baseline_results.json`, 3 splits cada:

| modelo | Parasitas | BraTS |
|---|---|---|
| MSCNet | 0,4450 | 0,8511 |
| MEANet | 0,4359 | 0,8049 |
| SAMNet | 0,4353 | 0,8226 |
| UNet | 0,4262 | 0,8208 |
| UNetFLIM | 0,3763 | 0,7421 |

**Advertência obrigatória:** estes números **não estão no registro canônico** e
`results/baselines/` **não está versionado**. A proveniência é mais fraca que
a do resto, e isso tem de estar dito onde eles aparecerem. Não os apresente na
mesma tabela que números do registro sem ressalva.

## 6.10 Resultados de natureza metodológica

São resultado de pesquisa, não constrangimento, e o capítulo 4 já tem seção
para eles. Valem expansão.

**Determinismo, medido.** Determinístico **dentro** de um processo: três
repetições dão Fβ idêntico até a décima casa. **Entre** execuções há ruído: a
reexecução completa do Schisto deu Fβ idêntico em **538 de 540** registros, e
nos 2 restantes o desvio máximo foi **0,0034**, duas ordens de grandeza abaixo
dos efeitos reportados.

**Quatro falsos positivos capturados pelo pré-registro:**

| achado aparente | depois | fonte |
|---|---|---|
| densidade na conjuntivite, p = 0,0016 | p = 0,586 | `confirmatorio2_densidade_2026-09-30.json` |
| dispersão da entropia, p = 0,082 | p = 0,5674 | `confirmatorio_estabilidade_2026-10-01.json` |
| NoC com α = 0, p = 0,0265 | p = 0,1385 | `confirmatorio_retencao_2026-10-05.json` |
| mecanismo de α = 0 | falseado pelo terciário pré-declarado | idem |

O segundo teve **errata própria**: a primeira versão reportava 24 células e
p = 0,8765, e as duas coisas estavam erradas, porque 4 das 12 células
misturavam dois conjuntos de teste. O limpo tem n = 16, p = 0,5674. **Uma
errata registrada é argumento de método** e vale um parágrafo.

**Quatro hipóteses refutadas** (`ESTADO_ATUAL.md` §3), cada uma custou
execução:

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
registro perdeu **293 registros**; `extract_encoder_features` assumia 3 bandas,
então **o CoreSet nunca rodou no BraTS** até 2026-09; a conjuntivite nunca
rodou com filtro de área válido. Resultado anterior a essas correções não
vale.

**Uma versão descartada antes de medir:** o primeiro desenho do gerador de
contorno era `faixa ∩ superpixel`, e um teste sintético mostrou **0% de
candidatos cruzando a borda**, porque as arestas do SLIC seguem as bordas do
objeto. Descartado **antes** de qualquer medição. Isso entra no texto.

**Qualidade da proveniência** (`proveniencia_registro.md`): `ganho_marginal` e
`tabela_k_por_modelo` têm 0% de execuções sem seed e sem commit;
`al_corrected_results` tem 42% sem seed e 100% sem commit;
`benchmark_conjunctiva`, 68% e 100%. As famílias sem commit são anteriores ao
registro de proveniência, e isso **limita** o que se pode afirmar a partir
delas. A dissertação declara a limitação.

# 7. Os conjuntos de dados e suas restrições

| conjunto | imagens | resolução | markers | partições |
|---|---|---|---|---|
| Schisto (Parasitas) | 1220 | 400×400 RGB | **31 reais** | 5train-70_30 |
| BraTS | 3753 | 240×240 L | 3753 | 50_50 (4 treino / 1872 val / 1877 teste) |
| Conjunctiva | 83 | 1079×863 RGB | 5 | 57 / 18 / 8 |

Restrições que afetam a leitura de quase toda tabela:

- **Marker real só para 31 imagens** do Schisto. AL de imagem sobre o pool
  completo **exige marker sintético**. Por isso o braço do artigo e os braços
  de AL sobre o pool completo **não são comparáveis** diretamente:
  `marker_origem` entra no pareamento para que esse par não se forme, porque
  mediria **quem desenhou o traço** e apresentaria como efeito da seleção.
  Coluna vazia significa comparação **impossível** com os dados existentes,
  não ausente por descuido.
- **49% do pool do Schisto não tem foreground nenhum**, e o critério do artigo
  prioriza justamente essas imagens. É o achado que sobreviveu da investigação
  inicial.
- **Filtro de área [1000, 9000] px** é prior rígido de tamanho.
- **Dynamic Trees não executa** nesta máquina.

# 8. Armadilhas do repositório (`CLAUDE.md` §2)

1. **Duas cópias do `pyflim`.** A da raiz **não é a que roda**;
   `flim_ad/libs/flim-python/pyflim/` é.
2. **faiss muda os resultados silenciosamente.** Todos os resultados vieram do
   fallback do sklearn. Use `requirements-dissertacao.txt`, não
   `requirements.txt`, que é o da biblioteca upstream e pede faiss.
3. **`flim_ad/` é um repositório git aninhado.** Alterações ali são invisíveis
   ao histórico; por isso existe `vendor/flim_ad/local-changes.patch`.
4. **`flim_ad/out/` tem 230 mil arquivos e é ignorado.**
5. **`flim_app/` não produz evidência científica.**
6. Scripts de experimento assumem **cwd = `flim_ad/`**.
7. **Antes de apagar qualquer coisa**, registre em `vendor/REMOVIDOS.txt`.

# 9. O que entregar

1. **Um diagnóstico do texto antes de reescrever**, capítulo por capítulo,
   dizendo o que está errado, o que está defasado e o que falta. Inclua a
   lista completa dos números que você conferiu e dos que não bateram.
2. **A introdução reescrita**, com objetivos, questões e hipóteses que
   correspondam ao trabalho realizado, derivados do capítulo 5.
3. **O capítulo 4 expandido**, com a contribuição da seção 2 deste prompt no
   centro e as tabelas ainda não citadas incorporadas onde fizerem sentido.
4. **O capítulo 2 reequilibrado**, com registro em `vendor/REMOVIDOS.txt` do
   que sair.
5. **`python scripts/conferir_dissertacao.py` passando.**
6. **Uma lista do que você NÃO fez** e por quê. Reduzir escopo é decisão do
   autor, não sua.

Não invente número em nenhuma circunstância. Se faltar dado para fechar um
parágrafo, deixe o parágrafo aberto e diga o que falta.
