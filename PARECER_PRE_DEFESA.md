# Parecer técnico pré-defesa

Dissertação: *Aplicação de aprendizado ativo, contínuo e interativo ao FLIM:
limites, custo e uma intervenção no gerador de candidatos*

> Título alterado em 2026-10-09, depois da elaboração deste parecer. A versão
> anterior era *Quanto vale o clique no lugar certo: limites do aprendizado
> ativo e contínuo em redes FLIM e o gargalo na geração de candidatos*. A
> análise do parecer não depende do título.

Elaborado em 2026-10-07 por verificação direta do texto contra o código, o
registro de execuções, as campanhas versionadas, as tabelas geradas e os
testes automatizados. Nenhum arquivo foi alterado na elaboração deste parecer.

---

## 1. Veredito em cinco frases

O trabalho é **defensável após correções**, e o parecer anterior é justo:
o núcleo experimental é mais forte que a consolidação teórica, e três
afirmações estão formuladas acima do que a evidência sustenta. A condição para
a defesa é corrigir os doze itens P0 da Seção 6, dos quais dois impedem a
compilação e quatro são sobreafirmações verificadas. A principal força é a
cadeia causal medida elo a elo, encerrada por uma intervenção com ganho
replicado em 10 de 10 sementes e com nulo previsto antes da execução. A
principal fragilidade é a distância entre o que algumas frases afirmam e o que
as campanhas efetivamente cobrem, com destaque para um desenho "3 × 4 × 7" que
não existe no registro. O grau de novidade é moderado e real: não há, na
literatura consultada, caracterização do porquê de o aprendizado ativo não
transferir para redes construídas por agrupamento, nem intervenção no gerador
de candidatos derivada desse diagnóstico.

---

## 2. Matriz das observações

| # | Observação da revisão anterior | Classificação | Evidência | Impacto | Correção |
|---|---|---|---|---|---|
| 1a | Capítulo 2 tem "Pricipais" no título | **Confirmada** | `capitulos/2_fundamentos.tex:1` | Baixo, mas é a primeira linha do capítulo | Corrigir ortografia |
| 1b | Cap. 2 afirma adotar destilação de conhecimento | **Confirmada** | `2_fundamentos.tex:339`: "opta-se prioritariamente pela terceira abordagem, fundamentada em destilação de conhecimento" | **Alto**: contradiz os cap. 3, 4 e 5, que mostram que destilação **não se aplica** a modelo sem gradiente | Reescrever o parágrafo |
| 1c | Discussão excessiva de RL | **Parcialmente confirmada** | Bloco de 13 linhas com MDP já foi comprimido em `f0cbc3c`; restam menções em trabalhos relacionados | Médio | Já mitigado; revisar remanescentes |
| 1d | Foco excessivo em VLM | **Confirmada** | 54 ocorrências de `\ac{VLM}` no cap. 2 | Médio | Condensar |
| 1e | Lacuna final do cap. 2 é sobre VLM | **Confirmada** | `2_fundamentos.tex:911`: lacuna "aplicado a \ac{VLM}" | **Alto**: o capítulo fecha apontando uma lacuna que não é a do trabalho | Reescrever a lacuna em termos de FLIM |
| 2 | Sobreafirmação de desenho 3 × 4 × 7 | **Confirmada** | Cruzamento do registro: os 5 decodificadores extras existem **só em schisto, com 1 semente**; BraTS e conjuntivite têm apenas `FLIM_lm` e `FLIM_pb`. A tabela `geral_al_lm` vem de 3 datasets × 4 orçamentos × **1 decodificador** × **3 sementes** | **Alto**: induz leitura de fatorial completo | Reescrever resumo, cap. 4 e cap. 5 |
| 3 | "oracle" chamado de teto | **Confirmada** | `geral_al_lm`: oracle com K=8 dá Δ = −0,232, p = 0,0107, vence em **0 de 9** células. Um procedimento que perde do sorteio de forma significativa não é teto | **Alto** | Trocar por "baseline supervisionado com informação privilegiada" |
| 4 | Inferência causal sobre o gerador | **Parcialmente confirmada** | A intervenção é experimental e isola o gerador (escore e orçamento fixos), o que é mais forte que associação. Mas não isola **tamanho** de **posição**: o próprio cap. 4 reconhece que os discos são menores e acrescentam menos filtros, e a correlação entre filtros acrescentados e ganho é negativa (ρ = −0,299, p = 0,0005) | **Médio-alto** | Moderar "o gargalo está na lista e não no escore" |
| 5 | Fronteira predita vs verdadeira | **Confirmada** | `gerador_de_contorno_2026-10-06.json`, `limitacao_confirmada_do_metodo`: fração de objeto dos candidatos foi 0,997 no schisto e 0,779 no BraTS, isto é, caem no **interior**. O cap. 3 diz "atravessam a fronteira por construção" sem qualificar na primeira menção | **Médio** | Qualificar toda ocorrência |
| 6 | BADGE não é BADGE | **Confirmada** | `coreset_badge.py:255`: `uncertainty = pred_scores - 0.5`. Em P = 0,5 (incerteza máxima) o vetor é **nulo**; em P = 0 ou 1 (confiança máxima) é máximo. O k-means++ sorteia proporcional ao quadrado da distância, logo favorece o **confiante**. O BADGE original usa `(p − e_ŷ) ⊗ h`, cuja magnitude é `|p − round(p)|`, máxima em P = 0,5 | **Alto para o rótulo, baixo para as conclusões**: BADGE aparece em **uma linha de uma tabela** (`comparacao_criterios`) e em **nenhum** resultado central | Renomear ou corrigir e reexecutar só aquela linha |
| 7 | Escores não são probabilidades calibradas | **Confirmada** | `region_al.py:389,482,756,832`: `Image.open(sal_path).convert("L") / 255.0`. É mapa de saliência normalizado e quantizado em 8 bits, não saída de sigmoide calibrada. O cap. 3 os chama de "probabilidade predita" | **Médio**: não invalida comparações (todos os braços usam o mesmo escore), mas muda o que "entropia" significa | Trocar por "pseudo-probabilidade" ou "escore normalizado" |
| 8 | n variável e teste t com poucas repetições | **Confirmada** | n por célula vai de 1 (conjuntivite em `noc_noc`) a 10. `geral_al_lm` tem 3 sementes; `ganho_marginal` tem 7 a 10; `gerador_contorno` tem 3 (BraTS), 5 (conjuntivite) e 10 (schisto) | **Alto** | Declarar n por campanha; acrescentar teste de sinais e permutação |
| 9 | Inconsistência de Benjamini-Hochberg | **Confirmada** | `custo_do_al.tex` destaca **5** células em negrito (p bruto < 0,05, incluindo p = 0,0495), e o texto afirma **4**. Recomputado: com BH sobre as 6, sobrevivem exatamente 4 (o maior k com p(k) ≤ k·α/m é k = 4) | **Alto**: tabela e texto discordam na mesma página | Destacar por q, ou acrescentar coluna de q |
| 10 | Monotonicidade | **Confirmada, já corrigida** | A curva tem quedas em 2, 5, 6 e 10 imagens. Corrigido em `1581a4b`, com a ressalva de que o intervalo de 9 e o de 10 são **disjuntos** e a diferença é de composição de amostra (26 execuções sobre 5 sementes contra 64 sobre 9) | Resolvido | Nenhuma |
| 11 | Valores 0,126 e 0,166 | **Confirmada, já corrigida** | Não vinham da tabela citada. A coluna `aleat−artigo` de `artigo_vs_regiao` vai de 0,075 a 0,182. Corrigido em `c7fe9dd` | Resolvido | Nenhuma |
| 12 | Direção do ancoramento no CL | **Não confirmada** | `cl_flim.py::_ancorar`: "Para cada centróide NOVO, o anterior mais próximo". `3_metodologia.tex:345`: "A âncora é tomada por centróide *novo*". **Código e texto concordam**, e o docstring explica por que a direção oposta quebra | Nenhum | Nenhuma |
| 13 | Teste "tocado uma única vez" | **Confirmada** | `3_metodologia.tex:464` afirma isso, mas `geral_al_lm` e `dissertacao_lm` declaram "medidos no mesmo conjunto de **teste**", com 5 critérios × 4 orçamentos × 3 datasets avaliados nele | **Alto**: a frase não é literalmente correta | Reescrever o protocolo |
| 14 | Natureza do pré-registro | **Não confirmada** | `3_metodologia.tex` diz "O documento é versionado junto com o código", sem alegar registro externo. 17 campanhas versionadas com timestamp em git | Baixo | Opcional: usar "pré-especificação versionada" |
| 15a | `\fbeta` indefinido | **Confirmada** | `3_metodologia.tex:109`. Não há `\newcommand` em `main.tex` nem em `ufscar.cls` | **Crítico: a compilação falha** | Definir ou substituir por `$F_\beta$` |
| 15b | Percentual escapado errado | **Confirmada** | `custo_do_al.tex`: `56\textbackslash{}\%`. Causa: `tabela_custo_do_al` emite `\%` e `_tex_escape` converte a barra. Também em `artigo_vs_regiao.nota.tex` | **Alto**: renderiza literalmente "56\%" | Emitir `%` puro e deixar o escape agir |
| 15c | Tabelas largas | **Confirmada** | `artigo_vs_regiao` 12 colunas, `k_por_modelo` 14, `comparacao_criterios` 10 com 41 linhas | **Alto**: estouram a caixa ABNT | `adjustbox`/`landscape`/`\small`, ou quebra em duas |

---

## 3. Mapa dos experimentos

Reconstruído do registro canônico (`evidencia/execucoes/execucoes.csv`, 14.193
execuções) e da proveniência de cada tabela.

| Campanha | Datasets | Orçamentos | Decodificadores | Critérios | Sementes | Teste | Natureza |
|---|---|---|---|---|---|---|---|
| `tabela_k_por_modelo` → `geral_al_lm` | 3 | 2, 3, 5, 8 | **1** (`FLIM_lm`) | random, coreset, entropy, LC, região, oracle | **3** | t pareado + BH | Confirmatória |
| `tabela_k_por_modelo` → `geral_al_pb` | 3 | 2, 3, 5, 8 | **1** (`FLIM_pb`) | idem | **3** | t pareado + BH | Confirmatória |
| `tabela_k_por_modelo` → `k_por_modelo` | 3 | 1 a 12 | **7, só em schisto** | 6 | **1** por célula de decodificador extra | nenhum | Descritiva (custo) |
| `ganho_marginal` | 3 | fixo (1 anotação) | 2 | teto, argmax entropia, argmax LC, argmax prob, sorteado, pior, imagem nova | 7 a 10 | t pareado | Confirmatória |
| `ganho_marginal` / contorno → `gerador_contorno` | 3 | ~300 px | 2 (colapsados) | sorteado vs contorno sorteado | **10 / 3 / 5** | t pareado + BH | Confirmatória |
| `criterio` → `criterio_sobre_contorno` | 1 (schisto) | ~300 px | 2 (colapsados) | argmax entropia sobre contorno | 10 | t pareado | **Exploratória, sem pré-registro** |
| `artigo_vs_regiao` | 1 (schisto) | fixo | 7 | artigo, AL região, aleatório | 6 pares (usuário × split) | t pareado | Confirmatória |
| `onde_marcar` | 1 (schisto) | 2000, 4000, 8000 px | 2 | incerteza, aleatória, borda, uniforme, balanceado | 5 | t pareado | Confirmatória |
| `cl_plasticidade` | 3 | — | 2 | α ∈ {0, 0,5, 1} | variável | t pareado | Confirmatória |
| `il_noc` → `noc_noc` | 3 | teto 12 cliques | 1 | α ∈ {0, 0,5, 1} | **38 / 40 / 1** imagens | t pareado + McNemar | Confirmatória |
| `noc_confirmatorio` | 2 | teto 12 | 1 | α ∈ {0, 1} | 18 / 20 | t pareado | Confirmatória (replicação) |
| `comparacao_criterios` | variável | variável | variável | todas, inclusive BADGE | 95 a 471 pares | t pareado | Descritiva agregada |
| baselines | 2 | treino completo | — | SAMNet, MSCNet, MEANet, UNet, UNetFLIM | 3 splits | nenhum | **Fora do registro canônico** |

**A tabela acima é, por si, uma correção P1 que deve entrar no texto.**

---

## 4. Conclusões permitidas

As nove conclusões propostas, avaliadas uma a uma:

1. **"O procedimento do artigo é um baseline supervisionado com informação
   privilegiada."** — **Pode permanecer, e deve substituir "teto".** Sustentada
   por `soares2026flim` (os autores declaram a abordagem supervisionada) e pelo
   registro (o procedimento perde do sorteio em K = 8).

2. **"Nos regimes testados, os critérios não superaram consistentemente o
   sorteio na seleção de imagens após correção."** — **Pode permanecer.**
   Qualificar: o regime é 3 datasets × 4 orçamentos × 2 decodificadores × 3
   sementes, não 3 × 4 × 7.

3. **"A posição da anotação apresenta grande valor potencial e grande risco,
   condicionados ao conjunto de candidatos examinado."** — **Pode permanecer.**
   É a formulação mais precisa das que o trabalho oferece: +0,1351 a +0,3089 de
   prêmio, −0,2075 a −0,4954 de risco, 6 de 6 células.

4. **"Os superpixels disponibilizam poucas regiões que atravessam a fronteira
   verdadeira."** — **Pode permanecer.** 2,5 %, 2,7 % e 12,4 % sobre 3768, 3964
   e 4381 regiões.

5. **"A geração de candidatos é um componente importante e potencialmente
   limitante."** — **Pode permanecer, e deve substituir "o gargalo está na
   lista e não no escore".**

6. **"Candidatos centrados no contorno predito melhoraram significativamente o
   resultado em Schisto."** — **Pode permanecer.** Qualificar: *predito*, não
   verdadeiro.

7. **"O ganho em Schisto ainda não demonstra generalização."** — **Pode
   permanecer.** Deve aparecer com mais destaque do que hoje.

8. **"Os resultados de CL e IL delimitam abordagens que não funcionaram nas
   condições avaliadas."** — **Pode permanecer.** Nunca escrever "equivalente".

9. **"O mecanismo completo ainda necessita de controle borda/interior, regiões
   de tamanho comparável, replicação independente, mais datasets, avaliação com
   especialistas e análise de calibração."** — **Pode permanecer, e deve ser
   promovida de "trabalhos futuros" para o corpo da discussão.**

---

## 5. Conclusões que devem ser moderadas ou removidas

| Onde | Afirmação atual | Problema | Substituir por |
|---|---|---|---|
| Resumo, cap. 5 | "três conjuntos, quatro orçamentos e sete decodificadores" | Fatorial inexistente | "três conjuntos de dados e quatro orçamentos de anotação, com dois decodificadores; os sete decodificadores foram avaliados em análise complementar, com uma repetição, apenas no conjunto de parasitas" |
| Cap. 3, 4, 5 | "oracle", "teto do procedimento" | Perde do sorteio com p = 0,0107 | "baseline supervisionado com informação privilegiada, que reproduz o Algoritmo 1" |
| Cap. 4 síntese | "o gargalo está na lista e não no escore" | Não isola tamanho de posição | "a geração de candidatos contribui materialmente e constitui alvo promissor de intervenção, embora o mecanismo causal completo ainda não esteja isolado" |
| Cap. 3 | "atravessam a fronteira por construção" | Fronteira *predita* | "atravessam o contorno **predito** por construção, mas não necessariamente a fronteira verdadeira" |
| Cap. 3 | "a probabilidade predita" | Não calibrada | "o escore de saliência normalizado, tratado como pseudo-probabilidade" |
| Cap. 3 | "tocado uma única vez" | Falso para a campanha de seleção de imagens | Ver Seção 7 |
| Cap. 2 | "opta-se prioritariamente pela destilação" | Contradiz cap. 3 a 5 | Ver Seção 7 |
| Cap. 2 | lacuna "aplicado a VLM" | Fora do escopo | Ver Seção 7 |
| Cap. 3 | "BADGE" | Implementação invertida | "híbrido inspirado em BADGE" + nota de divergência |

---

## 6. Correções

### P0 — obrigatórias antes da defesa

1. **Definir ou eliminar `\fbeta`** (`3_metodologia.tex:109`). *A compilação
   falha sem isso.*
2. **Corrigir o escape de percentual** em `tabela_custo_do_al` e em
   `artigo_vs_regiao.nota.tex`, e regerar.
3. **Corrigir o layout das tabelas largas** (item 26 do pedido):
   `artigo_vs_regiao` (12 colunas), `k_por_modelo` (14), `comparacao_criterios`
   (10 × 41 linhas), `noc_noc` e `gerador_contorno` (10).
4. **Remover a implicação de fatorial 3 × 4 × 7** do resumo, do cap. 4 e do
   cap. 5.
5. **Substituir "oracle como teto"** por terminologia precisa, nos quatro
   lugares onde aparece.
6. **Moderar "o gargalo está na lista e não no escore"**.
7. **Qualificar "fronteira predita" vs "fronteira verdadeira"** em todas as
   ocorrências.
8. **Renomear o BADGE** para "híbrido inspirado em BADGE", declarar a
   divergência, e decidir se a linha de `comparacao_criterios` é reexecutada.
9. **Declarar o n real por campanha** (a tabela da Seção 3 deste parecer).
10. **Acrescentar teste de sinais e permutação** nas células com n ≤ 5.
11. **Corrigir a inconsistência de BH** na tabela de custo: destacar por q, ou
    acrescentar coluna de q, ou mudar a legenda.
12. **Reescrever o parágrafo de destilação e a lacuna final do cap. 2.**
13. **Corrigir "Pricipais"** no título do cap. 2.
14. **Reescrever a frase do conjunto de teste.**

### P1 — importantes

- Incluir a matriz de campanhas da Seção 3 no cap. 3 ou num apêndice.
- Separar visualmente, no cap. 4, o que é confirmatório do que é exploratório.
- Análise não paramétrica de sensibilidade para os resultados principais.
- Apresentar a reprodução do artigo com tabela própria, hoje ausente.
- Declarar que os mapas de saliência são pseudo-probabilidades.
- Chamar o procedimento de "pré-especificação versionada no repositório".
- Seção de ameaças à validade (interna, externa, de construto).
- Explicitar qual conclusão depende de qual campanha.

### P2 — apresentação

- Reduzir a tabela de 16 datasets gerais do cap. 2 (Cityscapes, COCO, LVIS,
  GTA V não servem ao argumento).
- Uniformizar nomes de critérios e métricas.
- Remover texto comentado do cap. 2 (há blocos extensos).
- Figura única com o encadeamento causal.
- Revisão ortográfica.
- Respostas curtas preparadas para a banca.

---

## 7. Sugestões de redação

### 7.1 Desenho experimental (resumo e cap. 5)

> Os experimentos abrangeram três conjuntos de dados e quatro orçamentos de
> anotação, com dois decodificadores e três repetições por célula. Os sete
> decodificadores do trabalho de referência foram avaliados em análise
> complementar, com uma repetição por configuração e apenas no conjunto de
> parasitas, de modo que essa parte é descritiva e não sustenta inferência.

### 7.2 Terminologia do procedimento do artigo (cap. 3 e 4)

> O Algoritmo 1 do trabalho de referência é reproduzido aqui como **baseline
> supervisionado com informação privilegiada**: ele consulta a anotação de
> referência de todo o conjunto disponível para escolher o que anotar. Não se
> trata de um limite superior de desempenho. A medição mostra, aliás, que ele
> pode perder para o sorteio aleatório: com oito imagens, fica 0,232 de
> $F_\beta$ abaixo, com valor p de 0,0107 e desvantagem em todas as nove
> células. Ter acesso ao gabarito não é suficiente para escolher bem quando o
> critério de escolha não corresponde ao que o modelo precisa.

### 7.3 Força da conclusão sobre o gerador (cap. 4, síntese)

> A geração de candidatos contribui materialmente para o desempenho e
> constitui um alvo promissor de intervenção. A comparação é experimental e
> isola o gerador, porque escore e orçamento são mantidos fixos, mas ela não
> isola a **geometria** do **tamanho**: os discos de contorno são menores que
> os superpixels e acrescentam menos filtros ao banco, e o diagnóstico de ganho
> marginal mede correlação negativa entre filtros acrescentados e ganho
> ($\rho = -0{,}299$, valor p de 0,0005). O mecanismo causal completo, portanto,
> ainda não foi isolado, e o experimento que o isolaria está descrito no
> Capítulo \ref{cap_5}.

### 7.4 Contorno predito (cap. 3)

> Os discos atravessam o contorno **predito** por construção, mas não
> necessariamente a fronteira verdadeira do objeto. Quando o modelo
> subsegmenta, o contorno predito situa-se por dentro do real e os discos caem
> no interior: medido, a fração de objeto dos candidatos foi de 0,997 no
> conjunto de parasitas e 0,779 no BraTS. O ganho relatado no Capítulo
> \ref{cap_4} ocorre apesar disso, o que é parte da razão para não atribuí-lo
> integralmente à posição.

### 7.5 Escores (cap. 3)

> Seja $S(x,y) \in [0,1]$ o escore de saliência normalizado produzido pelo
> decodificador, obtido por normalização linear do mapa de ativação e
> quantizado em oito bits. Esse valor **não é uma probabilidade calibrada**:
> não provém de uma função logística ajustada nem foi submetido a calibração.
> As funções de aquisição abaixo são aplicadas a ele por analogia com a
> formulação probabilística, e os termos "entropia" e "least confidence" devem
> ser lidos nesse sentido. Como todos os braços comparados usam o mesmo escore,
> a comparação entre eles permanece válida; o que não se sustenta é a leitura
> da entropia como medida de incerteza epistêmica calibrada.

### 7.6 Conjunto de teste (cap. 3)

> O conjunto de teste não participou de nenhuma decisão de método: nenhum
> hiperparâmetro, critério ou limiar foi escolhido com base nele. As campanhas
> de diagnóstico, que são as que sustentam os resultados centrais deste
> trabalho, medem exclusivamente em validação. A campanha de seleção de
> imagens, por outro lado, reporta desempenho no conjunto de teste para todos
> os braços pré-especificados, o que corresponde a 5 critérios × 4 orçamentos ×
> 3 conjuntos de dados. Trata-se de reporte de braços declarados antes da
> execução, e não de busca sobre o teste, mas a expressão "tocado uma única
> vez" seria imprecisa e não é usada aqui.

### 7.7 Capítulo 2, parágrafo de aprendizado contínuo

> Os três grupos de métodos descritos acima compartilham um pressuposto:
> existe um otimizador que desloca pesos, e é sobre esse deslocamento que eles
> atuam. A regularização penaliza a mudança de pesos importantes, o ensaio
> reapresenta dados antigos ao otimizador, e a destilação faz o modelo aluno
> imitar as saídas do professor. **Nenhuma dessas operações está definida para
> o \ac{FLIM}**, que não possui perda nem otimizador. Este trabalho não adota,
> portanto, nenhum dos três. O mecanismo investigado no Capítulo \ref{cap_3}
> atua sobre o objeto que de fato se desloca no \ac{FLIM}, que é o banco de
> filtros reestimado por agrupamento a cada nova anotação.

### 7.8 Capítulo 2, lacuna final

> A literatura de aprendizado ativo foi construída quase inteiramente sobre
> modelos treinados por gradiente, nos quais a anotação adicional refina
> parâmetros existentes de forma incremental. Não há, nos trabalhos
> levantados, avaliação dessas estratégias sobre uma rede cujos filtros são
> estimados por agrupamento dos próprios dados anotados, regime em que a
> anotação adicional **reestima** o extrator em vez de refiná-lo, e em que o
> banco é compartilhado por todas as imagens. Essa é a lacuna que esta
> dissertação endereça: caracterizar se, e por quê, os critérios de seleção
> transferem para esse regime, e o que a resposta implica para a forma de
> propor anotação ao especialista.

---

## 8. Perguntas da banca

**1. Qual é exatamente a contribuição original?**
A caracterização medida do porquê de os critérios de \ac{AL} não transferirem
para o \ac{FLIM}, e a intervenção no gerador de candidatos que esse diagnóstico
indica. *Limitação a admitir:* a intervenção é significativa em um dos três
conjuntos. *Evitar:* "propomos um novo método de aprendizado ativo".

**2. Por que chamar o procedimento do artigo de oracle?**
Não se deve. Ele é um baseline supervisionado com informação privilegiada.
*Evidência:* perde do sorteio em K = 8 com p = 0,0107.

**3. Acesso ao ground truth transforma uma heurística em teto?**
Não. Teto exige maximizar o desfecho; essa heurística maximiza outra coisa (o
pior $F_\beta$), e pode escolher imagens insegmentáveis.

**4. O que foi realmente reproduzido do artigo?**
O arranjo codificador + decodificador adaptativo, a curva de orçamento e a
saturação precoce. *Admitir:* sem Dynamic Trees, os valores não se comparam aos
publicados. *Fragilidade atual:* falta tabela explícita de reprodução.

**5. Como saber que o resultado negativo não é bug?**
Três evidências: o baseline supervisionado também não se destaca, o que não se
explica por erro nos critérios; os critérios demonstram comportamento
estruturado (localizam o objeto com enriquecimento de uma a duas ordens de
grandeza); e defeitos reais foram encontrados e corrigidos, com os resultados
anteriores descartados. *Admitir:* o BADGE implementado está invertido.

**6. Se o BADGE foi adaptado de forma não padrão, o que continua válido?**
Tudo exceto a linha de `comparacao_criterios`. BADGE não participa de nenhum
resultado central.

**7. Por que entropia sobre um mapa não calibrado?**
Porque é o escore que o método disponibiliza. *Admitir:* a leitura
informação-teórica não se sustenta; a comparação entre braços, sim.

**8. O que é "custo do aprendizado ativo"?**
A diferença pareada entre o melhor candidato examinado e o que o critério
escolhe. É útil porque separa "não há o que ganhar" de "há e não se captura".

**9. O máximo entre candidatos é limite superior?**
Não. É máximo amostrado sobre os candidatos examinados.

**10. Como evitar viés ao escolher o melhor pela validação?**
Não se evita: por isso ele não é método. O viés é declarado e a linha é rotulada
como referência.

**11. Se só 2 % a 12 % cruzam a fronteira, por que um bom escore não os acha?**
Em princípio poderia, e essa é a objeção correta. O que a medição mostra é que
os escores testados não o fazem, e que o resultado exploratório sobre a lista
corrigida sugere que escore e gerador agem por caminhos distintos. *Evitar:*
"nenhum escore poderia".

**12. O experimento separa geometria de tamanho?**
Não. É a limitação mais importante do resultado principal.

**13. Os discos atravessam a fronteira verdadeira?**
Não necessariamente. Fração de objeto medida: 0,997 e 0,779.

**14. Por que ganho em apenas um dataset?**
No BraTS o mecanismo não pode operar, e isso foi previsto. Na conjuntivite não
há explicação mecanística, e o efeito vai na mesma direção sem significância.

**15. O nulo do BraTS foi previsto antes?**
Sim. `gerador_de_contorno_2026-10-06.json` declara o mecanismo antes da
execução, e o git tem o timestamp.

**16. Em que sentido o BraTS é controle?**
Controle negativo do mecanismo: a arquitetura fixa o banco em 8 filtros e ambos
os braços acrescentam 24.

**17. Dez repetições bastam?**
Para o efeito observado no schisto, sim, inclusive sob teste de sinais (10 de 10
dá p = 0,002 bicaudal). Para os demais, não.

**18. Como justificar t com três ou cinco pares?**
Não se justifica bem. Deve-se acrescentar permutação exata ou sinais, e a
conclusão daquelas células é "não decidido".

**19. O que sobrevive a não paramétrico?**
Precisa ser computado antes da defesa. É item P0.

**20. O desenho foi 3 × 4 × 7?**
Não. Essa é a correção mais importante do texto.

**21. O que é confirmatório e o que é exploratório?**
Confirmatório: seleção de imagem, ganho marginal, gerador, CL, IL.
Exploratório: critério sobre contorno, e a análise de sensibilidade.

**22. O que o CL acrescentou se o resultado foi negativo?**
A caracterização de que o esquecimento no \ac{FLIM} tem outra forma e que os
métodos consolidados não se aplicam. O mecanismo proposto funciona no que se
propõe, verificado por teste, sem render ganho.

**23. IL com usuário real ou simulado?**
Simulado, que acerta sempre. Limite superior otimista.

**24. Até onde generaliza?**
Até o regime avaliado: segmentação binária, objeto pequeno a médio, três a oito
imagens anotadas, dois decodificadores.

**25. Qual o experimento decisivo seguinte?**
Discos de contorno contra discos de mesmo tamanho no interior, com orçamento e
número de candidatos casados. Isola geometria de tamanho e decide a questão 12.

---

## 9. Parecer pessoal

O conteúdo é genuinamente interessante, e por uma razão que vale dizer: a
maioria das dissertações que encontram resultado negativo param ali e o
apresentam como limitação. Esta não para. Ela pergunta por que, decompõe a
pergunta em elos verificáveis, mede cada um, e chega a uma intervenção que
funciona. Esse percurso é mais instrutivo para quem for aplicar \ac{FLIM} do
que seria um ganho marginal de um critério sobre outro.

A contribuição mais original não é a intervenção no gerador, embora seja ela a
que dá o resultado positivo. É a **separação entre a margem que existe e a
margem que se captura**, com teto amostrado e piso medidos na mesma campanha.
Essa decomposição é transferível para qualquer estudo de aprendizado ativo, em
qualquer modelo, e responde a uma ambiguidade que a literatura de resultados
negativos em \ac{AL} costuma deixar aberta: quando um critério não ajuda, não
se sabe se é porque não há o que ganhar ou porque ele não alcança o ganho.

A fragilidade é de calibragem retórica, não de evidência. As campanhas são mais
honestas que algumas frases do texto.

---

## 10. Decisão final

**Defensável após correções.**

Não são necessários experimentos novos para defender. São necessários os
quatorze itens P0, dos quais dois são de compilação, quatro são sobreafirmações
verificadas contra o registro, e os demais são de terminologia e estatística.

O único experimento que eu recomendaria fortemente, e que não é condição para
a defesa mas fortaleceria muito a arguição, é o controle que separa geometria
de tamanho nos candidatos de contorno. Ele é barato, usa a infraestrutura
existente, e converte a resposta da pergunta 12 de "não separo" em "separo, e o
efeito é este".
