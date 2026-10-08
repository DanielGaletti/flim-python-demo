# Perguntas da banca

Vinte e oito perguntas, com resposta curta para falar, resposta técnica para
sustentar, a limitação a admitir, e a frase a evitar.

A regra geral: **admita a limitação antes que perguntem**. Uma limitação que
você enuncia é rigor; a mesma limitação arrancada por pergunta é falha.

---

### 1. Por que o máximo amostrado não é um método?

**Curta.** Porque para saber qual era o melhor candidato eu testei todos e olhei
o resultado da validação. Se eu já tenho o resultado, não preciso do critério.

**Técnica.** O máximo é tomado sobre os candidatos efetivamente examinados, com
o $F_\beta$ de validação de cada um em mãos. É um estimador do que existe para
capturar naquele conjunto de candidatos, e por construção não é computável sem o
rótulo. Ele entra como referência, nunca como braço comparável.

**Admitir.** Nem sequer é teto absoluto: é máximo sobre os candidatos
examinados, e um conjunto maior poderia conter posições melhores.

**Evitar.** "O melhor candidato alcança 0,3089."

---

### 2. Por que o baseline supervisionado não é um teto?

**Curta.** Porque ele perde do sorteio. Um teto, por definição, não perde.

**Técnica.** O Algoritmo 1 do trabalho de referência maximiza o *pior* $F_\beta$,
e não a informatividade. Com oito imagens ele fica 0,232 abaixo do sorteio,
$p = 0{,}0107$, perdendo em 9 de 9 células. A reimplementação mostra o motivo: o
critério tende a escolher imagens insegmentáveis, cujas componentes previstas
caem fora da faixa do filtro de área, de modo que nenhuma anotação ajudaria.

**Admitir.** Acesso ao gabarito é vantagem informacional real; o que não segue é
que vantagem informacional produza desempenho ótimo.

**Evitar.** "O oracle representa o limite superior."

---

### 3. Por que o aprendizado ativo não alcançou os 0,3 automaticamente?

**Curta.** Porque ele escolhe sem ver o resultado, e os 0,3 foram identificados
vendo o resultado. A diferença entre os dois é justamente o que eu meço.

**Técnica.** A lacuna vai de 0,0839 a 0,2667, significativa em 4 de 6 células
após Benjamini-Hochberg. A caracterização do comportamento mostra duas razões:
os critérios de incerteza localizam o objeto mas pousam no interior, e os de
diversidade não localizam o objeto quando ele é pequeno, porque a distribuição
de aparência é dominada pelo fundo.

**Admitir.** Em uma célula o critério capturou 84% da margem. O desempenho
depende do decodificador e do domínio.

**Evitar.** "Os critérios de incerteza são inúteis."

---

### 4. Como você defende usar entropia sobre algo que não é probabilidade?

**Curta.** Uso por analogia, e digo isso no texto. Como todos os braços recebem
a mesma representação, a comparação entre eles continua válida.

**Técnica.** Os valores vêm da normalização linear do mapa de ativação do
decodificador para $[0,1]$, quantizada em oito bits, e não de uma função
logística calibrada. Chamo-os de escore de saliência normalizado, tratado como
pseudo-probabilidade. A entropia calculada sobre eles não tem a interpretação
informação-teórica que teria sobre probabilidade calibrada: o máximo em 0,5 é
propriedade de onde a normalização situa o ponto médio.

**Admitir.** Uma calibração, por exemplo escalonamento de Platt sobre a
validação, poderia mudar a ordenação dos candidatos e não foi testada.

**Evitar.** "A entropia mede a incerteza do modelo."

---

### 5. O BADGE que você implementou é BADGE?

**Curta.** Não. É um híbrido inspirado nele, e a divergência está declarada no
texto com as duas fórmulas lado a lado.

**Técnica.** O BADGE original usa o gradiente da perda na última camada,
avaliado no pseudo-rótulo: $g = (P - e_{\hat y}) \otimes h$, cuja magnitude no
caso binário é $|P - \mathrm{round}(P)|$, máxima na incerteza. A implementação
avaliada usa $(S - 0{,}5)\,f$, cuja magnitude é $|S - 0{,}5|$: **nula** na
incerteza máxima e máxima na confiança máxima. Como o $k$-means++ sorteia
proporcional ao quadrado da distância, o efeito é favorecer o confiante,
invertendo a intenção.

**Admitir.** É um defeito, encontrado na auditoria do próprio trabalho. A linha
correspondente não avalia o BADGE.

**O que isso NÃO invalida.** O critério não participa de nenhum resultado
central: não aparece no ganho marginal, no custo do aprendizado ativo nem na
troca do gerador. Ele consta em uma linha de uma tabela.

**Evitar.** "Avaliei o BADGE e ele não funcionou."

---

### 6. O BALD é bayesiano?

**Curta.** Não. É discordância entre codificadores, e chamo assim no texto.

**Técnica.** A formulação original aproxima a posterior sobre parâmetros por
dropout de Monte Carlo. O FLIM é determinístico e não tem dropout, de modo que
não há posterior a aproximar. O comitê aqui são os mapas de saliência de
codificadores distintos, e a quantidade continua separando desacordo de
incerteza compartilhada, mas sem a interpretação bayesiana.

**Admitir.** O comitê é pequeno e os codificadores não são amostras de uma
posterior.

**Evitar.** "Apliquei BALD."

---

### 7. Teste $t$ com três ou cinco pares se justifica?

**Curta.** Sozinho, não. Por isso reporto também sinais e permutação exata.

**Técnica.** Na tabela do gerador, o desfecho primário vem com três testes.
Nos parasitas, $t = 0{,}0013$, sinais $= 0{,}0020$, permutação $= 0{,}0020$: os
três concordam, e a conclusão não depende da normalidade. Na conjuntivite,
0,2374, 0,3750 e 0,2500: nenhum decide. Com cinco pares, a permutação exata tem
$2^5 = 32$ atribuições e o menor $p$ atingível é $1/16$.

**Admitir.** As células de três e cinco sementes são **inconclusivas**, e o
desenho não tem poder para decidi-las.

**Evitar.** "Os resultados são equivalentes."

---

### 8. Dez repetições bastam?

**Curta.** Para o efeito do gerador nos parasitas, sim, inclusive sob teste de
sinais. Para os demais, não.

**Técnica.** Dez vitórias em dez dá $p = 0{,}00195$ no teste de sinais bicaudal,
sem suposição alguma. O pareamento por semente, que fixa partição, imagens e
codificador inicial, reduz a variância e é o que torna dez suficientes.

**Admitir.** Dez sementes compartilham o mesmo pool e a mesma validação; o
intervalo descreve a variação sob sorteio das imagens de treino, e não
generaliza para o conjunto.

**Evitar.** "Dez repetições garantem robustez."

---

### 9. O desenho foi $3 \times 4 \times 7$?

**Curta.** Não, e isso está corrigido no texto. Foram três conjuntos e quatro
orçamentos com **dois** decodificadores e três repetições por célula.

**Técnica.** Os cinco decodificadores adicionais aparecem apenas em
`tabela_k_por_modelo`, restritos ao conjunto de parasitas e com uma repetição
por configuração. A Tabela da matriz de campanhas, no Capítulo 3, permite
conferir isso diretamente.

**Admitir.** A versão anterior do texto sugeria um fatorial completo que não
ocorreu.

**Evitar.** Qualquer formulação que some os três fatores sem qualificar.

---

### 10. Se só 2% a 12% dos candidatos cruzam a fronteira, por que um bom escore não os acharia?

**Curta.** Poderia, em princípio. A escassez torna a tarefa difícil, não
impossível, e eu corrigi o texto para dizer isso.

**Técnica.** Um escore que ordenasse perfeitamente colocaria os raros candidatos
de fronteira no topo. O que a medição mostra é que os escores avaliados não
fazem isso: eles pousam no interior. E a campanha exploratória sobre a lista
corrigida sugere que gerador e escore agem por caminhos distintos, com efeitos
aproximadamente aditivos.

**Admitir.** É uma objeção correta, e a formulação anterior do texto, "nenhuma
escolha de escore resolveria", era forte demais.

**Evitar.** "O gargalo está na lista e não no escore."

---

### 11. O experimento do contorno separa geometria de tamanho?

**Curta.** Não. É a limitação mais importante do resultado principal.

**Técnica.** Os discos de contorno são menores que os superpixels e acrescentam
menos filtros ao banco. O diagnóstico de ganho marginal mede correlação negativa
entre filtros acrescentados e ganho ($\rho = -0{,}299$, $p = 0{,}0005$), de modo
que parte do efeito pode decorrer do tamanho, e não da posição.

**Admitir.** O mecanismo causal completo permanece parcialmente identificado.

**Experimento que resolveria.** Discos de contorno contra discos de **mesmo
tamanho** posicionados no interior, com orçamento e número de candidatos
casados. É barato e usa a infraestrutura existente.

**Evitar.** "Provei que a posição é a causa."

---

### 12. Os discos atravessam a fronteira verdadeira?

**Curta.** Não necessariamente. Atravessam o contorno **predito**.

**Técnica.** Quando o modelo subsegmenta, o contorno predito fica por dentro do
real e os discos caem no interior. Medido: fração de objeto dos candidatos em
0,997 nos parasitas e 0,779 no BraTS. O ganho ocorre **apesar** disso, o que é
parte da razão para não atribuí-lo integralmente à posição.

**Admitir.** O mecanismo pelo qual o ganho ocorre não está totalmente
estabelecido.

**Evitar.** "Os candidatos de contorno atravessam a fronteira por construção."

---

### 13. Por que houve ganho em apenas um dos três conjuntos?

**Curta.** No BraTS o mecanismo não pode operar, e isso foi previsto antes de
rodar. Na conjuntivite o desenho não tem poder.

**Técnica.** A arquitetura do BraTS fixa o banco em oito filtros por camada, e
ambos os braços acrescentam exatamente 24: o deslocamento que a intervenção
evita não pode ocorrer. A previsão está no arquivo de pré-especificação,
datado antes da execução.

**Admitir.** Um dos três conjuntos confirma. **Não há demonstração de
generalização ampla.**

**Evitar.** "O método funciona em segmentação biomédica."

---

### 14. O nulo do BraTS foi previsto ou explicado depois?

**Curta.** Previsto, e o arquivo está no repositório com data anterior à
execução.

**Técnica.** A pré-especificação `gerador_de_contorno_2026-10-06.json` declara o
mecanismo e a consequência esperada antes do resultado, e o histórico do git
registra a ordem.

**Admitir.** É pré-especificação versionada internamente, não registro externo
imutável. Um repositório sob controle do autor oferece garantia mais fraca.

**Evitar.** "Pré-registrei o experimento", sem qualificar.

---

### 15. Em que sentido o BraTS é um controle?

**Curta.** Controle negativo do mecanismo: é a condição em que o mecanismo
proposto não pode agir, e o efeito desaparece como esperado.

**Técnica.** Um nulo previsto por um mecanismo declarado antes é evidência a
favor desse mecanismo, porque o mecanismo poderia ter falhado ali e não falhou.

**Admitir.** Com três sementes, o nulo é compatível com a previsão mas não a
confirma com força. Ausência de efeito detectável não é prova de ausência de
efeito.

**Evitar.** "O BraTS prova o mecanismo."

---

### 16. Como você sabe que o resultado negativo não é bug na sua implementação?

**Curta.** Três razões: o baseline que lê o gabarito também não se destaca, os
critérios mostram comportamento estruturado, e eu encontrei e corrigi bugs reais
descartando os resultados anteriores.

**Técnica.** Se os critérios estivessem quebrados, eles escolheriam ao acaso. Não
escolhem: selecionam interior puro em 70%, 40% e 100% das vezes, contra taxa de
base de 1,5%, 2,4% e 8,3%. Isso é enriquecimento de uma a duas ordens de
grandeza, e é comportamento estruturado. Além disso, o procedimento supervisionado
também não supera o sorteio, o que não se explica por erro nos critérios não
supervisionados.

**Admitir.** O híbrido inspirado no BADGE está, de fato, com a ponderação
invertida.

**Evitar.** "Testei tudo e está correto."

---

### 17. O que foi realmente reproduzido do artigo?

**Curta.** O arranjo codificador com decodificador adaptativo, a curva de
orçamento e a saturação precoce entre três e quatro imagens.

**Técnica.** Sem Dynamic Trees, cujo binário não executa no ambiente usado, a
avaliação emprega Otsu com filtro de área, e os valores **não são diretamente
comparáveis** aos publicados na Tabela III.

**Admitir.** A reprodução não pode ser verificada contra os números publicados.

**Evitar.** "Reproduzi os resultados do artigo."

---

### 18. Você usou o conjunto de teste quantas vezes?

**Curta.** As campanhas de diagnóstico, que sustentam os resultados centrais,
medem só em validação. A campanha de seleção de imagens reporta no teste cinco
critérios em quatro orçamentos e três conjuntos.

**Técnica.** Nenhum hiperparâmetro, limiar ou faixa do filtro de área foi
escolhido com base no teste, e todos os braços foram declarados antes. É reporte
de configurações pré-especificadas, não busca sobre o teste.

**Admitir.** Dizer que o teste foi "tocado uma única vez" seria impreciso, e o
texto não usa essa formulação.

**Evitar.** "O teste foi usado uma vez só."

---

### 19. Qual é exatamente a contribuição original?

**Curta.** A separação, medida, entre a margem que existe e a margem que se
captura, mais a intervenção no gerador que esse diagnóstico indica.

**Técnica.** A decomposição responde a uma ambiguidade que a literatura de
resultados negativos em AL deixa aberta, e é transferível para qualquer estudo
de aprendizado ativo em qualquer modelo.

**Admitir.** A intervenção é confirmada em um conjunto, e o mecanismo não está
totalmente isolado.

**Evitar.** "Propus um novo método de aprendizado ativo."

---

### 20. O que o aprendizado contínuo acrescentou, se o resultado foi negativo?

**Curta.** A caracterização de que o esquecimento no FLIM tem forma própria e
que os métodos consolidados não se aplicam.

**Técnica.** Não há deslocamento de pesos por gradiente; há deslocamento de
filtros no reagrupamento, e a perturbação é global porque o banco é
compartilhado. Regularização, ensaio e destilação atuam sobre gradiente e não
têm sobre o que agir. O mecanismo de retenção proposto resolve o que se propõe,
verificado por teste, sem render ganho mensurável.

**Admitir.** A hipótese de que a retenção preservaria qualidade foi **falseada**
pelo critério declarado antes da execução.

**Evitar.** "O aprendizado contínuo melhorou a estabilidade."

---

### 21. O aprendizado interativo foi avaliado com pessoas?

**Curta.** Não. Usuário simulado que acerta sempre, pelo protocolo de robot user
de Xu et al.

**Técnica.** O clique vai ao máximo da transformada de distância dentro da maior
componente de erro, e o rótulo vem do gabarito. A métrica mede **número de
interações**, não tempo de especialista.

**Admitir.** É limite superior otimista. Nenhum resultado deste trabalho
autoriza afirmação sobre redução de tempo ou custo de anotação.

**Evitar.** "O sistema reduz o esforço do especialista."

---

### 22. Até onde os resultados generalizam?

**Curta.** Até o regime avaliado: segmentação binária, objeto pequeno a médio,
três a oito imagens anotadas, dois decodificadores.

**Técnica.** Os três conjuntos cobrem a fração de objeto em faixas separadas,
de 3,2% a 19,3%, que a investigação mostrou ser a propriedade determinante. Fora
dessa faixa, nada foi medido.

**Admitir.** Nenhuma afirmação sobre multiclasse, sobre objetos grandes, ou
sobre modelos treinados por gradiente.

**Evitar.** "O resultado vale para segmentação médica."

---

### 23. Qual seria o próximo experimento decisivo?

**Curta.** Discos de contorno contra discos de mesmo tamanho no interior, com
orçamento e número de candidatos casados.

**Técnica.** Isola geometria de tamanho, que é a confusão remanescente no
resultado principal, e usa a infraestrutura que já existe.

**Admitir.** Ele não foi feito, e por isso a afirmação causal está moderada no
texto.

---

### 24. Por que a conjuntivite não confirma?

**Curta.** Porque o desenho não tem poder, não porque o efeito seja nulo.

**Técnica.** Cinco sementes, com os três testes dando 0,2374, 0,3750 e 0,2500. A
permutação exata com $n = 5$ tem menor $p$ atingível de $1/16$. Além disso, uma
das sementes entregou **um único** candidato de contorno, e o desfecho
secundário de pior caso, que parecia significativo com $p = 0{,}0480$, passa a
0,1032 quando o número de candidatos é casado entre os braços.

**Admitir.** Esse desfecho secundário **não** deve ser lido como significativo, e
a nota da tabela diz isso.

**Evitar.** "A conjuntivite aponta na mesma direção, então corrobora."

---

### 25. Por que o teste de sinais discorda do $t$ numa célula?

**Curta.** Porque a lacuna é grande em média mas não consistente em direção.

**Técnica.** Nos parasitas com `FLIM_lm`, o $t$ dá 0,0230 e os sinais dão
0,3438. O $t$ responde à magnitude; os sinais, à direção. A divergência indica
que algumas sementes têm lacuna grande e outras têm lacuna negativa.

**Admitir.** Onde os dois divergem, a evidência é mais fraca do que o $t$
sozinho sugere, e o texto diz isso.

**Evitar.** Reportar só o $t$.

---

### 26. Por que a acurácia passa de 0,97 em quase tudo?

**Curta.** Porque o objeto ocupa 3% da imagem, e acertar o fundo já garante isso.

**Técnica.** A acurácia é $1 - \mathrm{MAE}$. Em conjuntos com objeto pequeno
ela é dominada pelo fundo e não separa braço nenhum. Quem separa é $F_\beta$ e
IoU.

**Evitar.** Usar acurácia para comparar métodos aqui.

---

### 27. Quantos falsos positivos o seu protocolo capturou?

**Curta.** Quatro, e eles estão no texto como resultado de método.

**Técnica.** Densidade na conjuntivite, de $p = 0{,}0016$ para 0,586; dispersão
da entropia, de 0,082 para 0,5674; NoC com $\alpha = 0$, de 0,0265 para 0,1385
ao separar por conjunto; e o mecanismo de $\alpha = 0$, falseado pelo critério
terciário pré-declarado. O segundo teve errata própria, registrada, porque a
primeira análise misturava dois conjuntos de teste em quatro das doze células.

**Admitir.** Que a errata foi necessária é, em si, sinal de que o procedimento
funciona.

---

### 28. Por que a plataforma, se nenhum número dela entra na dissertação?

**Curta.** Para que o fenômeno seja observável por um especialista, e para que a
comparação central possa ser operada por quem não escreveu o código.

**Técnica.** Ela implementa a comparação de posição de anotação com braço de
sorteio como controle. Roda em `localhost`, sem serviço externo, de modo que as
imagens de paciente não saem da máquina.

**Admitir.** Ela roda com orçamento pequeno e execução única, regime em que a
amplitude do braço de sorteio, de 0,10 a 0,22, é maior que qualquer efeito
observável. Por isso nenhum número dela é citado.

**Evitar.** Mostrar um número da tela e tratá-lo como resultado.
