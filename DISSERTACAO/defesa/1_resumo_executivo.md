# Resumo executivo

Uma página. Todos os números vêm do registro canônico de execuções e podem ser
conferidos nas tabelas geradas.

## Problema

O FLIM constrói uma rede convolucional de segmentação a partir de traços que o
especialista desenha sobre poucas imagens, estimando os filtros por agrupamento
dos trechos sob esses traços, sem retropropagação. Treina em segundos e opera
com três a cinco imagens anotadas.

O trabalho de referência, porém, escolhe essas poucas imagens por um
procedimento que mede o desempenho em **todo** o conjunto disponível e seleciona
a de pior $F_\beta$. Medir o desempenho exige a anotação. Escolher três imagens
entre 848 exige, portanto, as 848 anotadas, o que devolve exatamente o custo que
o método foi criado para evitar.

## Pergunta

Quanto desse benefício se recupera com critérios de seleção que **não consultam
anotação alguma**? E o que acontece quando se tenta acoplar aprendizado contínuo
e interativo a um modelo sem gradiente?

## Resultados principais

**1. A seleção de imagem não é a alavanca.** Nenhum critério avaliado supera o
sorteio aleatório de forma que sobreviva à correção para múltiplas comparações.
O baseline supervisionado com informação privilegiada, que lê o gabarito de todo
o conjunto, também não se destaca: com oito imagens fica **0,232 abaixo** do
sorteio, com $p = 0{,}0107$, perdendo em 9 de 9 células.

**2. A posição da anotação é.** Com as mesmas imagens e o mesmo orçamento de
pixels, anotar o melhor candidato examinado em vez de um sorteado vale de
**+0,1351 a +0,3089** de $F_\beta$, nas seis células. Anotar o pior custa de
**−0,2075 a −0,4954**. No FLIM, anotar mais **não é por si melhorar**.

**3. Os critérios capturam parte dessa margem.** A lacuna entre o máximo
amostrado e o que o critério entrega vai de **0,0839 a 0,2667**, e sobrevive à
correção em 4 das 6 células. É o que o trabalho define como *custo do aprendizado
ativo*. O desempenho depende do domínio e do decodificador: há uma célula em que
o *least confidence* captura 84% da margem.

**4. A geração de candidatos contribui materialmente.** Regiões que atravessam a
fronteira do objeto são 2,5% a 12,4% dos candidatos produzidos por superpixels.
Trocar o gerador por discos centrados no contorno predito, mantendo escore e
orçamento fixos, rende **+0,0772** de $F_\beta$ nos parasitas, em 10 de 10
repetições, com $t = 0{,}0013$, sinais $= 0{,}0020$ e permutação exata
$= 0{,}0020$.

## Contribuição

A separação entre **a margem que existe** e **a margem que se captura**, medida
na mesma campanha com o mesmo pareamento. Ela responde a uma ambiguidade que a
literatura de resultados negativos em AL costuma deixar aberta: quando um
critério não ajuda, não se sabe se é porque não há o que ganhar ou porque ele
não alcança o ganho. Essa decomposição é transferível para qualquer estudo de
aprendizado ativo, em qualquer modelo.

## Limitações

- O máximo amostrado é identificado **com o resultado de validação**: é
  referência de potencial, não método aplicável, e não é limite superior
  absoluto.
- O ganho do gerador é significativo em **um** dos três conjuntos. O nulo do
  BraTS é previsto pelo mecanismo; a conjuntivite é **inconclusiva**, não
  equivalente.
- A intervenção **não separa geometria de tamanho**: os discos são menores e
  acrescentam menos filtros, e a correlação entre filtros acrescentados e ganho é
  negativa ($\rho = -0{,}299$, $p = 0{,}0005$).
- Os escores de saliência **não são probabilidades calibradas**.
- O critério chamado BADGE diverge do método original e está renomeado como
  híbrido inspirado nele.
- Toda a avaliação de esforço usa **usuário simulado que acerta sempre**: é
  limite superior otimista, e nada aqui mede tempo de especialista.

## Conclusão defensável

O valor de uma nova anotação no FLIM depende fortemente de onde ela cai. Existe
margem substancial para a seleção regional, e os critérios atuais capturam apenas
parte dela. O problema científico em aberto é encontrar essa região **sem
consultar o gabarito**, e o trabalho identifica quais componentes precisam
melhorar para isso: a geração dos candidatos e a ordenação deles.
