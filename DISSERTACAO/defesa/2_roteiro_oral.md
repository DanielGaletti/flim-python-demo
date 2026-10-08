# Roteiro oral da defesa

Tempo-alvo: 30 a 35 minutos de apresentação. A frase que organiza tudo:

> **O clique certo pode valer até aproximadamente 0,3 de $F_\beta$. Encontrar
> esse local automaticamente continua sendo o problema científico.**

Repita essa frase três vezes ao longo da apresentação: ao abrir, ao apresentar o
resultado central, e ao fechar. É a única coisa que a banca precisa levar.

---

## A distinção que você não pode errar

Antes do roteiro, treine esta explicação até sair natural. É o ponto onde a
defesa se ganha ou se perde.

**Três números diferentes, que soam parecidos:**

| o que é | valor | como foi obtido |
|---|---|---|
| **Potencial do melhor local** | +0,1351 a +0,3089 | olhando o resultado de validação de cada candidato e pegando o melhor |
| **Desempenho do critério automático** | +0,0096 a +0,1741 | o critério escolhe sozinho, sem ver resultado nenhum |
| **Custo do aprendizado ativo** | 0,0839 a 0,2667 | a diferença entre os dois, pareada dentro da mesma semente |

**Como dizer isso em voz alta, sem jargão:**

> "Imaginem que eu tenho vinte lugares possíveis para o especialista clicar.
> Eu testei os vinte, um por um, e vi qual deu o melhor resultado. A diferença
> entre esse melhor lugar e um lugar sorteado ao acaso é de até 0,3 de $F_\beta$.
>
> Isso **não** quer dizer que meu método entrega 0,3. Eu só sei qual era o melhor
> lugar porque testei todos e olhei a resposta, e na vida real não dá para fazer
> isso: se eu já soubesse a resposta, não precisaria anotar.
>
> O que o meu critério automático entrega, escolhendo sozinho, é bem menos. A
> diferença entre o que estava disponível e o que ele pegou é o que eu chamo de
> custo do aprendizado ativo. E o valor desse número é mostrar que **havia o que
> ganhar**: o critério não falhou por falta de oportunidade, falhou por não
> encontrar a oportunidade."

**Se a banca insistir**, use a analogia do mapa: "medi que existe um atalho que
economiza trinta minutos. Não construí o GPS que acha o atalho. Mas antes de
medir, ninguém sabia se o atalho existia, e essa é a diferença entre desistir da
linha de pesquisa e saber o que precisa ser melhorado."

---

## Roteiro, bloco a bloco

### Bloco 1. O problema (3 min)

Comece pelo FLIM, não pelo aprendizado ativo. A banca precisa entender que este
não é um modelo comum.

- Rede convolucional cujos filtros **não** são aprendidos por retropropagação:
  são centróides de um agrupamento sobre os trechos de imagem sob os traços do
  especialista. Treina em segundos, com três a cinco imagens.
- **Figura:** `flim_pipeline.png`, percorrendo da entrada ao erro por pixel.
- O paradoxo do trabalho de referência: ele escolhe as três imagens medindo o
  desempenho nas 848. Para escolher o que anotar, precisa de tudo anotado.

**Frase de transição:** "A pergunta desta dissertação é quanto desse benefício
sobrevive quando a gente tira o acesso ao gabarito."

### Bloco 2. A primeira resposta, negativa (4 min)

- Avaliei incerteza, diversidade e híbrido. Piso: sorteio. Referência:
  o procedimento do artigo.
- **Nenhum supera o sorteio** após correção para múltiplas comparações.
- **Guarde o melhor para o fim deste bloco:** o procedimento que lê o gabarito
  também não supera. Com oito imagens ele fica 0,232 **abaixo** do sorteio,
  $p = 0{,}0107$, perdendo em 9 de 9 células.

**Frase de transição:** "Isso me deixou com uma pergunta que eu não podia
responder só com esses dados: a escolha não importa, ou os critérios não acham
a boa escolha?"

### Bloco 3. O resultado central (8 min, o coração da defesa)

- Mudei a pergunta de **quais imagens** para **onde dentro da imagem**.
- O desenho: mesmas imagens, mesmo orçamento de pixels, mesma base, mesmo
  codificador inicial. A única variável é **qual região recebe os ~300 px**.
- **Figura:** `valor_do_local.png`. Pare nela. Deixe a banca olhar.
- Leia os dois números em voz alta: melhor candidato **+0,3089**, pior candidato
  **−0,4954**.
- **Diga explicitamente:** "A barra verde não é o meu método. É o que existe
  para ser capturado, e eu só sei disso porque olhei a resposta."
- A consequência que surpreende: **no FLIM, anotar mais não é por si melhorar**.
  Uma anotação mal posicionada pode custar meio ponto. Isso não acontece numa
  rede treinada por retropropagação, e acontece aqui porque a anotação
  **reestima** o banco de filtros em vez de refinar pesos.

**Frase de transição:** "Então existe margem. A pergunta seguinte é quanto dela
os critérios pegam."

### Bloco 4. O custo do aprendizado ativo (5 min)

- **Figura:** `margem_captura.png`.
- Definição em uma frase: a diferença pareada entre o melhor candidato
  disponível e o que o critério escolheu.
- Faixa: **0,0839 a 0,2667**, significativa em 4 de 6 células após correção.
- **Seja honesto sobre o contraexemplo:** no BraTS com `FLIM_lm`, o
  *least confidence* captura 84% da margem, $p = 0{,}0012$. O escore não é
  inútil; o desempenho dele depende do decodificador e do domínio.
- Mencione, sem se demorar, que o teste de sinais discorda do $t$ numa das
  células. Mostra que você olhou.

### Bloco 5. Por que não capturam (5 min)

- **Figura:** `al_regiao_slic.png` e depois `gerador_candidatos.png`.
- Os critérios de incerteza **acham o objeto**: escolhem interior puro em 70%,
  40% e 100% das vezes, contra taxa de base de 1,5%, 2,4% e 8,3%. Isso é
  enriquecimento de uma a duas ordens de grandeza.
- Mas pousam no **interior**, e não na fronteira.
- E a fronteira quase não está na lista: só 2,5% a 12,4% dos superpixels
  atravessam.
- **Cuidado aqui:** não diga "nenhum escore resolveria". Diga "a lista limita o
  que o escore pode escolher, e isso torna a tarefa difícil, não impossível".

### Bloco 6. A intervenção (5 min)

- Se a lista é pobre, troque o gerador. Discos no **contorno predito**.
- **Diga a ressalva junto com o resultado**, não depois: eles atravessam o
  contorno *predito*, não necessariamente a fronteira verdadeira. O modelo
  subsegmenta, e a fração de objeto dos candidatos é 0,997 e 0,779.
- Resultado nos parasitas: **+0,0772**, IC [0,0393; 0,1151], 10 de 10,
  $t = 0{,}0013$, sinais $= 0{,}0020$, permutação $= 0{,}0020$.
- **Diga o escopo antes que perguntem:** confirmado em **um** dos três
  conjuntos. BraTS é nulo **e o nulo foi previsto antes de rodar**, porque lá a
  arquitetura trava o banco em 8 filtros. Conjuntivite é **inconclusiva**.

### Bloco 7. Contínuo, interativo e a plataforma (4 min)

- O esquecimento existe no FLIM, mas tem **forma própria**: não há deslocamento
  de pesos, há deslocamento de filtros no reagrupamento, e é global porque o
  banco é compartilhado.
- Os três grupos de métodos consolidados atuam sobre gradiente e **não se
  aplicam**.
- O mecanismo de retenção proposto funciona no que se propõe, verificado por
  teste, **mas não rendeu ganho mensurável**. Diga isso sem rodeio.
- **Capturas:** `plataforma_onde_anotar.jpg`. A plataforma implementa a
  comparação central, com braço de sorteio como controle.
- **Diga:** "nenhum número da plataforma entra na dissertação; ela serve para
  observar o fenômeno, não para medir."

### Bloco 8. Fechamento (3 min)

- Retome os quatro elos, um slide só.
- A contribuição metodológica: pré-especificação versionada capturou **quatro**
  falsos positivos que teriam entrado no texto.
- Feche com a frase de abertura, de novo.

---

## O que NÃO dizer, em nenhuma hipótese

| nunca diga | diga |
|---|---|
| "O aprendizado ativo melhorou o $F_\beta$ em 0,3" | "A seleção regional revelou potencial de até 0,3; capturá-lo sem gabarito é o problema em aberto" |
| "O oracle é o teto" | "É um baseline supervisionado com informação privilegiada, e ele chega a perder do sorteio" |
| "O gargalo está na lista, não no escore" | "A geração de candidatos contribui materialmente e é um alvo promissor" |
| "Os critérios são equivalentes ao sorteio" | "Não há evidência de superioridade nas condições avaliadas" |
| "Os discos atravessam a fronteira" | "Atravessam o contorno predito; a fronteira verdadeira, nem sempre" |
| "Avaliei em 3 datasets, 4 orçamentos e 7 decodificadores" | "3 conjuntos e 4 orçamentos com 2 decodificadores; os 7 só em análise complementar restrita" |
| "A conjuntivite confirma a direção" | "A conjuntivite é inconclusiva: o desenho não tem poder para decidir" |
| "Avaliei o BADGE" | "Avaliei um híbrido inspirado no BADGE, cuja ponderação diverge do original" |
