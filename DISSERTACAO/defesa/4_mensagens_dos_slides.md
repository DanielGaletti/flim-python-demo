# Mensagens principais para os slides

Uma frase por slide, no título. Cada uma vem com o número que a sustenta e com
a figura ou tabela que a mostra. **Nenhuma frase aqui deve aparecer sem a sua
qualificação ao lado**, no corpo do slide ou na fala.

---

## Bloco 1 — o problema

**"O FLIM aprende com três imagens. O procedimento que escolhe essas três
precisa das 848 anotadas."**
Qualificação: os próprios autores declaram a abordagem como supervisionada.
Figura: `flim_pipeline.png`.

---

## Bloco 2 — a primeira resposta

**"Nenhum critério de seleção de imagem superou o sorteio."**
Qualificação: ausência de evidência de superioridade nas condições avaliadas,
não demonstração de equivalência.
Tabela: `geral_al_lm`.

**"Nem o procedimento que lê o gabarito superou."**
Número: com oito imagens, 0,232 **abaixo** do sorteio, $p = 0{,}0107$, 9 de 9
células.
Qualificação: ele maximiza o pior $F_\beta$, não a informatividade.

---

## Bloco 3 — o resultado central

**"O local da anotação altera o banco de filtros."**
Qualificação: no FLIM a anotação reestima o extrator; o banco é compartilhado por
todas as imagens, inclusive as não anotadas.

**"O melhor clique avaliado ganhou até 0,3089 de $F_\beta$."**
Qualificação obrigatória, no mesmo slide: **identificado com o resultado de
validação em mãos. É potencial disponível, não desempenho de método.**
Figura: `valor_do_local.png`.

**"O pior clique perdeu até 0,4954."**
Qualificação: prêmio e risco são da mesma ordem. Anotar mais não é, por si,
melhorar.

**"Existe grande margem. O desafio é capturá-la sem consultar o gabarito."**
Esta é a frase da defesa. Repita-a.

---

## Bloco 4 — o custo

**"O custo do aprendizado ativo: 0,0839 a 0,2667."**
Qualificação: diferença pareada entre o máximo amostrado e o que o critério
entrega; significativa em 4 de 6 células após Benjamini-Hochberg.
Figura: `margem_captura.png`.

**"O AL regional capturou parte da margem, com forte dependência do domínio e do
decodificador."**
Número que sustenta os dois lados: 84% da margem numa célula, praticamente nada
em outra.

---

## Bloco 5 — por que não capturam

**"Os critérios acham o objeto. Erram a geometria."**
Números: interior puro em 70%, 40% e 100% das escolhas, contra taxa de base de
1,5%, 2,4% e 8,3%.
Figura: `al_regiao_slic.png`.

**"O gerador de candidatos é um componente limitante, não necessariamente o
único gargalo."**
Número: 2,5% a 12,4% dos superpixels atravessam a fronteira.
Qualificação: um escore ideal poderia, em princípio, priorizar os raros.
Figura: `gerador_candidatos.png`.

---

## Bloco 6 — a intervenção

**"A intervenção no contorno predito melhorou o conjunto de parasitas em
0,0772."**
Números: IC [0,0393; 0,1151]; 10 de 10; $t = 0{,}0013$; sinais $= 0{,}0020$;
permutação exata $= 0{,}0020$.
Qualificação no mesmo slide: **contorno predito, não fronteira verdadeira.**

**"Confirmado em um dos três conjuntos."**
Qualificação: BraTS é nulo **e o nulo foi previsto antes de rodar**;
conjuntivite é **inconclusiva**, o desenho não tem poder.

**"O resultado positivo ainda requer replicação e controle entre geometria e
tamanho."**
Diga isso você, antes que perguntem.

---

## Bloco 7 — contínuo e interativo

**"O esquecimento existe no FLIM, com forma própria, e os métodos consolidados
não se aplicam."**
Qualificação: o mecanismo de retenção proposto funciona no que se propõe,
verificado por teste, **sem render ganho mensurável**.

**"Quatro falsos positivos foram capturados pela pré-especificação."**
Qualificação: pré-especificação **versionada no repositório**, não registro
externo.

---

## Bloco 8 — fechamento

**"O valor do clique depende de onde ele é dado."**

**"O aprendizado ativo regional é promissor não porque já entregue os 0,3, mas
porque a margem existe e agora se sabe o que precisa melhorar para capturá-la."**

---

## Slide de apoio: a distinção, caso perguntem

| | valor | como se obtém |
|---|---|---|
| potencial do melhor local | +0,1351 a +0,3089 | testando todos e olhando o resultado |
| o que o critério entrega | +0,0096 a +0,1741 | escolhendo sozinho, sem ver resultado |
| **custo do aprendizado ativo** | **0,0839 a 0,2667** | a diferença, pareada por semente |

Mantenha este slide escondido e traga-o se a pergunta vier. Ele encerra a
discussão em dez segundos.
