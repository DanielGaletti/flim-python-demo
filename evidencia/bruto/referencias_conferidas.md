# Conferência das referências de FLIM

Cada entrada foi conferida em pelo menos duas fontes independentes, conforme
exigido. Data da conferência: 2026-10-05.

Nenhuma entrada foi escrita de memória. Quando uma fonte não pôde ser lida, o
fato está registrado abaixo em vez de contornado.

---

## souza2020flim — o artigo que introduz o FLIM

**Feature Learning from Image Markers for Object Delineation.**
de Souza, Italos Estilon; Benato, Barbara C.; Falcão, Alexandre Xavier.
2020 33rd SIBGRAPI Conference on Graphics, Patterns and Images, p. 116–123,
IEEE, 2020. DOI 10.1109/SIBGRAPI51738.2020.00024

| conferência | fonte | resultado |
|---|---|---|
| 1ª | resolução do DOI | `https://doi.org/10.1109/SIBGRAPI51738.2020.00024` responde 302 para `ieeexplore.ieee.org/document/9265976`, que é o documento cujo título é esse. Liga título e DOI. |
| 2ª | lista de referências do próprio `soares2026flim` | entrada [13] traz autores, venue, páginas e DOI **idênticos** |
| 3ª | busca indexada | mesmos autores, páginas 116–123, mesmo DOI |

A página do IEEE bloqueia leitura automatizada e o servidor de anais do
SIBGRAPI (`sibgrapi.sid.inpe.br`) recusou conexão. A conferência foi feita
pela resolução do DOI mais a lista de referências dos autores que reproduzimos,
que é fonte de primeira mão sobre a própria linhagem.

**Atenção:** este é o artigo que **introduz** o FLIM. Um erro fácil aqui seria
citar `benato2020cnnmarkers` como origem; o resumo dele diz explicitamente que
a técnica "foi recentemente proposta", ou seja, ele **estende** e não origina.

---

## soares2026flim — o artigo reproduzido nesta dissertação

**FLIM-based Salient Object Detection Networks with Adaptive Decoders.**
Soares, Gilson Junior; Cerqueira, Matheus Abrantes; Gomes, Jancarlo F.;
Najman, Laurent; Guimarães, Silvio Jamil F.; Falcão, Alexandre Xavier.
Journal of the Brazilian Computer Society, v. 32, n. 1, p. 750–765, 2026.
DOI 10.5753/jbcs.2026.5893

| conferência | fonte | resultado |
|---|---|---|
| 1ª | página do periódico (JBCS/SBC) | título, os seis autores na ordem, v. 32, n. 1, p. 750–765, 2026, DOI confirmados |
| 2ª | arXiv:2504.20872v1 | título e os seis autores na mesma ordem; submetido 29/04/2025 |
| 3ª | bibtex no repositório dos próprios autores (`flim_ad/README.md`) | mesmos autores e título |

**Decisão de citação:** citar a versão do **periódico**, não o preprint. O
trabalho saiu revisado por pares no JBCS, e o repositório do projeto trazia
apenas a entrada de arXiv. Citar preprint quando existe versão publicada
subestima a referência.

---

## souza2020coconut — FLIM aplicado a classificação

**Learning CNN Filters from User-Drawn Image Markers for Coconut-Tree Image
Classification.** de Souza, Italos Estilon; Falcão, Alexandre Xavier.
IEEE Geoscience and Remote Sensing Letters, v. 19, p. 1–5, 2020.
DOI 10.1109/LGRS.2020.3020098

| conferência | fonte | resultado |
|---|---|---|
| 1ª | lista de referências de `soares2026flim`, entrada [12] | autores, periódico, volume, páginas e DOI |
| 2ª | busca indexada | mesmos dados |

---

## benato2020cnnmarkers — extensão do FLIM com camadas densas

**Convolutional Neural Networks from Image Markers.** Benato, Barbara C.;
de Souza, Italos Estilon; Galvão, Felipe L.; Falcão, Alexandre Xavier.
arXiv:2012.12108, 2020. DOI 10.48550/arXiv.2012.12108

| conferência | fonte | resultado |
|---|---|---|
| 1ª | página do arXiv | título e os quatro autores na ordem, 2020 |
| 2ª | o próprio resumo | declara que o FLIM "foi recentemente proposto", confirmando que este artigo estende e não introduz |

Sem versão publicada localizada. Fica citado como preprint, que é o que ele é.

---

## joao2025flyweight — o artigo pedido pela orientadora

**Flyweight FLIM Networks for Salient Object Detection in Biomedical Images.**
Joao, Leonardo M.; Gomes, Jancarlo F.; Guimaraes, Silvio J. F.; Kijak, Ewa;
Falcao, Alexandre X. arXiv:2504.11112, 2025. DOI 10.48550/arXiv.2504.11112

| conferência | fonte | resultado |
|---|---|---|
| 1ª | página do arXiv | título e os cinco autores na ordem, 2025 |
| 2ª | busca indexada | mesmos dados |

**Correção de premissa, e ela importa.** Este artigo foi pedido como "um
trabalho que usa IL e CL como base de aprendizado". Ele **não usa** nenhum dos
dois: é outra rede FLIM, com kernels separáveis dilatados e decodificador
adaptativo, voltada a imagens biomédicas. É baseline de FLIM relevante e entra
como tal, mas citá-lo como fonte de Interactive ou Continual Learning seria
atribuir ao artigo uma afirmação que ele não sustenta.

---

## Referências já existentes no `.bib` e usadas nos capítulos novos

Não foram reconferidas, por já estarem no arquivo do projeto desde a
qualificação: `xu2016deep_arxiv` (protocolo do usuário simulado),
`xie2022ripu`, `kim2023adaptive`, `mackowiak2018cereals` (AL por região),
`kirkpatrick2017overcoming`, `rebuffi2017icarl`, `hinton2015distilling`
(aprendizado contínuo), `menze2014multimodal` (BraTS).

Se alguma delas entrar em afirmação nova, precisa passar pela mesma conferência
de duas fontes antes.
