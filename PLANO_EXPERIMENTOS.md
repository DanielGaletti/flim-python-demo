# Plano de experimentos — dissertação FLIM + Active Learning

Referência: Soares et al., *FLIM-based Salient Object Detection Networks with
Adaptive Decoders* (arXiv:2504.20872).
Prazos: depósito 05–16/10/2026 · defesa 14/11/2026.

---

## 1. A reformulação que muda a contribuição

O **Algoritmo 1 do paper já é um laço de active learning**, e isso não estava
explícito no projeto até agora. Reproduzido do artigo:

```
1  T ← {imagem aleatória}
3  enquanto o usuário não estiver satisfeito:
4      treina uma CNN FLIM sobre T
5      avalia em Z1\T
6      x ← Fβ médio em Z1\T
7      entre as imagens de MENOR Fβ, escolhe a próxima z
8      se x < x_anterior: remove a última        (backtracking)
11     senão:             T ← T ∪ {z}
```

O passo 7 é o ponto: o artigo diz textualmente *"We adopted a **supervised**
approach to select a few representative images"*. Ele precisa de ground truth
de **todo o Z₁\T** — 848 imagens no split do Parasites — para escolher 3.

Isso derrota o propósito do FLIM. Se houvesse GT de 848 imagens, não haveria
razão para anotar apenas 3 com markers.

**A pergunta da dissertação passa a ser:**

> Quanto do benefício do Algoritmo 1 se recupera com um critério de seleção
> que não usa ground truth nenhum?

| braço | usa GT do pool? | papel |
|---|---|---|
| Algoritmo 1 (paper) | sim, em todo Z₁\T | **teto** — oráculo |
| entropy / coreset / badge / BALD | não | proposta |
| aleatório | não | **piso** |

Comparar com aleatório é fraco; comparar com o oráculo do paper é a pergunta
real. Isso também responde ao que o orientador pediu em 18/06 — *"por que 3 a
5 imagens?"*: porque o Algoritmo 1 para quando acrescentar imagem deixa de
melhorar o Fβ de validação, e empiricamente isso satura em 3–4.

---

## 2. A restrição de anotação, e o que ela permite

Existem markers **reais** para **31 imagens** (união de `data/markers/`,
`data/markers_oracle/` e os splits de `user_A`/`user_B`) — provavelmente todas
as que os usuários do paper anotaram enquanto rodavam o Algoritmo 1. O pool de
seleção do paper tem 848.

Consequência direta: **não é possível rodar AL de imagem sobre o pool completo
com markers reais.** Qualquer imagem escolhida fora dessas 31 precisa de
markers sintéticos — que é exatamente o que degradou o Experimento A.

Daí dois desenhos, e o primeiro é o que sustenta a dissertação:

| desenho | pool | markers | confundidor |
|---|---|---|---|
| **A — pool real** | 31 imagens | reais | nenhum |
| B — pool completo | 848 | sintéticos | markers degradam o encoder |

O desenho A é pequeno, mas é fiel ao paradigma de anotação do FLIM e não tem
confundidor. É o experimento principal. O B vira análise de sensibilidade.

---

## 3. Experimentos, por prioridade

### E1 — Algoritmo 1 com critério trocável ✅ implementado
`flim_al/paper_selection.py`

Laço iterativo honesto: a cada rodada o encoder é retreinado e as saliências
do pool são **regeradas com o encoder atual**, então o critério enxerga o
estado corrente do modelo. Inclui o backtracking dos passos 8–12.

```bash
for c in oracle entropy coreset badge random; do
  python ../flim_al/paper_selection.py --criterion $c --pool real \
     --splits 1 2 3 --seeds 3 --device cuda:0
done
```

Entrega: curva Fβ × |T| por critério → responde "por que 3–5 imagens" e
posiciona cada método entre o piso e o teto.

### E2 — Markers sintéticos realistas
Hoje o gerador sorteia 100 pontos de fg + 300 de bg. O marker real tem ~1215
seeds em traços contíguos seguindo bordas (ver `figs/fig2_markers.png`). O
k-means do FLIM aprende filtros de borda a partir da *vizinhança* dos seeds —
pontos isolados não carregam essa informação.

Implementar: erosão do GT → eixo medial → caminhos contíguos; bg em faixas
próximas à borda. Validar comparando com os markers reais das 31 imagens
(distribuição de seeds, contiguidade). Só então refazer o desenho B.

É o que pode transformar o Experimento A de negativo em positivo, e habilita
AL sobre o pool completo.

### E3 — AL em pixels/regiões, com orçamento de anotação real
Pedido do orientador: *"explorar conjunto de pixels ao invés de imagens"*.

O `region_al.py` já seleciona superpixels por incerteza, mas a cobertura é
fixa em 12,8% (5 patches de 64×64), então não há curva de custo. Parametrizar
o orçamento em **número de seeds** e varrer 100 / 300 / 600 / 1215 seeds,
comparando: seeds em regiões de alta entropia vs seeds distribuídos vs os
markers reais. Isso ataca junto o pedido *"intervenção do especialista com
marcador em imagens de alta confusão"*.

Também resolve a ressalva do slide de eficiência: com custo em seeds, dá para
comparar honestamente com as 3 imagens do baseline.

### E4 — Segundo dataset: BraTS
Encoders já treinados em `out/trained_models/brats/`. O paper usa Parasites +
BraTS, então replicar E1 no BraTS dá validade externa e alinha com o artigo.
Atenção: BraTS é 16-bit grayscale (`bits=16`) e o encoder tem 3 blocos, não 4.

### E5 — Baselines do paper
`baselines/models/` já tem `samnet.py`, `unet.py`, `unet_flim.py`,
`meanet.py`, `mscnet.py`. Falta rodá-los sob o mesmo protocolo (mesmo T, mesmo
split, mesma métrica) para a tabela comparativa que o paper apresenta.

### E6 — Visual antes vs depois, por decoder
Pedido de 18/06. `flim_al/make_figures.py` já gera pipeline, markers, seleção,
região e curvas. Falta um grid: mesma imagem × 7 decoders × (antes do AL,
depois do AL).

---

## 4. Cronograma sugerido

| período | foco |
|---|---|
| ago (2ª quinzena) | E1 completo nos 3 splits × 5 critérios · E2 implementado |
| set (1ª quinzena) | E2 validado · E3 (curva de custo em seeds) |
| set (2ª quinzena) | E4 BraTS · E5 baselines |
| até 05/10 | E6 figuras · escrita · congelamento de resultados |
| 05–16/10 | depósito |

E1 e E2 são os que uma banca vai cobrar. E4 e E5 são o que alinha a
dissertação com o paper reproduzido. E3 é o que torna a afirmação de
eficiência defensável.

---

## 5. O que já está pronto

- Experimento B corrigido: 396 execuções, comparação pareada, piso de ruído
  medido (±0,023) — `RESULTADOS_DISSERTACAO.md`
- Protocolo e defeitos corrigidos documentados — `PROTOCOLO_REVISADO.md`
- Figuras de processo — `figs/`
- Apresentação — `FLIM_AL_Apresentacao_v4.pptx`
- Ambiente GPU (RTX 5070, sm_120) — `Dockerfile.gpu`

## 6. Limitações a declarar na dissertação

1. Markers reais existem para 31 imagens; o desenho B usa markers sintéticos.
2. O decoder 1×1 do Experimento B tem teto de Fβ ≈ 0,567, abaixo dos 0,744 do
   FLIM com 3 markers. O Exp. B mede seleção dentro desse teto.
3. Avaliação com Otsu + filtro de área, não com Dynamic Trees — os números não
   são diretamente comparáveis à Tabela III do paper.
4. O filtro de área [1000, 9000] px é um prior rígido de tamanho: ovos fora
   dessa faixa são insegmentáveis por construção (ver `figs/fig7_falha.png`).
