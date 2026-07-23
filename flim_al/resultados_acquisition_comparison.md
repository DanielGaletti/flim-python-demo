# Comparação de Acquisition Functions — AL para Encoder FLIM

**Data:** 2026-07-23 | **Splits:** 1, 2, 3 | **Budgets:** K=3, 5, 10 | **n_seeds:** 3 | **Status:** ✅ Validado (encoders frescos)

---

## Resultados Resumo (média 3 splits)

| Método           | Baseline | AL Fβ | Random Fβ | Δ(AL−Rand) | Status  |
|---|---|---|---|---|---|
| **Entropy**          | 0.698 | **0.627** | 0.488 | **+0.139** | ✅ Funciona |
| **Least Confidence** | 0.698 | 0.485 | 0.487 | −0.002 | ❌ Colapsa |
| **Margin**           | 0.698 | 0.479 | 0.485 | −0.006 | ❌ Colapsa |

> Margin e LC selecionam as **mesmas imagens** (fórmulas matematicamente idênticas para binário).  
> Diferença residual (0.479 vs 0.485) é ruído estocástico do k-means no treinamento do encoder.

---

## Resultados por Split — Entropy

### Split 1 — Baseline: 0.713

| K | AL Entropy | Random | Δ |
|---|---|---|---|
| 3  | 0.611 | 0.478 | **+0.133** |
| 5  | 0.631 | 0.479 | **+0.152** |
| **10** | **0.698** | 0.479 | **+0.219** |

### Split 2 — Baseline: 0.668

| K | AL Entropy | Random | Δ |
|---|---|---|---|
| 3  | 0.591 | 0.479 | **+0.112** |
| 5  | 0.596 | 0.479 | **+0.117** |
| **10** | **0.735** | 0.479 | **+0.256** |

### Split 3 — Baseline: 0.712

| K | AL Entropy | Random | Δ |
|---|---|---|---|
| 3  | 0.578 | 0.515 | **+0.063** |
| 5  | 0.589 | 0.516 | **+0.073** |
| **10** | **0.614** | 0.491 | **+0.123** |

---

## Resultados por Split — Least Confidence (validado)

### Split 1 — Baseline: 0.713

| K | AL LC | Random | Δ |
|---|---|---|---|
| 3  | 0.494 | 0.482 | +0.012 |
| 5  | 0.479 | 0.479 | 0.000 |
| 10 | 0.479 | 0.479 | 0.000 |

### Split 2 — Baseline: 0.668

| K | AL LC | Random | Δ |
|---|---|---|---|
| 3  | 0.493 | 0.479 | +0.014 |
| 5  | **0.503** | 0.484 | **+0.019** |
| 10 | 0.479 | 0.479 | 0.000 |

### Split 3 — Baseline: 0.712

| K | AL LC | Random | Δ |
|---|---|---|---|
| 3  | 0.479 | 0.512 | −0.033 |
| 5  | 0.479 | 0.512 | −0.033 |
| 10 | 0.479 | 0.479 | 0.000 |

## Resultados por Split — Margin (validado)

Colapsa para Fβ=0.479 em todos os splits e budgets sem exceção (AL ≡ trivial).

---

## Análise: Por que Margin e Least Confidence colapsam?

### Causa raiz: natureza esparsa dos saliency maps do FLIM

O encoder FLIM produz saliency maps **bimodais**: pixels próximos de 0 (fundo) ou próximos de 1 (ovo). Não há saída difusa em 0.5.

Comparação das funções de scoring para esse tipo de saliency:

| Situação | Entropy | Margin = 1 − |2p − 1| |
|---|---|---|
| Imagem com ovo: pixels em {0, 1} | **ALTA** (mix de valores) | **BAIXA** (pixels confiantes em 0 ou 1) |
| Imagem sem ovo: pixels ≈ 0 | Baixa | Baixa |
| Imagem com saliency difusa ≈ 0.5 | Alta | **ALTA** (pixels ambíguos) |

**Entropy seleciona imagens com ovos** — regiões ativas (valor alto) cercadas de fundo (valor baixo) geram entropia alta.

**Margin seleciona imagens com saliency difusa/ruidosa** — imagens onde o encoder produz valores próximos de 0.5 em muitos pixels, que são imagens ambíguas mas não necessariamente com ovos.

### Evidência: imagens selecionadas (confirmado)

| Acquisition | Split 1 top-3 | Split 2 top-3 | Split 3 top-3 |
|---|---|---|---|
| **Entropy** | 000002, 000003, 000004 | (low-idx, egg+) | (low-idx, egg+) |
| **Margin** | 001077, 000513, 000218 | 001167, 000140, 000247 | 001167, 000634, 000338 |
| **Least Confidence** | **idêntico ao Margin** | **idêntico ao Margin** | **idêntico ao Margin** |

Margin e Least Confidence selecionam exatamente as mesmas imagens — confirmado em todos os splits.  
As fórmulas `1 − |2p−1|` (margin) e `1 − 2|p−0.5|` (LC) são matematicamente idênticas.  
Diferenças residuais nos Fβ finais são ruído de k-means estocástico no treinamento do encoder.

---

## Implicação para o FLIM

Para encoders flyweight com saída esparsa (como o FLIM), a função de aquisição deve capturar **presença de regiões de interesse**, não **ambiguidade global**:

- **Entropy** funciona porque detecta imagens com alguma ativação (potencial ovo)
- **Margin/LC** falha porque detecta imagens com saída difusa, que podem não ter ovos

**Contribuição:** esta análise demonstra que a escolha da acquisition function deve considerar a distribuição dos valores de saída do modelo. Para modelos com saída esparsa/bimodal, entropy é superior a margin e least confidence.

---

## Próximos passos

1. **RIPU**: selecionar regiões dentro das imagens top-entropy (onde a confusão é nas bordas dos ovos)
2. **Entropy + CoreSet**: combinar — entropia filtra imagens informativas, CoreSet garante diversidade
3. **Análise qualitativa**: visualizar os saliency maps das imagens selecionadas por cada método
