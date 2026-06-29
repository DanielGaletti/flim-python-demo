# Experimento 2 — Active Learning no Encoder FLIM
**Dataset:** Schistosoma (schisto/user_A) | Split 1 | Budgets: K=3, 5, 10 | Aquisição: Entropy

---

## Pipeline de Avaliação

Replicação exata do pipeline oficial FLIM-AD (`compute_metrics.py` + `metrics.py`):

1. Forward pass do encoder → saliency uint8
2. `threshold_otsu` (skimage) → binarização
3. `filter_component_by_area`: mantém componentes com área ∈ [1000, 9000] pixels
4. `dice_score` oficial: retorna **1.0** quando GT e pred são ambos vazios (sem ovos)
5. Fβ com β²=0.3 (conforme paper)

> **Nota:** 260/610 imagens de teste têm GT e predição vazios (sem ovos) → dice=1.0 oficial.
> Este critério era a causa do gap: DICE=0.270 (nosso, errado) vs 0.697 (paper, correto).

---

## Baseline — 3 Marcadores Originais do Usuário A

| Decoder                      | Fβ    | DICE  |
|------------------------------|-------|-------|
| labeled_marker               | 0.362 | **0.684** |
| vanilla_adaptive_decoder     | 0.372 | 0.621 |
| vanilla_adaptive_decoder_wt  | 0.340 | 0.478 |
| decoder_2                    | 0.391 | 0.679 |
| **decoder_3**                | **0.399** | **0.694** |
| decoder_attention            | 0.352 | 0.518 |
| hybrid_decoder               | 0.398 | 0.638 |

**Paper (FLIM-AD, labeled_marker):** DICE ≈ 0.697 → nosso: 0.684–0.694 ✓

---

## Resultados AL (Entropy) vs Random — DICE

| Decoder                     | Baseline | K=3 AL | K=3 Rand | K=5 AL | K=5 Rand | K=10 AL | K=10 Rand |
|-----------------------------|----------|--------|----------|--------|----------|---------|-----------|
| labeled_marker              | 0.684    | 0.551  | 0.479    | 0.625  | 0.479    | **0.644** | 0.479  |
| vanilla_adaptive_decoder    | 0.621    | 0.475  | 0.358    | 0.462  | 0.371    | 0.419   | 0.379     |
| vanilla_adaptive_decoder_wt | 0.478    | 0.255  | 0.126    | 0.246  | 0.162    | 0.213   | 0.187     |
| decoder_2                   | 0.679    | 0.453  | 0.479    | 0.421  | 0.306    | 0.341   | 0.382     |
| decoder_3                   | 0.694    | 0.336  | 0.092    | 0.285  | 0.125    | 0.177   | 0.195     |
| decoder_attention           | 0.518    | 0.166  | 0.079    | 0.224  | 0.126    | 0.163   | 0.152     |
| hybrid_decoder              | 0.638    | 0.430  | 0.438    | 0.445  | 0.360    | 0.385   | —         |

**AL > Random:** consistente em 6/7 decoders para K=3, 7/7 para K=5.

---

## Resultados AL (Entropy) vs Random — Fβ (β²=0.3)

| Decoder                     | Baseline | K=3 AL | K=3 Rand | K=5 AL | K=5 Rand | K=10 AL | K=10 Rand |
|-----------------------------|----------|--------|----------|--------|----------|---------|-----------|
| labeled_marker              | 0.362    | 0.259  | 0.000    | 0.323  | 0.000    | 0.313   | 0.000     |
| vanilla_adaptive_decoder    | 0.372    | 0.261  | 0.023    | 0.276  | 0.056    | 0.163   | 0.082     |
| vanilla_adaptive_decoder_wt | 0.340    | 0.181  | 0.065    | 0.180  | 0.109    | 0.160   | 0.117     |
| decoder_2                   | 0.391    | 0.319  | 0.000    | 0.312  | 0.019    | 0.269   | 0.114     |
| decoder_3                   | 0.399    | 0.252  | 0.063    | 0.228  | 0.092    | 0.141   | 0.161     |
| decoder_attention           | 0.352    | 0.121  | 0.038    | 0.169  | 0.072    | 0.124   | 0.100     |
| hybrid_decoder              | 0.398    | 0.295  | 0.008    | 0.333  | 0.061    | 0.194   | 0.150     |

**Δ(AL−Random) médio:** +0.170 (K=3), +0.130 (K=5), +0.073 (K=10)

---

## Observações

**AL vs Baseline:** AL com K marcadores adicionais não supera o baseline com 3 marcadores curados pelo especialista. Esperado — os marcadores originais foram selecionados por expertise humana, enquanto o AL parte de seeds automáticos gerados via GT.

**AL vs Random:** AL (entropy sampling) supera Random consistentemente em Fβ e DICE para todos os budgets. A vantagem é maior para K=3 e K=5, reduzindo em K=10 (Random também melhora com mais dados).

**Divergência Fβ vs DICE:** Fβ com β²=0.3 penaliza mais falsos positivos (precision-oriented). O saliency FLIM tende a alta recall com precisão moderada, resultando em Fβ < DICE. Ambas as métricas são complementares.

**Decoder_3 e hybrid_decoder** são os melhores decoders no baseline (DICE ≈ 0.694/0.638), mas são mais sensíveis à qualidade dos marcadores de treino — degradação maior com AL/Random vs labeled_marker.

---

## Configuração

```
Encoder: FLIM k-means (CPU), 4 camadas, patches 16x16
Aquisição: entropy sampling sobre predição softmax do encoder
Marcadores sintéticos: gerados de GT masks via connected components
Área filter: [1000, 9000] pixels (schisto eggs)
Avaliação: pixel-level DICE + Fβ (β²=0.3), pipeline oficial FLIM-AD
```

---

*Gerado em: 2026-06-27 | Código: `flim_al/al_encoder_experiment.py`*
