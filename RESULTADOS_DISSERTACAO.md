# Resultados — FLIM + Active Learning

## Experimento B — Backprop AL (encoder FIXO)

### B.1 Piso de ruído e teto do decoder

Braço `full`: todas as N imagens do pool, variando apenas a semente de inicialização. O desvio é o ruído irredutível; nenhum ΔFβ menor que ele deve ser interpretado.

| método | split | N | Fβ (teto) | desvio (ruído) |
|---|---|---|---|---|
| badge | 1 | 177 | 0.4666 | ±0.0099 |
| badge | 2 | 196 | 0.6401 | ±0.0007 |
| badge | 3 | 189 | 0.5931 | ±0.0585 |
| coreset | 1 | 177 | 0.4666 | ±0.0099 |
| coreset | 2 | 196 | 0.6401 | ±0.0007 |
| coreset | 3 | 189 | 0.5931 | ±0.0585 |
| entropy | 1 | 177 | 0.4666 | ±0.0099 |
| entropy | 2 | 196 | 0.6401 | ±0.0007 |
| entropy | 3 | 189 | 0.5931 | ±0.0585 |
| region_entropy | 1 | 177 | 0.4298 | ±0.0063 |
| region_entropy | 2 | 196 | 0.4772 | ±0.0059 |
| region_entropy | 3 | 189 | 0.5241 | ±0.0117 |

### B.2 ΔFβ pareado (AL − Random), por budget

Cada par usa a MESMA inicialização; n = splits × sementes.

| método | K | AL Fβ | Random Fβ | ΔFβ pareado | IC95% | p (t) | p (Wilcoxon) | Δ>0 | colapsos |
|---|---|---|---|---|---|---|---|---|---|
| badge | 3 | 0.4975 | 0.4845 | +0.0130 | [-0.056, +0.082] | 0.693 | 0.804 | 7/15 | 0/15 |
| badge | 5 | 0.5032 | 0.5025 | +0.0007 | [-0.059, +0.060] | 0.979 | 0.925 | 8/15 | 0/15 |
| badge | 10 | 0.5634 | 0.4832 | **+0.0803** | [+0.050, +0.111] | <0.001 | <0.001 | 15/15 | 3/15 ⚠ |
| coreset | 3 | 0.5424 | 0.4845 | +0.0579 | [-0.006, +0.121] | 0.071 | 0.074 | 10/15 | 4/15 ⚠ |
| coreset | 5 | 0.5518 | 0.5025 | +0.0493 | [-0.003, +0.102] | 0.063 | 0.073 | 11/15 | 3/15 ⚠ |
| coreset | 10 | 0.5752 | 0.4832 | **+0.0920** | [+0.059, +0.125] | <0.001 | 0.002 | 13/15 | 2/15 ⚠ |
| entropy | 3 | 0.4090 | 0.4845 | **-0.0755** | [-0.132, -0.019] | 0.012 | 0.018 | 3/15 | 0/15 |
| entropy | 5 | 0.4230 | 0.5025 | -0.0795 | [-0.168, +0.009] | 0.076 | 0.107 | 4/15 | 0/15 |
| entropy | 10 | 0.5258 | 0.4832 | +0.0426 | [-0.014, +0.099] | 0.130 | 0.135 | 10/15 | 0/15 |
| region_entropy | 3 | 0.2930 | 0.4459 | **-0.1529** | [-0.224, -0.082] | <0.001 | <0.001 | 1/15 | 0/15 |
| region_entropy | 5 | 0.3621 | 0.4198 | -0.0577 | [-0.164, +0.049] | 0.266 | 0.277 | 6/15 | 0/15 |
| region_entropy | 10 | 0.5067 | 0.4274 | **+0.0793** | [+0.001, +0.158] | 0.048 | 0.035 | 11/15 | 0/15 |

`**` = IC95% não cruza zero **e** efeito acima do piso de ruído. `~` = consistente, porém dentro do ruído. ⚠ = alguma execução do braço AL colapsou (predição toda-fundo): nesses casos o 'ganho' não representa aprendizado.

### B.3 Agregado por método (todos os budgets de anotação)

| método | ΔFβ pareado | IC95% | p (t) | Δ>0 | veredicto |
|---|---|---|---|---|---|
| badge | +0.0292 | [-0.004, +0.063] | 0.085 | 27/42 | indistinguível do random |
| coreset | **+0.0722** | [+0.038, +0.106] | <0.001 | 27/36 | supera o random |
| entropy | -0.0375 | [-0.078, +0.003] | 0.068 | 17/45 | indistinguível do random |
| region_entropy | -0.0438 | [-0.098, +0.011] | 0.112 | 18/45 | indistinguível do random |

_Execuções com colapso do braço AL são excluídas do agregado._

### B.4 Eficiência de anotação: AL vs pool inteiro

Compara o maior budget de AL contra treinar com TODAS as imagens do pool. Diferença dentro do piso de ruído (B.1) significa desempenho equivalente com uma fração da anotação.

| método | K | split | N do pool | AL Fβ | pool Fβ | AL − pool | veredicto |
|---|---|---|---|---|---|---|---|
| badge | 10 | 1 | 177 | 0.4622 | 0.4666 | -0.0044 | |
| badge | 10 | 2 | 196 | 0.6127 | 0.6401 | -0.0274 | |
| badge | 10 | 3 | 189 | 0.6155 | 0.5931 | +0.0224 | |
| **badge** | **10** | média | | **0.5634** | **0.5666** | **-0.0032** | **equivalente** (|Δ| ≤ ruído 0.023) |
| coreset | 10 | 1 | 177 | 0.5282 | 0.4666 | +0.0616 | |
| coreset | 10 | 2 | 196 | 0.6041 | 0.6401 | -0.0359 | |
| coreset | 10 | 3 | 189 | 0.5931 | 0.5931 | +0.0000 | |
| **coreset** | **10** | média | | **0.5752** | **0.5666** | **+0.0086** | **equivalente** (|Δ| ≤ ruído 0.023) |
| entropy | 10 | 1 | 177 | 0.5820 | 0.4666 | +0.1154 | |
| entropy | 10 | 2 | 196 | 0.5133 | 0.6401 | -0.1267 | |
| entropy | 10 | 3 | 189 | 0.4820 | 0.5931 | -0.1111 | |
| **entropy** | **10** | média | | **0.5258** | **0.5666** | **-0.0408** | AL abaixo do pool |
| region_entropy | 10 | 1 | 177 | 0.5846 | 0.4298 | +0.1548 | |
| region_entropy | 10 | 2 | 196 | 0.4776 | 0.4772 | +0.0004 | |
| region_entropy | 10 | 3 | 189 | 0.4580 | 0.5241 | -0.0662 | |
| **region_entropy** | **10** | média | | **0.5067** | **0.4770** | **+0.0297** | AL acima do pool |

## Experimento A — Encoder AL (encoder retreinado)

### A.1 Imagens efetivamente usadas por braço

Métodos `region_*` descartam imagens sem região de foreground. Se o braço AL treinar com menos imagens que o Random, parte do Δ vem de ter adicionado menos markers sintéticos, não da seleção.

| método | split | K | AL | random | random_region |
|---|---|---|---|---|---|
| coreset | 1 | 3 | 6 | 6 | — |
| coreset | 1 | 5 | 8 | 8 | — |
| coreset | 1 | 10 | 13 | 13 | — |
| coreset | 2 | 3 | 6 | 6 | — |
| coreset | 2 | 5 | 8 | 8 | — |
| coreset | 2 | 10 | 13 | 13 | — |
| coreset | 3 | 3 | 6 | 6 | — |
| coreset | 3 | 5 | 8 | 8 | — |
| coreset | 3 | 10 | 13 | 13 | — |

Braços balanceados em todas as configurações.

### A.2 Decomposição: seleção vs geometria dos seeds

Só faz sentido para métodos `region_*`, que possuem o braço de controle `random_region` (imagens aleatórias + seeds de região).

| método | decoder | AL | random_region | random | seleção<br>(AL−rand_reg) | geometria<br>(rand_reg−rand) |
|---|---|---|---|---|---|---|
| region_bald | labeled_marker | 0.575 | 0.605 | 0.517 | -0.030 | +0.088 |
| region_bald | decoder_2 | 0.623 | 0.586 | 0.441 | +0.036 | +0.145 |
| region_bald | decoder_3 | 0.594 | 0.560 | 0.277 | +0.033 | +0.284 |
| region_bald | decoder_attention | 0.350 | 0.259 | 0.162 | +0.091 | +0.096 |
| region_bald | hybrid_decoder | 0.503 | 0.425 | 0.377 | +0.079 | +0.048 |
| region_bald | vanilla_adaptive_decoder | 0.466 | 0.436 | 0.409 | +0.030 | +0.027 |
| region_bald | vanilla_adaptive_decoder_wt | 0.338 | 0.226 | 0.161 | +0.112 | +0.065 |

### A.3 Fβ por decoder (média entre splits e budgets)

| método | decoder | baseline | AL | random (±dp) | Δ(AL−rand) | colapsos AL |
|---|---|---|---|---|---|---|
| badge | labeled_marker | 0.744 | 0.518 | 0.518±0.052 | -0.000 | 5/9 |
| badge | decoder_2 | 0.728 | 0.434 | 0.439±0.097 | -0.005 | 0/9 |
| badge | decoder_3 | 0.731 | 0.225 | 0.274±0.097 | -0.049 | 0/9 |
| badge | decoder_attention | 0.561 | 0.132 | 0.162±0.059 | -0.029 | 0/9 |
| badge | hybrid_decoder | 0.686 | 0.379 | 0.377±0.049 | +0.002 | 0/9 |
| badge | vanilla_adaptive_decoder | 0.635 | 0.415 | 0.408±0.043 | +0.006 | 0/9 |
| badge | vanilla_adaptive_decoder_wt | 0.531 | 0.128 | 0.161±0.056 | -0.033 | 0/9 |
| coreset | labeled_marker | 0.744 | 0.504 | 0.521±0.050 | -0.017 | 6/9 |
| coreset | decoder_2 | 0.728 | 0.549 | 0.437±0.099 | +0.112 | 0/9 |
| coreset | decoder_3 | 0.731 | 0.367 | 0.273±0.100 | +0.094 | 0/9 |
| coreset | decoder_attention | 0.561 | 0.162 | 0.162±0.061 | +0.000 | 0/9 |
| coreset | hybrid_decoder | 0.686 | 0.367 | 0.376±0.047 | -0.009 | 0/9 |
| coreset | vanilla_adaptive_decoder | 0.635 | 0.392 | 0.409±0.045 | -0.016 | 0/9 |
| coreset | vanilla_adaptive_decoder_wt | 0.531 | 0.139 | 0.161±0.055 | -0.022 | 0/9 |
| entropy | labeled_marker | 0.744 | 0.502 | 0.516±0.051 | -0.014 | 5/9 |
| entropy | decoder_2 | 0.728 | 0.510 | 0.441±0.098 | +0.069 | 0/9 |
| entropy | decoder_3 | 0.732 | 0.400 | 0.277±0.098 | +0.124 | 0/9 |
| entropy | decoder_attention | 0.562 | 0.247 | 0.163±0.061 | +0.085 | 0/9 |
| entropy | hybrid_decoder | 0.687 | 0.371 | 0.377±0.047 | -0.006 | 0/9 |
| entropy | vanilla_adaptive_decoder | 0.636 | 0.453 | 0.409±0.045 | +0.044 | 0/9 |
| entropy | vanilla_adaptive_decoder_wt | 0.531 | 0.210 | 0.161±0.055 | +0.049 | 0/9 |
| least_confidence | labeled_marker | 0.744 | 0.498 | 0.518±0.050 | -0.020 | 5/9 |
| least_confidence | decoder_2 | 0.728 | 0.529 | 0.439±0.098 | +0.091 | 1/9 |
| least_confidence | decoder_3 | 0.731 | 0.420 | 0.275±0.100 | +0.145 | 0/9 |
| least_confidence | decoder_attention | 0.561 | 0.250 | 0.162±0.059 | +0.089 | 0/9 |
| least_confidence | hybrid_decoder | 0.686 | 0.375 | 0.378±0.049 | -0.002 | 0/9 |
| least_confidence | vanilla_adaptive_decoder | 0.635 | 0.456 | 0.408±0.043 | +0.047 | 0/9 |
| least_confidence | vanilla_adaptive_decoder_wt | 0.531 | 0.214 | 0.161±0.056 | +0.053 | 0/9 |
| region_bald | labeled_marker | 0.744 | 0.575 | 0.517±0.050 | +0.058 | 2/9 |
| region_bald | decoder_2 | 0.728 | 0.623 | 0.441±0.098 | +0.182 | 0/9 |
| region_bald | decoder_3 | 0.731 | 0.594 | 0.277±0.100 | +0.317 | 0/9 |
| region_bald | decoder_attention | 0.561 | 0.350 | 0.162±0.061 | +0.187 | 0/9 |
| region_bald | hybrid_decoder | 0.686 | 0.503 | 0.377±0.048 | +0.126 | 0/9 |
| region_bald | vanilla_adaptive_decoder | 0.635 | 0.466 | 0.409±0.045 | +0.057 | 0/9 |
| region_bald | vanilla_adaptive_decoder_wt | 0.531 | 0.338 | 0.161±0.055 | +0.178 | 0/9 |
