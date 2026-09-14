# E1 — Seleção iterativa (pool: real)

Algoritmo 1 de arXiv:2504.20872 com o critério do passo 7 trocado.

## Curva Fβ × |T|

Melhor Fβ acumulado com |T| imagens, média ± desvio sobre (split × semente).

| critério | usa GT? | \|T\|=1 | \|T\|=2 | \|T\|=3 | \|T\|=4 | \|T\|=5 | \|T\|=6 |
|---|---|---|---|---|---|---|---|
| oracle | **sim** | 0.5931±0.074 | 0.7036±0.080 | 0.7305±0.051 | 0.7522±0.037 | 0.7515 | — |
| badge | não | 0.5931±0.074 | 0.6355±0.088 | 0.6653±0.035 | 0.7150±0.039 | 0.7293±0.033 | — |
| coreset | não | 0.5931±0.074 | 0.6504±0.101 | 0.7067±0.064 | 0.6631±0.030 | 0.7232 | — |
| entropy | não | 0.5931±0.074 | 0.7268±0.076 | 0.7577±0.032 | 0.7450±0.018 | 0.7366 | — |
| random | não | 0.5931±0.074 | 0.6804±0.034 | 0.7298±0.029 | 0.7200±0.027 | 0.7514 | — |

## Diferença para o aleatório, e quanto do oráculo se recupera

A coluna principal é `critério − random`, sempre interpretável. A recuperação `(critério − random) / (oracle − random)` só é reportada quando o oráculo supera o aleatório por pelo menos 0.02 — abaixo disso o denominador é ruído e a razão explode para valores sem sentido.

| |T| | oracle − random | badge − random | coreset − random | entropy − random | recuperação |
|---|---|---|---|---|---|
| 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | _oráculo ≈ random_ |
| 2 | +0.0231 | -0.0449 | -0.0301 | +0.0464 | badge -194%, coreset -130%, entropy +201% |
| 3 | +0.0007 | -0.0645 | -0.0230 | +0.0280 | _oráculo ≈ random_ |
| 4 | +0.0322 | -0.0050 | -0.0569 | +0.0250 | badge -15%, coreset -177%, entropy +78% |
| 5 | +0.0001 | -0.0221 | -0.0282 | -0.0148 | _oráculo ≈ random_ |
| 6 | — | — | — | — | _oráculo ≈ random_ |

## Anotações desperdiçadas

Imagens que o critério mandou anotar, foram anotadas, pioraram o modelo e acabaram descartadas pelo backtracking. Cada uma é esforço real do especialista jogado fora.

| critério | descartadas por execução | rodadas | taxa |
|---|---|---|---|
| oracle | 2.0 | 6.0 | 33% |
| badge | 1.4 | 6.0 | 24% |
| coreset | 1.9 | 6.0 | 31% |
| entropy | 1.9 | 6.0 | 31% |
| random | 2.2 | 6.0 | 37% |

## Imagens mais escolhidas

| critério | top-5 (frequência entre execuções) |
|---|---|
| oracle | `000017` (30), `000659` (18), `000675` (18), `000332` (18), `000004` (12) |
| badge | `000043` (30), `000675` (27), `000659` (20), `000332` (18), `000601` (17) |
| coreset | `000043` (30), `000659` (18), `000675` (18), `000332` (18), `000601` (15) |
| entropy | `000017` (29), `000204` (28), `000659` (18), `000675` (18), `000332` (18) |
| random | `000156` (27), `000659` (18), `000675` (18), `000332` (18), `000941` (15) |
