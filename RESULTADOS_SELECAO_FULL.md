# E1 — Seleção iterativa (pool: full)

Algoritmo 1 de arXiv:2504.20872 com o critério do passo 7 trocado.

## Curva Fβ × |T|

Melhor Fβ acumulado com |T| imagens, média ± desvio sobre (split × semente).

| critério | usa GT? | \|T\|=1 | \|T\|=2 | \|T\|=3 |
|---|---|---|---|---|
| oracle | **sim** | 0.4994±0.031 | 0.5290±0.065 | 0.6576 |
| coreset | não | 0.4994±0.031 | — | — |
| entropy | não | 0.4994±0.031 | 0.5042±0.059 | 0.5820 |

## Anotações desperdiçadas

Imagens que o critério mandou anotar, foram anotadas, pioraram o modelo e acabaram descartadas pelo backtracking. Cada uma é esforço real do especialista jogado fora.

| critério | descartadas por execução | rodadas | taxa |
|---|---|---|---|
| oracle | 2.7 | 5.0 | 53% |
| coreset | 3.0 | 5.0 | 60% |
| entropy | 2.8 | 5.0 | 57% |

## Imagens mais escolhidas

| critério | top-5 (frequência entre execuções) |
|---|---|
| oracle | `000107` (10), `000006` (9), `000010` (6), `000722` (5), `000723` (5) |
| coreset | `000107` (10), `000149` (6), `000722` (5), `000374` (5), `000723` (5) |
| entropy | `000107` (10), `000722` (5), `000006` (5), `000723` (5), `000726` (5) |
