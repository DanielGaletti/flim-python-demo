# AL por imagem vs AL por região — Schistosoma (Parasites)

Dois níveis de granularidade da anotação ativa. O de imagem escolhe QUAIS imagens anotar; o de região escolhe, dentro da imagem, ONDE anotar.

## Experimento B — encoder fixo, decoder 1×1 por backprop

ΔFβ **pareado**: AL e aleatório do mesmo índice compartilham a semente de inicialização, então a diferença isola a seleção.

| nível | método | K | AL | aleatório | ΔFβ pareado | p | colapsos |
|---|---|---|---|---|---|---|---|
| imagem | badge | 3 | 0.497 | 0.484 | +0.0130 | 0.693 | 0/15 |
| imagem | badge | 5 | 0.503 | 0.502 | +0.0007 | 0.979 | 0/15 |
| imagem | badge | 10 | 0.563 | 0.483 | +0.0803 | <0.001 | 3/15 ⚠ |
| imagem | coreset | 3 | 0.542 | 0.484 | +0.0579 | 0.071 | 4/15 ⚠ |
| imagem | coreset | 5 | 0.552 | 0.502 | +0.0493 | 0.063 | 3/15 ⚠ |
| imagem | coreset | 10 | 0.575 | 0.483 | +0.0920 | <0.001 | 2/15 ⚠ |
| imagem | entropy | 3 | 0.409 | 0.484 | -0.0755 | 0.012 | 0/15 |
| imagem | entropy | 5 | 0.423 | 0.502 | -0.0795 | 0.076 | 0/15 |
| imagem | entropy | 10 | 0.526 | 0.483 | +0.0426 | 0.130 | 0/15 |
| região | region_entropy | 3 | 0.293 | 0.446 | -0.1529 | <0.001 | 0/15 |
| região | region_entropy | 5 | 0.362 | 0.420 | -0.0577 | 0.266 | 0/15 |
| região | region_entropy | 10 | 0.507 | 0.427 | +0.0793 | 0.048 | 0/15 |

## Experimento A — encoder FLIM retreinado

Decoder de referência: `labeled_marker` (o do paper). Δ = AL − aleatório, média sobre splits e budgets.

| nível | método | AL | aleatório | Δ | p | colapsos |
|---|---|---|---|---|---|---|
| imagem | badge | 0.518 | 0.518 | -0.000 | 0.045 | 5/9 ⚠ |
| imagem | coreset | 0.504 | 0.521 | -0.017 | 0.006 | 6/9 ⚠ |
| imagem | entropy | 0.502 | 0.516 | -0.014 | 0.007 | 5/9 ⚠ |
| imagem | least_confidence | 0.498 | 0.518 | -0.020 | 0.004 | 5/9 ⚠ |
| região | region_bald | 0.575 | 0.517 | +0.058 | 0.622 | 2/9 ⚠ |

⚠ = alguma execução do braço AL colapsou (predição toda-fundo, Fβ=DICE=IoU≈0.479). O 'ganho' dessas linhas não é aprendizado.

### Decomposição do ganho dos métodos de região

`seleção` isola quais imagens foram escolhidas; `geometria` isola como os seeds foram desenhados. Os dois braços aleatórios usam a mesma semente, logo sorteiam as mesmas imagens — só muda a anotação.

| método | decoder | AL | random_região | random | seleção | geometria |
|---|---|---|---|---|---|---|
| region_bald | labeled_marker | 0.575 | 0.605 | 0.517 | -0.030 | +0.088 |
| region_bald | decoder_2 | 0.623 | 0.586 | 0.441 | +0.036 | +0.145 |
| region_bald | decoder_3 | 0.594 | 0.560 | 0.277 | +0.033 | +0.284 |
| region_bald | decoder_attention | 0.350 | 0.259 | 0.162 | +0.091 | +0.096 |
| region_bald | hybrid_decoder | 0.503 | 0.425 | 0.377 | +0.079 | +0.048 |
| region_bald | vanilla_adaptive_decoder | 0.466 | 0.436 | 0.409 | +0.030 | +0.027 |
| region_bald | vanilla_adaptive_decoder_wt | 0.338 | 0.226 | 0.161 | +0.112 | +0.065 |
| **média** | | | | | **+0.050** | **+0.108** |

> A geometria dos seeds pesa **2.1×** a seleção de imagens. O ganho dos métodos de região vem sobretudo de COMO anotar, não de QUAIS imagens anotar.
