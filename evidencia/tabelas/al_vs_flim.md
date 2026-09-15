### Melhora sobre o FLIM do artigo e sobre o sorteio, pareada por semente

| critério | K | decoder | n | Fβ médio | Δ vs artigo | IC95% | p | Δ vs sorteio | p  |
|---|---|---|---|---|---|---|---|---|---|
| medoide | 3 | FLIM_lm | 5 | 0.630 | 0.154 | [+0.084, +0.224] | 0.004 | 0.095 | 0.218 |
| medoide | 3 | FLIM_pb | 5 | 0.561 | -0.089 | [-0.230, +0.052] | 0.154 | -0.033 | 0.663 |
| medoide | 5 | FLIM_lm | 5 | 0.694 | 0.100 | [-0.004, +0.204] | 0.055 | 0.144 | 0.053 |
| medoide | 5 | FLIM_pb | 5 | 0.542 | -0.136 | [-0.230, -0.041] | 0.016 | -0.121 | 0.050 |
| coreset | 3 | FLIM_lm | 5 | 0.562 | 0.087 | [-0.031, +0.205] | 0.109 | 0.028 | 0.683 |
| coreset | 3 | FLIM_pb | 5 | 0.542 | -0.108 | [-0.225, +0.009] | 0.063 | -0.052 | 0.080 |
| coreset | 5 | FLIM_lm | 5 | 0.654 | 0.060 | [-0.074, +0.194] | 0.283 | 0.103 | 0.164 |
| coreset | 5 | FLIM_pb | 5 | 0.576 | -0.101 | [-0.213, +0.011] | 0.066 | -0.087 | 0.008 |
| entropy | 3 | FLIM_lm | 5 | 0.421 | -0.054 | [-0.417, +0.309] | 0.699 | -0.113 | 0.506 |
| entropy | 3 | FLIM_pb | 5 | 0.631 | -0.019 | [-0.147, +0.109] | 0.699 | 0.037 | 0.501 |
| entropy | 5 | FLIM_lm | 5 | 0.562 | -0.032 | [-0.251, +0.186] | 0.703 | 0.011 | 0.929 |
| entropy | 5 | FLIM_pb | 5 | 0.639 | -0.038 | [-0.104, +0.027] | 0.178 | -0.024 | 0.362 |
| least_confidence | 3 | FLIM_lm | 5 | 0.407 | -0.068 | [-0.422, +0.286] | 0.622 | -0.127 | 0.435 |
| least_confidence | 3 | FLIM_pb | 5 | 0.640 | -0.010 | [-0.116, +0.097] | 0.816 | 0.047 | 0.352 |
| least_confidence | 5 | FLIM_lm | 5 | 0.543 | -0.051 | [-0.269, +0.167] | 0.552 | -0.008 | 0.951 |
| least_confidence | 5 | FLIM_pb | 5 | 0.619 | -0.058 | [-0.124, +0.008] | 0.071 | -0.044 | 0.298 |

Diferença pareada por semente: dentro de cada semente os braços compartilham partição, conjunto de teste e gerador de traços, então a diferença isola a seleção. `flim_paper` é o braço do artigo; `random` é o sorteio. Bater o artigo sem bater o sorteio não demonstra seleção — demonstra que as imagens do artigo não eram especiais. p vem de t pareado bicaudal; com poucas sementes o teste tem pouco poder, e p alto significa **não decidido**, não 'igual'. Uma única campanha (`tabela_k_por_modelo/pool40/teste60/iterativa`).
