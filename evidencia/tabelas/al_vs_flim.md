### Melhora sobre o FLIM do artigo e sobre o sorteio, pareada por semente

| critério | K | decoder | n | Fβ médio | Δ vs artigo | IC95% | p | Δ vs sorteio | p  |
|---|---|---|---|---|---|---|---|---|---|
| coreset | 3 | FLIM_lm | 10 | 0.220 | — | — | — | -0.110 | 0.324 |
| coreset | 3 | FLIM_pb | 10 | 0.415 | — | — | — | -0.053 | 0.557 |
| coreset | 5 | FLIM_lm | 10 | 0.352 | — | — | — | -0.037 | 0.840 |
| coreset | 5 | FLIM_pb | 10 | 0.466 | — | — | — | -0.095 | 0.373 |
| entropy | 3 | FLIM_lm | 10 | 0.016 | — | — | — | -0.314 | 0.015 |
| entropy | 3 | FLIM_pb | 10 | 0.149 | — | — | — | -0.319 | 0.004 |
| entropy | 5 | FLIM_lm | 10 | 0.284 | — | — | — | -0.105 | 0.536 |
| entropy | 5 | FLIM_pb | 10 | 0.407 | — | — | — | -0.154 | 0.262 |
| least_confidence | 3 | FLIM_lm | 10 | 0.049 | — | — | — | -0.281 | 0.044 |
| least_confidence | 3 | FLIM_pb | 10 | 0.192 | — | — | — | -0.276 | 0.012 |
| least_confidence | 5 | FLIM_lm | 10 | 0.282 | — | — | — | -0.107 | 0.427 |
| least_confidence | 5 | FLIM_pb | 10 | 0.406 | — | — | — | -0.155 | 0.199 |

Diferença pareada por semente: dentro de cada semente os braços compartilham partição, conjunto de teste e gerador de traços, então a diferença isola a seleção. `flim_paper` é o braço do artigo; `random` é o sorteio. Bater o artigo sem bater o sorteio não demonstra seleção — demonstra que as imagens do artigo não eram especiais. p vem de t pareado bicaudal; com poucas sementes o teste tem pouco poder, e p alto significa **não decidido**, não 'igual'. Uma única campanha (`tabela_k_por_modelo/pool40/teste60/iterativa/estabilidade`).
