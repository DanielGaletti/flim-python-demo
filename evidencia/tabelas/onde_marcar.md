### Onde anotar, com orçamento fixo em pixels: Fβ e as duas diferenças

| braço | px/imagem | decoder | n | Fβ médio | Δ vs uniforme | p | Δ vs balanceado | p  |
|---|---|---|---|---|---|---|---|---|
| regiao_incerteza | 2000 | FLIM_lm | 5 | 0.247 | -0.215 | 0.022 | -0.256 | 0.013 |
| regiao_incerteza | 2000 | FLIM_pb | 5 | 0.476 | -0.167 | 0.019 | 0.209 | 0.051 |
| regiao_incerteza | 4000 | FLIM_lm | 5 | 0.055 | -0.283 | 0.022 | -0.054 | 0.263 |
| regiao_incerteza | 4000 | FLIM_pb | 5 | 0.488 | -0.142 | 0.023 | 0.305 | 0.037 |
| regiao_incerteza | 8000 | FLIM_lm | 5 | 0.000 | -0.011 | 0.264 | 0.000 | 1.000 |
| regiao_incerteza | 8000 | FLIM_pb | 5 | 0.410 | -0.189 | 0.129 | 0.239 | 0.121 |
| regiao_aleatoria | 2000 | FLIM_lm | 5 | 0.437 | -0.024 | 0.566 | -0.066 | 0.489 |
| regiao_aleatoria | 2000 | FLIM_pb | 5 | 0.599 | -0.044 | 0.048 | 0.331 | 0.002 |
| regiao_aleatoria | 4000 | FLIM_lm | 5 | 0.387 | 0.049 | 0.412 | 0.278 | 0.037 |
| regiao_aleatoria | 4000 | FLIM_pb | 5 | 0.622 | -0.008 | 0.688 | 0.439 | 0.006 |
| regiao_aleatoria | 8000 | FLIM_lm | 5 | 0.070 | 0.059 | 0.303 | 0.070 | 0.229 |
| regiao_aleatoria | 8000 | FLIM_pb | 5 | 0.602 | 0.003 | 0.974 | 0.431 | 0.003 |
| regiao_borda | 2000 | FLIM_lm | 5 | 0.501 | 0.039 | 0.389 | -0.002 | 0.968 |
| regiao_borda | 2000 | FLIM_pb | 5 | 0.620 | -0.024 | 0.247 | 0.352 | 0.002 |
| regiao_borda | 4000 | FLIM_lm | 5 | 0.391 | 0.052 | 0.490 | 0.281 | 0.003 |
| regiao_borda | 4000 | FLIM_pb | 5 | 0.631 | 0.001 | 0.960 | 0.448 | 0.008 |
| regiao_borda | 8000 | FLIM_lm | 5 | 0.056 | 0.045 | 0.375 | 0.056 | 0.352 |
| regiao_borda | 8000 | FLIM_pb | 5 | 0.477 | -0.122 | 0.402 | 0.306 | 0.122 |
| uniforme_balanceado | 2000 | FLIM_lm | 5 | 0.503 | 0.042 | 0.606 | — | — |
| uniforme_balanceado | 2000 | FLIM_pb | 5 | 0.268 | -0.376 | 0.001 | — | — |
| uniforme_balanceado | 4000 | FLIM_lm | 5 | 0.109 | -0.229 | 0.111 | — | — |
| uniforme_balanceado | 4000 | FLIM_pb | 5 | 0.183 | -0.447 | 0.009 | — | — |
| uniforme_balanceado | 8000 | FLIM_lm | 5 | 0.000 | -0.011 | 0.264 | — | — |
| uniforme_balanceado | 8000 | FLIM_pb | 5 | 0.171 | -0.428 | 0.008 | — | — |

Orçamento em **pixels anotados por imagem**, não em imagens: o que limita o FLIM é a quantidade de pixels marcados — a mesma imagem rende 54 kernels por camada com traço esparso e 200 com traço denso. Todos os braços gastam exatamente o mesmo número de pixels, nas MESMAS imagens (as do artigo); o que muda é onde eles caem. `uniforme` é o FLIM de hoje. `uniforme_balanceado` tem a mesma proporção objeto/fundo que a incerteza produz, com os pixels sorteados — é ele que separa *onde se anotou* de *quanto de cada classe se anotou*, porque escolher região incerta escolhe junto muito mais foreground (~73% contra ~16%, medido antes de rodar). `regiao_borda` usa o gabarito e é teto, não método. Diferenças pareadas por semente; p de t pareado bicaudal. Campanha `onde_marcar/k3/teste60/seg300`.
