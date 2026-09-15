### Onde anotar, com orçamento fixo em pixels: Fβ e as duas diferenças

| braço | px/imagem | decoder | n | Fβ médio | Δ vs uniforme | p | Δ vs balanceado | p  |
|---|---|---|---|---|---|---|---|---|
| regiao_incerteza | 4000 | FLIM_lm | 2 | 0.016 | -0.334 | 0.206 | 0.016 | 0.500 |
| regiao_incerteza | 4000 | FLIM_pb | 2 | 0.484 | -0.146 | 0.298 | 0.481 | 0.008 |
| regiao_aleatoria | 4000 | FLIM_lm | 2 | 0.382 | 0.032 | 0.868 | 0.382 | 0.092 |
| regiao_aleatoria | 4000 | FLIM_pb | 2 | 0.609 | -0.021 | 0.745 | 0.607 | 0.019 |
| regiao_borda | 4000 | FLIM_lm | 2 | 0.313 | -0.037 | 0.525 | 0.313 | 0.113 |
| regiao_borda | 4000 | FLIM_pb | 2 | 0.652 | 0.022 | 0.742 | 0.649 | 0.017 |
| uniforme_balanceado | 4000 | FLIM_lm | 2 | 0.000 | -0.350 | 0.171 | — | — |
| uniforme_balanceado | 4000 | FLIM_pb | 2 | 0.002 | -0.627 | 0.069 | — | — |

Orçamento em **pixels anotados por imagem**, não em imagens: o que limita o FLIM é a quantidade de pixels marcados — a mesma imagem rende 54 kernels por camada com traço esparso e 200 com traço denso. Todos os braços gastam exatamente o mesmo número de pixels, nas MESMAS imagens (as do artigo); o que muda é onde eles caem. `uniforme` é o FLIM de hoje. `uniforme_balanceado` tem a mesma proporção objeto/fundo que a incerteza produz, com os pixels sorteados — é ele que separa *onde se anotou* de *quanto de cada classe se anotou*, porque escolher região incerta escolhe junto muito mais foreground (~73% contra ~16%, medido antes de rodar). `regiao_borda` usa o gabarito e é teto, não método. Diferenças pareadas por semente; p de t pareado bicaudal. Campanha `onde_marcar/k3/teste60/seg300`.
