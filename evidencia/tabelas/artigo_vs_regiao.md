### Tabela do artigo reproduzida × os mesmos cliques realocados por AL de região

| decoder | n | Fβ artigo | Fβ AL região | Δ Fβ | p | MAE artigo | MAE AL região | Fβ publicado (A) |
|---|---|---|---|---|---|---|---|---|
| FLIM_lm | 6 | 0.744 | 0.622 | -0.122 | 0.0370 | 0.0096 | 0.0152 | 0.860 |
| FLIM_pb | 6 | 0.720 | 0.494 | -0.225 | 0.0140 | 0.0108 | 0.0248 | 0.857 |
| FLIM_mb | 6 | 0.718 | 0.502 | -0.217 | 0.0106 | 0.0110 | 0.0236 | 0.843 |
| FLIM_at | 6 | 0.518 | 0.303 | -0.214 | 0.0051 | 0.0184 | 0.0419 | 0.740 |
| FLIM_ts | 6 | 0.591 | 0.466 | -0.124 | 0.0031 | 0.0136 | 0.0191 | 0.747 |
| FLIM_lt | 6 | 0.649 | 0.549 | -0.100 | 0.0043 | 0.0120 | 0.0154 | 0.810 |
| FLIM_ts* | 6 | 0.506 | 0.349 | -0.158 | 0.0029 | 0.0195 | 0.0336 | — |

Mesmas imagens — as que os especialistas A e B anotaram — e o MESMO número de cliques nos dois braços; o traço de fundo é copiado, não regerado. A única diferença é a posição dos cliques de objeto: onde o especialista os pôs, ou onde o modelo está mais incerto. **Não há seleção de imagem**: é seleção de região. Cada célula é um par (usuário, split), n = 6. Δ e p são pareados por esse par. `Fβ publicado (A)` é o valor da Tabela III do artigo para o usuário A, transcrito — ele **não** é comparável com a coluna reproduzida, porque o artigo aplica Dynamic Trees e o binário não executa nesta máquina; está ali só para situar a ordem de grandeza. A avaliação usa 250 imagens de Z₁\T, as mesmas em todos os braços e splits.
