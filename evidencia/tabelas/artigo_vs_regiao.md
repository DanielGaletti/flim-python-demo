### Tabela do artigo × cliques realocados por AL × realocados ao acaso

| decoder | n | Fβ artigo | Fβ AL região | Fβ aleatório | Δ AL−artigo | p | Δ aleat−artigo | p  | Δ AL−aleat | p   | Fβ publicado (A) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLIM_lm | 6 | 0.744 | 0.622 | 0.669 | -0.122 | 0.0370 | -0.075 | 0.0480 | -0.047 | 0.3088 | 0.860 |
| FLIM_pb | 6 | 0.720 | 0.494 | 0.537 | -0.225 | 0.0140 | -0.182 | 0.0240 | -0.043 | 0.4223 | 0.857 |
| FLIM_mb | 6 | 0.718 | 0.502 | 0.544 | -0.217 | 0.0106 | -0.174 | 0.0121 | -0.042 | 0.3997 | 0.843 |
| FLIM_at | 6 | 0.518 | 0.303 | 0.354 | -0.214 | 0.0051 | -0.164 | 0.0015 | -0.050 | 0.1178 | 0.740 |
| FLIM_ts | 6 | 0.591 | 0.466 | 0.491 | -0.124 | 0.0031 | -0.099 | 0.0000 | -0.025 | 0.2922 | 0.747 |
| FLIM_lt | 6 | 0.649 | 0.549 | 0.575 | -0.100 | 0.0043 | -0.075 | 0.0238 | -0.026 | 0.2467 | 0.810 |
| FLIM_ts* | 6 | 0.506 | 0.349 | 0.391 | -0.158 | 0.0029 | -0.115 | 0.0005 | -0.042 | 0.1427 | — |

Mesmas imagens — as que os especialistas A e B anotaram —, mesmo número de cliques e mesmo traço de fundo (copiado, não regerado) nos três braços. Só muda para onde vão os cliques de objeto. **Não há seleção de imagem: é seleção de região.** As três diferenças decompõem o efeito — `aleat−artigo` isola o ato de tirar o clique da borda, e `AL−aleat` isola a escolha do Active Learning. Sem a terceira, um Δ negativo na primeira seria creditado ao AL quando pode ser todo da segunda. Cada célula é um par (usuário, split), n = 6, pareado. `Fβ publicado (A)` vem da Tabela III do artigo e **não** é comparável com a coluna reproduzida: o artigo aplica Dynamic Trees, cujo binário não executa nesta máquina. Avaliação em 250 imagens de Z₁\T, as mesmas em todos os braços.
