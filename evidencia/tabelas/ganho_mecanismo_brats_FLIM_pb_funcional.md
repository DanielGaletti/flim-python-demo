### O que prevê o ganho de anotar uma região — BraTS, FLIM_pb

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.192 | [0.091, 0.293] | 8 | 0.0028 |
| least confidence | 0.194 | [0.093, 0.295] | 8 | 0.0027 |
| fração de foreground | 0.197 | [0.066, 0.329] | 8 | 0.0093 |
| Δ kernels do encoder | constante | — | 8 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.0246 | [-0.1854, 0.1361] | 8 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0570 | [-0.2016, 0.0877] | 8 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
