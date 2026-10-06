### O que prevê o ganho de anotar uma região — BraTS, FLIM_lm

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.167 | [-0.000, 0.333] | 7 | 0.0501 |
| least confidence | 0.169 | [0.001, 0.337] | 7 | 0.0491 |
| fração de foreground | 0.137 | [-0.150, 0.425] | 7 | 0.2868 |
| Δ kernels do encoder | constante | — | 7 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.1831 | [-0.2520, -0.1141] | 7 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.2246 | [-0.3690, -0.0803] | 7 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
