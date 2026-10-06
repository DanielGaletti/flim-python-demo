### O que prevê o ganho de anotar uma região — Conjuntivite, FLIM_lm

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.209 | [0.111, 0.306] | 10 | 0.0009 |
| least confidence | 0.200 | [0.097, 0.302] | 10 | 0.0017 |
| fração de foreground | 0.288 | [0.198, 0.377] | 10 | 0.0000 |
| Δ kernels do encoder | constante | — | 10 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.0674 | [-0.1665, 0.0317] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0795 | [-0.1968, 0.0378] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
