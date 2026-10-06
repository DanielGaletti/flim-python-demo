### O que prevê o ganho de anotar uma região — Conjuntivite, FLIM_pb

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.174 | [0.025, 0.324] | 10 | 0.0271 |
| least confidence | 0.180 | [0.028, 0.332] | 10 | 0.0252 |
| fração de foreground | 0.134 | [0.009, 0.259] | 10 | 0.0381 |
| Δ kernels do encoder | constante | — | 10 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.0517 | [-0.1289, 0.0254] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0514 | [-0.1172, 0.0144] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
