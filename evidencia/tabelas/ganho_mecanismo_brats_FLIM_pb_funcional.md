### O que prevê o ganho de anotar uma região — BraTS, FLIM_pb

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.184 | [0.077, 0.291] | 8 | 0.0047 |
| least confidence | 0.186 | [0.079, 0.293] | 8 | 0.0045 |
| fração de foreground | 0.171 | [0.025, 0.318] | 8 | 0.0279 |
| Δ kernels do encoder | constante | — | 8 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.0208 | [-0.1867, 0.1452] | 8 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0519 | [-0.1838, 0.0800] | 8 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
