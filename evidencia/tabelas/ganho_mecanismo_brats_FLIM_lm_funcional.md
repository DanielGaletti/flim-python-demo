### O que prevê o ganho de anotar uma região — BraTS, FLIM_lm

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.147 | [-0.019, 0.313] | 7 | 0.0734 |
| least confidence | 0.150 | [-0.016, 0.315] | 7 | 0.0695 |
| fração de foreground | 0.119 | [-0.168, 0.407] | 7 | 0.3481 |
| Δ kernels do encoder | constante | — | 7 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.1812 | [-0.2590, -0.1034] | 7 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.2113 | [-0.3618, -0.0607] | 7 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
