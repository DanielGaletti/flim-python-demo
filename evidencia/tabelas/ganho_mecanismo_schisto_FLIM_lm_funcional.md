### O que prevê o ganho de anotar uma região — Parasitas, FLIM_lm

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.427 | [0.292, 0.562] | 10 | 0.0001 |
| least confidence | 0.427 | [0.292, 0.562] | 10 | 0.0001 |
| fração de foreground | 0.308 | [0.221, 0.395] | 10 | 0.0000 |
| Δ kernels do encoder | -0.299 | [-0.429, -0.170] | 10 | 0.0005 |
| (Δ Fβ médio ao anotar região de objeto) | -0.0740 | [-0.1661, 0.0181] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.1077 | [-0.1816, -0.0339] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
