### O que prevê o ganho de anotar uma região — Parasitas, FLIM_pb

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | -0.026 | [-0.224, 0.172] | 10 | 0.7758 |
| least confidence | -0.022 | [-0.222, 0.178] | 10 | 0.8076 |
| fração de foreground | 0.019 | [-0.190, 0.228] | 10 | 0.8394 |
| Δ kernels do encoder | -0.038 | [-0.176, 0.100] | 10 | 0.5530 |
| (Δ Fβ médio ao anotar região de objeto) | -0.0364 | [-0.0666, -0.0062] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0357 | [-0.0720, 0.0005] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
