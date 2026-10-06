### O que prevê o ganho de anotar uma região — Parasitas, FLIM_pb

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | -0.045 | [-0.254, 0.165] | 10 | 0.6400 |
| least confidence | -0.043 | [-0.253, 0.168] | 10 | 0.6562 |
| fração de foreground | 0.031 | [-0.190, 0.252] | 10 | 0.7597 |
| Δ kernels do encoder | -0.021 | [-0.130, 0.089] | 10 | 0.6830 |
| (Δ Fβ médio ao anotar região de objeto) | -0.0360 | [-0.0607, -0.0113] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0389 | [-0.0785, 0.0007] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
