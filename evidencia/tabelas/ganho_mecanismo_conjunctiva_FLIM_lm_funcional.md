### O que prevê o ganho de anotar uma região — Conjuntivite, FLIM_lm

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.192 | [0.077, 0.307] | 10 | 0.0043 |
| least confidence | 0.188 | [0.064, 0.312] | 10 | 0.0076 |
| fração de foreground | 0.277 | [0.169, 0.384] | 10 | 0.0002 |
| Δ kernels do encoder | constante | — | 10 | não estimável |
| (Δ Fβ médio ao anotar região de objeto) | -0.0612 | [-0.1625, 0.0401] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.0786 | [-0.2014, 0.0441] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
