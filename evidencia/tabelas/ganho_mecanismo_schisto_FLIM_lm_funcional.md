### O que prevê o ganho de anotar uma região — Parasitas, FLIM_lm

| variável | ρ de Spearman com Δ Fβ | IC95% | n (sementes) | p (ρ ≠ 0) |
|---|---|---|---|---|
| entropia da região | 0.424 | [0.271, 0.576] | 10 | 0.0001 |
| least confidence | 0.424 | [0.271, 0.577] | 10 | 0.0001 |
| fração de foreground | 0.288 | [0.184, 0.392] | 10 | 0.0001 |
| Δ kernels do encoder | -0.317 | [-0.463, -0.171] | 10 | 0.0008 |
| (Δ Fβ médio ao anotar região de objeto) | -0.0741 | [-0.1616, 0.0134] | 10 | — |
| (Δ Fβ médio ao anotar região de fundo) | -0.1023 | [-0.1622, -0.0425] | 10 | — |

ρ calculado **dentro** de cada semente, sobre os candidatos sorteados daquela semente; são os ρ por semente que entram no teste, com n = sementes. Agregar candidatos de sementes diferentes num ρ único trataria observações correlacionadas como independentes. `entropia da região` é o escore que o Active Learning usa para decidir — um ρ indistinguível de zero significa que o escore não ordena as regiões por utilidade, e é isso que explicaria todos os resultados negativos das campanhas anteriores. `Δ kernels` mede capacidade, não posição: é quantos filtros a mais o encoder extraiu com o marcador extra.
