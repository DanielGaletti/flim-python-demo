### Técnicas de Active Learning contra o sorteio

| técnica | nível | Fβ (células pareadas) | n pares | Δ vs random | p | Δ vs controle do nível | veredito |
|---|---|---|---|---|---|---|---|
| badge | imagem | 0.430 | 95 | **-0.019** | 0.0105 | — | suportado |
| coreset | imagem | 0.450 | 94 | 0.003 | 0.7144 | — | sem suporte |
| entropy | imagem | 0.457 | 211 | **0.046** | 0.0000 | — | suportado |
| least_confidence | imagem | 0.424 | 135 | **0.039** | 0.0006 | — | suportado |
| margin | imagem | 0.451 | 72 | 0.023 | 0.2110 | — | sem suporte |
| random | imagem | — | — | — | — | — | piso |
| oracle | imagem (usa GT) | 0.686 | 31 | 0.011 | 0.0889 | — | indicativo |
| random_region | região | 0.442 | 252 | **0.108** | 0.0000 | — | suportado |
| region_bald | região | 0.462 | 135 | **0.077** | 0.0000 | 0.050 | suportado |
| region_entropy | região | 0.437 | 66 | 0.014 | 0.4295 | — | sem suporte |

`random` é o piso — sem critério nenhum. O pareamento casa split, decoder, orçamento e semente **dentro da mesma família de experimento**; a média mostrada vem dessas mesmas células, para que média e Δ falem do mesmo conjunto. **Toda técnica que rodou está aqui, inclusive as que perderam do sorteio** — no experimento de encoder retreinado isso aconteceu com as quatro de imagem. Negrito só com p < 0,05. Para técnicas de REGIÃO, leia a coluna do controle de nível (`random_region`): o Δ contra `random` soma seleção e geometria, e a geometria pesa ~2× a seleção neste projeto.
