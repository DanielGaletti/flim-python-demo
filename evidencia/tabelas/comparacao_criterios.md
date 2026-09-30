### Técnicas de Active Learning contra o sorteio

| técnica | nível | Fβ (células pareadas) | n pares | Δ vs random | p | Δ vs FLIM (K=3) | p (FLIM) | Δ vs controle do nível | veredito |
|---|---|---|---|---|---|---|---|---|---|
| badge | imagem | 0.430 | 95 | **-0.019** | 0.0105 | — | — | — | suportado |
| coreset | imagem | 0.458 | 234 | -0.011 | 0.3354 | — | — | — | sem suporte |
| entropy | imagem | 0.457 | 351 | 0.016 | 0.0728 | — | — | — | indicativo |
| least_confidence | imagem | 0.419 | 249 | -0.000 | 0.9991 | — | — | — | sem suporte |
| margin | imagem | 0.451 | 72 | 0.023 | 0.2110 | — | — | — | sem suporte |
| random | imagem | — | — | — | — | — | — | — | piso |
| oracle | imagem (usa GT) | 0.514 | 103 | -0.020 | 0.4456 | — | — | — | sem suporte |
| flim_3img | linha de base | — | 0 | — | — | — | — | — | sem pares suficientes |
| random_region | região | 0.442 | 252 | **0.108** | 0.0000 | — | — | — | suportado |
| region_bald | região | 0.462 | 135 | **0.077** | 0.0000 | — | — | 0.050 | suportado |
| region_entropy | região | 0.437 | 66 | 0.014 | 0.4295 | — | — | — | sem suporte |
| 1_imagens | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| 2_imagens | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| 3_imagens | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| 5_imagens | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| 8_imagens | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| al_regiao | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| aleatorio_objeto | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| flim_paper | ? | 0.467 | 69 | -0.015 | 0.4074 | — | — | — | sem suporte |
| flim_puro | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| medoide | ? | 0.482 | 94 | **-0.043** | 0.0054 | — | — | — | suportado |
| regiao_aleatoria | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| regiao_borda | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| regiao_confusa | ? | 0.396 | 71 | -0.075 | 0.0766 | — | — | — | indicativo |
| regiao_incerteza | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| uniforme | ? | — | 0 | — | — | — | — | — | sem pares suficientes |
| uniforme_balanceado | ? | — | 0 | — | — | — | — | — | sem pares suficientes |

`random` é o piso — sem critério nenhum. O pareamento casa split, decoder, orçamento e semente **dentro da mesma família de experimento**; a média mostrada vem dessas mesmas células, para que média e Δ falem do mesmo conjunto. **Toda técnica que rodou está aqui, inclusive as que perderam do sorteio** — no experimento de encoder retreinado isso aconteceu com as quatro de imagem. Negrito só com p < 0,05. Para técnicas de REGIÃO, leia a coluna do controle de nível (`random_region`): o Δ contra `random` soma seleção e geometria, e a geometria pesa ~2× a seleção neste projeto. **`flim_3img` é o FLIM do artigo** — as 3 imagens que os autores escolheram, sem seleção automática. A coluna `Δ vs FLIM` responde à pergunta central: aplicar AL melhora sobre isso? Ela só aparece em orçamento casado (K=3), porque o braço do artigo só existe com 3 imagens; comparar AL com 10 imagens contra FLIM com 3 mediria orçamento, não seleção. **Coluna toda vazia significa comparação IMPOSSÍVEL com os dados existentes, não ausente por descuido**: o braço do artigo usa os 31 markers REAIS do especialista, e os braços de AL sobre o pool completo usam sintéticos, porque marker real só existe para 31 imagens. `marker_origem` entra no pareamento para que esse par não se forme — ele mediria quem desenhou o traço e apresentaria como efeito da seleção. Para responder à pergunta, rode `scripts/varrer_criterios.py --modo imagem`: marker real nos dois lados, com `oracle` (o passo 7 do Algoritmo 1) como referência.
