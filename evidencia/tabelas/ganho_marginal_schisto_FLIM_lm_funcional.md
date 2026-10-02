### Ganho marginal de um marcador adicional — Parasitas, FLIM_lm, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 10 | 0.2156 | [0.0817, 0.3494] | 0.0300 | 0.5777 | 0.3025 | 0.0000 |
| argmax entropia (o AL) | 10 | 0.0843 | [-0.0577, 0.2264] | -0.1196 | 0.4676 | 0.1712 | 0.0175 |
| argmax least confidence | 10 | 0.1078 | [-0.0267, 0.2423] | -0.1168 | 0.4647 | 0.1947 | 0.0076 |
| candidato sorteado (média) | 10 | -0.0869 | [-0.1606, -0.0132] | -0.2315 | 0.1348 | — | — |
| pior região | 10 | -0.3490 | [-0.4764, -0.2216] | -0.6091 | -0.0433 | -0.2621 | 0.0000 |
| imagem nova inteira (orçamento MAIOR) | 10 | 0.0506 | [-0.0402, 0.1414] | -0.1427 | 0.3172 | 0.1375 | 0.0007 |
| sorteio uniforme (reponderado, sem GT) | 10 | -0.1002 | [-0.1615, -0.0389] | -0.2392 | 0.0787 | -0.0133 | 0.1356 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
