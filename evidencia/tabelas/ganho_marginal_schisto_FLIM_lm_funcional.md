### Ganho marginal de um marcador adicional — Parasitas, FLIM_lm, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 10 | 0.2191 | [0.0851, 0.3530] | 0.0300 | 0.5777 | 0.3089 | 0.0000 |
| argmax entropia (o AL) | 10 | 0.0843 | [-0.0577, 0.2264] | -0.1196 | 0.4676 | 0.1741 | 0.0157 |
| argmax least confidence | 10 | 0.1078 | [-0.0267, 0.2423] | -0.1168 | 0.4647 | 0.1976 | 0.0071 |
| candidato sorteado (média) | 10 | -0.0898 | [-0.1697, -0.0099] | -0.2672 | 0.1461 | — | — |
| pior região | 10 | -0.3490 | [-0.4764, -0.2216] | -0.6091 | -0.0433 | -0.2592 | 0.0000 |
| imagem nova inteira (orçamento MAIOR) | 10 | 0.0506 | [-0.0402, 0.1414] | -0.1427 | 0.3172 | 0.1404 | 0.0002 |
| sorteio uniforme (reponderado, sem GT) | 10 | -0.1050 | [-0.1789, -0.0311] | -0.3038 | 0.1034 | -0.0152 | 0.1930 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
