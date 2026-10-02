### Ganho marginal de um marcador adicional — BraTS, FLIM_pb, base colapsada (Fβ₀=0)

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 2 | 0.7329 | [0.3158, 1.1500] | 0.7001 | 0.7657 | 0.5479 | 0.0358 |
| argmax entropia (o AL) | 2 | 0.0735 | [-0.8609, 1.0080] | 0.0000 | 0.1471 | -0.1114 | 0.5657 |
| argmax least confidence | 2 | 0.1764 | [-1.9129, 2.2658] | 0.0120 | 0.3409 | -0.0085 | 0.9762 |
| candidato sorteado (média) | 2 | 0.1850 | [-0.6235, 0.9935] | 0.1213 | 0.2486 | — | — |
| pior região | 2 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | -0.1850 | 0.2109 |
| imagem nova inteira (orçamento MAIOR) | 2 | 0.2811 | [-2.5822, 3.1444] | 0.0557 | 0.5064 | 0.0961 | 0.6586 |
| sorteio uniforme (reponderado, sem GT) | 2 | 0.0940 | [-0.2791, 0.4672] | 0.0647 | 0.1234 | -0.0909 | 0.2294 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base colapsada (Fβ₀=0)**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
