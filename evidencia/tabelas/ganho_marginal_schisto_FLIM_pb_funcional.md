### Ganho marginal de um marcador adicional — Parasitas, FLIM_pb, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 10 | 0.0979 | [0.0725, 0.1233] | 0.0457 | 0.1638 | 0.1351 | 0.0000 |
| argmax entropia (o AL) | 10 | -0.0276 | [-0.0803, 0.0252] | -0.1287 | 0.0744 | 0.0096 | 0.7313 |
| argmax least confidence | 10 | -0.0207 | [-0.0704, 0.0291] | -0.1268 | 0.0877 | 0.0165 | 0.5164 |
| candidato sorteado (média) | 10 | -0.0371 | [-0.0636, -0.0107] | -0.0932 | 0.0139 | — | — |
| pior região | 10 | -0.2447 | [-0.3050, -0.1844] | -0.4133 | -0.1540 | -0.2075 | 0.0000 |
| imagem nova inteira (orçamento MAIOR) | 10 | -0.0002 | [-0.0111, 0.0106] | -0.0313 | 0.0191 | 0.0369 | 0.0113 |
| sorteio uniforme (reponderado, sem GT) | 10 | -0.0388 | [-0.0762, -0.0014] | -0.1236 | 0.0396 | -0.0017 | 0.8359 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
