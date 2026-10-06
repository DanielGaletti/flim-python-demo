### Ganho marginal de um marcador adicional — Conjuntivite, FLIM_pb, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 10 | 0.0919 | [0.0272, 0.1565] | -0.0137 | 0.2497 | 0.1436 | 0.0001 |
| argmax entropia (o AL) | 10 | 0.0080 | [-0.0411, 0.0571] | -0.1120 | 0.1292 | 0.0598 | 0.0762 |
| argmax least confidence | 10 | -0.0543 | [-0.1482, 0.0396] | -0.2684 | 0.1671 | -0.0025 | 0.9330 |
| candidato sorteado (média) | 10 | -0.0518 | [-0.1219, 0.0184] | -0.2237 | 0.0928 | — | — |
| pior região | 10 | -0.2859 | [-0.4222, -0.1496] | -0.6573 | -0.0589 | -0.2341 | 0.0004 |
| imagem nova inteira (orçamento MAIOR) | 10 | -0.0670 | [-0.1632, 0.0293] | -0.2570 | 0.2395 | -0.0152 | 0.6022 |
| sorteio uniforme (reponderado, sem GT) | 10 | -0.0514 | [-0.1185, 0.0157] | -0.2236 | 0.0970 | 0.0003 | 0.9435 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
