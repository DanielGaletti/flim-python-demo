### Ganho marginal de um marcador adicional — BraTS, FLIM_lm, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 7 | 0.0748 | [-0.0049, 0.1544] | 0.0080 | 0.2581 | 0.2696 | 0.0036 |
| argmax entropia (o AL) | 7 | -0.1134 | [-0.3468, 0.1199] | -0.6785 | 0.0532 | 0.0814 | 0.3225 |
| argmax least confidence | 7 | 0.0301 | [-0.0027, 0.0630] | -0.0344 | 0.0680 | 0.2250 | 0.0021 |
| candidato sorteado (média) | 7 | -0.1948 | [-0.2955, -0.0941] | -0.3571 | -0.0406 | — | — |
| pior região | 7 | -0.6972 | [-0.7849, -0.6095] | -0.7768 | -0.4949 | -0.5024 | 0.0002 |
| imagem nova inteira (orçamento MAIOR) | 7 | -0.2158 | [-0.4314, -0.0001] | -0.5292 | 0.0363 | -0.0209 | 0.8321 |
| sorteio uniforme (reponderado, sem GT) | 7 | -0.2101 | [-0.3531, -0.0671] | -0.4400 | 0.0098 | -0.0152 | 0.5279 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
