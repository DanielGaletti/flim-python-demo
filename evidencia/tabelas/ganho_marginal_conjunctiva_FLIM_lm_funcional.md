### Ganho marginal de um marcador adicional — Conjuntivite, FLIM_lm, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 10 | 0.1266 | [0.0303, 0.2229] | 0.0031 | 0.4514 | 0.1958 | 0.0003 |
| argmax entropia (o AL) | 10 | 0.0391 | [-0.0725, 0.1506] | -0.1786 | 0.4294 | 0.1082 | 0.0067 |
| argmax least confidence | 10 | 0.0268 | [-0.0771, 0.1307] | -0.2204 | 0.3591 | 0.0959 | 0.0107 |
| candidato sorteado (média) | 10 | -0.0691 | [-0.1783, 0.0400] | -0.3248 | 0.2256 | — | — |
| pior região | 10 | -0.4694 | [-0.5757, -0.3630] | -0.7023 | -0.2655 | -0.4002 | 0.0000 |
| imagem nova inteira (orçamento MAIOR) | 10 | -0.0290 | [-0.2288, 0.1708] | -0.6899 | 0.3608 | 0.0401 | 0.5630 |
| sorteio uniforme (reponderado, sem GT) | 10 | -0.0738 | [-0.1894, 0.0419] | -0.3512 | 0.2254 | -0.0046 | 0.4623 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
