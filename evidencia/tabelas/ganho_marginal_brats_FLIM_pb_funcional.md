### Ganho marginal de um marcador adicional — BraTS, FLIM_pb, base funcional

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 8 | 0.1990 | [-0.0337, 0.4318] | 0.0260 | 0.6691 | 0.2384 | 0.0031 |
| argmax entropia (o AL) | 8 | -0.0677 | [-0.3439, 0.2086] | -0.6860 | 0.4616 | -0.0283 | 0.7698 |
| argmax least confidence | 8 | 0.0678 | [-0.0563, 0.1920] | -0.0663 | 0.4147 | 0.1072 | 0.1450 |
| candidato sorteado (média) | 8 | -0.0394 | [-0.1902, 0.1115] | -0.2708 | 0.2284 | — | — |
| pior região | 8 | -0.4498 | [-0.6677, -0.2319] | -0.7334 | -0.0602 | -0.4104 | 0.0000 |
| imagem nova inteira (orçamento MAIOR) | 8 | -0.0583 | [-0.1998, 0.0831] | -0.3217 | 0.1491 | -0.0189 | 0.6833 |
| sorteio uniforme (reponderado, sem GT) | 8 | -0.0549 | [-0.1999, 0.0901] | -0.3035 | 0.1837 | -0.0155 | 0.2206 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base funcional**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
