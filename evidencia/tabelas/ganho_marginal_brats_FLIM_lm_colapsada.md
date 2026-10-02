### Ganho marginal de um marcador adicional — BraTS, FLIM_lm, base colapsada (Fβ₀=0)

| o que recebeu o marcador | n (sementes) | Δ Fβ médio | IC95% | pior | melhor | Δ vs sorteado | p |
|---|---|---|---|---|---|---|---|
| melhor região (teto amostrado) | 3 | 0.7415 | [0.7018, 0.7812] | 0.7242 | 0.7557 | 0.5605 | 0.0103 |
| argmax entropia (o AL) | 3 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | -0.1809 | 0.1017 |
| argmax least confidence | 3 | 0.0091 | [-0.0300, 0.0482] | 0.0000 | 0.0273 | -0.1718 | 0.1350 |
| candidato sorteado (média) | 3 | 0.1809 | [-0.0883, 0.4501] | 0.0742 | 0.2909 | — | — |
| pior região | 3 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | -0.1809 | 0.1017 |
| imagem nova inteira (orçamento MAIOR) | 3 | 0.2275 | [-0.2542, 0.7093] | 0.0036 | 0.3426 | 0.0466 | 0.5542 |
| sorteio uniforme (reponderado, sem GT) | 3 | 0.0850 | [-0.1892, 0.3591] | 0.0096 | 0.2116 | -0.0960 | 0.0591 |

Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. Dentro de cada semente, a partição, as imagens iniciais, os marcadores base, o encoder inicial e o conjunto de validação são idênticos — a única variável é qual região recebeu os ~300 px adicionais. O treino é determinístico **dentro de um processo**, a partir dos mesmos arquivos de marcador (verificado: três repetições dão Fβ idêntico até a décima casa). **Entre** execuções há ruído: a reexecução completa do schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 restantes o desvio máximo foi 0,0034 — duas ordens de grandeza abaixo dos efeitos reportados. **n é o número de sementes**: os candidatos de uma mesma semente compartilham encoder e são correlacionados. `melhor região (teto)` escolhe pelo Fβ da validação e **não é uma estratégia** — mede o que existe para capturar, **entre os 22 candidatos examinados** de ~380 superpixels; não é teto absoluto. Linhas restritas a: **base colapsada (Fβ₀=0)**. A separação por regime é post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações não têm média comparável. As sementes compartilham o pool e o conjunto de validação: o IC95% vale para **esta** validação sob sorteio das imagens de treino, e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
