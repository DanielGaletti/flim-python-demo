### O custo do Active Learning: a margem que existe e a que se captura

| conjunto | decodificador | n | teto amostrado | o critério | lacuna | p da lacuna | captura |
|---|---|---|---|---|---|---|---|
| Parasitas | FLIM_lm | 10 | 0.3089 | 0.1741 | **0.1347** | 0.0230 | 56\% |
| Parasitas | FLIM_pb | 10 | 0.1351 | 0.0096 | **0.1255** | 0.0006 | 7\% |
| BraTS | FLIM_lm | 7 | 0.2766 | 0.0884 | 0.1882 | 0.1233 | 32\% |
| BraTS | FLIM_pb | 8 | 0.2384 | -0.0283 | **0.2667** | 0.0495 | -12\% |
| Conjuntivite | FLIM_lm | 10 | 0.1997 | 0.1121 | **0.0876** | 0.0038 | 56\% |
| Conjuntivite | FLIM_pb | 10 | 0.1436 | 0.0598 | **0.0839** | 0.0117 | 42\% |

Todas as colunas são diferenças de Fβ contra o **candidato sorteado**, medidas na validação, com a partição, as imagens, os marcadores base e o codificador inicial idênticos dentro de cada semente. `teto amostrado` é o melhor candidato entre os examinados, escolhido **pelo Fβ da validação**: não é estratégia, e mede o que existe para capturar, não um teto absoluto. `o critério` é o que o Active Learning de fato escolhe, sem olhar resultado. A **lacuna** entre os dois é o custo de o critério errar o lugar do clique, e é a coluna com teste: t pareado por semente, com n = sementes, porque os candidatos de uma mesma semente compartilham codificador. `captura` é razão entre duas médias, fica instável com denominador pequeno e serve de ilustração, não de estatística. Linhas restritas à **base funcional**; a estratificação por regime é post-hoc e forçada pela aritmética, porque Δ a partir de Fβ=0 exato não pode ser negativo. As sementes compartilham pool e validação, então o IC vale para esta validação sob sorteio das imagens de treino e não generaliza para o dataset. O conjunto de teste não foi tocado por esta campanha.
