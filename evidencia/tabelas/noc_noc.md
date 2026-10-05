### Cliques até a IoU alvo, por plasticidade do banco de filtros

| dataset | α | n | NoC médio | NoC mediano | taxa de sucesso | IoU final | Δ NoC vs α=1 | p | McNemar |
|---|---|---|---|---|---|---|---|---|---|
| Parasitas | 0.00 | 38 | 11.03 | 12.0 | 13% | 0.5267 | -0.53 | 0.1031 | 3/0 (p=0.250) |
| Parasitas | 0.50 | 22 | 11.09 | 12.0 | 14% | 0.5490 | -0.55 | 0.2876 | 2/0 (p=0.500) |
| Parasitas | 1.00 | 38 | 11.55 | 12.0 | 5% | 0.5501 | — | — | — |
| BraTS | 0.00 | 40 | 7.42 | 12.0 | 48% | 0.4096 | -0.35 | 0.5829 | 4/3 (p=1.000) |
| BraTS | 0.50 | 20 | 6.00 | 3.0 | 60% | 0.5273 | 0.05 | 0.7715 | 1/1 (p=1.000) |
| BraTS | 1.00 | 40 | 7.78 | 12.0 | 45% | 0.4064 | — | — | — |
| Conjuntivite | 0.00 | 1 | 12.00 | 12.0 | 0% | 0.0000 | — | — | 0/1 (p=1.000) |
| Conjuntivite | 0.50 | 1 | 6.00 | 6.0 | 100% | 0.7617 | — | — | 0/0 (p=1.000) |
| Conjuntivite | 1.00 | 1 | 6.00 | 6.0 | 100% | 0.7764 | — | — | — |

NoC@X é o número de cliques até a IoU passar do alvo, com teto de 12; é o eixo de RITM (arXiv:2102.06583) e SimpleClick (arXiv:2210.11006), e **não** se compara com o NoC publicado deles, que usa outros conjuntos e outra definição de alvo. Imagem que não atinge o alvo entra com o teto, e por isso a **taxa de sucesso** aparece em toda linha: um braço pode mostrar NoC baixo por desistir mais cedo. `α=1` é o FLIM puro, que refaz o banco de filtros a cada clique; `α=0` congela o banco, e o clique chega ao modelo só pelos rótulos que o decoder `labeled_marker` lê. Os testes são **separados por dataset**: agregar Schisto e BraTS, com 7% e 30% de sucesso, inflou um achado exploratório a p=0,0265 que separado não passa de p=0,10. O clique é simulado do ground truth pelo protocolo de Xu et al. (2016), com usuário que acerta sempre, então a tabela mede **número de interações e não tempo de especialista**.
