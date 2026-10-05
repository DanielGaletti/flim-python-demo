### Cliques até a IoU alvo, por plasticidade do banco de filtros

| dataset | α | n | NoC médio | NoC mediano | taxa de sucesso | IoU final | Δ NoC vs α=1 | p | McNemar |
|---|---|---|---|---|---|---|---|---|---|
| Parasitas | 0.00 | 18 | 11.50 | 12.0 | 6% | 0.5179 | 0.00 | 1.0000 | 0/0 (p=1.000) |
| Parasitas | 1.00 | 18 | 11.50 | 12.0 | 6% | 0.5646 | — | — | — |
| BraTS | 0.00 | 20 | 9.50 | 12.0 | 30% | 0.2515 | -0.10 | 0.9138 | 2/2 (p=1.000) |
| BraTS | 1.00 | 20 | 9.60 | 12.0 | 30% | 0.2770 | — | — | — |

NoC@X é o número de cliques até a IoU passar do alvo, com teto de 12; é o eixo de RITM (arXiv:2102.06583) e SimpleClick (arXiv:2210.11006), e **não** se compara com o NoC publicado deles, que usa outros conjuntos e outra definição de alvo. Imagem que não atinge o alvo entra com o teto, e por isso a **taxa de sucesso** aparece em toda linha: um braço pode mostrar NoC baixo por desistir mais cedo. `α=1` é o FLIM puro, que refaz o banco de filtros a cada clique; `α=0` congela o banco, e o clique chega ao modelo só pelos rótulos que o decoder `labeled_marker` lê. Os testes são **separados por dataset**: agregar Schisto e BraTS, com 7% e 30% de sucesso, inflou um achado exploratório a p=0,0265 que separado não passa de p=0,10. O clique é simulado do ground truth pelo protocolo de Xu et al. (2016), com usuário que acerta sempre, então a tabela mede **número de interações e não tempo de especialista**.
