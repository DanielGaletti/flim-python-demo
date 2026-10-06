### Gerador de candidatos: contorno previsto contra superpixel

| dataset | n (sementes) | Δ médio | IC95% | vitórias | p | Δ pior caso | p  | Δ dispersão | p   |
|---|---|---|---|---|---|---|---|---|---|
| Parasitas | 10 | 0.0772 | [0.0393, 0.1151] | 10 de 10 | 0.0013 | 0.2153 | 0.0001 | -0.1091 | 0.0000 |
| BraTS | 3 | -0.0089 | [-0.3263, 0.3084] | 1 de 3 | 0.9145 | -0.1063 | 0.2703 | 0.0464 | 0.1376 |
| Conjuntivite | 5 | 0.0452 | [-0.0452, 0.1357] | 4 de 5 | 0.2374 | 0.1759 | 0.0480 | -0.0577 | 0.1492 |

Δ é a variação de Fβ na validação ao acrescentar **uma** anotação, do braço de contorno menos o braço de superpixel. Os dois sorteiam o candidato e gastam os mesmos ~300 px: a **única** diferença é o gerador da lista, e por isso a comparação isola o gerador e não diz nada sobre qual escore usar. A unidade é a **semente**; os dois decodificadores de uma semente compartilham o encoder e entram colapsados por média. Pior caso e dispersão foram declarados no pré-registro, com o mecanismo escrito antes de rodar: o candidato de contorno é pequeno e local, acrescenta menos filtros, e o ganho marginal mede correlação negativa entre filtros acrescentados e ganho. O nulo do BraTS é **previsto**: a arquitetura dele fixa o banco em 8 filtros por camada, os dois braços acrescentam exatamente 24, e o deslocamento que a intervenção evita não pode ocorrer. Semente cuja base degenerou (Fβ=0) sai, porque Δ a partir de zero não pode ser negativo.
