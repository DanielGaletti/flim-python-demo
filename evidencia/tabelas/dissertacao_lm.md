### FLIM puro × Active Learning nos três datasets (FLIM_lm)

| dataset | K | braço | n | Fβ | acurácia | IoU | treino (s) | teste (s) | Δ Fβ | p |
|---|---|---|---|---|---|---|---|---|---|---|
| Parasitas | 2 | FLIM puro | 3 | 0.531 | 0.980 | 0.322 | 3.7 | 23.6 | — | — |
| Parasitas | 2 | coreset | 3 | 0.543 | 0.980 | 0.314 | 3.5 | 24.9 | 0.012 | 0.913 |
| Parasitas | 2 | entropy | 3 | 0.457 | 0.978 | 0.270 | 3.7 | 22.0 | -0.074 | 0.589 |
| Parasitas | 2 | least_confidence | 3 | 0.420 | 0.975 | 0.253 | 3.1 | 23.5 | -0.111 | 0.447 |
| Parasitas | 2 | regiao_confusa | 3 | 0.142 | 0.976 | 0.074 | 5.3 | 24.0 | -0.389 | 0.019 |
| Parasitas | 3 | FLIM puro | 3 | 0.469 | 0.980 | 0.294 | 10.4 | 32.7 | — | — |
| Parasitas | 3 | coreset | 3 | 0.538 | 0.981 | 0.322 | 10.9 | 37.8 | 0.069 | 0.548 |
| Parasitas | 3 | entropy | 3 | 0.474 | 0.980 | 0.288 | 6.4 | 35.6 | 0.005 | 0.976 |
| Parasitas | 3 | least_confidence | 3 | 0.451 | 0.979 | 0.275 | 5.5 | 28.9 | -0.018 | 0.904 |
| Parasitas | 3 | regiao_confusa | 3 | 0.280 | 0.978 | 0.168 | 9.3 | 26.4 | -0.189 | 0.391 |
| Parasitas | 5 | FLIM puro | 3 | 0.542 | 0.981 | 0.333 | 40.4 | 42.7 | — | — |
| Parasitas | 5 | coreset | 3 | 0.640 | 0.983 | 0.423 | 47.1 | 35.3 | 0.099 | 0.352 |
| Parasitas | 5 | entropy | 3 | 0.599 | 0.982 | 0.367 | 61.3 | 35.9 | 0.057 | 0.623 |
| Parasitas | 5 | least_confidence | 3 | 0.583 | 0.981 | 0.355 | 66.2 | 36.0 | 0.041 | 0.675 |
| Parasitas | 5 | regiao_confusa | 3 | 0.039 | 0.974 | 0.020 | 79.6 | 37.9 | -0.502 | 0.040 |
| Parasitas | 8 | FLIM puro | 3 | 0.674 | 0.981 | 0.475 | 87.6 | 38.8 | — | — |
| Parasitas | 8 | coreset | 3 | 0.690 | 0.983 | 0.481 | 94.4 | 35.9 | 0.015 | 0.624 |
| Parasitas | 8 | entropy | 3 | 0.681 | 0.983 | 0.471 | 97.1 | 53.7 | 0.007 | 0.807 |
| Parasitas | 8 | least_confidence | 3 | 0.684 | 0.983 | 0.484 | 126.2 | 43.0 | 0.010 | 0.607 |
| Parasitas | 8 | regiao_confusa | 3 | 0.412 | 0.979 | 0.240 | 117.6 | 46.7 | -0.262 | 0.028 |
| BraTS | 2 | FLIM puro | 3 | 0.257 | 0.964 | 0.193 | 1.3 | 4.2 | — | — |
| BraTS | 2 | coreset | 3 | 0.000 | 0.958 | 0.000 | 1.3 | 2.0 | -0.257 | 0.423 |
| BraTS | 2 | entropy | 3 | 0.001 | 0.958 | 0.000 | 1.4 | 1.8 | -0.255 | 0.423 |
| BraTS | 2 | least_confidence | 3 | 0.001 | 0.958 | 0.000 | 4.7 | 1.9 | -0.255 | 0.423 |
| BraTS | 2 | regiao_confusa | 3 | 0.252 | 0.962 | 0.196 | 2.1 | 1.9 | -0.005 | 0.993 |
| BraTS | 3 | FLIM puro | 3 | 0.070 | 0.902 | 0.058 | 3.5 | 7.1 | — | — |
| BraTS | 3 | coreset | 3 | 0.033 | 0.958 | 0.015 | 1.9 | 2.0 | -0.038 | 0.423 |
| BraTS | 3 | entropy | 3 | 0.000 | 0.958 | 0.000 | 7.9 | 2.1 | -0.070 | 0.423 |
| BraTS | 3 | least_confidence | 3 | 0.000 | 0.958 | 0.000 | 2.9 | 4.7 | -0.070 | 0.423 |
| BraTS | 3 | regiao_confusa | 3 | 0.257 | 0.964 | 0.172 | 4.5 | 10.9 | 0.187 | 0.423 |
| BraTS | 5 | FLIM puro | 3 | 0.351 | 0.964 | 0.171 | 3.3 | 1.9 | — | — |
| BraTS | 5 | coreset | 3 | 0.489 | 0.962 | 0.401 | 3.2 | 2.0 | 0.137 | 0.743 |
| BraTS | 5 | entropy | 3 | 0.498 | 0.963 | 0.387 | 6.7 | 10.4 | 0.147 | 0.727 |
| BraTS | 5 | least_confidence | 3 | 0.498 | 0.963 | 0.387 | 10.3 | 7.1 | 0.147 | 0.727 |
| BraTS | 5 | regiao_confusa | 3 | 0.499 | 0.970 | 0.326 | 11.5 | 2.0 | 0.148 | 0.729 |
| BraTS | 8 | FLIM puro | 3 | 0.300 | 0.949 | 0.194 | 8.3 | 4.1 | — | — |
| BraTS | 8 | coreset | 3 | 0.230 | 0.956 | 0.202 | 9.1 | 1.8 | -0.071 | 0.801 |
| BraTS | 8 | entropy | 3 | 0.000 | 0.958 | 0.000 | 5.5 | 8.7 | -0.300 | 0.191 |
| BraTS | 8 | least_confidence | 3 | 0.000 | 0.958 | 0.000 | 13.1 | 4.2 | -0.300 | 0.191 |
| BraTS | 8 | regiao_confusa | 3 | 0.525 | 0.971 | 0.376 | 9.4 | 2.2 | 0.225 | 0.212 |
| Conjuntivite | 2 | FLIM puro | 3 | 0.040 | 0.872 | 0.012 | 22.3 | 52.7 | — | — |
| Conjuntivite | 2 | coreset | 3 | 0.038 | 0.872 | 0.013 | 31.9 | 57.6 | -0.001 | 0.964 |
| Conjuntivite | 2 | entropy | 3 | 0.070 | 0.873 | 0.024 | 27.4 | 65.2 | 0.030 | 0.108 |
| Conjuntivite | 2 | least_confidence | 3 | 0.070 | 0.873 | 0.024 | 24.8 | 46.3 | 0.030 | 0.108 |
| Conjuntivite | 2 | regiao_confusa | 3 | 0.082 | 0.875 | 0.028 | 30.4 | 46.3 | 0.042 | 0.070 |
| Conjuntivite | 3 | FLIM puro | 3 | 0.010 | 0.868 | 0.003 | 30.3 | 43.9 | — | — |
| Conjuntivite | 3 | coreset | 3 | 0.075 | 0.874 | 0.025 | 36.1 | 54.9 | 0.065 | 0.087 |
| Conjuntivite | 3 | entropy | 3 | 0.055 | 0.873 | 0.019 | 28.5 | 58.3 | 0.044 | 0.191 |
| Conjuntivite | 3 | least_confidence | 3 | 0.055 | 0.873 | 0.019 | 29.7 | 41.7 | 0.044 | 0.191 |
| Conjuntivite | 3 | regiao_confusa | 3 | 0.026 | 0.868 | 0.009 | 52.4 | 34.1 | 0.016 | 0.624 |
| Conjuntivite | 5 | FLIM puro | 3 | 0.030 | 0.870 | 0.011 | 53.0 | 53.9 | — | — |
| Conjuntivite | 5 | coreset | 3 | 0.038 | 0.873 | 0.015 | 62.2 | 48.9 | 0.008 | 0.734 |
| Conjuntivite | 5 | entropy | 3 | 0.044 | 0.874 | 0.015 | 48.6 | 65.2 | 0.014 | 0.726 |
| Conjuntivite | 5 | least_confidence | 3 | 0.044 | 0.874 | 0.015 | 47.8 | 52.9 | 0.014 | 0.726 |
| Conjuntivite | 5 | regiao_confusa | 3 | 0.054 | 0.872 | 0.019 | 75.8 | 36.1 | 0.024 | 0.695 |
| Conjuntivite | 8 | FLIM puro | 3 | 0.010 | 0.872 | 0.003 | 78.8 | 56.9 | — | — |
| Conjuntivite | 8 | coreset | 3 | 0.045 | 0.874 | 0.014 | 90.2 | 48.8 | 0.035 | 0.204 |
| Conjuntivite | 8 | entropy | 3 | 0.051 | 0.873 | 0.017 | 83.0 | 49.5 | 0.041 | 0.126 |
| Conjuntivite | 8 | least_confidence | 3 | 0.089 | 0.875 | 0.030 | 70.0 | 53.1 | 0.080 | 0.074 |
| Conjuntivite | 8 | regiao_confusa | 3 | 0.068 | 0.872 | 0.025 | 127.1 | 44.6 | 0.059 | 0.192 |

`random` é o FLIM puro: sorteia as imagens, como o artigo faz na primeira rodada. Todos os braços recebem o MESMO número de imagens (K) e são medidos no mesmo conjunto de teste. **`regiao_confusa` é diferente dos outros três**: ele usa exatamente as imagens que o sorteio usaria naquela semente e muda só ONDE o traço cai, com o mesmo número de pixels — então o Δ dele mede posição de anotação, e o dos outros mede escolha de imagem. **Acurácia é `1 − MAE`**, a fração de pixels certos; ela passa de 0,97 em quase tudo porque o objeto ocupa uma fração mínima da imagem, e acertar o fundo já garante isso — quem separa os braços é Fβ e IoU. Δ e p são pareados por semente contra o FLIM puro, no mesmo dataset e no mesmo K; com poucas sementes o teste tem pouco poder, e p alto significa **não decidido**, não 'igual'. Decoder FLIM_lm.
