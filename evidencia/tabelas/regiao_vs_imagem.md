### Active Learning por região: braços e controle, por decoder

| decoder | al (média) | random (média) | n células | Δ | p | veredito |
|---|---|---|---|---|---|---|
| decoder_2 | 0.532 | 0.461 | 9 | 0.036 | 0.5113 | sem suporte |
| decoder_3 | 0.395 | 0.316 | 9 | 0.033 | 0.6159 | sem suporte |
| decoder_attention | 0.217 | 0.176 | 9 | 0.091 | 0.1743 | sem suporte |
| hybrid_decoder | 0.394 | 0.384 | 9 | 0.079 | 0.1107 | sem suporte |
| labeled_marker | 0.512 | 0.524 | 9 | -0.030 | 0.2452 | sem suporte |
| vanilla_adaptive_decoder | 0.429 | 0.413 | 9 | 0.030 | 0.4782 | sem suporte |
| vanilla_adaptive_decoder_wt | 0.195 | 0.170 | 9 | **0.112** | 0.0471 | suportado |

Δ = al − random, teste **pareado**, pareado por semente. Só entram células em que os dois braços existem no **mesmo orçamento** — o braço do artigo para em 3–4 imagens e o de AL vai a 9, e comparar orçamentos diferentes mediria duas coisas ao mesmo tempo. Negrito só com p < 0,05. Cada linha é um decoder; eles compartilham o encoder dentro de cada célula, então **não são observações independentes entre si** — não some os p.
