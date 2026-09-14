### FLIM puro contra seleção por Active Learning, por decoder

| decoder | al (média) | paper (média) | n células | Δ | p | veredito |
|---|---|---|---|---|---|---|
| FLIM_at | 0.618 | 0.541 | 4 | 0.119 | 0.1123 | sem suporte |
| FLIM_lm | 0.737 | 0.782 | 4 | -0.015 | 0.3110 | sem suporte |
| FLIM_lt | 0.689 | 0.687 | 4 | 0.034 | 0.0526 | indicativo |
| FLIM_mb | 0.717 | 0.784 | 4 | -0.012 | 0.2926 | sem suporte |
| FLIM_pb | 0.740 | 0.777 | 4 | -0.010 | 0.3242 | sem suporte |
| FLIM_ts | 0.649 | 0.612 | 4 | 0.054 | 0.1275 | sem suporte |
| FLIM_ts* | 0.568 | 0.526 | 4 | 0.065 | 0.2705 | sem suporte |
| todos (agregado) | 0.709 | 0.676 | 28 | 0.033 | 0.0187 | suportado |

Δ = al − paper, teste **pareado**, médias por célula (split × orçamento) contra a referência. Só entram células em que os dois braços existem no **mesmo orçamento** — o braço do artigo para em 3–4 imagens e o de AL vai a 9, e comparar orçamentos diferentes mediria duas coisas ao mesmo tempo. Negrito só com p < 0,05. Cada linha é um decoder; eles compartilham o encoder dentro de cada célula, então **não são observações independentes entre si** — não some os p.
