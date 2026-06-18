# AL + FLIM: Arquitetura e Contribuição

## O que mudou e por quê

### Versão errada (al_select_train.py)
- Treina `SaliencyRefiner` (UNet auxiliar) sobre saliency maps → GT
- **Problema:** não é o decoder do FLIM. Compara com FLIMpb (Fβ=0.694) usando um modelo diferente — comparação injusta e sem relevância para o paper

### Versão correta (al_flim_backprop.py)
- Usa **`model.fit_backprop_decoder()`** do pyflim — o decoder real
- O decoder é uma **convolução 1×1** treinada sobre feature maps FLIM reais com GT masks
- Compara diretamente com FLIMpb original → mesma métrica, mesmo modelo, só muda *quais imagens* treinar

---

## Arquitetura

```
FLIM Encoder (fixo, já treinado com k-means)
     │
     ▼ features [B, C, H, W] em target_layer=3
     │
┌────────────────────────────────────────┐
│         Active Learning Loop           │
│                                        │
│  Pool = 610 imagens (saliency maps)    │
│       ↓ entropy score (sem GT)         │
│  Ranking por informatividade           │
│       ↓                                │
│  Budget K → seleciona top-K (AL)       │
│          ou K aleatório (Random)       │
│       ↓                                │
│  "Oracle": GT masks disponíveis        │
│  (schistossoma-eggs/label/)            │
└────────────────────────────────────────┘
     │ K pares (imagem, GT)
     ▼
fit_backprop_decoder(dataset, target_layer=3, epochs=100)
     │  → treina decoder_weights [1, C, 1, 1]
     │  → salva layer3_weight.pth
     ▼
Inferência no val set (848 imagens):
     features = FLIM_encoder(imagem)[:layer3]
     pred = conv2d(features, decoder_weights)
     pred = relu(pred) → normalize → sigmoid
     ↓
     Fβ, DICE, MAE
```

---

## Contribuição para o artigo

**Pergunta de pesquisa:** Com K imagens anotadas selecionadas por AL, FLIMpb atinge desempenho próximo ao treinado com o dataset completo (610 imagens)?

**Hipótese:** entropy scoring dos saliency maps do `labeled_marker` (que não usa GT) identifica imagens mais informativas para treinar o `backprop_decoder`.

**Baseline interno:** 3-5 imagens com seed markers (configuração original do FLIM-AD)

**Ponto ótimo esperado:** budget ~10–30 imagens >> budget original de 3-5

**Figura central:** curva AL Fβ vs K (AL vs Random), comparada ao FLIMpb original (linha horizontal)

---

## Comandos

```bash
# Rodar do container Docker, dentro de flim_ad/
bash scripts/schisto/al_flim_backprop.sh cuda:0 100 5

# Saída: out/al_flim_curve/schisto-user_A_al_flim_curve.csv
```

### Colunas do CSV
| coluna | descrição |
|---|---|
| split | fold (1, 2, 3) |
| budget | K imagens selecionadas |
| al_fb | Fβ com seleção AL (entropy) |
| rand_fb | Fβ com seleção aleatória (média n_seeds) |
| delta_fb | al_fb − rand_fb (ganho do AL) |

---

## Por que isso é uma contribuição válida

1. **FLIM usa pouquíssimas anotações** (3-5 seed markers). AL questiona: *quais* imagens anotar com GT completo?
2. **Nenhum trabalho anterior** aplicou AL ao FLIM backprop_decoder
3. **Custo zero de anotação extra** para scoring: proxy = saliency maps do labeled_marker (sem GT)
4. **Resultado esperado:** AL com K=10–30 → Fβ próximo a K=610, justificando redução de anotações em 20-60×
