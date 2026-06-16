# Reprodução FLIM-AD — Parasites (Schistosoma mansoni)

**Paper:** arXiv:2504.20872 — "FLIM-AD: Adaptive Decoders for FLIM-based Segmentation"  
**Repo:** github.com/LIDS-UNICAMP/flim_ad  
**Dataset:** Parasites (Schistossoma), user_A + user_B, splits 1/2/3  
**Ambiente:** Docker `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, RTX 5070 (sm_120)

---

## Tabela 1 — Reprodução vs Paper (decoders originais)

| Modelo     | User | MAE paper | MAE real | Fβ paper | Fβ real | ΔFβ    |
|------------|------|-----------|----------|----------|---------|--------|
| FLIMts     | A    | 0.010     | 0.010    | 0.747    | 0.691   | -0.056 |
| FLIMts     | B    | 0.013     | 0.010    | 0.687    | 0.681   | -0.006 |
| FLIMat     | A    | 0.011     | 0.014    | 0.740    | 0.576   | -0.164 |
| FLIMat     | B    | 0.014     | 0.018    | 0.660    | 0.461   | -0.199 |
| FLIMpb     | A    | 0.006     | 0.010    | 0.857    | 0.694   | -0.163 |
| FLIMpb     | B    | 0.006     | 0.011    | 0.847    | 0.671   | -0.176 |

> Valores reais = média splits 1/2/3, test set, `filtered_saliencies/1000-9000`.  
> Gap Fβ sistemático: best-layer selecionada apenas via split 2 (único com delineação IFT).

---

## Tabela 2 — Todos decoders (user_A, média splits 1/2/3)

| Decoder (paper) | Interno              | MAE   | Fβ    | layer |
|-----------------|----------------------|-------|-------|-------|
| FLIMts          | labeled_marker       | 0.010 | 0.691 | 3     |
| FLIMad3         | decoder_3            | 0.010 | 0.699 | 3     |
| FLIMad2         | decoder_2            | 0.011 | 0.685 | 3     |
| FLIMpb          | backprop_decoder     | 0.010 | 0.694 | 4     |
| FLIMhyb         | hybrid_decoder       | 0.011 | 0.635 | 3     |
| FLIMat          | vanilla_adaptive     | 0.014 | 0.576 | 3     |
| FLIMatt         | decoder_attention    | 0.016 | 0.529 | 3     |
| FLIMat_wt       | vanilla_adaptive_wt  | 0.019 | 0.473 | 3     |

---

## Tabela 3 — Contribuição AL (FLIM-AL, pendente execução)

| Modelo         | User | MAE   | Fβ    | Δ vs FLIMpb | acquisition | rounds |
|----------------|------|-------|-------|-------------|-------------|--------|
| FLIM-AL(pb)    | A    | -     | -     | -           | entropy     | 10     |
| FLIM-AL(pb)    | B    | -     | -     | -           | entropy     | 10     |
| FLIM-AL(ts)    | A    | -     | -     | -           | entropy     | 10     |
| FLIM-AL(ts)    | B    | -     | -     | -           | entropy     | 10     |
| FLIM-AL(BALD)  | A    | -     | -     | -           | bald        | 10     |

> Execute `bash scripts/schisto/al_train.sh && bash scripts/schisto/al_eval.sh`  
> Os valores serão preenchidos automaticamente pelo script `flim_al/eval_al_decoder.py`.

---

## Pipeline Executado

```
Step  1  → train_flim_encoders       (schisto, user_A)
Step  2  → train_backprop_decoder
Step  3  → train_flim_unet
Step  6  → val_run_flim_decoders     (split 2, all decoders)
Step  7  → val_run_delineation       (IFT, split 2)
Step 11  → val_compute_metrics       (split 2, all decoders, layers 1-4)
Step 12  → val_get_best_layers       (split 2, comp_range 1000-9000)
Step 13  → test_flim_decoders        (splits 1/2/3, user_A + user_B)
Step 14  → test_compute_metrics      (splits 1/2/3, layer_4 pb / layer_3 outros)
Step AL  → al_train.sh + al_eval.sh  (FLIM-AL, contribuição nova)
```

---

## Bugs Corrigidos no Repo

| Arquivo | Bug | Fix |
|---|---|---|
| `pyflim/flim.py` | `from torchvision.transforms import v2` — dead import, crash | Removido |
| `run_flim_decoders.py` | CUDA tensor em spawn multiprocessing | `map_location='cpu'` + `model.to(device)` no worker |
| `run_smansoni_delineation.py` | Paths hardcoded `/home/gilson/` | Relativos `out/` |
| `run_smansoni_delineation.py` | Race condition mkdir em paralelo | `os.makedirs(exist_ok=True)` antes do spawn |
| `get_best_layers.py` | `AxisError` em best_layer_ vazio | Check `len == 0` → default layer 1 |
| `get_best_layers.py` | Double `//` no path | `{marker}/val` |
| Scripts val | `--delineation_name` ausente | Adicionado `delination` (typo do repo) |
| `test_flim_decoders.sh` | `--decoders` faltando no primeiro call | Corrigido |

---

## Ambiente Docker

```bash
docker run --gpus all --shm-size=8g \
  -v /caminho/local:/workspace \
  pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime \
  bash /workspace/flim-python-demo/reproduce_flim_ad.sh --step=N
```

Python 3.11.9 | PyTorch 2.7.0+cu128 | CUDA 12.8 | RTX 5070 (sm_120/Blackwell)
