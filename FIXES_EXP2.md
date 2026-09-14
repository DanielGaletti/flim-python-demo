# FIXES — Experimento 2 (al_encoder_experiment.py)

## Bug 1 — Device Mismatch em `evaluate_decoder`

**Erro:** `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`  
**Afeta:** `vanilla_adaptive_decoder`, `hybrid_decoder`, `decoder_attention`, `decoder_2`, `decoder_3`  
**Não afeta:** `labeled_marker` (usa tensores CUDA direto, sem `torch.from_numpy`)

**Causa raiz:**  
Os decoders adaptativos do pyflim (ex: `vanilla_adaptive_decoder`) fazem internamente:
```python
interp_feature_array = interp_feature.cpu().detach().numpy()   # feature → numpy
adapted_weights = self.adaptation_function(interp_feature_array)  # numpy
kernel = torch.from_numpy(adapted_weights).float().to(self.device) # volta pra tensor
```
O `self.device` do decoder original salvo no checkpoint pode não coincidir com o device
de carregamento, gerando mismatch em `F.conv2d(img_cuda, kernel_cpu)`.

**Fix aplicado:**  
`evaluate_decoder` agora carrega sempre em `"cpu"`, independente do `device` de treino:
```python
eval_device = "cpu"
model = torch.load(encoder_path, map_location=eval_device, weights_only=False)
model.decoder = layers.FLIMAdaptiveDecoderLayer(..., device=eval_device, ...)
x = torch.tensor(arr).unsqueeze(0)   # sem .to(device)
gt = torch.tensor(gt_arr)...          # sem .to(device)
```
Avaliação em CPU é correta e suficientemente rápida (inference, não treino).

---

## Bug 2 — `AssertionError: No training images were loaded`

**Erro:** `AssertionError: No training images were loaded` em `retrain_encoder`  
**Causa raiz:**  
`pyflim/data.py → FLIMData.filter_from_folder` usa concatenação de string:
```python
marker_file = marker_folder + filename.split(".")[0] + marker_ext
# ex: "/tmp/al_markers" + "000479" + "-seeds.txt"
#  →  "/tmp/al_markers000479-seeds.txt"   ← ERRADO (sem barra!)
```
Resultado: nenhum arquivo é encontrado → `file_list` fica vazio → assertion falha.

**Fix aplicado:**  
Em `retrain_encoder`, adicionar trailing slash antes de passar para `FLIMData`:
```python
marker_dir_slash = marker_dir.rstrip("/") + "/"
train_ds = flimdata.FLIMData(
    marker_folder=marker_dir_slash,   # ← garante barra
    ...
)
```

---

## Como rodar o Experimento 2

```bash
cd flim_ad

# Teste rápido (1 split, budget 3, sem seeds random, CPU)
python3 ../flim_al/al_encoder_experiment.py \
    --markers schisto/user_A \
    --splits 1 \
    --budgets 3 \
    --acquisition entropy \
    --n_seeds 1 \
    --device cpu \
    --save_dir out/al_encoder_results

# Experimento completo (Docker com CUDA)
bash scripts/schisto/al_encoder_experiment.sh cuda:0 3
```

---

## O que o experimento produz

**Por budget K (ex: K=3, 5, 10, 20, 30):**
1. **AL select**: escolhe K imagens do pool por entropia/coreset/badge
2. **Synthetic markers**: gera seeds.txt automático a partir da GT mask
3. **Retrain encoder**: FLIM re-treinado com 3 imgs originais + K selecionadas
4. **Eval ALL decoders**: 7 decoders avaliados (sem treino extra)
5. **Random baseline**: repete N_SEEDS vezes com K aleatórias → média

**Output:**
- `out/al_encoder_results/schisto-user_A_entropy_encoder_al.csv`
- Tabela no terminal: `Decoder | Original (3 imgs) | AL-K | Random-K | Δ(AL-Random)`

**Hipótese científica validada se** Fβ(AL) > Fβ(Random) para múltiplos decoders.
