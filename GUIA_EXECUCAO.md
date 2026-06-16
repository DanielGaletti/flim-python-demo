# FLIM-Python — Guia de Execução no Container

## 1. Subir o container com o projeto montado

```bash
docker run -it \
  -v /caminho/local/flim-python-demo:/workspace \
  -p 8888:8888 \
  jleomelo/flim-python:v1 bash
```

> Substitua `/caminho/local/flim-python-demo` pelo caminho real do projeto na sua máquina.

---

## 2. Dentro do container: navegar e instalar pyflim

```bash
cd /workspace
pip install -e .
```

---

## 3. Verificar bugs corrigidos (já aplicados no repositório)

Dois bugs foram corrigidos diretamente no código-fonte:

### Bug 1 — `pyflim/flim.py` linha 446
`forward()` era chamado com 3 argumentos posicionais, mas aceita apenas 2.

**Antes (quebrado):**
```python
y = self.forward(X.unsqueeze(0), self.layers[decoder_layer].marker_labels.clone(), decoder_layer)
```
**Depois (corrigido):**
```python
y = self.forward(X.unsqueeze(0), decoder_layer=decoder_layer)
```

### Bug 2 — `pyflim/metrics.py` linha 689
`saliency` era referenciado fora do bloco `if`, causando `UnboundLocalError` quando o arquivo não existia.

**Corrigido:** adicionado `continue` quando arquivo de saliência ou label não é encontrado.

---

## 4. Rodar o notebook via Jupyter

```bash
cd /workspace
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Acesse no browser: `http://localhost:8888`  
Abra `flim.ipynb` e execute **Kernel → Restart & Run All**.

> **Importante:** remover as células de monkey patch (cells 5, 9 e 13) antes de rodar — elas sobrescrevem o `run` corrigido com versões sem salvamento de imagens.

---

## 5. Alternativa: rodar como script (sem Jupyter)

```bash
cd /workspace
python3 flim.py \
  --architecture arch.json \
  --input "data/orig/" \
  --markers "data/markers/" \
  --output "results/" \
  --save_model_at "saved_models/"
```

---

## 6. Verificar saídas geradas

```bash
ls -la /workspace/out/
```

Devem aparecer as imagens de saliência com os mesmos nomes de `data/orig/`.

---

## 7. Calcular métricas manualmente (se necessário)

```python
from pyflim import flim, arch, data, metrics, util

file_list = "./val.txt"
label_folder = "data/label/"
results_folder = "out/"

metricas = metrics.FLIMMetrics()
metricas.evaluate_saliency_results(results_folder, label_folder, file_list=util.readFileList(file_list))
metricas.print_results()
```

---

## Diagnóstico rápido de erros comuns

| Erro | Causa | Fix |
|------|-------|-----|
| `TypeError: forward() takes 2 positional arguments` | Bug original `flim.py:446` | Já corrigido no repo |
| `UnboundLocalError: saliency referenced before assignment` | Bug original `metrics.py:689` | Já corrigido no repo |
| `out/` tem apenas `result.png` | Monkey patch no notebook sobrescrevia `run` sem salvar | Remover cells 5, 9, 13 do notebook |
| `ModuleNotFoundError: pyflim` | pyflim não instalado no container | Rodar `pip install -e .` |
| `faiss` não encontrado | Imagem docker não tem faiss ou env errado | Usar a imagem oficial `jleomelo/flim-python:v1` |
