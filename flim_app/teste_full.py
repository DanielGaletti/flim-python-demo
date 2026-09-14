"""Testa lote (curva) e validacao (matriz de confusao)."""
import json, sys, time, urllib.request
API = "http://localhost:8000"


def post(r, b):
    q = urllib.request.Request(API + r, json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(q, timeout=1800))


def get(r):
    return json.load(urllib.request.urlopen(API + r, timeout=120))


print("== LOTE: curva de aprendizado ==", flush=True)
print(post("/api/lote/iniciar", {"ds": "schisto", "ks": [2, 4, 8],
                                 "n_test": 40}), flush=True)
while True:
    time.sleep(10)
    r = get("/api/lote")
    if not r["rodando"]:
        break
    print(f'  {r["feito"]}/{r["total"]}', flush=True)
for x in r["resultados"]:
    if "erro" in x:
        print(f'  K={x["k"]} ERRO: {x["erro"]}')
    else:
        print(f'  K={x["k"]:>2}  Fb={x["fb"]:.4f}  IoU={x["iou"]:.3f}  '
              f'selecao={x["t_selecao"]}s treino={x["t_treino"]}s '
              f'teste={x["t_teste"]}s ({x["ms_por_imagem"]} ms/img)', flush=True)

print("\n== VALIDACAO: matriz de confusao ==", flush=True)
import numpy as np
from PIL import Image
sys.path.insert(0, "/workspace/flim-python-demo")
# treina rapido no laco guiado
post("/api/k_alvo", {"k": 2})
for _ in range(2):
    p = post("/api/proxima", {})
    img = p["alvo"]
    gt = np.array(Image.open(f"datasets/schistossoma-eggs/label/{img}.png")
                  .convert("L")) > 0
    ys, xs = np.nonzero(gt); yb, xb = np.nonzero(~gt)
    post("/api/marker", {"id": img,
                         "fg": [[int(c), int(r)] for c, r in zip(xs[::12], ys[::12])][:150],
                         "bg": [[int(c), int(r)] for c, r in zip(xb[::700], yb[::700])][:350]})
    post("/api/treinar", {})
v = post("/api/validar/iniciar", {"n": 6})
n = 0
while not v.get("fim"):
    # o "especialista" responde pela anotacao real
    lp = f'datasets/schistossoma-eggs/label/{v["id"]}.png'
    tem = bool((np.array(Image.open(lp).convert("L")) > 0).any())
    print(f'  {v["id"]}: modelo={"detectou" if v["detectou"] else "nao"} '
          f'· especialista={"tem" if tem else "nao tem"}', flush=True)
    v = post("/api/validar/responder", {"tem_objeto": tem})
    n += 1
print("  matriz:", v["matriz"], flush=True)
print("OK — lote e validacao funcionam")
