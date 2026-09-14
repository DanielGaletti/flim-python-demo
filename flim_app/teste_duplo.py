"""Testa a comparacao com AL vs sem AL."""
import json, sys, urllib.request
sys.path.insert(0, "/workspace/flim-python-demo")
import numpy as np
from PIL import Image
API = "http://localhost:8000"


def post(r, b):
    q = urllib.request.Request(API + r, json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(q, timeout=1800))


def rabisca(braco, img):
    gt = np.array(Image.open(f"datasets/schistossoma-eggs/label/{img}.png")
                  .convert("L")) > 0
    ys, xs = np.nonzero(gt); yb, xb = np.nonzero(~gt)
    return post("/api/duplo/marker", {
        "braco": braco, "id": img,
        "fg": [[int(c), int(r)] for c, r in zip(xs[::12], ys[::12])][:150],
        "bg": [[int(c), int(r)] for c, r in zip(xb[::700], yb[::700])][:350]})


K = 3
r = post("/api/duplo/iniciar", {"k": K})
for rodada in range(K):
    print(f'rodada {r["rodada"]}/{r["k"]}')
    for b in ("al", "sem"):
        print(f'  {b:4s}: {r[b]["alvo"]}  ({r[b]["motivo"][:60]})', flush=True)
        rabisca(b, r[b]["alvo"])
    if rodada < K - 1:
        r = post("/api/duplo/proxima", {})
print("\ntreinando os dois...", flush=True)
t = post("/api/duplo/treinar", {"sorteios": 3})
print(f'  com AL : Fb={t["al"].get("fb")} IoU={t["al"].get("iou")} '
      f'treino={t["al"].get("t_treino")}s')
print(f'  sem AL : Fb={t["sem"].get("fb")} IoU={t["sem"].get("iou")} '
      f'treino={t["sem"].get("t_treino")}s')
print(f'  delta  : {t.get("delta")}')
m = t.get("sem_multiplos", {})
if "fb_mediana" in m:
    print(f'  sorteios extras: pior={m["pior"]["fb"]} mediana={m["fb_mediana"]} '
          f'melhor={m["melhor"]["fb"]} amplitude={m["amplitude"]}')
est = json.load(urllib.request.urlopen(API + "/api/estado"))
p = post("/api/duplo/prever", {"id": est["pool"][0]})
print(f'\nmesma imagem {p["id"]}: com AL Fb={p["al"].get("fb")} · '
      f'sem AL Fb={p["sem"].get("fb")}')
print("OK — comparacao duplo funciona")
