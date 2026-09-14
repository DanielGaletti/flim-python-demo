"""Teste do laço guiado: o SISTEMA escolhe, o especialista anota, treina."""
import json, sys, urllib.request
sys.path.insert(0, "/workspace/flim-python-demo")
import numpy as np
from PIL import Image

API = "http://localhost:8000"


def post(rota, corpo):
    q = urllib.request.Request(API + rota, json.dumps(corpo).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(q, timeout=900))


def rabisca(img):
    """Simula o especialista: traços sobre o ovo e sobre o fundo."""
    gt = np.array(Image.open(f"datasets/schistossoma-eggs/label/{img}.png")
                  .convert("L")) > 0
    ys, xs = np.nonzero(gt)
    yb, xb = np.nonzero(~gt)
    fg = [[int(c), int(r)] for c, r in zip(xs[::12], ys[::12])][:150]
    bg = [[int(c), int(r)] for c, r in zip(xb[::700], yb[::700])][:350]
    return post("/api/marker", {"id": img, "fg": fg, "bg": bg})


post("/api/k_alvo", {"k": 3})
for rodada in range(3):
    p = post("/api/proxima", {})
    if "erro" in p:
        print("erro:", p["erro"]); break
    print(f'rodada {p["rodada"]}/{p["k_alvo"]}: sistema escolheu {p["alvo"]}')
    print(f'   motivo: {p["motivo"]}', flush=True)
    rabisca(p["alvo"])
    t = post("/api/treinar", {})
    a = post("/api/avaliar", {})
    print(f'   anotada e treinada em {t["t_treino"]}s -> Fb={a["fb"]} '
          f'IoU={a["iou"]}  (|T|={t["n_imgs"]})', flush=True)
print("OK — laço guiado funciona")
