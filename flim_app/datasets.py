#!/usr/bin/env python3
"""
datasets.py — registro de datasets da aplicação
===============================================
Padrão adotado: um diretório com **`orig/`** (imagens) e, opcionalmente,
**`label/`** (segmentação de referência).

O `label/` é opcional de propósito. Um dataset enviado pelo especialista tem
só as imagens — é exatamente o cenário do FLIM, em que a anotação não existe
ainda. Sem `label/` a aplicação continua treinando e prevendo; o que não dá
para fazer é calcular Fβ automaticamente, e aí a validação passa a ser o
julgamento do especialista (matriz de confusão).

Os datasets existentes no repositório não seguem todos o mesmo layout — o
Conjunctiva usa `images/`+`labels/`. O registro normaliza isso sem mexer nos
arquivos.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPLOADS = REPO / "flim_app" / "uploads"

# Cada entrada: onde estão as imagens, onde está a anotação (se houver), qual
# arquitetura usar e qual filtro de área o pós-processamento aplica.
#
# O filtro de área não é detalhe de implementação: é um prior de tamanho de
# objeto, e usar o do Schisto no BraTS zeraria quase toda predição. Vem da
# Tabela I do artigo.
REGISTRO: dict[str, dict] = {
    "schisto": {
        "nome": "Schistosoma (ovos)",
        "orig": "flim_ad/datasets/schistossoma-eggs/orig",
        "label": "flim_ad/datasets/schistossoma-eggs/label",
        "arch": "flim_ad/data/schisto/user_A/split1/arch2D.json",
        "area": [1000, 9000],
        "bloco": 2,
        "descricao": "1220 imagens 400x400 RGB. 49% não têm ovo.",
    },
    "brats": {
        "nome": "BraTS (tumor cerebral)",
        "orig": "data/brats/orig",
        "label": "data/brats/label",
        "arch": "arch_best_brats.json",
        "area": [100, 20000],
        "bloco": 2,
        "descricao": "3753 imagens 240x240 em tons de cinza. Todas com tumor.",
    },
    "conjunctiva": {
        "nome": "Conjuntiva",
        "orig": "data/conjunctiva/images",
        "label": "data/conjunctiva/labels",
        "arch": "arch_conjunctiva.json",
        "area": [1000, 9000],
        "bloco": 2,
        "descricao": "83 imagens 1079x863 RGB.",
    },
}


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(REPO / p)


def _conta(d: str, exts=(".png", ".jpg", ".jpeg")) -> int:
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith(exts))


def listar() -> list[dict]:
    """Datasets do registro + os enviados por upload, só os que existem."""
    out = []
    for chave, c in REGISTRO.items():
        orig = _abs(c["orig"])
        n = _conta(orig)
        if n == 0:
            continue
        lab = _abs(c["label"]) if c.get("label") else ""
        out.append({
            "id": chave, "nome": c["nome"], "n_imagens": n,
            "tem_label": _conta(lab) > 0, "descricao": c["descricao"],
            "area": c["area"], "enviado": False,
        })
    for d in sorted(UPLOADS.glob("*")) if UPLOADS.is_dir() else []:
        if not d.is_dir():
            continue
        n = _conta(str(d / "orig"))
        if n == 0:
            continue
        meta = {}
        mp = d / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        out.append({
            "id": f"upload:{d.name}", "nome": meta.get("nome", d.name),
            "n_imagens": n, "tem_label": _conta(str(d / "label")) > 0,
            "descricao": "enviado pelo especialista",
            "area": meta.get("area", [1000, 9000]), "enviado": True,
        })
    return out


def resolver(ds_id: str) -> dict:
    """Devolve caminhos absolutos e parâmetros de um dataset."""
    if ds_id.startswith("upload:"):
        d = UPLOADS / ds_id.split(":", 1)[1]
        meta = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return {
            "id": ds_id, "nome": meta.get("nome", d.name),
            "orig": str(d / "orig"),
            "label": str(d / "label") if _conta(str(d / "label")) else "",
            # Sem arquitetura própria, usa a do Schisto: 4 blocos, kernels 3x3.
            # É a mais genérica das três e serve para imagens coloridas.
            "arch": meta.get("arch", _abs(REGISTRO["schisto"]["arch"])),
            "area": meta.get("area", [1000, 9000]),
            "bloco": meta.get("bloco", 2),
        }
    if ds_id not in REGISTRO:
        raise KeyError(f"dataset desconhecido: {ds_id}")
    c = REGISTRO[ds_id]
    lab = _abs(c["label"]) if c.get("label") else ""
    return {
        "id": ds_id, "nome": c["nome"], "orig": _abs(c["orig"]),
        "label": lab if _conta(lab) else "", "arch": _abs(c["arch"]),
        "area": c["area"], "bloco": c["bloco"],
    }


def imagens(ds_id: str) -> list[str]:
    """Nomes de arquivo, ordenados. Só as que têm anotação, se houver label."""
    c = resolver(ds_id)
    exts = (".png", ".jpg", ".jpeg")
    fs = sorted(f for f in os.listdir(c["orig"]) if f.lower().endswith(exts))
    if c["label"]:
        # o label pode ter extensão diferente da imagem (jpg -> png)
        labs = {os.path.splitext(f)[0] for f in os.listdir(c["label"])}
        fs = [f for f in fs if os.path.splitext(f)[0] in labs]
    return fs


def caminho_label(ds_id: str, fname: str) -> str:
    """Caminho da anotação, tolerando extensão diferente da imagem."""
    c = resolver(ds_id)
    if not c["label"]:
        return ""
    base = os.path.splitext(fname)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(c["label"], base + ext)
        if os.path.exists(p):
            return p
    return ""


def criar_upload(nome: str, existente: str | None = None) -> str:
    """
    Cria um dataset de upload vazio e devolve seu id.

    Com `existente`, devolve o dataset indicado sem criar nada -- e o que
    permite enviar um lote agora e outro depois para o MESMO dataset. Sem isso,
    o segundo lote virava um dataset separado, e trocar de dataset descarta o
    encoder: o modelo treinado no primeiro lote se perdia justamente quando se
    queria validar com o segundo.
    """
    if existente and existente.startswith("upload:"):
        d = UPLOADS / existente.split(":", 1)[1]
        if d.is_dir():
            return existente
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in nome)[:40] or "dataset"
    d = UPLOADS / slug
    i = 1
    while d.exists():
        i += 1
        d = UPLOADS / f"{slug}_{i}"
    (d / "orig").mkdir(parents=True)
    (d / "label").mkdir()
    (d / "meta.json").write_text(json.dumps({"nome": nome}, ensure_ascii=False),
                                 encoding="utf-8")
    return f"upload:{d.name}"


def marcar_validacao(ds_id: str, nomes: list) -> int:
    """
    Registra nomes de imagens como conjunto de validacao fixo do upload.

    Guardado no meta.json do dataset para sobreviver a reinicios do servidor.
    Quando esta lista existe, `_init_conjuntos` a usa no lugar do sorteio: o
    especialista decide o que e treino e o que e teste, em vez de a aplicacao
    sortear.
    """
    if not ds_id.startswith("upload:"):
        return 0
    d = UPLOADS / ds_id.split(":", 1)[1]
    mp = d / "meta.json"
    if not mp.exists():
        return 0
    meta = json.loads(mp.read_text(encoding="utf-8"))
    val = set(meta.get("val", []))
    val.update(os.path.splitext(n)[0] for n in nomes)
    meta["val"] = sorted(val)
    mp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return len(meta["val"])


def validacao_fixa(ds_id: str) -> list:
    """Os nomes marcados como validacao, ou lista vazia."""
    if not ds_id.startswith("upload:"):
        return []
    mp = UPLOADS / ds_id.split(":", 1)[1] / "meta.json"
    if not mp.exists():
        return []
    try:
        return list(json.loads(mp.read_text(encoding="utf-8")).get("val", []))
    except Exception:
        return []


def apagar_upload(ds_id: str) -> bool:
    if not ds_id.startswith("upload:"):
        return False
    d = UPLOADS / ds_id.split(":", 1)[1]
    if d.is_dir() and UPLOADS in d.parents:
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False
