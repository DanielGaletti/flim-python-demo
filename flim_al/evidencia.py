#!/usr/bin/env python3
"""
evidencia.py — o registro canônico de execução experimental

Problema que isto resolve
    Os 476 CSVs de resultado deste projeto estão em 16 formatos diferentes
    (mais 355 arquivos de detalhe por imagem, que são outra granularidade).
    Nenhum deles registra, de forma consistente, quem produziu o número: falta
    `seed` em 12 dos 16, e não há em nenhum o commit do código nem a
    configuração. Isso torna impossível responder "qual versão do código
    produziu este resultado?" — e escrever agregação determinística em cima
    de 16 formatos é escrever 16 parsers que envelhecem.

O contrato
    Uma linha = uma execução avaliada. Colunas fixas, ordem fixa, um arquivo.
    Tudo que não se sabe vale `UNKNOWN` — nunca um valor plausível inventado.
    `UNKNOWN` é informação: diz que a proveniência daquele número se perdeu, e
    é isso que distingue evidência recuperada de evidência registrada.

    Formato CSV, não Parquet: 3 mil linhas cabem, o git faz diff legível, e a
    banca consegue abrir o arquivo. Parquet vira binário opaco no histórico.

Uso
    from flim_al import evidencia as ev

    ev.registrar([
        ev.execucao(experimento="curva_orcamento", dataset="schisto",
                    split=1, braco="al", criterio="coreset",
                    decoder="labeled_marker", bloco=2, orcamento=16,
                    seed=0, imagens=["000982", "000463"],
                    fb=0.781, dice=0.72, iou=0.61, mae=0.009),
    ])
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO = os.path.join(RAIZ, "evidencia", "execucoes", "execucoes.csv")

UNKNOWN = "UNKNOWN"

# Ordem fixa. Acrescentar campo no FIM; nunca reordenar nem renomear, porque o
# run_id e os diffs do git dependem disso.
CAMPOS = [
    # ── identidade ──────────────────────────────────────────────────────────
    "run_id",          # sha256[:16] dos campos identificadores
    "experimento",     # familia logica: paper_selection, final_comparison, ...
    "hipotese",        # H### quando houver; UNKNOWN nos migrados
    # ── condicao experimental ───────────────────────────────────────────────
    "dataset",         # schisto | brats | conjunctiva
    "usuario",         # A | B | UNKNOWN  (quem desenhou os markers)
    "split",           # 1 | 2 | 3 | UNKNOWN
    "braco",           # al | random | oracle | random_regiao | ...
    "variante",        # rotulo cru do braco na origem (random_region_seed0)
    "criterio",        # coreset | entropy | badge | bald | medoide | random
    "decoder",         # nome no codigo: labeled_marker, decoder_2, ...
    "decoder_paper",   # nome no artigo: FLIM_lm, FLIM_pb, ...
    "bloco",           # camada do encoder usada na avaliacao
    "orcamento",       # numero de imagens anotadas (K / budget / n_train)
    "seed",            # semente da execucao
    "imagens",         # ids do conjunto de treino, separados por "|"
    "marker_origem",   # real | sintetico | UNKNOWN — quem desenhou os traços
    # ── medidas ─────────────────────────────────────────────────────────────
    "fb",              # Fbeta, beta^2 = 0.3 — a metrica principal do artigo
    "dice",
    "iou",
    "mae",
    "fb_val",          # Fbeta na validacao, quando o bloco foi escolhido nela
    "colapsou",        # 1 quando a predicao degenerou (Fb=DICE=IoU~0.479)
    # marker_origem e condicao experimental, nao metadado. Ver o comentario
    # em CAMPOS_ID.
    "segundos",
    # ── proveniencia ────────────────────────────────────────────────────────
    "git_commit",      # commit do codigo que produziu o numero
    "config_hash",     # sha256[:16] da configuracao completa
    "esquema_origem",  # cabecalho do CSV de origem, nos registros migrados
    "fonte",           # caminho relativo do arquivo de origem
    "linha_origem",    # numero da linha no CSV de origem; UNKNOWN nos novos
    "registrado_em",   # ISO-8601 UTC
    # Campos acrescentados depois; por isso vem no fim. `segundos` continua
    # sendo treino + avaliacao, para nao invalidar nenhum registro existente.
    "segundos_treino", # estimar os kernels por k-means, sem backpropagacao
    "segundos_aval",   # rodar o decoder no conjunto de teste
    # Orcamento em PIXELS anotados por imagem.
    #
    # `orcamento` conta imagens, que e como o artigo e quase toda a literatura
    # de AL contam. Nao serve para o experimento de ONDE anotar: ali o numero
    # de imagens e fixo de proposito, e o que varia e quanto se traca dentro
    # delas. Sem campo proprio, dois orcamentos diferentes com o mesmo braco e
    # a mesma semente colidiriam no run_id.
    #
    # Fica FORA de CAMPOS_ID de proposito. Acrescentar campo identificador
    # recalcula o run_id de todo registro ja gravado, e `validar` compara o id
    # armazenado com o recalculado: os 9 mil registros historicos passariam a
    # falhar de uma vez, e a proveniencia das tabelas apontaria para ids que
    # nao existem mais. Quem precisa distinguir orcamentos no id usa
    # `variante`, que ja e identificador e carrega o rotulo cru do braco.
    "orcamento_px",
]

# Campos que entram no run_id.
#
# O run_id identifica o REGISTRO, não a configuração. A distinção importa e
# custou uma rodada de migração para ficar clara: com um id de configuração,
# 1.645 linhas de CSVs históricos colidiram e foram descartadas como
# duplicatas — mas eram execuções distintas cuja configuração a origem não
# registrava por inteiro (o `method` guardava a seed dentro do nome, e o
# adaptador a perdia). Descartar dado por identidade incompleta é pior que
# registrar duas linhas parecidas.
#
# Por isso `fonte` + `linha_origem` entram: duas linhas de um mesmo CSV são,
# por construção, dois registros. Nos registros NOVOS `linha_origem` é
# UNKNOWN, e a identidade volta a ser a configuração.
#
# Comparar configurações é trabalho da agregação, agrupando pelas colunas de
# condição experimental — não do id.
CAMPOS_ID = [
    "experimento", "dataset", "usuario", "split", "braco", "variante",
    "criterio", "decoder", "bloco", "orcamento", "seed", "imagens",
    "marker_origem", "fonte", "linha_origem",
]

NUMERICOS = {"fb", "dice", "iou", "mae", "fb_val", "segundos",
             "segundos_treino", "segundos_aval"}

# As MEDIDAS -- e so elas -- disparam alerta de divergencia quando o mesmo
# run_id reaparece com valor diferente.
#
# `segundos*` esta em NUMERICOS porque precisa da mesma normalizacao numerica,
# mas e relogio de parede: reexecutar a mesma configuracao SEMPRE da um tempo
# diferente. Usar NUMERICOS na deteccao fazia todo merge de reexecucao acusar
# divergencia -- 258 alertas numa reexecucao de 540, dos quais 256 eram so
# tempo. Alerta que soa sempre nao informa nada, e foi preciso conferir os 258
# a mao para achar os 2 que importavam.
MEDIDAS = {"fb", "dice", "iou", "mae", "fb_val"}
INTEIROS = {"split", "bloco", "orcamento", "seed", "colapsou",
            "orcamento_px"}


# ── construcao ──────────────────────────────────────────────────────────────

def _norm(valor, campo: str) -> str:
    """Normaliza para string, preservando UNKNOWN e vazio como coisas distintas."""
    if valor is None:
        return ""
    if isinstance(valor, str):
        v = valor.strip()
        return v
    if isinstance(valor, (list, tuple)):
        return "|".join(str(x) for x in valor)
    if isinstance(valor, bool):
        return "1" if valor else "0"
    if campo in INTEIROS:
        return str(int(valor))
    if campo in NUMERICOS:
        return f"{float(valor):.6g}"
    return str(valor)


def run_id(rec: dict) -> str:
    """Id estavel e deterministico: mesma configuracao, mesmo id."""
    base = "|".join(f"{c}={_norm(rec.get(c, UNKNOWN), c)}" for c in CAMPOS_ID)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def config_hash(cfg: dict) -> str:
    """Hash da configuracao completa; ordena as chaves para ser estavel."""
    canon = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def git_commit_atual() -> str:
    """O commit do codigo que esta rodando, ou UNKNOWN se o git nao responder."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip()
        sujo = subprocess.run(
            ["git", "status", "--porcelain"], cwd=RAIZ, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip()
        # O sufixo importa: um numero produzido com a arvore suja nao e
        # reproduzivel a partir do commit sozinho, e omitir isso seria mentir
        # sobre a proveniencia.
        return sha[:12] + ("-sujo" if sujo else "")
    except Exception:
        return UNKNOWN


def execucao(**campos) -> dict:
    """
    Monta um registro completo a partir do que se sabe.

    O que nao for informado vira UNKNOWN, exceto as metricas, que ficam vazias
    — nao medir e diferente de ter perdido o registro de quem mediu.
    """
    rec = {}
    for c in CAMPOS:
        if c in ("run_id", "registrado_em"):
            continue
        if c in campos:
            rec[c] = _norm(campos[c], c)
        elif c in NUMERICOS:
            rec[c] = ""
        elif c == "git_commit":
            rec[c] = git_commit_atual()
        else:
            rec[c] = UNKNOWN
    rec["run_id"] = run_id(rec)
    rec["registrado_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {c: rec.get(c, "") for c in CAMPOS}


# ── validacao ───────────────────────────────────────────────────────────────

class RegistroInvalido(ValueError):
    pass


def validar(rec: dict) -> None:
    """
    Recusa registro malformado ANTES de ele entrar no arquivo.

    Um registro invalido que passa contamina toda agregacao que o leia depois,
    e o erro so aparece na tabela — longe da causa.
    """
    faltando = [c for c in CAMPOS if c not in rec]
    if faltando:
        raise RegistroInvalido(f"campos ausentes: {faltando}")
    extras = [c for c in rec if c not in CAMPOS]
    if extras:
        raise RegistroInvalido(f"campos desconhecidos: {extras}")

    for c in NUMERICOS:
        v = rec[c]
        if v in ("", UNKNOWN):
            continue
        try:
            float(v)
        except ValueError:
            raise RegistroInvalido(f"{c}={v!r} nao e numero") from None

    for c in INTEIROS:
        v = rec[c]
        if v in ("", UNKNOWN):
            continue
        try:
            int(float(v))
        except ValueError:
            raise RegistroInvalido(f"{c}={v!r} nao e inteiro") from None

    fb = rec["fb"]
    if fb not in ("", UNKNOWN) and not (0.0 <= float(fb) <= 1.0):
        raise RegistroInvalido(f"fb={fb} fora de [0,1]")

    if not rec["experimento"] or rec["experimento"] == UNKNOWN:
        raise RegistroInvalido("experimento e obrigatorio")

    esperado = run_id(rec)
    if rec["run_id"] != esperado:
        raise RegistroInvalido(
            f"run_id {rec['run_id']} nao corresponde aos campos ({esperado}) "
            "— algum campo identificador foi editado a mao")


# ── leitura e escrita ───────────────────────────────────────────────────────

def carregar(arquivo: str = None) -> list:
    """
    Le o registro, completando campos que o arquivo ainda nao tem.

    `CAMPOS` so cresce pelo fim, e um arquivo gravado antes de um campo novo
    nao tem a coluna dele. Sem completar aqui, todo registro antigo falharia a
    validacao por "campos ausentes" no instante em que o esquema crescesse —
    e o custo de estender o esquema viraria uma migracao do arquivo inteiro.

    Vazio e o valor certo para isso: diz "nao foi medido", que e diferente de
    UNKNOWN ("mediram, nao sabemos qual") e diferente de zero.
    """
    arquivo = arquivo or ARQUIVO
    if not os.path.isfile(arquivo):
        return []
    with open(arquivo, encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    for r in linhas:
        for c in CAMPOS:
            r.setdefault(c, "")
    return linhas


@contextlib.contextmanager
def _trava(arquivo: str, espera: float = 180.0):
    """
    Trava entre processos para o registro.

    `registrar` le o arquivo inteiro e o reescreve inteiro. Com duas campanhas
    rodando ao mesmo tempo — o que e o normal neste projeto, uma na GPU e uma
    na CPU — duas leituras podem intercalar e a segunda escrita apaga as
    linhas que a primeira acabou de gravar. Pior: uma escrita interrompida no
    meio deixa CSV truncado. As duas coisas ja aconteceram aqui (ver
    `vendor/REMOVIDOS.txt`: dois fragmentos corrompidos de `paper_selection`).

    Um arquivo `.lock` criado com O_EXCL e atomico em NTFS e em POSIX, o que
    basta: o custo e de milissegundos e a alternativa e perder evidencia.

    Trava orfa (processo morto antes de liberar) e removida apos `espera`.
    """
    lock = arquivo + ".lock"
    pasta = os.path.dirname(arquivo)
    if pasta:
        os.makedirs(pasta, exist_ok=True)
    t0 = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > espera:
                    os.unlink(lock)          # orfa
                    continue
            except OSError:
                pass                          # outro processo venceu a corrida
            if time.time() - t0 > espera:
                raise TimeoutError(
                    f"a trava de {lock} nao liberou em {espera:.0f}s. Se nenhum "
                    f"processo esta escrevendo, apague o arquivo a mao.")
            time.sleep(0.05 + random.random() * 0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(lock)


def _escrever_atomico(arquivo: str, linhas: list) -> None:
    """
    Grava num temporario ao lado e troca por `os.replace`, que e atomico.

    Escrever direto no destino deixa o arquivo truncado se o processo morrer no
    meio — e o registro inteiro da dissertacao esta neste CSV.
    """
    tmp = f"{arquivo}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS, lineterminator="\n")
        w.writeheader()
        for r in linhas:
            w.writerow({c: r.get(c, "") for c in CAMPOS})
    os.replace(tmp, arquivo)


def registrar(recs, arquivo: str = None, substituir: bool = False) -> dict:
    """
    Acrescenta registros, validando cada um e detectando duplicata por run_id.

    Duplicata nao e erro: reexecutar a mesma configuracao acontece. Mas ela
    precisa ser VISIVEL, porque duas linhas com o mesmo run_id e metricas
    diferentes significam que algo mudou fora da configuracao registrada —
    versao de biblioteca, dado, ordem de execucao. Agregar as duas como se
    fossem repeticoes independentes seria errado.
    """
    arquivo = arquivo or ARQUIVO
    recs = list(recs)
    for r in recs:
        validar(r)

    # A leitura e a escrita ficam na MESMA trava. Separa-las reintroduz
    # exatamente a corrida que a trava existe para impedir: dois processos
    # leriam o mesmo estado e o segundo apagaria as linhas do primeiro.
    with _trava(arquivo):
        existentes = [] if substituir else carregar(arquivo)
        vistos = {r["run_id"]: r for r in existentes}

        novos, duplicados, divergentes = [], [], []
        for r in recs:
            anterior = vistos.get(r["run_id"])
            if anterior is None:
                novos.append(r)
                vistos[r["run_id"]] = r
                continue
            duplicados.append(r["run_id"])
            if any(anterior.get(c, "") != r.get(c, "") for c in MEDIDAS):
                divergentes.append(r["run_id"])

        linhas = existentes + novos
        _escrever_atomico(arquivo, linhas)

    return {"novos": len(novos), "duplicados": len(duplicados),
            "divergentes": divergentes, "total": len(linhas)}


def resumo(arquivo: str = None) -> str:
    """Uma visao rapida do que esta registrado, e de quanto e UNKNOWN."""
    arquivo = arquivo or ARQUIVO
    recs = carregar(arquivo)
    if not recs:
        return "nenhum registro"
    linhas = [f"{len(recs)} execucoes registradas", ""]
    for campo in ("experimento", "dataset", "criterio", "seed", "git_commit"):
        vals = {}
        for r in recs:
            vals[r.get(campo, UNKNOWN)] = vals.get(r.get(campo, UNKNOWN), 0) + 1
        desc = ", ".join(f"{k}={v}" for k, v in
                         sorted(vals.items(), key=lambda kv: -kv[1])[:6])
        desconhecido = vals.get(UNKNOWN, 0)
        pct = 100 * desconhecido / len(recs)
        linhas.append(f"  {campo:<14} {desc}"
                      + (f"   [{pct:.0f}% UNKNOWN]" if desconhecido else ""))
    return "\n".join(linhas)


if __name__ == "__main__":
    print(resumo(sys.argv[1] if len(sys.argv) > 1 else ARQUIVO))
