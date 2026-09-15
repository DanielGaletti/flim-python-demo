#!/usr/bin/env python3
"""
migrar_evidencia.py — traz os CSVs históricos para o registro canônico

Roda uma vez. A partir da Fase B os experimentos gravam direto no formato
canônico via `flim_al.evidencia.registrar()`; este script existe para que os
resultados já produzidos não fiquem de fora.

Princípio que governa cada adaptador abaixo
    **Não inventar.** Onde o CSV de origem não registra a informação, o campo
    vira UNKNOWN. Doze dos dezesseis formatos não têm `seed`; nenhum tem o
    commit do código. Preencher isso com um palpite plausível transformaria
    perda de proveniência em proveniência falsa, que é pior — um número com
    `seed=UNKNOWN` avisa que não dá para reproduzir aquela linha; um número com
    `seed=0` inventado afirma que dá.

    Os arquivos de detalhe por imagem (355, em metrics/, metrics_delination/ e
    saliencies_delination/) NÃO entram: são outra granularidade — uma linha por
    imagem, não por execução — e vêm do pipeline do flim_ad, não dos
    experimentos de AL. Ficam em evidencia/bruto/ como estão.

Uso
    python scripts/migrar_evidencia.py            # migra
    python scripts/migrar_evidencia.py --dry-run  # só relata
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al import evidencia as ev  # noqa: E402

BRUTO = os.path.join(RAIZ, "evidencia", "bruto")
UNK = ev.UNKNOWN

# Famílias que são detalhe por imagem, não execução.
IGNORAR_DIRS = {"metrics", "metrics_delination", "saliencies_delination"}

# Arquivos superados, excluídos com motivo — nunca em silêncio.
#
# `*_BACKUP_pre_repair.csv`: o backup anterior a um reparo de cabeçalho. Tem
# 10 colunas de dado para 9 nomes, então `n_train_imgs` cai em `fb` e 189 das
# 525 linhas ficam com fb entre 2 e 12. Os DADOS são idênticos aos do arquivo
# reparado, que já entra na migração; o que mudou foi só o cabeçalho. Foi o
# validador de faixa de fb que apontou isto — a razão de ele existir.
IGNORAR_ARQUIVOS = ("_BACKUP_pre_repair.csv",)

# Dataset por família de experimento.
#
# Inferência ESTRUTURAL, não adivinhação: estas campanhas rodaram sobre um
# dataset só, e o caminho não carrega o nome porque na época havia só um. As
# campanhas de BraTS e Conjunctiva têm o nome no diretório (paper_selection_brats,
# brats, benchmark_conjunctiva) e são detectadas pelo regex; o que sobra aqui é
# Schisto. Sem este mapa, 67% dos registros ficavam com dataset=UNKNOWN e o
# registro inteiro deixava de ser agrupável por dataset — o campo mais básico
# de qualquer tabela.
#
# Família que NÃO estiver aqui e não casar com o regex continua UNKNOWN.
DATASET_POR_FAMILIA = {
    "final_comparison": "schisto",
    "final_comparison_v2": "schisto",
    "final_comparison_v3": "schisto",
    "paper_selection_v2": "schisto",
    "paper_selection_v3": "schisto",
    "paper_selection_pac2": "schisto",
    "paper_selection_rand": "schisto",
    "paper_selection_coreset": "schisto",
    "paper_selection_POOL_COM_VAZAMENTO": "schisto",
    "paper_selection_uB_POOL_COM_VAZAMENTO": "schisto",
    "sel_cs_grande": "schisto",
    "sel_degen": "schisto",
    "sel_medoide": "schisto",
    "sel_proposto": "schisto",
    "curva": "schisto",
    "orcamento": "schisto",
    "al_curve": "schisto",
    "al_flim_curve": "schisto",
    "al_region_results": "schisto",
}


# ── inferência a partir do caminho ──────────────────────────────────────────
#
# O caminho carrega informação que o conteúdo do CSV não tem: qual dataset,
# qual usuário, qual campanha. É inferência de estrutura de diretório, não
# adivinhação de valor — e onde o padrão não casa, o campo fica UNKNOWN.

def do_caminho(rel: str) -> dict:
    p = rel.replace(os.sep, "/")
    fora = {}

    fora["dataset"] = (
        "brats" if re.search(r"brats", p, re.I) else
        "conjunctiva" if re.search(r"conjunctiva", p, re.I) else
        "schisto" if re.search(r"schisto", p, re.I) else UNK
    )

    m = re.search(r"user[_-]?([AB])", p)
    fora["usuario"] = m.group(1) if m else UNK

    m = re.search(r"_s(\d)\.csv$|split(\d)", p)
    fora["split_caminho"] = (m.group(1) or m.group(2)) if m else UNK

    fora["experimento"] = p.split("/")[0]
    if fora["dataset"] == UNK:
        fora["dataset"] = DATASET_POR_FAMILIA.get(fora["experimento"], UNK)

    # Campanhas cujo nome no diretório carrega um aviso: essas linhas existiam
    # com vazamento do conjunto de teste, e o próprio nome registra isso.
    fora["suspeito"] = "POOL_COM_VAZAMENTO" in p or "obsoleto" in p
    return fora


def _f(v):
    """float ou vazio — nunca 0 por omissão, que viraria uma medida falsa."""
    if v is None or str(v).strip() in ("", "nan", "None"):
        return ""
    try:
        return float(v)
    except ValueError:
        return ""


def _i(v):
    if v is None or str(v).strip() in ("", "nan", "None"):
        return UNK
    try:
        return int(float(v))
    except ValueError:
        return UNK


def _marker(v):
    """
    Normaliza a origem do marker: quem desenhou os traços.

    Campo decisivo e quase sempre ausente. O braço do artigo treina com os 31
    markers REAIS que um especialista desenhou; os braços de AL sobre o pool
    completo usam SINTÉTICOS, porque marker real só existe para 31 imagens.
    Comparar os dois sem casar este campo mede a qualidade do traço e
    apresenta o resultado como efeito da seleção.
    """
    if v is None:
        return UNK
    t = str(v).strip().lower()
    if t in ("real", "reais", "user", "usuario"):
        return "real"
    if t in ("synthetic", "sintetico", "sintético", "generated", "full"):
        return "sintetico"
    return UNK


def _origem_por_braco(braco: str, experimento: str):
    """
    Inferência ESTRUTURAL da origem do marker, onde a linha não a registra.

    Vale só para as famílias `al_*`, e vem do que o próprio
    `al_encoder_experiment.py` documenta no cabeçalho: o braço `original` são
    as 3 imagens do artigo, com os markers reais; os braços de AL e de sorteio
    recebem markers sintéticos ("Gera markers sintéticos (simula usuário
    desenhando seeds)").

    Fora dessas famílias, UNKNOWN — inferir por analogia seria adivinhar.
    """
    if not experimento.startswith("al_"):
        return UNK
    if braco == "original":
        return "real"
    if braco in ("al", "random"):
        return "sintetico"
    return UNK


def _seed_do_arm(arm: str):
    """
    Extrai a seed embutida no rótulo do braço.

    Vários formatos guardam a seed DENTRO do nome — `al_seed0`,
    `random_region_seed1` — em vez de numa coluna. Perder isso fazia execuções
    de seeds diferentes virarem o mesmo registro: foi a causa das 1.645
    colisões da primeira migração.
    """
    m = re.search(r"seed(\d+)", str(arm))
    return (int(m.group(1)), re.sub(r"_?seed\d+", "", str(arm)) or "al") \
        if m else (UNK, arm)


# ── adaptadores, um por schema ──────────────────────────────────────────────

def ad_paper_selection(linha, ctx):
    """Algoritmo 1 reimplementado — o único formato com seed E round."""
    return dict(
        braco="oracle" if linha.get("criterion") == "oracle" else "al",
        criterio=linha.get("criterion", UNK),
        seed=_i(linha.get("seed")),
        split=_i(linha.get("split")),
        orcamento=_i(linha.get("n_images")),
        imagens=linha.get("selected", UNK) or UNK,
        # A unica familia que registra a origem do marker na propria linha.
        marker_origem=_marker(linha.get("marker_source")),
        fb=_f(linha.get("fb")), dice=_f(linha.get("dice")),
        iou=_f(linha.get("iou")), mae=_f(linha.get("mae")),
    )


def ad_per_model(linha, ctx):
    """final_comparison / brats — um decoder por linha, seed dentro do braço."""
    seed, braco = _seed_do_arm(linha.get("arm", UNK))
    return dict(
        variante=str(linha.get("arm", UNK)),
        usuario=linha.get("user", ctx["usuario"]),
        dataset=linha.get("dataset", ctx["dataset"]),
        split=_i(linha.get("split")),
        braco=braco, seed=seed,
        decoder=linha.get("decoder", UNK),
        decoder_paper=linha.get("paper_name", UNK),
        bloco=_i(linha.get("block")),
        orcamento=_i(linha.get("n_train")),
        imagens=linha.get("train_ids", UNK) or UNK,
        fb=_f(linha.get("fb")), dice=_f(linha.get("dice")),
        iou=_f(linha.get("iou")), mae=_f(linha.get("mae")),
        fb_val=_f(linha.get("fb_val")),
    )


def ad_budget_method(linha, ctx):
    """al_*_results — o braço vem no campo `method`, às vezes com a seed junto."""
    met = str(linha.get("method", UNK))
    seed, _ = _seed_do_arm(met)
    return dict(
        split=_i(linha.get("split")),
        orcamento=_i(linha.get("budget") or linha.get("n_train_imgs")),
        braco="random" if "rand" in met.lower() else
              "original" if "original" in met.lower() else "al",
        variante=met,
        seed=seed,
        criterio=linha.get("acquisition", UNK),
        decoder=linha.get("decoder", UNK),
        dataset=linha.get("dataset", ctx["dataset"]),
        fb=_f(linha.get("fb")), dice=_f(linha.get("dice")),
        iou=_f(linha.get("iou")), mae=_f(linha.get("mae")),
    )


def ad_runs(linha, ctx):
    """al_backprop_results/*_runs.csv — tem seed E marca de colapso."""
    return dict(
        split=_i(linha.get("split")),
        orcamento=_i(linha.get("budget") or linha.get("n_train_imgs")),
        braco="random" if linha.get("arm") == "rand" else linha.get("arm", UNK),
        variante=str(linha.get("arm", UNK)),
        criterio=linha.get("acquisition", UNK),
        seed=_i(linha.get("seed")),
        colapsou=_i(linha.get("collapsed")),
        fb=_f(linha.get("fb")), dice=_f(linha.get("dice")),
        iou=_f(linha.get("iou")), mae=_f(linha.get("mae")),
    )


def ad_curva_larga(linha, ctx):
    """
    Formato largo: al_* e rand_* na MESMA linha.

    Vira duas execuções, porque são duas — juntá-las numa linha foi o que
    tornou este formato inagregável junto com os outros.
    """
    saida = []
    for pref, braco in (("al_", "al"), ("rand_", "random")):
        if f"{pref}fb" not in linha:
            continue
        saida.append(dict(
            split=_i(linha.get("split")),
            orcamento=_i(linha.get("budget")),
            braco=braco,
            criterio=linha.get("acquisition", UNK),
            fb=_f(linha.get(f"{pref}fb")), dice=_f(linha.get(f"{pref}dice")),
            iou=_f(linha.get(f"{pref}iou")), mae=_f(linha.get(f"{pref}mae")),
        ))
    return saida


def ad_curva_k(linha, ctx):
    """curva/ — Fbeta x K, com tempo de execução."""
    return dict(
        split=_i(linha.get("split")),
        orcamento=_i(linha.get("k")),
        braco="al", criterio="coreset",
        decoder=linha.get("decoder", UNK),
        decoder_paper=linha.get("paper_name", UNK),
        bloco=_i(linha.get("block")),
        imagens=linha.get("imagens", UNK) or UNK,
        fb=_f(linha.get("fb")), dice=_f(linha.get("dice")),
        iou=_f(linha.get("iou")), mae=_f(linha.get("mae")),
        segundos=_f(linha.get("segundos")),
    )


def ad_orcamento(linha, ctx):
    """orcamento/ — orçamento de pinceladas x nº de imagens."""
    return dict(
        split=_i(linha.get("split")),
        orcamento=_i(linha.get("n_imgs")),
        braco="al" if linha.get("criterio") not in ("aleatorio",) else "random",
        criterio=linha.get("criterio", UNK),
        imagens=linha.get("imagens", UNK) or UNK,
        fb=_f(linha.get("fb")), dice=_f(linha.get("dice")),
        iou=_f(linha.get("iou")), mae=_f(linha.get("mae")),
        segundos=_f(linha.get("segundos")),
    )


def ad_treino(linha, ctx):
    """al_metrics.csv — curva de TREINO, não avaliação. Não é execução."""
    return []


ADAPTADORES = {
    "split,criterion,seed,round,n_images,fb,dice,iou,mae,improved,selected,"
    "marker_source,pool_size,val_used,rejected": ad_paper_selection,
    "user,split,arm,decoder,paper_name,block,n_train,train_ids,fb,dice,mae,"
    "iou,fb_val": ad_per_model,
    "split,arm,decoder,paper_name,block,n_train,train_ids,fb,dice,mae,iou,"
    "fb_val": ad_per_model,
    "dataset,split,arm,decoder,paper_name,block,n_train,train_ids,fb,dice,"
    "mae,iou,fb_val": ad_per_model,
    "split,budget,method,acquisition,decoder,fb,dice,mae,iou": ad_budget_method,
    "split,budget,method,acquisition,decoder,fb,dice,mae": ad_budget_method,
    "split,budget,method,acquisition,decoder,n_train_imgs,fb,dice,mae,iou":
        ad_budget_method,
    "dataset,split,budget,method,acquisition,decoder,eval_mode,fb,dice,mae,"
    "iou": ad_budget_method,
    "split,budget,arm,seed,acquisition,n_train_imgs,coverage,fb,dice,mae,iou,"
    "collapsed": ad_runs,
    "split,budget,acquisition,al_dice,al_fb,al_mae,al_iou,rand_dice,rand_fb,"
    "rand_mae,rand_iou,delta_fb": ad_curva_larga,
    "split,budget,al_dice,al_fb,al_mae,rand_dice,rand_fb,rand_mae,delta_fb":
        ad_curva_larga,
    "split,k,decoder,paper_name,block,imagens,fb,dice,mae,iou,segundos":
        ad_curva_k,
    "split,orcamento,alocacao,n_imgs,dabs_por_img,criterio,imagens,n_seeds,"
    "fb,dice,mae,iou,segundos": ad_orcamento,
    "round,labeled_size,train_loss,proxy_dice": ad_treino,
    "split,dice,fscore,mae,unc_mean": ad_treino,
    "fname,papel,fb_medio,desvio,fb_lm,delta_lm": ad_treino,
}


def _veio_do_bruto(fonte: str) -> bool:
    """
    Um registro é migrado quando sua `fonte` aponta para um arquivo real
    dentro de evidencia/bruto/. Campanhas registradas pelo runner usam um
    rótulo lógico (ex.: "tabela_k_por_modelo/semente0"), que não existe lá.
    """
    if not fonte:
        return False
    return os.path.isfile(os.path.join(BRUTO, fonte.replace("/", os.sep)))


def migrar(dry: bool = False) -> int:
    registros, pulados, sem_adaptador, superados = [], [], {}, []

    for raiz, _, arquivos in os.walk(BRUTO):
        rel_dir = os.path.relpath(raiz, BRUTO).replace(os.sep, "/")
        if rel_dir.split("/")[0] in IGNORAR_DIRS:
            continue
        for nome in sorted(arquivos):
            if not nome.endswith(".csv"):
                continue
            if any(nome.endswith(sufixo) for sufixo in IGNORAR_ARQUIVOS):
                superados.append(os.path.relpath(
                    os.path.join(raiz, nome), BRUTO).replace(os.sep, "/"))
                continue
            caminho = os.path.join(raiz, nome)
            rel = os.path.relpath(caminho, BRUTO)
            with open(caminho, encoding="utf-8", errors="replace",
                      newline="") as fh:
                cab = fh.readline().strip()
                if cab not in ADAPTADORES:
                    sem_adaptador.setdefault(cab, []).append(rel)
                    continue
                fh.seek(0)
                linhas = list(enumerate(csv.DictReader(fh), start=2))

            ctx = do_caminho(rel)
            adapt = ADAPTADORES[cab]
            for n_linha, linha in linhas:
                saida = adapt(linha, ctx)
                if not saida:
                    pulados.append(rel)
                    continue
                for campos in (saida if isinstance(saida, list) else [saida]):
                    if campos.get("split", UNK) == UNK:
                        campos["split"] = ctx["split_caminho"]
                    campos.setdefault("dataset", ctx["dataset"])
                    campos.setdefault("usuario", ctx["usuario"])
                    campos["experimento"] = ctx["experimento"]
                    campos["esquema_origem"] = cab
                    campos["fonte"] = rel.replace(os.sep, "/")
                    campos["linha_origem"] = n_linha
                    if campos.get("marker_origem", UNK) == UNK:
                        campos["marker_origem"] = _origem_por_braco(
                            campos.get("braco", UNK), ctx["experimento"])
                    # Migrado: o commit que produziu o número se perdeu.
                    campos["git_commit"] = UNK
                    campos["config_hash"] = UNK
                    campos["hipotese"] = UNK
                    registros.append(ev.execucao(**campos))

    print(f"{len(registros)} execuções extraídas")
    if sem_adaptador:
        print(f"\n{len(sem_adaptador)} schema(s) sem adaptador "
              f"({sum(len(v) for v in sem_adaptador.values())} arquivos):")
        for cab, fs in sorted(sem_adaptador.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(fs):>3}x {cab[:88]}")
    if pulados:
        print(f"\n{len(set(pulados))} arquivo(s) de curva de treino "
              "ignorados (não são execuções avaliadas)")
    if superados:
        print(f"\n{len(superados)} arquivo(s) superado(s), excluído(s):")
        for sup in superados:
            print(f"  {sup}")

    if dry:
        print("\n--dry-run: nada gravado")
        return 0

    # Preserva o que NÃO veio de evidencia/bruto/.
    #
    # `substituir=True` reescreve o arquivo inteiro, e a migração só reconstrói
    # os registros históricos. Sem esta linha, rodar a migração depois de uma
    # campanha nova apagaria a campanha — e a ordem que este projeto
    # documenta ("arquivar, migrar, gerar tabelas") faz exatamente isso.
    #
    # Registros migrados são reconhecíveis pela `fonte`, que é o caminho
    # relativo dentro de evidencia/bruto/. Qualquer outra origem é campanha
    # registrada direto pelo runner e sobrevive.
    fontes_brutas = {r["fonte"] for r in registros}
    preservados = [r for r in ev.carregar()
                   if r.get("fonte") not in fontes_brutas
                   and not _veio_do_bruto(r.get("fonte", ""))]
    if preservados:
        print(f"\n{len(preservados)} registro(s) de campanha preservados "
              "(não vieram de evidencia/bruto/)")

    r = ev.registrar(registros + preservados, substituir=True)
    print(f"\ngravados: {r['novos']} novos, {r['duplicados']} duplicados "
          f"(mesma configuração), total {r['total']}")
    if r["divergentes"]:
        print(f"ATENÇÃO: {len(r['divergentes'])} run_id repetidos com "
              f"métricas DIFERENTES — mesma configuração, resultado outro:")
        for rid in r["divergentes"][:5]:
            print(f"  {rid}")
    print()
    print(ev.resumo())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(migrar(ap.parse_args().dry_run))
