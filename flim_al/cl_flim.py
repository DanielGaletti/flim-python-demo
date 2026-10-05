#!/usr/bin/env python3
"""
cl_flim.py — aprendizado contínuo para o FLIM, sem backpropagação

O problema, medido
    Acrescentar uma anotação ao FLIM pode destruir o encoder. Na campanha
    `ganho_marginal` deste projeto, um único acréscimo de 300 pixels na região
    errada levou o Fbeta de validação de 0,78 para 0,00, em 6 de 6 células
    (dataset x decoder), com p <= 0,0015.

    Isso é esquecimento catastrófico. O que ele NÃO é, aqui, é esquecimento por
    gradiente: o FLIM não tem backpropagação. Então Destilação de Conhecimento,
    EWC e replay, que supõem um modelo treinado por descida de gradiente, não
    se aplicam diretamente.

Onde o esquecimento acontece
    `FLIMModel.learn_layer_weights` junta os candidatos de kernel de todas as
    imagens e de todos os marcadores num conjunto só, e então roda um k-means
    FINAL que o reduz a `noutput_channels`:

        kernels, _ = self.kmeans(combined_kernel_candidates,
                                 layer_parameters["noutput_channels"])

    É esse k-means final. Quando entra uma anotação nova, novos candidatos
    entram no conjunto e o k-means REPARTICIONA tudo: os centróides que estavam
    funcionando se movem ou desaparecem. O banco de filtros é reescrito, não
    estendido.

O mecanismo proposto
    Em vez de aceitar o novo banco, interpola-o com o anterior. Para cada
    centróide anterior, acha o novo mais próximo e devolve

        banco[j] = (1 - alpha) * anterior[j] + alpha * novo[casado(j)]

    `alpha` é a taxa de plasticidade, e ele expõe o dilema
    estabilidade-plasticidade na própria mecânica do FLIM:

        alpha = 0    retenção total. O banco não muda; a anotação nova não
                     entra. Imune ao esquecimento, incapaz de aprender.
        alpha = 1    FLIM puro. O banco é o novo, e é por isso que este valor
                     serve de TESTE DE CORRETUDE: com o mesmo número de
                     centróides e casamento bijetivo, alpha=1 devolve uma
                     permutação do banco que o FLIM puro produziria, e o Fbeta
                     tem de bater.
        0 < a < 1    o banco se move parcialmente na direção da anotação nova.

    A âncora é por centróide NOVO: cada um é puxado na direção do centróide
    anterior de que mais se parece. O banco de saída tem o tamanho que a rodada
    corrente espera, e não o do banco anterior, porque
    `learn_layer_weights` encolhe `noutput_channels` quando há poucos
    candidatos e reconstrói o Conv2d junto com as estatísticas de
    normalização. Forçar o tamanho antigo quebra a contabilidade da
    biblioteca.

    Com alpha=0 o banco passa a conter apenas direções já presentes no
    anterior, eventualmente repetidas, e nenhuma informação nova entra.

Uso
    # 1) treino normal, gravando o banco cru
    treinar(arch, markers0, orig, label, dev, enc0, gravar_banco=banco0)

    # 2) anotação nova, com protecao
    treinar(arch, markers1, orig, label, dev, enc1,
            banco=banco0, alpha=0.3)
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(RAIZ, "flim_ad", "libs", "flim-python") not in sys.path:
    sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))


# ── o banco ─────────────────────────────────────────────────────────────────

class Banco:
    """
    Intercepta o k-means final de cada camada: grava ou protege.

    `self.kmeans` do FLIMModel atende DUAS chamadas por camada, com números de
    cluster diferentes: `nkernels_per_marker` (por componente de marcador) e
    `noutput_channels` (a final). Só a final é protegida, e ela é identificada
    pela camada corrente mais a primeira chamada cujo `n_clusters` bate com o
    `noutput_channels` daquela camada.

    Identificar pelo valor de `n_clusters` sozinho não serve: se a arquitetura
    pedir `noutput_channels` igual a `nkernels_per_marker`, as duas chamadas
    ficariam indistinguíveis. O BraTS pede 8 canais de saída e 5 por marcador,
    e a conjuntivite pede 16/12/8/6 — perto o bastante para o risco ser real.
    """

    def __init__(self, anterior: dict = None, alpha: float = 1.0):
        self.anterior = anterior or {}
        self.alpha = float(alpha)
        self.gravado = {}
        self._camada = None
        self._final_vista = set()
        self._orig = None
        self.protegidas = []      # (camada, n_ant, n_novo, n_casados)

    # ── instalação ──────────────────────────────────────────────────────────

    def instalar(self, modelo):
        """Troca `modelo.kmeans` e envolve `learn_layer_weights`."""
        from pyflim.flim import FLIMModel

        self._orig = modelo.kmeans
        orig_learn = modelo.learn_layer_weights

        def learn(X, M, l, _o=orig_learn):
            self._camada = l
            try:
                return _o(X, M, l)
            finally:
                self._camada = None

        def kmeans(patches, n_clusters, _m=modelo):
            l = self._camada
            esperado = None
            if l is not None and l < len(_m.architecture.layers):
                esperado = _m.architecture.layers[l].get("noutput_channels")
            e_final = (l is not None and esperado == n_clusters
                       and l not in self._final_vista)
            saida = self._orig(patches, n_clusters)
            if not e_final:
                return saida
            self._final_vista.add(l)
            centroides, closest = saida
            novo = self._aplicar(l, np.asarray(centroides))
            return novo, closest

        self._orig_learn = orig_learn
        self._modelo = modelo
        modelo.learn_layer_weights = learn
        modelo.kmeans = kmeans
        return modelo

    def desinstalar(self):
        """
        Devolve os métodos originais ao modelo.

        Obrigatório antes de `torch.save`: o encoder é serializado como OBJETO
        inteiro, e `learn`/`kmeans` são funções locais de `instalar`, que o
        pickle não consegue referenciar. Sem isto o treino roda e a gravação
        falha com `Can't get local object`.
        """
        m = getattr(self, "_modelo", None)
        if m is None:
            return
        if self._orig is not None:
            m.kmeans = self._orig
        if getattr(self, "_orig_learn", None) is not None:
            m.learn_layer_weights = self._orig_learn
        self._modelo = None

    # ── o nucleo ────────────────────────────────────────────────────────────

    def _aplicar(self, camada: int, novo: np.ndarray) -> np.ndarray:
        self.gravado[camada] = np.array(novo, copy=True)
        ant = self.anterior.get(camada)
        if ant is None or self.alpha >= 1.0:
            return novo                      # sem banco anterior = FLIM puro
        ant = np.asarray(ant)
        if ant.shape[1:] != novo.shape[1:]:
            # forma de patch diferente: a camada anterior mudou de
            # dimensionalidade, e interpolar seria comparar coisas distintas.
            return novo
        ancora = self._ancorar(ant, novo)
        saida = (1.0 - self.alpha) * ancora + self.alpha * novo
        self.protegidas.append((camada, ant.shape[0], novo.shape[0],
                                int(novo.shape[0])))
        self.gravado[camada] = np.array(saida, copy=True)
        return saida.astype(novo.dtype)

    @staticmethod
    def _ancorar(ant: np.ndarray, novo: np.ndarray) -> np.ndarray:
        """
        Para cada centróide NOVO, o anterior mais próximo.

        A direção importa. Casar o anterior com o novo (um par para cada
        anterior) obrigaria a saída a ter o tamanho do banco anterior, e isso
        quebra: `learn_layer_weights` encolhe `noutput_channels` quando há
        poucos candidatos e RECONSTRÓI o Conv2d, e as médias e desvios de
        normalização passam a ter o tamanho da rodada nova. Forçar o tamanho
        antigo gerava `shape '[1, 153, 1, 1]' is invalid for input of size
        157`.

        Ancorando na direção oposta, a saída tem o tamanho que a rodada
        corrente espera, a contabilidade da biblioteca continua válida, e a
        semântica de retenção se mantém: cada filtro novo é puxado na direção
        do filtro antigo que ele mais se parece. Com alpha=0 o banco passa a
        conter apenas direções já presentes no anterior, eventualmente
        repetidas, e nenhuma informação nova entra.

        Sem repetição também seria possível, por Hungarian, mas custa caro e
        não muda o que o experimento mede: a versão gulosa com proibição de
        repetição levava 39 s por treino contra 8 s desta.
        """
        a = ant.reshape(ant.shape[0], -1)
        b = novo.reshape(novo.shape[0], -1)
        # ||b_i - a_j||^2 = |b_i|^2 - 2 b_i.a_j + |a_j|^2; o termo |b_i|^2 nao
        # muda o argmin em j, entao sai da conta.
        d = (a * a).sum(1)[None, :] - 2.0 * (b @ a.T)
        viz = np.argmin(d, axis=1)
        return ant[viz].reshape(novo.shape)


# ── persistencia ────────────────────────────────────────────────────────────

def salvar_banco(banco: Banco, caminho: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    with open(caminho, "wb") as fh:
        pickle.dump({int(k): np.asarray(v) for k, v in banco.gravado.items()},
                    fh)


def carregar_banco(caminho: str) -> dict:
    with open(caminho, "rb") as fh:
        return pickle.load(fh)


# ── treino ──────────────────────────────────────────────────────────────────

def treinar(arch_file: str, marker_dir: str, orig_folder: str,
            label_folder: str, device: str, saida: str,
            banco: str = None, alpha: float = 1.0,
            gravar_banco: str = None, bits: int = 8):
    """
    Treina o encoder, opcionalmente protegendo o banco de filtros.

    Reimplementa o caminho de `al_encoder_experiment.retrain_encoder` porque
    precisa do objeto modelo ANTES do `fit`, para instalar a interceptação.
    Os parâmetros do dataset são os mesmos, de propósito: trocar qualquer um
    tornaria os números incomparáveis com as campanhas já registradas.
    """
    import torch
    from pyflim import flim as flimlib, data as flimdata, arch as flimarch

    imagens = [f.replace("-seeds.txt", "") for f in os.listdir(marker_dir)
               if f.endswith("-seeds.txt")]
    # Todos os argumentos abaixo repetem `retrain_encoder` exatamente. Trocar
    # qualquer um tornaria os numeros incomparaveis com as campanhas ja
    # registradas, que e o unico motivo de nao chamar aquela funcao: aqui o
    # objeto modelo e preciso ANTES do fit, para instalar a interceptacao.
    ds = flimdata.FLIMData(
        orig_folder=orig_folder,
        marker_folder=marker_dir.rstrip("/") + "/",
        images_list=imagens,
        label_folder=label_folder,
        orig_ext=".png", marker_ext="-seeds.txt", label_ext=".png",
        transform=flimdata.transforms.Compose([flimdata.ToTensor()]),
        bits=bits,
        convert_gray_to_lab=False,
    )
    arq = flimarch.FLIMArchitecture(arch_file)
    modelo = flimlib.FLIMModel(
        arq,
        device=device,
        decoder_type="vanilla_adaptive_decoder",
        adj_radius=1.5,
        multi_layer=False,
    )

    b = Banco(carregar_banco(banco) if banco else None, alpha)
    b.instalar(modelo)

    # Uma thread. Sem isso o treino nao e reproduzivel quando o numero de
    # candidatos passa do teto da arquitetura: o k-means final cai em outra
    # bacia e o banco de filtros inteiro muda. Medido em flim_al/determinismo.
    from flim_al.determinismo import fixar
    with fixar():
        modelo.fit(ds)
    b.desinstalar()
    if os.path.dirname(saida):
        os.makedirs(os.path.dirname(saida), exist_ok=True)
    torch.save(modelo, saida)
    if gravar_banco:
        salvar_banco(b, gravar_banco)
    return modelo, b
