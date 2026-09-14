"""
Testes de sanidade do ambiente — o que muda resultado sem avisar.

Estes testes não verificam o método; verificam que o ambiente em que ele roda
é o mesmo que produziu os números da dissertação. Cada um deles corresponde a
uma divergência que já aconteceu de verdade neste projeto.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(RAIZ, "flim_ad", "libs", "flim-python")


def test_pyflim_vem_do_vendor():
    """
    O pyflim que roda tem de ser o de flim_ad/libs/flim-python.

    Há DUAS cópias de pyflim no repositório: `pyflim/` na raiz e a vendorizada
    em `flim_ad/libs/flim-python/pyflim/`. Os scripts inserem a segunda primeiro
    no sys.path, e é ela que tem os patches locais. Importar a da raiz por
    engano roda outro código com o mesmo nome.
    """
    if not os.path.isdir(VENDOR):
        pytest.skip("flim_ad/libs/flim-python ausente neste checkout")
    sys.path.insert(0, VENDOR)
    spec = importlib.util.find_spec("pyflim")
    assert spec is not None and spec.origin is not None
    assert os.path.normcase(VENDOR) in os.path.normcase(spec.origin), (
        f"pyflim veio de {spec.origin}, não do vendor em {VENDOR}"
    )


def test_kmeans_usa_o_mesmo_caminho_dos_resultados():
    """
    O k-means do encoder precisa ser o fallback do sklearn, não o faiss.

    `FLIMModel.cluster_patches_faiss` escolhe faiss quando o import funciona e
    cai no `sklearn.KMeans` quando não. Os dois dão centróides diferentes, e
    portanto kernels diferentes, e portanto Fβ diferente.

    TODOS os resultados da dissertação vieram do fallback do sklearn: na imagem
    `flim-ad-env` o faiss está quebrado (binding SWIG), e o `Dockerfile.gpu`
    deliberadamente NÃO o instala para manter a comparabilidade. Um ambiente
    com faiss funcional — como um `pip install` ingênuo num venv local — troca
    o caminho silenciosamente e produz números que não se comparam com os
    antigos.

    Se este teste falhar, os números gerados aqui não pertencem à mesma série
    experimental. Ou desinstale o faiss, ou rotule os resultados como de outra
    série.
    """
    if not os.path.isdir(VENDOR):
        pytest.skip("flim_ad/libs/flim-python ausente neste checkout")
    sys.path.insert(0, VENDOR)
    from pyflim import flim as flimlib

    assert flimlib.faiss is None, (
        "faiss está disponível e o pyflim vai usá-lo no lugar do sklearn "
        "KMeans. Todos os resultados existentes vieram do sklearn. "
        "Rode `pip uninstall faiss-cpu faiss-gpu` no ambiente de experimento, "
        "ou registre explicitamente que esta série usa outro k-means."
    )


def test_numpy_shim_presente():
    """
    O shim de numpy 2.x precisa estar no pyflim vendorizado.

    Sem ele, `int(kernel_labels[f])` levanta
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`
    quando a imagem de markers tem dimensão de canal. Funcionava em numpy 1.x
    (o do container) e quebra em numpy 2.x (o do ambiente nativo).
    """
    caminho = os.path.join(VENDOR, "pyflim", "flim.py")
    if not os.path.isfile(caminho):
        pytest.skip("pyflim vendorizado ausente neste checkout")
    with open(caminho, encoding="utf-8") as fh:
        fonte = fh.read()
    assert "_rotulo_escalar" in fonte, (
        "o shim de numpy 2.x sumiu de flim_ad/libs/flim-python/pyflim/flim.py "
        "— reaplique vendor/flim_ad/local-changes.patch"
    )


@pytest.mark.parametrize("pacote,minimo", [("numpy", 1), ("torch", 1)])
def test_dependencias_minimas_importam(pacote, minimo):
    """Falha cedo e com mensagem clara em vez de estourar dentro do treino."""
    mod = pytest.importorskip(pacote)
    assert int(mod.__version__.split(".")[0]) >= minimo
