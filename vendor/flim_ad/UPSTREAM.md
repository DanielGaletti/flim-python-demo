# flim_ad — dependência vendorizada

`flim_ad/` é um **clone de outro repositório**, não parte deste. O git deste
projeto não desce nela (repositório aninhado), então nada lá dentro entra no
histórico daqui. Este diretório existe para que isso não custe reprodutibilidade.

| campo | valor |
|---|---|
| upstream | https://github.com/LIDS-UNICAMP/flim_ad |
| commit fixado | `caba55623a36d5837cb397d28e32278c7201f07c` |
| capturado em | 2026-09-14 |

## Por que importa

O `pyflim` que roda de verdade é o de `flim_ad/libs/flim-python/`, **não** o
`pyflim/` da raiz — os scripts inserem aquele caminho primeiro no `sys.path`.
Ele tem patches locais que **mudam resultados experimentais**, entre eles o
shim de compatibilidade com numpy 2.x em `flim.py` (sem ele, treinar com
markers de certas geometrias levanta `TypeError: only 0-dimensional arrays`).

## Como reconstruir o ambiente

```bash
git clone https://github.com/LIDS-UNICAMP/flim_ad
cd flim_ad && git checkout caba55623a36d5837cb397d28e32278c7201f07c
git apply ../vendor/flim_ad/local-changes.patch
```

O patch cobre `libs/flim-python/`, `src/` e `scripts/`, ignorando diferenças
de fim de linha (o clone no Windows marca ~40 arquivos como modificados só por
CRLF). Datasets e saídas não entram — vêm de fora.

## O que NÃO está capturado aqui

- `flim_ad/datasets/` (2,1 GB) e `flim_ad/data/`
- `flim_ad/out/` (230 mil arquivos); os CSVs de resultado estão em `evidencia/`
- `libs/ift/` — binários e fontes C do IFT, sem alteração nossa relevante
