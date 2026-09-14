# CLAUDE.md — dissertação FLIM + Active Learning

Contexto de leitura obrigatória antes de qualquer alteração. Complementa
`ESTADO_ATUAL.md`, que é a fonte de verdade sobre o **estado científico**;
este arquivo trata de **como trabalhar** neste repositório.

Defesa 14/11/2026 · depósito 05–16/10/2026.

---

## 1. Integridade científica — regras absolutas

Estas não cedem a pressa, a pedido, nem a "só para o slide de amanhã".

- **NUNCA fabricar referência.** Todo paper citado precisa de DOI, arXiv ID ou
  URL verificável. Se não der para verificar, escrever `UNKNOWN`, não um
  palpite plausível.
- **NUNCA alterar resultado experimental à mão.** Números vêm de arquivo de
  resultado, sempre.
- **NUNCA esconder resultado negativo.** `ESTADO_ATUAL.md` §3 lista quatro
  hipóteses refutadas, com o custo de cada uma. Isso é o padrão, não a
  exceção — hipótese refutada é resultado de pesquisa.
- **NUNCA afirmar significância sem o teste.** "Melhorou" e "melhorou
  significativamente" são afirmações diferentes. Sem teste pareado e sem n,
  a segunda não se escreve.
- **NUNCA escrever na dissertação uma afirmação que não passou por validação
  independente.** Observação ≠ interpretação ≠ claim.
- **NUNCA otimizar contra o conjunto de teste.** O teste comum de 364 imagens
  existe para ser tocado uma vez por configuração final.
- **NUNCA apagar proveniência** de experimento — configs, seeds, CSVs brutos.
- **NUNCA deixar o mesmo agente ser implementador e validador** de um mesmo
  claim científico.
- **NUNCA copiar número à mão para tabela oficial** quando ele pode ser lido
  de um resultado estruturado.
- **Reprodutibilidade vale mais que número bonito.** Entre um resultado melhor
  e um resultado que se reproduz, o segundo.

### Relatar resultado parcial como final é um erro grave

Já aconteceu três vezes neste projeto: o "empate" apareceu e sumiu conforme as
execuções subiram de 1 → 3 → 9; o CoreSet mostrou 55% com 7 de 9 seeds e
fechou em 16%. **Não relatar número de execução incompleta.** Se precisar
falar antes do fim, dizer explicitamente quantas de quantas rodaram.

---

## 2. Armadilhas deste repositório

Cada uma destas já custou tempo ou resultado errado.

### 2.1 Existem DUAS cópias do `pyflim`

```
pyflim/                            ← a da raiz, NÃO é a que roda
flim_ad/libs/flim-python/pyflim/   ← esta é a que roda
```

Os scripts inserem a segunda primeiro no `sys.path`. Patches aplicados só na
primeira não têm efeito nenhum. Ao corrigir algo no `pyflim`, **corrigir nas
duas** ou verificar qual está sendo importada.

### 2.2 faiss muda os resultados silenciosamente

`FLIMModel.cluster_patches_faiss` usa `faiss.Kmeans` quando o import funciona
e cai no `sklearn.KMeans` quando não. Centróides diferentes → kernels
diferentes → Fβ diferente.

**Todos os resultados da dissertação vieram do fallback do sklearn.** O
`Dockerfile.gpu` não instala faiss de propósito. Um `pip install -r
requirements.txt` ingênuo instala, e a partir daí os números deixam de ser
comparáveis sem nenhum aviso.

`tests/test_ambiente_experimental.py` falha se faiss estiver presente.

### 2.3 `flim_ad/` é um repositório git aninhado

É um clone de `LIDS-UNICAMP/flim_ad`, não parte deste repositório. O git daqui
não desce nele. Alterações locais ali são invisíveis ao histórico deste
projeto — por isso existe `vendor/flim_ad/local-changes.patch`. **Ao mexer em
`flim_ad/`, regenerar o patch.**

### 2.4 `flim_ad/out/` tem 230 mil arquivos e é ignorado

Os experimentos escrevem lá. A evidência tabular (CSV/JSON) é arquivada em
`evidencia/bruto/` por `scripts/arquivar_resultados.sh`, e **é essa cópia que
entra no git**. Rodar o script depois de cada campanha de experimentos.

### 2.5 O `requirements.txt` não é deste projeto

É o da biblioteca FLIM-Python upstream: incompleto (falta `monai`,
`transformers`, `scipy`) e pede faiss. Use `requirements-dissertacao.txt`.

### 2.6 A aplicação interativa não produz evidência científica

`flim_app/` é demonstração. Os números que ela mostra vêm de uma execução
única com K pequeno, onde a amplitude do braço aleatório (0,10–0,22) é maior
que qualquer efeito observável. **Nunca citar número da aplicação na
dissertação.**

---

## 3. Como as coisas rodam

```bash
# aplicação interativa (nativo, sem Docker — o engine desta máquina está quebrado)
cd flim_ad && "C:/Users/galet/anaconda3/envs/flim-python/python.exe" -u ../flim_app/server.py

# testes
python -m pytest

# arquivar resultados novos para dentro do git
bash scripts/arquivar_resultados.sh

# experimentos (dentro do container, com GPU)
bash run_all_dissertation.sh --phases "1 2"
```

Os scripts de experimento assumem **cwd = `flim_ad/`**. Rodar da raiz quebra
os caminhos relativos dos datasets.

---

## 4. Convenções

- **Português** em documentação, comentários e mensagens de interface;
  identificadores de código em inglês quando já for a convenção do arquivo.
- **Comentários explicam o porquê**, não o quê. O padrão do repositório é
  registrar a medição que motivou a decisão (ver `Dockerfile.gpu`,
  `flim_app/duplo.py`). Manter esse padrão.
- **Nada de reescrita ampla.** Este repositório carrega meses de trabalho
  experimental; mudança grande de estrutura invalida caminhos que scripts e
  resultados já assumem. Preferir adição.
- **Antes de apagar qualquer coisa**, registrar em `vendor/REMOVIDOS.txt` o
  que era e por quê.

---

## 5. Separação entre observação e afirmação

Quatro níveis distintos, que não devem colapsar um no outro:

| nível | exemplo |
|---|---|
| **dado bruto** | Fβ de cada seed, em `evidencia/` |
| **observação** | "a média subiu de 0,688 para 0,697" |
| **resultado agregado** | "média ± desvio, IC95%, teste pareado p = 0,33" |
| **interpretação** | "CoreSet parece mais eficiente em amostra" |
| **claim** | "sob o dataset D e orçamento B, CoreSet melhorou a eficiência de amostra em relação ao sorteio" |

Observação nunca vira claim automaticamente.

---

## 6. Git

- Nunca `force push`, nunca reescrever histórico, nunca apagar branch com
  trabalho não consolidado.
- Nunca commitar credencial. O remote já teve um PAT em texto puro; o
  histórico é público.
- Datasets (14 GB) e saídas de predição (2 GB) ficam fora do git por regra
  explícita no `.gitignore` — não contornar com `git add -f`.
