# Protocolo revisado — FLIM + Active Learning

Documento de referência para o capítulo de metodologia. Descreve o que estava
errado no protocolo anterior, o que mudou, e como ler os resultados.

---

## 1. Defeitos corrigidos

### 1.1 Ruído de inicialização não controlado (Experimento B) — **crítico**

O decoder 1×1 era inicializado com `xavier_uniform_` sem semente. O `pyflim`
chama `torch.manual_seed(1234)` no import, então cada treino consumia a posição
seguinte do stream global: o braço AL (1º treino) recebia uma inicialização e o
braço Random (2º em diante), outra.

**Como isso foi detectado.** Em `budget = N` (pool inteiro) o AL seleciona
`ranking[:N]` = todas as imagens e `random.sample(fnames, N)` = as mesmas
imagens. Os dois braços treinam com **dados idênticos**, então ΔFβ deveria ser
zero. Medido:

| método | Split 1 | Split 2 | Split 3 |
|---|---|---|---|
| entropy | −0.046 | +0.109 | −0.109 |
| badge | −0.046 | +0.109 | −0.109 |

`entropy` e `badge` deram valores idênticos até a 4ª casa — prova de que os
dados eram os mesmos e o pipeline é determinístico dada a posição no stream.
**A inicialização sozinha movia o Fβ em até 0.109**, mais que qualquer ΔFβ
reportado. Um teste posterior com K=3 mostrou o mesmo conjunto de 3 imagens
indo de Fβ=0.6228 a Fβ=0.4793 (colapso total) só trocando a semente.

Agravante: o braço AL rodava **uma vez** e o Random era a média de 3. Uma
amostra única contra uma média de três.

**Correção.** `init_seed` explícito em ambos os treinadores. O par de índice `s`
usa `torch.manual_seed(1000 + s)` nos dois braços, então

```
Δ_s = Fβ(AL, s) − Fβ(Random, s)
```

cancela a inicialização e isola a seleção de imagens. Ambos os braços rodam
`--n_seeds` vezes. Validado em `tests/test_paired_seeding.py`: mesma semente →
`max|Δw| = 0`; sementes diferentes → `max|Δw| = 0.44`.

### 1.2 Loss mascarada travada (region_entropy)

A máscara era aplicada como `pred * mask` e `gt * mask`, o que zera os
**logits** fora da região — e `sigmoid(0) = 0.5`. O `DiceLoss` passava a
enxergar meio-foreground em ~87% da imagem, o que domina o denominador e trava
o termo Dice em ≈1.0.

A loss observada batia com a previsão do defeito:

```
dice(≈1.0) + BCE_fora(0.693 × 0.872) = 1.604      observado: 1.54 – 1.61
```

**Correção.** `flim_al/losses.py` aplica a máscara como **peso por pixel**:
fora da região não entra no numerador nem no denominador do Dice, nem na média
da BCE. Verificado em `tests/test_masked_dice_ce.py`:

- `masked_dice_ce(mask=None)` é **numericamente idêntico** a
  `DiceCELoss(sigmoid=True)` do MONAI (diferença 0.00e+00), então `entropy` e
  `region_entropy` diferem só pela máscara;
- com predição **perfeita** dentro da região, a loss antiga fica em 1.34 e a
  nova cai para 0.0005.

### 1.3 Dice sobre logits crus

`DiceCELoss()` do MONAI tem `sigmoid=False` por padrão — o flag é usado
*apenas* pelo `DiceLoss`, não pela BCE. O termo BCE tratava logits
corretamente, mas o Dice recebia valores não-limitados (podendo ser negativos).
Corrigido para `sigmoid=True` (agora embutido em `masked_dice_ce`).

### 1.4 CoreSet devolvendo duplicatas

`coreset_select` pedia `n_init + budget` pontos de um pool de `N`, descartando
o ponto semente. Com `budget ≥ N` isso devolvia índices repetidos — a mesma
imagem entrava duas vezes no treino. Corrigido: o ponto semente conta para o
budget (padrão de Sener & Savarese) e o alvo é limitado a `N`. Há um `assert`
contra duplicatas.

> **Consequência:** a seleção do CoreSet mudou em todos os budgets. Os
> resultados anteriores de CoreSet estão obsoletos e precisam ser reexecutados.

### 1.5 Braços não comparáveis no Experimento A — **crítico**

Para `region_*`, o braço AL usava seeds de superpixel e o braço Random usava
**sempre** seeds em pontos aleatórios. Os dois braços diferiam em duas
variáveis ao mesmo tempo: *quais* imagens e *como* anotá-las. O Δ não media
seleção.

Além disso, o caminho de região descarta imagens sem região de foreground,
enquanto o de pontos nunca descarta — então o braço AL frequentemente treinava
com menos imagens. Exemplo de uma execução do `region_bald`:

| config | imagens adicionadas | AL treinou com | Random treinou com |
|---|---|---|---|
| S3 K=3 | 1 de 3 | 4 | 6 |
| S3 K=5 | 1 de 5 | 4 | 8 |
| S3 K=10 | 3 de 10 | 6 | 13 |

Como markers sintéticos degradam o encoder, adicionar menos deles é uma
vantagem artificial.

**Correção.** Terceiro braço `random_region` — imagens aleatórias com **o mesmo
gerador de seeds do AL** e o mesmo descarte. Ele decompõe o efeito:

```
AL_region  − random_region  →  efeito da SELEÇÃO
random_region − random      →  efeito da GEOMETRIA dos seeds
```

Os dois braços aleatórios usam a mesma semente, logo sorteiam exatamente as
mesmas imagens; só muda a anotação.

Coluna `n_train_imgs` adicionada ao CSV para tornar o desbalanceamento
auditável em vez de invisível.

### 1.6 Conjunto de validação inconsistente entre A e B

O Exp A removia do val as imagens com markers reais (fix B2); o Exp B não. Os
val sets diferiam e as tabelas não eram comparáveis. Mesmo filtro aplicado nos
dois.

### 1.7 `run_region_al` ausente

Removida por engano no commit `b6b3af8`, mas ainda importada no topo de
`al_flim_backprop.py` — o `ImportError` derrubava os **quatro** métodos da
Fase 2. Restaurada, com a conversão para LAB que a versão original não tinha
(ela usava `FLIMData`, que entrega RGB — espaço errado para o encoder FLIM).

### 1.8 Detecção de GPU enganosa

`torch.cuda.is_available()` retorna `True` mesmo sem kernel compilado para a
arquitetura. Numa RTX 50xx (sm_120) com PyTorch < 2.7, a falha só apareceria no
meio do experimento. As checagens agora executam uma `conv2d` real.

---

## 2. Predições degeneradas

Um decoder que prevê fundo em tudo produz, no val set do schisto,

```
Fβ = DICE = IoU = 406/848 ≈ 0.4788
```

porque só as imagens vazias pontuam (empty-empty → 1.0) e as demais dão 0.

Isso não é desempenho — é ausência de aprendizado. E acontecia com frequência:
no Exp A, **~50% das avaliações** do decoder `labeled_marker` colapsavam, tanto
no braço AL quanto no Random. No Exp B, 3 das 9 "vitórias" do CoreSet eram
casos em que o braço AL colapsou e ainda assim superou o Random.

Agora há uma coluna `collapsed` no CSV, essas execuções são marcadas com ⚠ nas
tabelas e **excluídas do agregado**.

---

## 3. Como executar

### Experimento B (principal)

```bash
bash run_dissertation.sh --device cpu --skip-encoder
```

Rápido, sem a referência do pool inteiro (~6× mais rápido, mas sem teto nem
piso de ruído):

```bash
bash run_dissertation.sh --device cpu --skip-encoder --skip-full-pool
```

### Experimento A — só o que mudou

O braço de controle novo e a correção do CoreSet não exigem reexecutar tudo.
O resume pula o que já existe:

```bash
# adiciona só o braço random_region ao region_bald
bash run_schisto_corrected.sh --device cpu --methods region_bald

# CoreSet mudou de seleção: mova o CSV antigo e rode de novo
mv flim_ad/out/al_corrected_results/coreset/schisto-user_A_coreset_encoder_al.csv \
   flim_ad/out/al_corrected_results/coreset/OBSOLETO_pre_fix_seedpoint.csv
bash run_schisto_corrected.sh --device cpu --methods coreset
```

`entropy`, `least_confidence` e `badge` não foram afetados.

### Tabelas

```bash
cd flim_ad && python3 ../flim_al/analyze_dissertation.py --out ../RESULTADOS_DISSERTACAO.md
```

### GPU (RTX 50xx / sm_120) — **resolvido: use `flim-ad-env:gpu`**

```bash
docker exec flim-gpu bash -c \
  "cd /workspace/flim-python-demo && bash run_dissertation.sh --device cuda:0 --skip-encoder"
```

A imagem é construída por [`Dockerfile.gpu`](Dockerfile.gpu) a partir da
`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` (Ubuntu 22.04, glibc 2.35),
que traz `sm_120` compilado. Código e dados vêm do bind mount — a imagem só
carrega dependências, com versões fixadas no mesmo conjunto que produziu os
resultados atuais.

**Validação da migração** (`tests/test_gpu_migration.py`): os encoders salvos
como pickle sob torch 2.6 / Python 3.10 carregam em 2.7 / 3.11, e a rota CPU da
imagem nova reproduz a antiga com erro relativo **0.00e+00** — bit-idêntico. A
divergência CPU↔GPU é 5.3e-05 relativo, ruído float32 esperado e desprezível
frente à dispersão de Fβ (±0.1) medida no experimento.

Duas ressalvas:

1. `faiss` **não** é instalado de propósito. Na imagem antiga ele falha
   (binding SWIG) e o pyflim cai no fallback de KMeans do sklearn — caminho que
   gerou todos os resultados existentes. Instalar um faiss funcional mudaria o
   treino do encoder e quebraria a comparabilidade.
2. O ganho de velocidade é **modesto no Exp A**: `evaluate_decoder`
   ([al_encoder_experiment.py:358](flim_al/al_encoder_experiment.py:358)) fixa
   `eval_device = "cpu"`, então a avaliação dos 7 decoders continua em CPU. A
   GPU acelera o Exp B por inteiro e, no Exp A, só o treino do encoder. Num
   mini-benchmark (split 1, K=3, 2 sementes, 20 épocas) a GPU levou 7m03s
   contra ~13m estimados em CPU. O gargalo restante é o laço de avaliação, que
   carrega e converte 847 imagens para LAB uma a uma na CPU.

---

### Histórico: por que a imagem antiga não servia

Investigado e fechado. A cadeia de fatos:

| | |
|---|---|
| GPU | RTX 5070, compute capability **sm_120** |
| PyTorch instalado | 2.6.0+cu124, kernels só até **sm_90** |
| Suporte a sm_120 | entra no PyTorch **2.7 + CUDA 12.8** |
| Wheels cu128 | publicados com a tag **`manylinux_2_28`** (glibc ≥ 2.28) |
| Container | **Ubuntu 18.04, glibc 2.27** |

2.27 < 2.28, então o pip recusa os wheels e reporta `from versions: none` —
mensagem que parece "essa versão não existe", mas significa "nenhuma
compatível com este sistema". Confirmado: os wheels cp310/cu128 existem
(2.7.0 a 2.11.0), e o índice cu124 é lido normalmente pelo mesmo pip. Não é
problema de versão do pip (testado com 22.3.1 e 26.2.1).

Nenhuma flag resolve. As opções são:

1. **Continuar em CPU.** Resultados idênticos, só mais lentos.
2. **Reconstruir a imagem** sobre base com glibc ≥ 2.28 (Ubuntu 22.04 = 2.35,
   Debian 12 = 2.36) e reinstalar pyflim, monai, scikit-image e as libs IFT.
3. **Imagem NGC da NVIDIA** (`nvcr.io/nvidia/pytorch:25.01-py3`), que já traz
   suporte a Blackwell, reinstalando as dependências do projeto por cima.

`bash setup_gpu_blackwell.sh` detecta essa incompatibilidade e para, em vez de
tentar uma instalação que não muda nada.

> Nota: o pip do container foi atualizado de 22.3.1 para 26.2.1 durante o
> diagnóstico. Não alterou o PyTorch e todas as dependências seguem
> importando (torch 2.6.0+cu124, monai 1.5.2, skimage 0.22.0, sklearn 1.3.2,
> numpy 1.26.4, scipy 1.11.4).

---

## 4. Como ler os resultados

`analyze_dissertation.py` emite:

| seção | conteúdo |
|---|---|
| B.1 | teto do decoder e piso de ruído (braço `full`) |
| B.2 | ΔFβ pareado por budget, com IC95%, teste t e Wilcoxon |
| B.3 | agregado por método, excluindo execuções colapsadas |
| A.1 | imagens por braço — expõe desbalanceamento |
| A.2 | decomposição seleção vs geometria dos seeds |
| A.3 | Fβ por decoder, com contagem de colapsos |

Marcadores:

- `**` — IC95% não cruza zero **e** efeito maior que o piso de ruído medido
- `~` — direção consistente, mas magnitude dentro do ruído
- `⚠` — alguma execução do braço AL colapsou

**Regra de leitura:** nenhum ΔFβ menor que o desvio do braço `full` deve ser
interpretado como efeito real, por mais consistente que pareça o sinal.

---

## 5. Limitações que permanecem

1. **AL é one-shot.** O ranking sai do encoder original e não é recalculado
   após cada seleção. Limita o ganho teórico esperado.
2. **Teto baixo do decoder do Exp B.** Na execução anterior, o decoder 1×1
   treinado com o pool inteiro (177–196 imagens com GT completo) atingia
   Fβ ≈ 0.54, contra 0.744 do `labeled_marker` com 3 imagens de markers. O
   Exp B mede seleção de imagens dentro desse teto — não é uma comparação
   com o FLIMpb do paper.
3. **Markers sintéticos degradam o encoder** (Exp A). Todos os métodos ficam
   abaixo do baseline; o que se compara é AL vs Random, não vs baseline.
4. **DT desativado.** `iftSMansoniDelineation` existe mas faltam `liblapack.so.3`
   e `libblas.so.3`. Toda a avaliação usa Otsu+AF, internamente consistente mas
   não comparável à Tabela III do paper.
5. **Cobertura de anotação fixa** no `region_entropy`: 5 patches de 64×64 dão
   12.8% em toda imagem, então não há curva de custo de anotação para reportar.
