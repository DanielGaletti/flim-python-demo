# Como esta pasta está organizada

Projeto da dissertação em LaTeX, classe `ufscar.cls` (PPGCC UFSCar).

**No Overleaf, o documento principal é `main.tex`.** Se o Overleaf não o
escolher sozinho, ajustar em *Menu → Settings → Main document*.

```
main.tex              documento principal: capa, folha de aprovação, resumo,
                      sumário e a ordem dos capítulos
ufscar.cls            classe do programa, não alterar

capitulos/            o texto, um arquivo por capítulo
  1_introducao.tex
  2_fundamentos.tex     FLIM, Active Learning, IL, CL, trabalhos relacionados
  3_metodologia.tex     formulação, métricas e os experimentos
  4_resultados.tex      as medições e as tabelas
  5_conclusao.tex

pretextual/
  abreviaturas.tex    lista de siglas, usada por \listasiglas

referencias/
  referencias.bib     uma entrada por referência, com DOI ou URL verificável

tabelas/              GERADO. Cada tabela sai de scripts/tabelas.py a partir
                      do registro canônico, junto com um arquivo .nota.tex
                      com a proveniência. Não editar à mão: a edição é
                      perdida na próxima geração, e o número deixa de
                      corresponder ao resultado bruto.

figuras/              as 9 figuras que o texto referencia
  nao_usadas/         56 figuras que sobraram da qualificação (exemplos do
                      PascalVOC, em sua maioria). Ficam fora do
                      \graphicspath, então não entram na compilação. Para
                      voltar a usar alguma, mover um nível para cima.

notas/
  Reuniao.tex         anotação de reunião, não faz parte do texto
```

## Antes de subir para o Overleaf

```bash
python scripts/conferir_dissertacao.py
```

Pega o que a leitura humana cansa de não pegar: citação sem entrada no
`.bib`, `\ref` para label que não existe, sigla usada sem declarar, lista de
autores que o BibTeX recusa, arquivo `.tex` que ninguém inclui, e travessão
usado como pontuação.
