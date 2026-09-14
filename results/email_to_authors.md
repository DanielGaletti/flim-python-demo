# Correspondência com os Autores

## E-mail enviado
**Assunto:** Tentativa de reprodução: bugs encontrados, resultados parcialmente divergentes e recursos ausentes

*(enviado anteriormente — ver versão anterior)*

---

## Resposta recebida

**De:** Autor (LIDS-UNICAMP)
**Assunto:** Re: Tentativa de reprodução

> Olá Daniel, bom dia, desculpe a demora pelo retorno, esses últimos dias foram bem corridos por aqui.
> Em relação aos problemas apontados, vi que você está utilizando o flim-python-demo, seria isso? se sim o problema pode ser esse, pois essa versão não está 100% atualizada. Você está tentando reproduzir os resultados do artigo: FLIM-based Salient Object Detection Networks with Adaptive Decoders? nesse caso pode dar uma olhada no repositório https://github.com/LIDS-UNICAMP/flim_ad pois contém o código da versão utilizada, e em /data/ estão os arquivos de split além dos marcadores(--seeds.txt).
> Qualquer dúvida pode me contactar novamente.

---

## Conclusão

**flim-python-demo NÃO é o repositório correto.**  
O repositório oficial do artigo é: **https://github.com/LIDS-UNICAMP/flim_ad**

Diferenças confirmadas:
- Versão da biblioteca: `libs/flim-python/` (diferente da versão pip)
- Splits corretos: 50-50 (não 70-30)
- Marker files prontos em `data/schisto/user_A/split{1,2,3}/markers/`
- Biblioteca IFT necessária para Parasites (Dynamic Trees)
- Pipeline de 5 etapas com seleção de melhor camada na validação
