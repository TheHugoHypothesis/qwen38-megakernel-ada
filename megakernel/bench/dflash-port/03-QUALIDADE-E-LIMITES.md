# 03 — Qualidade e limites do primeiro porte

Suíte existente de 20 prompts, greedy, até 128 tokens por resposta, um
target/draft reutilizado entre prompts com reset de ambos:

- 10/10 problemas numéricos objetivos corretos.
- 19/20 textos iguais ao MK MTP anterior.
- 15/20 textos iguais ao llama.cpp sem especulação.

A única diferença de texto frente ao MK anterior foi `generation-4`:
o docstring escreveu “preserving the order of first occurrences” em vez
de “preserving the first occurrence order”. O restante do código coincide.
Isso não foi atribuído a uma causa numérica específica sem um probe.
A diferença de profundidade/batch entre os dois caminhos também impede
tratar identidade greedy como consequência automática da especulação.

O verificador externo passou o oracle MK escalar em 64 tokens com rejeição
em cada posição; isso é distinto da comparação de qualidade com llama.cpp.
A divergência numérica longa já conhecida no target MK continua pendente.
Features das camadas profundas foram verificadas quanto a forma, posições
e finitude, ainda sem um oracle independente de todas as cinco camadas.
Não há resultado amplo de perplexidade ou equivalência estatística de
sampling nesta etapa.

Resultados completos: `quality/results.json`, `.out`, `.log`. Os runners
preservam prompts/IDs, respostas, comandos, contadores e tempos.

## Baseline novo

llama-server DFlash2 com quatro propostas, três repetições, mesmos
modelos/IDs/contexto Q8/128 tokens: 74,13 tok/s curto e 59,52 longo.
Híbrido: 67,05 / 55,65, aproximadamente −9,6% / −6,5%. As continuações
longas dos dois targets diferem; comparar tokens/s não isola o custo em
uma sequência idêntica. O baseline com sete propostas ainda é mais forte
e não é diretamente equivalente a este protótipo de quatro propostas.

## Trabalho ainda necessário

1. Comparar logits e as cinco features em teacher forcing, com as mesmas
   sequências, antes de declarar qualidade numericamente equivalente.
2. Medir o verificador T=8 e revisar reduções/atenção/shared. Sete drafts
   não são habilitados pelo protótipo atual.
3. Reduzir custo do target e medir tamanho de grupo, spills e tráfego IQ.
   Aproximadamente 88% do decode atual está no target/injeção; a injeção
   sozinha representa ~1,4%. Fusão integral do draft não é o primeiro
   experimento indicado pelos tempos atuais.
4. Avaliar memória sem pesos/buffers MTP no caminho externo e contextos
   preenchidos maiores, antes da integração HTTP/multi-turn.

Não foi alterado o launcher de produção nem os clocks. O porte inicial é
executável e validado nos limites acima; ainda não atingiu o objetivo de
superar llama.cpp/DFlash2.
