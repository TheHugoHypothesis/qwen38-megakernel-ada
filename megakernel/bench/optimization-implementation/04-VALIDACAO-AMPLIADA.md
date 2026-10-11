# Validação ampliada: vinte prompts

O conjunto contém dez problemas numéricos com respostas conhecidas e dez
prompts de código/prosa em português e inglês. A referência foi gerada no
llama.cpp sem especulação; cada engine usa os mesmos IDs de entrada e
sampling greedy. Contexto 4096/Q8, até 128 tokens. É um conjunto local
pequeno, não um benchmark amplo de qualidade.

| Engine | Tarefas numéricas corretas | Gerações idênticas à referência |
|---|---:|---:|
| llama.cpp sem especulação | 10/10 | referência |
| llama.cpp DFlash2 Q4, 7 propostas | 10/10 | 17/20 |
| Megakernel IQ16, MTP 4, chunk auto | 10/10 | 16/20 |

As divergências observadas incluem redação de prosa, detalhes de formatação e
continuação truncada. DFlash2 também não produz identidade universal dos IDs
greedy com o caminho sem especulação, embora tenha concordado nos dois
prompts iniciais. O batching muda caminhos/reduções numéricos do target.
Isso é uma hipótese para as diferenças, não uma causa isolada em cada caso.

O megakernel foi validado através da C API, reutilizando a mesma engine e
resetando-a entre os vinte pedidos. As respostas conhecidas se mantiveram;
perplexidade e acurácia em conjuntos maiores continuam não medidas.

O processo DFlash2 usou 14103 MiB após carregar e aproximadamente 14395 MiB
após pedidos neste teste. GPU/clock snapshots estão em `quality-dflash2`.
Clocks observados são o boost dinâmico de fábrica; offsets permanecem zero.
Ainda não foi validado DFlash2 com contexto efetivamente preenchido de 32K.

Os 31 testes Python de backend, GGUF e build passaram. O A/B final com chunk
automático está em andamento. `serve_ada.py --check` confirmou a compatibilidade
dos tokenizers e imprimiu a configuração de execução sem carregar a GPU.

Evidência: `quality-nospec/results.json`, `quality-dflash2/results.json`,
`quality-megakernel/results.json`, `quality-suite.json` e `host-tests.log`.
