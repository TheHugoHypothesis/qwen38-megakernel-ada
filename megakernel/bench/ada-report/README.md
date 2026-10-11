# Gráfico de desempenho e correção

**Gráfico histórico, anterior às otimizações IQ16/prefill e ao teste DFlash2.**
DFlash2 foi posteriormente baixado e validado neste workspace via llama.cpp.
O gráfico e relatório atuais estão em
[optimization-implementation/report](../optimization-implementation/report/ganhos_implementados.png)
e [IMPLEMENTATION_RESULTS.md](../../IMPLEMENTATION_RESULTS.md).
As observações abaixo sobre ausência de draft/testes descrevem a investigação
anterior e foram mantidas como registro.

Atualizado em 2026-10-10 com novos benchmarks de MTP, perfis e varredura de
chunks. O script do gráfico apenas lê os registros; os clocks permaneceram
padrão durante toda a investigação.

- `ganhos_e_correcao.png`: visualização raster.
- `ganhos_e_correcao.svg` e `.pdf`: versões para exportação.
- `chart_data.json`: valores derivados dos registros brutos.
- `../plot_ada_results.py`: script para regeneração (Python, matplotlib, numpy).

O gráfico atualizado compara quatro propostas em ambos os motores:
megakernel versus llama.cpp = −23,49% greedy curto (48,17 versus 62,96 tok/s),
−23,73% sampling (47,24 versus 61,93) e −12,27% greedy longo (39,93 versus
45,51). No caso longo as continuações diferem, afetando a aceitação.
Os ganhos anteriores de +4,63% e +2,67% usavam MTP 2 no llama.cpp e não
representavam vantagem sobre o baseline ajustado. O decode sem MTP também
continua mais lento no megakernel.
São duas repetições por configuração, exceto o baseline greedy curto com MTP,
que possui três. Não há intervalo de confiança nem avaliação de estabilidade
prolongada. A geração exclui carregamento e prefill.

Concordância de tokens não é acurácia em tarefas. Na referência inicial,
foram 1.280/1.280 tokens
iguais em dez gerações do prompt curto e 512/512 em quatro gerações do prompt
mais longo. As repetições não representam 1.792 exemplos independentes. São
dois prompts, com continuações de 128 tokens. Teacher forcing inicial concordou
em 128/128 nos dois prompts; o prompt longo foi conferido com chunks 512 e 128.

A nova referência MTP 4 concordou com o llama.cpp sem especulação no prompt
longo e divergiu da referência MTP 2 anterior e do megakernel a partir do
índice 41 (base zero). O gráfico agora mostra teacher forcing da nova
referência: 128/128 no curto e **127/128 = 99,22% no longo**. A causa ainda
não foi isolada; não há comprovação de equivalência numérica geral.
Teacher forcing usa `--score`, cujo caminho padrão é o prefill FP16;
essa porcentagem não representa concordância entre duas gerações livres.

Perplexidade, benchmarks de tarefas e equivalência estatística da distribuição
de amostragem não foram avaliados. O teste de alocação de contexto 64K utilizou
um prompt curto; não mede a qualidade com entrada de 64K tokens.

## DFlash2

Nenhum caminho DFlash/DFlash2 foi encontrado no código do `open-jet` durante
esta inspeção. A implementação entregue usa o bloco MTP do próprio GGUF.
`--draft 2` significa duas propostas MTP; não significa DFlash2.

O checkout local do llama.cpp contém suporte de código para DFlash2:

- `llama.cpp/common/speculative.cpp:1018`: exige contextos target e draft;
  lê camadas de extração do draft e detecta DFlash2 por `selector_top_k > 0`.
- `llama.cpp/common/speculative.cpp:1296`: lê o lattice do selector DFlash2.
- `llama.cpp/src/models/dflash.cpp:139`: carrega tensores selector e valida
  metadados de convoluções/selector.
- `llama.cpp/common/arg.cpp:4276`: recebe um modelo draft separado; a estratégia
  correspondente é `draft-dflash`, com detecção de DFlash2 pelos metadados.

No cache do modelo neste workspace foram encontrados apenas o GGUF target
IQ3_S-mtp e o mmproj BF16, sem GGUF draft DFlash2. Portanto não houve teste de
execução DFlash2 e não é possível afirmar que funciona com este target ou
quantificar sua VRAM e desempenho. Para testar, é necessário obter um draft
treinado/compatível com o target real e verificar tokenizer, dimensões,
camadas de extração, tensores e orçamento de memória. Um draft para outro
modelo não deve ser considerado compatível apenas pela semelhança do nome.

A busca desta investigação encontrou o [draft oficial Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF),
incluindo Q4_K_M de aproximadamente 1,14 GB. Ainda não foi baixado nem testado
com este target GSQ-RCO. Isso fornece um candidato concreto, sem garantir
aceitação, qualidade ou ganho na RTX 4060 Ti.

Integrar esse caminho no megakernel exigiria extração de estados das camadas
do target, execução do draft com suas convoluções e selector, e verificação
dos blocos propostos. O suporte existente no llama.cpp é a primeira referência
para testar essa hipótese antes de portar novos kernels.

Diagnóstico completo, fontes e plano priorizado:
[OPTIMIZATION_ANALYSIS.md](../../OPTIMIZATION_ANALYSIS.md).
