# Análise de desempenho e próximas otimizações — RTX 4060 Ti 16 GB

**Registro da investigação anterior à implementação.** As mudanças autorizadas
posteriormente foram implementadas e medidas: IQ16/prefill/chunk auto deram
aproximadamente +25% de decode MK; DFlash2 Q4 funciona via llama.cpp e atingiu
91,35 tok/s no prompt curto. O protótipo GEMM tiled foi mais lento e ficou
opt-in. Consulte [IMPLEMENTATION_RESULTS.md](IMPLEMENTATION_RESULTS.md) e
[registros intermediários](bench/optimization-implementation/README.md).
Afirmações de trabalho ainda não realizado neste documento descrevem o momento
desta investigação, não o estado atual do código.

Investigação e benchmarks em 2026-10-10, clocks padrão, target GSQ-RCO IQ3_S-mtp
existente, CUDA 13.4, sm_89. Nesta etapa foram produzidos perfis, benchmarks e
relatórios; os kernels e binários de inferência não foram modificados.

## Resultado principal e correção do baseline

O megakernel funciona no orçamento de memória, mas **ainda perde para o
llama.cpp com MTP ajustado**. A comparação anterior de +4,6% usava duas
propostas no llama.cpp e quatro no megakernel. Ela descrevia aquelas
configurações, não uma vantagem sobre o melhor baseline disponível.

| Geração de 128 tokens, contexto 4096/KV Q8 | llama.cpp MTP 4 | Megakernel MTP 4 | Diferença do megakernel |
|---|---:|---:|---:|
| Greedy, prompt de 24 tokens | 62,956 tok/s | 48,170 tok/s | −23,49% |
| Sampling, temp 1/top_k 20/top_p 0,95/seed 123 | 61,931 tok/s | 47,235 tok/s | −23,73% |
| Greedy, prompt de 1365 tokens¹ | 45,509 tok/s | 39,925 tok/s | −12,27% |

Médias de três repetições do novo baseline greedy curto e duas nos demais
casos. Megakernel: registros anteriores com o mesmo binário, configuração e
clocks; não são execuções intercaladas de um ensaio controlado. Throughput de
decode exclui carregamento/prefill. Não há intervalo de confiança. Os prompts
foram conferidos pelos IDs, evitando diferenças de template/tokenização.

¹ As continuações do caso longo diferem: esta é uma comparação das gerações
observadas, não de uma continuação fixa. O texto muda a aceitação especulativa.

A varredura curta do llama.cpp produziu: MTP 3 = 54,10; MTP 4 = 62,96;
MTP 5 = 63,02; MTP 6 = 62,90; MTP 7 = 63,21 tok/s. A diferença abaixo de
0,5% entre 4–7 propostas não permite concluir que 7 seja melhor. MTP 4 é uma
referência comparável em profundidade. Sem MTP, a medição anterior era
20,92 tok/s no llama.cpp contra 16,42 no megakernel.

Registros: [novos benchmarks](bench/optimization-analysis/),
[dados do gráfico](bench/ada-report/chart_data.json),
[gráfico atualizado](bench/ada-report/ganhos_e_correcao.png).

## Limitação de corretude encontrada

O llama.cpp sem especulação e com MTP 4 gerou os mesmos 128 IDs no prompt
longo. Ambos diferem da referência anterior com MTP 2 e do megakernel a partir
do índice 41, base zero. Em teacher forcing da nova continuação, o megakernel
concordou em **127/128 argmax**, contra 128/128 no prompt curto. O comando
`--score` usa `MKEngine::score` com o caminho de prefill FP16 por padrão;
essa porcentagem não conta tokens de duas gerações livres alinhadas.
A diferença
inicial altera comentários/docstrings do código gerado; isso não mede acurácia
em tarefas, mas invalida uma afirmação geral de identidade greedy.

A causa ainda não foi isolada. São candidatos as diferenças de precisão das
ativações, ordem de redução, prefill e seleção de kernels por tamanho do lote.
Não há evidência suficiente para chamar a divergência de um bug específico
de MTP. O resultado de 100% do relatório inicial era limitado àquelas duas
referências, incluindo a execução MTP 2. O oráculo dos pesos dequantizados
continua válido, mas não demonstra equivalência de toda a inferência.

Evidência: `bench/optimization-analysis/llama-mtp4-long/results.json` e
`llama-mtp0-long/results.json`, com IDs e saída do teacher forcing. Não foram
medidas perplexidade, acurácia em tarefas ou equivalência estatística de
sampling. Essas avaliações continuam necessárias.

## Onde o tempo é gasto

O perfil interno instrumentado separa as fases do decode; inclui o custo de
sincronização associado às fases, sem isolar a latência de cada barreira.
Usar as médias de ciclos reais para diagnóstico, não a seção impressa
`min-composed`, que compõe mínimos de fases e é uma estimativa.

| Fases, prompt curto/MTP 4 | Tempo médio/ciclo | Fração aproximada |
|---|---:|---:|
| FFN gate/up + down | 51,775 ms | 59,24% |
| Projeções DeltaNet + saída residual | 18,487 ms | 21,15% |
| Projeções attention + saída residual | 5,440 ms | 6,22% |
| Núcleo recorrente DeltaNet | 1,927 ms | 2,20% |
| LM head target | 2,650 ms | 3,03% |
| Todas as fases do MTP | 6,814 ms | 7,80% |
| Atenção propriamente dita + combinação | 0,296 ms | 0,34% |

Ciclo médio: 87,400 ms, 30 ciclos, 4,23 tokens emitidos/ciclo. No prompt
longo: 85,310 ms, 37 ciclos, 3,43 tokens/ciclo; atenção fica em aproximadamente
0,71%. A aceitação cai de 88,6% para 68,7%. Portanto a queda de throughput
nesses exemplos vem sobretudo de menos tokens aceitos por ciclo, não de um
crescimento dominante do custo de atenção. Não extrapolar isso para entradas
reais de 32K/64K tokens, que não foram perfiladas nesta etapa.

O novo llama.cpp MTP 4 curto registrou 92,6% de aceitação e comprimento médio
aceito de aproximadamente 4,70. A métrica do log e tokens emitidos/ciclo não
têm exatamente o mesmo tratamento de aquecimento/término. O vocabulário draft
restrito e diferenças numéricas são hipóteses para a diferença, não causas
isoladas experimentalmente.

O Nsight Systems registrou prefill do prompt de 1365 tokens, chunk 512:

| Trabalho na GPU | Tempo | Fração |
|---|---:|---:|
| Dequantização dos pesos nativos para matriz FP16 | 628,169 ms | 25,60% |
| GEMMs cuBLAS | 1411,757 ms | 57,53% |
| Conversões de entrada/saída e GLU | 242,833 ms | 9,90% |
| Demais kernels | aproximadamente 171 ms | 6,97% |

Foram 1086 lançamentos de dequantização e 1086 GEMMs: 362 matrizes nativas
processadas em três chunks. A escrita global da matriz FP16 seguida de sua
leitura pelo GEMM é um custo concreto do fallback atual.

No trecho de decode do trace, dez `mk_cycle` somaram 848,500 ms em uma janela
de 849,067 ms. Os intervalos entre kernels somaram **0,567 ms = 0,067%**.
Reduzir apenas overhead de lançamento tem pouco espaço nesse caminho. O
tempo de uma API síncrona de cópia também inclui espera pela GPU: a tabela
de APIs CUDA inteira não é evidência de gargalo PCIe no decode.

Evidência: [perfis e comandos](bench/optimization-analysis/profiles.json),
[resumo do Nsight](bench/optimization-analysis/nsys-summary.json),
`long-nsys.nsys-rep` e `long-nsys.sqlite` no mesmo diretório.

## Limites da Ada e do kernel atual

`mk_cycle` usa 512 threads, 128 registradores/thread e 92544 bytes de shared
memory dinâmica, mais 16 estáticos. São 34 CTAs cooperativas, uma por SM,
16 warps residentes para um máximo arquitetural de 48: occupancy teórica
de 33,3%. Os 65536 registradores alocados por CTA e a shared memory impedem
colocar uma segunda CTA. Occupancy não é porcentagem de utilização da GPU
nem um objetivo isolado. Os limites de 64K registradores e 100 KiB de shared
por SM constam no [guia Ada da NVIDIA](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

`ptxas.log` informa, para a entrada `mk_cycle`, stack de 2560 bytes, 1288 bytes
de spill stores e 1632 de spill loads. Isso indica spills na compilação, não
quantifica tráfego dinâmico. O campo de local memory zero do trace não elimina
essa evidência. Ainda faltam contadores de DRAM/L1, instruções e stalls; `ncu`
não estava no PATH. Coletar esses contadores com [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
antes de atribuir percentuais de perda a bandwidth ou spills.

Um teste cego de `-maxrregcount` pode aumentar os spills. O pragma CUDA 13
`enable_smem_spilling` também **não é aplicável diretamente**: a documentação
proíbe seu uso com shared memory dinâmica. O kernel atual depende dela e já
consome quase o limite por CTA. Ver [PTX ISA, enable_smem_spilling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#pragma-strings-enable-smem-spilling).

A RTX 4060 Ti 16 GB tem bandwidth nominal de 288 GB/s, segundo a
[ficha técnica do fabricante ELSA](https://www.elsa-jp.co.jp/wp-content/uploads/2023/09/Datasheet_A4_4060_Ti_SAC_16GB.pdf).
O payload das matrizes trunk + head é 11353307136 bytes. Um modelo simples de
uma leitura desses pesos por token dá 288e9/11353307136 = **25,37 tok/s**,
antes de dequantização, ativações, estados, KV e outros tráfegos. Não é um
limite do decode especulativo: vários tokens compartilham essa leitura. Também
não é uma medida de bandwidth efetiva. A capacidade de 16 GB permite alojar
o modelo; não aumenta a velocidade com que seus pesos são lidos.

GSQ-RCO escolhe quantizações por tensor no GGUF. O arquivo mistura IQ3_S,
IQ3_XXS, IQ4_XS e outros formatos; o nome IQ3_S não descreve todos os pesos.
Não foi encontrado um custo adicional de rotação RCO no runtime. A descrição
do processo está no [model card do target](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/raw/main/README.md).

## Otimização de configuração já medida

Mesmo prompt de 1365 tokens, MTP 4, contexto 4096/Q8, duas repetições por chunk:

| Chunk | Prefill tok/s | Latência de prefill | Decode tok/s | VRAM usada pelo dispositivo |
|---:|---:|---:|---:|---:|
| 128 | 272,35 | 5,012 s | 40,20 | 12,844 GiB |
| 256 | 412,00 | 3,313 s | 40,08 | 12,895 GiB |
| 512 | 530,50 | 2,5725 s | 40,04 | 13,004 GiB |
| 1024 | 575,40 | 2,3725 s | 40,01 | 13,217 GiB |
| 2048 | 630,20 | 2,166 s | 40,00 | 13,633 GiB |

Chunk 2048: **+18,79% throughput de prefill**, **−15,80% latência**, com
aproximadamente +0,629 GiB, sem ganho no decode. Todos os dez conjuntos de
128 IDs concordaram com a referência inicial MTP 2; isso não resolve a
divergência frente à nova referência. O tempo CLI inclui produção do primeiro
token e preparação do checkpoint, de forma consistente entre os chunks.

É possível experimentar `--prefill-chunk 2048` no CLI existente, sem alterar
kernels. Recomendo seleção automática por comprimento e memória livre, não
trocar o default universalmente. Em 32K/Q8, a estimativa é aproximadamente
14,64 GiB de uso total; não foi executada essa combinação. Em 64K/Q8, o teste
anterior com chunk 512 já deixava apenas 0,516 GiB livre, menos que o aumento
observado; chunk 2048 pode não caber. Conservar o preflight de memória.

Registros completos: [chunk-sweep.json](bench/optimization-analysis/chunk-sweep.json).

## Mudanças priorizadas

| Prioridade | Mudança concreta | Alvo e impacto | Complexidade / risco |
|---|---|---|---|
| 0 | Corrigir referências e isolar a divergência numérica | Permite avaliar cada ganho sem aceitar regressão de qualidade | média / baixo para diagnóstico |
| 1 | Escolher chunk 1024/2048 quando houver memória e prompt suficiente | Prefill: ganho de até 18,8% já observado neste prompt | baixa / baixo, com preflight |
| 2 | Especializar produtos IQ×Q8 das matrizes dominantes | Decode: FFN e projeções, maior alvo medido | média–alta / médio |
| 3 | Reduzir liveness/spills e especializar entrada por T | Decode: menos tráfego local e melhor uso dos recursos | média–alta / médio |
| 4 | GEMM de prefill com dequantização por tile, fusion limitada de GLU | Evita materialização global FP16; ataca aproximadamente 35,5% do prefill mais possíveis melhorias do GEMM | alta / médio–alto |
| 5 | Adaptar profundidade MTP 1–4 e avaliar vocabulário draft | Mais tokens úteis por tempo de ciclo, sobretudo em texto menos previsível | média / médio |
| 6 | Testar DFlash2 no llama.cpp antes de portar | Explora propostas em paralelo com implementação já existente | média para baseline, alta para porte / alto |
| 7 | Arquitetura híbrida: GEMMs separados + CUDA Graphs, pequenos ops fundidos | Libera matrizes grandes dos recursos comuns do megakernel | alta / alto |

### Produtos quantizados: reutilização prioritária

Arquivos/funções: `qwen38_mk.cu::rows_native` (linha 597), `ffn_gu` (848),
helpers de down/projeções; `native_quant.cuh::quant8` e loaders.
Comparar com `llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh`, `mmvq.cu` e `mmq.cuh`.

O caminho atual já usa DP4A, decodifica fragmentos de oito pesos, aplica
escalas/acumulação float por fragmento e repete parte do trabalho de
índices/metadados. O GGML tem produtos IQ3_S/IQ3_XXS com
loads empacotados, acumulação inteira por grupos maiores e aplicação de
escalas após a redução. Portar essas operações para o layout QBuf do
megakernel; escalas e permutações devem ser conferidas explicitamente.
Começar pelos tipos 18/21/23 já especializados e pelos shapes FFN
5120×17408, medindo T=1…5. Gate/up podem ter tipos diferentes: manter o
dispatch separado quando necessário.

Testar loads vetoriais dos headers, reutilização entre tokens, e repack
**sem perda** de layout, sem converter GSQ-RCO para outro esquema. Não ampliar
todos os pesos para FP16. O [Marlin](https://github.com/IST-DASLab/marlin)
serve de referência para pipelining, loads assíncronos, double buffering e
layout offline em Ampere/Ada. Seu formato FP16×INT4 não aceita diretamente
os IQ mistos deste GGUF; seus ganhos publicados não são estimativas para nós.

Se o tempo do FFN cair 20%, mantendo todo o resto e a aceitação constantes,
a lei de Amdahl dá aproximadamente +13,4% no throughput total, de 48 para
55 tok/s. É uma simulação, não previsão. Para chegar aos aproximadamente
63 do baseline só pelo FFN seria necessária redução de aproximadamente 40%
no tempo dessa fase. Otimizar somente a atenção atual dificilmente daria 1%.

### Recursos e prefill

`qwen38_mk.cu::mk_cycle` (1974) agrega variantes T=1…5 e o draft no mesmo
kernel. Avaliar entradas templadas por T, reduzir arrays temporários vivos,
unrolling e número de linhas acumuladas R conforme dtype. Reduzir
registradores sem mexer na shared não permite automaticamente duas CTAs.
Validar SASS e tráfego local, não apenas o número de registradores.
O `Makefile` usa `-Xptxas -dlcm=cg`: comparar políticas de cache por medição,
depois de identificar quais loads são afetados.

`pf_init` (2613), `gemm` (2650), `gemm_glu` (2676) e `prefill.cuh` são os
alvos do prefill. Reutilizar loaders/tile MMA do GGML MMQ para os IQ suportados;
há um tensor IQ1_M que precisa manter fallback. Um caminho com ativações
Q8_1 mudaria a precisão atual FP16 e exige avaliação própria. Uma alternativa
com dequantização FP16 por tile preserva essa precisão, mas também precisa de
validação de reduções. Fusões de conversão/GLU devem respeitar os formatos
diferentes de gate/up. Double buffering da matriz temporária custa mais
aproximadamente 170 MiB e só deve ser avaliado quando houver memória/overlap.

### MTP e DFlash2

`MKEngine::step` (2962) reduz o draft apenas de 4 para 3 depois de rejeição.
Avaliar política 1–4 com médias de aceitação por posição e custo medido;
maximizar tokens verificados por segundo, não a taxa de aceitação isolada.
`dv_add` (2103) e o subset inicial de 32K são alvos de uma varredura de
8K/16K/32K/64K/vocabulário completo **somente do draft**. O head target deve
continuar verificando o vocabulário inteiro. Medir o benefício contra custo
de head, memória e mudança das propostas.

Foi encontrado o [draft oficial DFlash2 para Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF).
Há GGUF Q4_K_M de aproximadamente 1,14 GB e Q8_0 de 2,06 GB na
[árvore publicada](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main).
O draft não está no workspace e não foi executado. Seu funcionamento e
aceitação com o target GSQ-RCO permanecem hipóteses.

O [config publicado](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json)
indica cinco camadas, hidden 5120, blocos de oito, convolução de dois taps,
selector e extração de features das camadas target [5,19,33,47,61]. O
conversor local soma um ao índice para armazenar inputs de camadas GGUF:
[6,20,34,48,62]. Conferir os metadados do draft real antes de qualquer porte.

Reutilizar primeiro `common/speculative.cpp` (1018/1296),
`src/models/dflash.cpp` (139) e `common/arg.cpp` (4276) do llama.cpp: já há
`draft-dflash` com detecção de DFlash2 pelo selector. Comparar ao MTP 4/5
no mesmo target e GPU, começando em contexto 4K/8K. Não combinar ganho
publicado contra autoregressivo com a nossa referência já acelerada por MTP.

Q4 ocupa aproximadamente 1,06 GiB só no arquivo; estimativa adicional de KV
do draft: aproximadamente 40 MiB FP16 ou 21 MiB Q8 para cinco camadas,
8 KV heads, dimensão 128 e janela 2048, além de features e workspace.
O target a 32K já consome cerca de 14 GiB no megakernel; Q4 deixaria pouco
espaço e Q8 não caberia no orçamento CUDA observado de 15,65 GiB com esses
valores, antes dos extras. Isso é estimativa, não medição de runtime DFlash2.

Portar DFlash2 para o megakernel exigiria taps de features, execução do draft,
selector, verificação e rollback DeltaNet. `MAXT=5` e o padding atual de
atenção também precisariam mudar para blocos maiores. Aumentar o limite de
draft no CLI não implementa esse caminho e pode exceder shared memory.

### Arquitetura híbrida e prioridades menores

Se os microkernels dentro do megakernel não vencerem o baseline, separar os
GEMMs grandes permite tile/grid/registradores próprios. Manter norm, residual
e DeltaNet fundidos onde for útil; usar graphs para a sequência de lançamentos.
O llama.cpp já usa CUDA Graphs. O [artigo da NVIDIA](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
explica sua finalidade; adicionar graphs ao caminho atual de um kernel por
ciclo não ataca o gargalo encontrado.

DeltaNet tem 48 heads distribuídos por 34 SMs: 14 SMs fazem dois heads e os
demais um. Há desequilíbrio, mas o core é apenas cerca de 2,2% neste perfil.
KV Q4/compressão de estado devem ficar para pressão de memória em contexto
realmente longo, pois alteram precisão sem atacar o principal custo atual.
FP16 de todos os pesos não cabe. TMA/WGMMA e NVFP4 próprios de gerações
posteriores não são soluções diretas para sm_89. Clocks continuam padrão.

## Sequência de validação e critérios de sucesso

1. Fixar baseline sem especulação e MTP 4, mesmos IDs de entrada, contexto,
   KV, sampling e hashes. Isolar a divergência no índice 41: comparar logits,
   margem top-1/top-2 e estados intermediários por camada, variando o chunk e
   o tamanho do lote de verificação. Conferir replay/checkpoint DeltaNet.
2. Criar microbenchmarks com pesos reais para FFN e projeções, T=1…5.
   Para cada candidato, validar oráculo de dequantização e saídas versus o
   helper atual e GGML; depois validar inferência completa. Testar fronteiras
   de blocos, linhas, chunks, rejeição total/parcial e contexto cheio.
3. Medir candidatos sem instrumentação, com aquecimento, pelo menos cinco
   repetições intercaladas e um conjunto de 20–50 prompts representativos:
   código, português, prosa, raciocínio e tokens fora do subset draft.
   Gerar 512–1024 tokens quando couber; separar média/mediana/p95, prefill,
   TTFT e latência da requisição inteira. Registrar clocks, temperatura,
   potência, VRAM, tokens/ciclo e aceitação por posição.
4. Avaliar contexto efetivamente preenchido de 4K/8K/32K. Uma alocação de
   cache 64K com prompt curto não substitui esse teste. Executar memcheck
   nos caminhos alterados e testar repetição/reuso do servidor.
5. Medir perplexidade/NLL no mesmo corpus e testes de tarefas com respostas
   verificáveis. Não tratar argmax de dois prompts ou seeds repetíveis como
   comprovação de qualidade. Definir tolerâncias numéricas a partir dos
   caminhos de referência e pares de kernels; investigar regressões antes
   de aceitar novos ganhos.

Critério de desempenho: superar o llama.cpp MTP ajustado no conjunto de uso
real, com diferença maior que a variabilidade medida e dentro do orçamento
de memória, mantendo os critérios de qualidade aprovados. Para uma mudança
local, exigir melhoria consistente da fase alvo e ausência de regressão
material na latência completa. Os aproximadamente 63 tok/s do prompt curto
são uma referência específica, não uma promessa de throughput em todo texto.

Como limites iniciais de qualidade para avaliar novos caminhos aproximados,
proponho aumento relativo de perplexidade inferior a 0,5% e perda de acurácia
inferior a um ponto percentual, avaliando intervalos de confiança e tamanho
do conjunto. São critérios propostos, não resultados obtidos. Mudanças de
layout de pesos devem manter a dequantização exata do formato original.

## Evidência e reprodutibilidade

`bench/optimization-analysis/` contém comandos, logs, IDs, JSONs e trace do
Nsight. `bench/ada_summary.json` mantém resultados/hashes originais e registra
esta revisão em `optimization_followup`. `bench/ada-report/README.md` e o
gráfico foram corrigidos para explicitar o novo baseline e a divergência.
As fontes externas são primárias e foram consultadas em 2026-10-10.
