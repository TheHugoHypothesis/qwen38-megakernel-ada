# 02 — Draft real e orçamento de memória

O executável `qwen38_dflash` usa MK para o target e o graph DFlash2
existente do llama.cpp para o draft. Reutiliza quantizações, convoluções,
Flash Attention, selector e CUDA Graphs do draft. O bloco anchor + mask
é removido do KV draft antes de injetar as features aceitas pelo target.
Não há segundo target, requantização nem alteração de clocks.

O bridge valida arquitetura, dimensões, número de blocos, conv/selector,
taps e tabelas de tokens/atributos. O runner de benchmark também compara
os metadados completos de tokens, merges, BOS/EOS via GGUFReader.
O bridge depende de `llama-model.h` e `llama-ext.h` internos do checkout
local; uma atualização do llama.cpp exige recompilar e validar esse ABI.

Três repetições após aquecimento, 128 tokens, contexto 4096 Q8, chunk 512,
profundidade 4; load/prefill/primeiro token excluídos do throughput:

| Caso | tok/s | Propostas aceitas | Pico NVML amostrado |
|---|---:|---:|---:|
| Curto greedy | 67,05 | 103/104 por repetição | 14931 MiB |
| Longo greedy | 55,65 | 97/124 por repetição | 14945 MiB |
| Curto sampling 0,7 | 64,41 | 102/108 por repetição | 14931 MiB |

IDs repetíveis nas três repetições de cada caso. O sampling usa a
verificação existente para drafts determinísticos (distribuição delta),
incluindo o sorteio residual na rejeição. Repetibilidade não prova por si
só equivalência estatística.

Tempos médios decode curto: draft 0,202 s; verificação incluindo injeção
1,692 s; injeção 0,027 s, já contida na verificação. Longo: 0,249 / 2,033 /
0,031 s. A maior parte do custo está no target, e eliminar só a cópia de
features não resolve a diferença frente ao llama.cpp/DFlash2.

Alocações: target com chunk 512 ~12,214 GiB; draft compute ~549 MiB,
pesos draft ~1,064 GiB, KV draft e demais buffers completam o orçamento.
Feature workspace: 512 × 5 × 5120 × 4 = 50 MiB, reutilizado por chunk;
não cresce com o contexto. Device CUDA antes de ativar esse workspace:
14,569 GiB usados e 1,087 GiB livres. NVML e cudaMemGetInfo possuem
denominadores/reservas distintos; os picos da tabela incluem todo o device.
Contextos maiores ainda não foram validados para o híbrido.

Resultados e comandos: `hybrid/results.json`; logs por caso no mesmo
diretório. Testes host: 34 passaram, em `host-tests.out`. Comparação A/B
do MK normal e suíte ampliada de saída serão registradas na etapa seguinte.
