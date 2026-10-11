# 18 — IQ2 em pares preservando a ordem numérica

2026-10-10/11. Segunda revisão de 57 camadas; candidata, ainda em compilação/validação.

O primeiro candidato reduziu o target para aproximadamente67–68ms por launch e chegou a89,985/59,921 tokens/s. Entretanto, três respostas abertas mudaram em relação ao native anterior. As10 tarefas com gabarito continuaram corretas. A revisão foi preservada em `optimization-v2/fused-six-v1/`, com hashes, antes de modificar novamente as fontes.

## Mudança

`native_iq_mma.cuh::iq_dp4a_dual` ganhou parâmetros de template `PRESERVE_IQ2_ORDER` e `ZERO`, mais `element_offset`. A opção ordered decodifica IQ2_XXS/IQ2_XS/IQ2_S em pares de oito, mantendo dois updates F32 e a associação anterior `d * (weight_scale * dot8)`. IQ3/IQ4 continuam com seu dot16 existente. A opção ZERO=false carrega a soma anterior e o offset seleciona a posição correta dos pesos em slices K.

`qwen38_mk.cu::rows_native_impl` usa o helper para duas linhas IQ2 simultâneas em T>5, compartilhando input/metadados e a leitura dos fragmentos de dezesseis pesos. `ffn_gu_iq` seleciona a ordem IQ2 original para gate/up. `MK_IQ2_PAIRED=0` permite comparar com a revisão anterior no mesmo binário; default do candidato é1. T<=5 preserva o caminho anterior, incluindo MTP.

Não mudam pesos, escalas de ativação, KV, DeltaNet, layouts de buffers globais, grid cooperativo ou alocações CUDA. Preservar a ordem de operações no código não basta para prometer identidade bit a bit entre variantes compiladas: verificar os IDs, logits/estados relevantes e as gerações reais.

## Validação prevista para esta revisão

`tests/iq_mma.cu` usa o mesmo helper de produção contra pesos CPU GGML e dot double. Acrescenta variantes ordered e sliced em36 pares, com K5120 dividido em64+64+32 unidades. O teste sliced cobre stride compacto, offsets de pesos não nulos e somas carregadas entre os slices, necessários no down FFN. Entradas zero, Q16 de escala variável e Q32.

Depois: A/B paired/unpaired no modelo completo, rejeição/replay, suite20 com IDs comparados ao native anterior, memcheck do kernel wide, regressão MTP e HTTP com contexto32K alocado. Benchmark final antes/depois/llama deve usar snapshots identificados, mesmos parâmetros e clocks padrão. O draft continua em CUDA Graphs; não declarar fusão persistente monolítica.

## Primeiro resultado

`optimization-v2/paired-oracle/`: PASS em917.504 resultados,36 pares por variante, incluindo ordered e sliced. NRMSE máximo3,57842195e-7; erro absoluto máximo1,91037506e-6, iguais aos máximos anteriores dos protótipos. Arquivos registram comando, hash do executável, exit0 e duração. A compilação do modelo completo ainda está em andamento; não há medida de throughput desta revisão neste ponto.

## A/B do modelo completo

Snapshot `paired-iq2-v1/` e `paired-ab/`: três repetições por caso, no mesmo binário, alterando apenas `MK_IQ2_PAIRED`.

| Caso | Paired0 | Paired1 | Ganho |
|---|---:|---:|---:|
| Curto |89,799|90,589|0,88%|
| Longo |59,610|60,350|1,24%|
| Sampling |89,179|90,092|1,02%|

Todos os128 IDs coincidiram com a fixture anterior em todas as repetições. Ciclos/aceitação continuaram17/111,25/106,17/114. O ganho é pequeno mas consistente nesta sessão; ainda abaixo do llama.cpp. A suite20 desse snapshot está em execução separada.

## Próxima revisão: pares em todos os seis IQ

O helper ordered/sliced já foi oracularmente validado nos36 pares, incluindo IQ3/IQ4. O caminho R2 está sendo estendido para TYPE18/21/23, preservando seus dots16 e sua associação F32. A chave principal passa a `MK_IQ_PAIRED`; `MK_IQ2_PAIRED` fica como alias para reproduzir o protótipo anterior. A chave principal tem precedência quando as duas estão definidas. Não muda T<=5, buffers ou CUDA Graphs do draft.

`paired-iq2-quality/`:10/10 tarefas;17/20 IDs/textos iguais ao native anterior;16/20 textos iguais ao llama sem spec e18/20 iguais ao MK MTP anterior. As diferenças agora ocorrem em `generation-1`, `generation-5`, `generation-8`. A ordem IQ2 no código não restaurou identidade de todas as gerações entre esses binários. Reportar essa limitação; identidade de texto não é uma medida de acurácia ampla.
