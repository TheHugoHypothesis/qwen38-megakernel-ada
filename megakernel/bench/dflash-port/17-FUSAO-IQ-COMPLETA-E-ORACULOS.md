# 17 — Fusão DP4A dos seis formatos IQ e resultados dos pipelines

2026-10-10. Clocks padrão; não foram alterados limites de potência, LACT ou driver.

## Candidato em validação

`native_quant.cuh::native_iq16<TYPE>` fornece um fragmento de dezesseis pesos para IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S e IQ4_XS. Os três primeiros concatenam dois fragmentos IQ8 existentes; todos têm offset zero e escala uniforme dentro de cada fragmento de dezesseis. Não há conversão dos pesos GGUF para outra quantização.

`native_iq_mma.cuh::iq_dp4a_dual<T,U,GT,UT>` e `qwen38_mk.cu::ffn_gu_iq`, `ffn_gu_iq_up`, `ffn_gu` passam a selecionar os 36 pares desses formatos. No GGUF real, a cobertura gate/up aumenta de 41 para 57 das 64 camadas. As sete restantes continuam usando o caminho anterior. A seleção é para batches T>5; o MTP de até quatro propostas preserva o caminho existente.

Gate e up passam a compartilhar as leituras das ativações e deixam de materializar acumuladores intermediários na shared memory. O epílogo SwiGLU e a quantização das ativações continuam os existentes. Para IQ2, agrupar dois dots de oito em um dot de dezesseis pode alterar o arredondamento da acumulação F32; identidade integral de saída ainda precisa dos testes de integração e qualidade.

`optimization-v2/fused-six-oracle/status.json` registra comandos, SHA256 e exits. PASS: 696.320 resultados dos produtos, incluindo 36 pares DP4A, referências CPU GGML dequantizadas e dot double, entradas zero/Q16/Q32. NRMSE máximo 3,57842195e-7; erro absoluto máximo 1,91037506e-6. PASS: 8.945.664 valores de pesos coincidiram com a dequantização CPU. Esses resultados não estabelecem qualidade ampla do modelo.

## Pipelines IMMA: medidos e rejeitados como default

Os protótipos permanecem em `native_iq_pipeline.cuh`, no oráculo e no microbenchmark. Não são incluídos pelo target de produção. Nenhum deles reserva cópias persistentes dos pesos em VRAM.

- `warp-pipeline-micro/`: cada warp produz e consome seu tile; IMMA K32 apenas quando pesos e ativação têm escala uniforme por 32 valores. Para IQ3_S/IQ3_S, K32 levou aproximadamente 0,598 ms, contra 0,311 ms do DP4A. IQ4_XS/IQ4_XS: 0,649 contra 0,361 ms.
- `weight-padding-micro/`: padding para stride de 17 linhas nos pesos compartilhados. IQ3_S/IQ3_S K32 caiu para aproximadamente 0,544 ms, mas permaneceu mais lento que DP4A. A shared memory do protótipo passou a 89.920 bytes, exigindo mover a região temporária SM_MISC antes de uma eventual integração.
- `unrolled-pipeline-micro/`: desenrolamento dos loops regrediu; IQ3_S/IQ3_S K32 aproximadamente 0,673 ms contra 0,312 ms; IQ4_XS/IQ4_XS aproximadamente 0,723 contra 0,362 ms. Os 613.376 resultados do oráculo dessa revisão passaram.

Tempos acima são de microbenchmarks dot-only de matrizes FFN reais, não de um ciclo completo nem de geração. Há IMMA real no SASS; presença de Tensor Cores não prova ganho. Contadores NCU continuam indisponíveis por `ERR_NVGPUCTRPERM`; não atribuir os tempos a um gargalo único sem medição.

## Baseline atualizado

`optimization-v2/llama-fresh-a/results.json`: três repetições por caso, llama.cpp intacto, mesmo target/draft, contexto 4K, KV Q8, Flash Attention e sete propostas. Médias: 92,161 tokens/s curto; 62,743 longo. A continuação curta coincide com o MK anterior. A longa diverge a partir do índice 41, limitando a comparação de throughput entre engines. A divergência já existia antes da fusão IQ desta etapa.

O candidato de 57 camadas ainda precisa de benchmark completo, rejeição/replay, suite de qualidade e memcheck. Os resultados de 86,321/57,500 tokens/s em [16](16-IQ2-ESPECIALIZADO-E-PIPELINE.md) são da revisão anterior, com cobertura de 41 camadas.

## Resultados da primeira revisão de 57 camadas

Snapshot `optimization-v2/fused-six-v1/`; três repetições em `fused-six-ab/`. Com inline ligado, 89,985/59,921/89,571 tokens/s em curto/longo/sampling. Todos os IDs e ciclos das três fixtures coincidiram com os anteriores. A variante inline desligado mudou a continuação longa e executou 24 ciclos: seus 62,036 tokens/s não representam uma aceleração no mesmo trabalho.

`validation-status.json` e `external-verify.out`: PASS nas 15 posições de rejeição para depths1/4/7, continuação contra scalar, bytes compartilhados, propostas inválidas, 243 callbacks e prefill longo em chunks128.

`optimization-v2/fused-six-quality/` desta revisão: 10/10 tarefas corretas; IDs/texto iguais ao native anterior em17/20 gerações; texto igual ao baseline llama sem spec em16/20, contra17/20 do native anterior. As três diferenças são respostas abertas: `generation-1` a partir do token88, `generation-8` no token127 e `generation-9` no token32. Não há evidência aqui de queda de acurácia das tarefas, nem identidade integral de saída. A revisão seguinte investiga preservar a ordem IQ2 original.

`fused-six-profile/` usa instrumentação separada: target67,256/68,007ms por launch curto/longo, contra76,65/77,39ms do perfil inicial. Gate/up23,223/23,346ms; down17,901/17,974ms. Não misturar TPS instrumentados com os benchmarks sem timers.
