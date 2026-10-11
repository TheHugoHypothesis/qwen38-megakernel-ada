# 14 — FFN IQ: fusão DP4A e protótipo IMMA

2026-10-10. Continuação da etapa13, usando clocks de fábrica.

## Primeira comparação A/B

Executável da variante DP4A preservado em `optimization-v2/fused-v1/`; comandos, IDs, tempos e amostras de GPU em `optimization-v2/fused-v1-ab/`. Cada caso teve três repetições com `MK_FFN_IQ_FUSED=0/1`, no mesmo binário. A ordem entre as duas configurações foi invertida para o caso longo. Os 128 IDs de todos os casos e repetições coincidiram com a fixture anterior.

| Caso | FFN anterior, tokens/s | FFN fused, tokens/s | Ganho |
|---|---:|---:|---:|
| Curto | 80,640 | 81,404 | +0,95% |
| Longo | 53,567 | 54,260 | +1,29% |
| Sampling | 79,965 | 80,952 | +1,23% |

Resultado pequeno e consistente nesta comparação, ainda inferior ao baseline histórico llama.cpp/DFlash2. Medição final requer comparações intercaladas e baseline llama na mesma sessão. Não extrapolar para outros contextos, seeds, batches ou qualidade geral.

## Protótipo com Tensor Cores inteiros

`native_iq_mma.cuh` implementa `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` para oito tokens. O K=16 coincide com uma escala das ativações Q16 do target: não há requantização extra. Os seis formatos IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S e IQ4_XS têm escala uniforme nesse trecho e nenhum offset aditivo. Tipos com offsets, como Q2_K/Q4_K/IQ1_M, continuam no caminho anterior. Gate/up podem ter tipos distintos: cobertura de 57 das 64 camadas deste GGUF.

O warp calcula oito linhas por oito tokens e reutiliza a ativação entre gate/up; o epílogo transpõe os resultados para o layout Q16 já existente. Sem cópia de pesos dequantizados, sem novos buffers persistentes e sem alteração da quantização GGUF. `MK_FFN_IQ_MMA=1` habilita o experimento somente em T=8; ainda desligado por padrão até validar integração e throughput. Também será comparado `MK_IQ_INLINE=0/1` para reduzir transporte de acumuladores nas projeções com duas linhas.

Referência primária para o mapeamento de fragmentos: [NVIDIA PTX, mma.m8n8k16](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8816).

## Oráculo independente

`tests/iq_mma.cu` usa linhas reais de matrizes FFN do GGUF, dequantização CPU `ggml_get_type_traits()->to_float`, e dot acumulado em double. Compara o helper CUDA de produção em 36 combinações de tipos gate/up, oito tokens, 64 linhas, entradas Q16 nulas e não nulas com escalas variáveis a cada 16 elementos.

Primeira execução: PASS em 73.728 valores; NRMSE máximo 3,4340e-7; erro absoluto máximo 1,9104e-6. Limites: NRMSE <=1e-5 e pico <=1e-4*(1+RMS da referência). Artefatos em `optimization-v2/mma-oracle/`.

Este teste cobre o produto matricial, mas não o epílogo integrado nem rollback/replay do target. Próximos gates: teste externo de aceitação/rejeição, fixtures completas e A/B com IMMA ligado/desligado. Uma diferença na ordem das somas float é esperada e exige validação; o teste não demonstra equivalência geral de qualidade.
