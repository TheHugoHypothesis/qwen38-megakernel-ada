# 13 — Continuação do porte MK: perfil e baseline do FFN

2026-10-10. Objetivo: acelerar o target de oito posições do DFlash2 na RTX 4060 Ti 16 GB, preservando os formatos de pesos, validação e clocks padrão.

Baseline de fontes, objetos, executáveis e biblioteca preservado em `optimization-v2/baseline/`, com `sha256.json`. Nenhuma alteração no checkout llama.cpp. A comparação final precisa reexecutar MK anterior, MK novo e llama.cpp/DFlash2 com as mesmas condições; a execução na 3090 não determina o ganho na 4060 Ti.

O perfil de fases da etapa12 localiza ~29,4 ms em gate/up, ~19,7 ms em down e ~11,9 ms em projeções DeltaNet. Primeira investigação: custos do leitor IQ e do transporte de acumuladores através das chamadas device, tiling e restaging do down. O compilador registra spills na função gate/up T8 e na saída residual T8; esse dado estático não mede tráfego nem prova o gargalo.

Nsight Compute 2026.3.1 está disponível em `/usr/local/cuda/bin/ncu`. Coleta planejada com `--clock-control none`, preservando o controle normal de clocks pelo driver; `--cache-control none` evita limpar artificialmente o cache. Tempos instrumentados não serão usados como benchmark de throughput.

Experimentos serão isolados e comparados com os 128 IDs das fixtures curto/longo/sampling. Alterações matemáticas também exigem oráculos CPU/CUDA, teste de rejeição/replay e comparação de features/logits; identidade em poucos prompts não substitui avaliação de qualidade.

Resultados intermediários serão acrescentados em documentos seguintes, incluindo variantes rejeitadas. Não atribuir eventual melhoria às otimizações sem A/B no mesmo ambiente.

## Coleta de contadores

Nsight Compute terminou com `ERR_NVGPUCTRPERM`: o driver não permite ao usuário atual acessar os contadores de desempenho. Comando, saída e status estão em `optimization-v2/profile-baseline/`. Não foram alteradas permissões do driver nem usados privilégios de administrador. Continuidade com timers de fases e experimentos A/B; bandwidth/stalls permanecem sem medição direta.

## Primeira variante

Adicionada uma variante `ffn_gu_iq<T,TYPE>` para batches T>5 e gate/up do mesmo tipo IQ3_XXS, IQ3_S ou IQ4_XS. Reutiliza o fragmento de ativação entre os dois produtos e mantém seus acumuladores locais até a redução. Conserva a ordem das somas de cada dot e o epílogo original de SwiGLU/Q16. `MK_FFN_IQ_FUSED=0` seleciona o caminho anterior para comparação; o snapshot anterior também permanece disponível.

O GGUF tem mistura por tensor, incluindo gate/up de tipos diferentes na mesma camada. A primeira variante cobre 24 das 64 camadas; não assumir quantização uniforme a partir do nome IQ3_S. Inspeção dos 64 trios e tamanhos em `optimization-v2/ffn-tensor-types.txt`: gate/up total4.800.430.080 bytes; down total2.597.273.600 bytes nos encodings originais.

O runner `dflash_port_bench.py` agora aceita `--binary` e `--cases`, registra flags de otimização e resolve as bibliotecas do checkout local também para executáveis dos snapshots. Isso permite A/B sem restaurar arquivos sobre a implementação atual.
