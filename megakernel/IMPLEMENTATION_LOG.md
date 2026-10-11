# Implementação das otimizações Ada

## 2026-10-10 — início da continuação

Autorização: implementar as mudanças propostas e registrar o trabalho em
Markdown. Clocks devem permanecer padrão. Não alterar os pesos GSQ-RCO nem
o checkout de referência llama.cpp. A GPU estava livre, com 41 MiB usados.

Baseline preservado em `bench/optimization-implementation/baseline/`, incluindo
fontes, objeto, binários e hashes. Isso permite A/B e restauração de candidatos
que não apresentem ganhos reais.

Referências atuais: llama.cpp MTP 4 ~62,96 tok/s no prompt curto; megakernel
MTP 4 ~48,17. Teacher forcing longo 127/128, com primeira divergência no
índice 41. A equivalência numérica geral ainda não foi demonstrada.

### Ordem de trabalho

1. Agrupar decode IQ3_S/IQ3_XXS/IQ4_XS em 16 pesos, mantendo DP4A e o GGUF
   original; comparar dequantização e produtos com oráculo independente.
2. Seleção automática de chunk com orçamento de VRAM, preservando override
   explícito e a preferência de precisão do KV existente.
3. Medir A/B e diagnosticar a divergência, sem inferir qualidade a partir de
   poucos argmax.
4. Avaliar redução de recursos, prefill por tile e adaptação MTP conforme os
   ganhos medidos. Candidatos experimentais só viram padrão após validação.
5. Obter o draft DFlash2 Q4 oficial e testar o suporte existente no llama.cpp
   com este target, antes de decidir o porte para o megakernel.

Logs por etapa serão mantidos em `bench/optimization-implementation/`.
O critério principal é desempenho de geração observado, com memória e
corretude documentadas; flags ou kernels adicionais não constituem ganho.

## 2026-10-10 — implementação concluída e validada

IQ16 está ativo para IQ3_XXS/IQ3_S/IQ4_XS. Chunk automático, prefill cuBLAS
com menos cópias, MTP 4 por padrão e probe de logits foram entregues.
A/B intercalado final: +25,54% curto, +24,81% longo, +25,05% sampling,
com mesmos IDs do engine original. Prefill longo: −22,78% de latência.
GEMM tiled validado, mas mais lento; permanece experimental e desligado.

DFlash2 Q4 oficial foi baixado e compatibilidade real verificada. A execução
reutiliza o llama.cpp local: 91,35/62,29 tok/s greedy curto/longo e 92,56 tok/s
sampling. `serve_ada.py` escolhe esse caminho por padrão. API real dos dois
launchers passou; DFlash2 também passou com contexto alocado 32K e fit off.
Nenhum porte do draft para dentro de `mk_cycle` foi realizado.

Validação: 8.945.664 valores reais no oráculo, GEMMs contra CPU/cuBLAS,
34 testes host, memcheck de quatro ciclos sem erros. Os três engines acertaram
10/10 problemas numéricos; DFlash2 teve 17/20 gerações idênticas ao baseline
sem especulação e MK 16/20. A divergência teacher forcing longa MK continua
127/128 e não foi corrigida. PPL/qualidade ampla continuam não medidas.

Registros intermediários e evidências:
[índice das seis etapas](bench/optimization-implementation/README.md).
Entrega: [IMPLEMENTATION_RESULTS.md](IMPLEMENTATION_RESULTS.md) e
[gráfico final](bench/optimization-implementation/report/ganhos_implementados.png).
Clocks padrão mantidos; servidores de teste encerrados ao final.

## Investigação posterior — porte DFlash2

A avaliação foi registrada em
[07 — viabilidade do porte](bench/optimization-implementation/07-VIABILIDADE-PORTE-DFLASH2.md).
Constatado limite MAXT=5 e ausência de embedding/head próprios no draft real;
o híbrido exige ponte de pesos/features, não apenas duas chamadas de API.
Recomendação: perfilar e validar um protótipo antes do porte completo.
Nesta investigação adicional não foram modificados kernels nem executados
novos benchmarks.

## Início do porte DFlash2 para MK

Implementado e executado o híbrido target MK + draft GGML/CUDA real, com
pesos compartilhados, captura de features e verificação externa até quatro
propostas. Build sm_89, 34 testes host, rejeições/replay, suíte de 20 saídas
e A/B MK normal passaram nos limites documentados. O protótipo cabe em 16 GB
e melhora MK MTP4, mas ainda fica abaixo do DFlash2 4 do llama.cpp.
Registro e resultados: [DFLASH_PORT_LOG.md](DFLASH_PORT_LOG.md).
Execução e próximos contratos: [DFLASH_PORT.md](DFLASH_PORT.md).
