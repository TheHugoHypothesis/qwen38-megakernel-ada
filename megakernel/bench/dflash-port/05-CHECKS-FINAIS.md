# 05 — Checks finais

Build final sm_89: `all`, `qwen38_dflash` e `external_verify_test` passaram.
Os hashes de fontes/binários e ptxas estão em `sha256-final.json`.

`external_verify_test` foi repetido no build final e novamente passou:
rejeição nas posições 0–3, aceitação das quatro propostas, continuação
com 64 IDs iguais ao decode escalar, propostas inválidas, amostras dos
bytes compartilhados e 89 callbacks incluindo prefill longo por chunks.
Saídas: `external-verify-final.out/.log`.

Compute Sanitizer memcheck executou o híbrido real, com filtro
`--kernel-name kns=mk_cycle --launch-count 4 --error-exitcode 99`:
**exit 0, ERROR SUMMARY: 0 errors**. Inclui captura de features, uso dos
pesos compartilhados e verificação externa no caminho MK. O filtro cobre
quatro lançamentos MK; não valida todos os kernels GGML do draft nem uma
geração longa. Os tempos sob instrumentação não contam como benchmark.
Saídas: `memcheck.out/.log`.

34 testes host passaram; Python dos runners compila; `git diff --check`
passou. O checkout llama.cpp continua sem alterações. Servidores de
benchmark e processos GPU criados nesta etapa terminaram. Clocks padrão
mantidos durante todo o trabalho.

O gráfico `report/dflash-port.png` foi inspecionado visualmente. SVG/PDF e
hashes dos quatro JSONs de origem estão no mesmo diretório. Ele separa
throughput, tarefas numéricas e identidade de textos, com os limites de
comparabilidade das continuações longas indicados.

Esta entrega conclui o protótipo híbrido de quatro propostas. Sete
propostas/T=8, kernels draft fundidos, HTTP/multi-turn, contextos maiores
e oracle numérico profundo ainda não foram implementados/validados.
