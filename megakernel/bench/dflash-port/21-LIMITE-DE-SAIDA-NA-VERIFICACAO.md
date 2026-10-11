# 21 — Limitar o trabalho no fim da resposta

2026-10-11. MMQ typedR4 selecionado; target de19 preservado.

`dflash_main.cpp` e `capi.cpp::mk_step` limitam o número de propostas verificadas ao espaço restante na saída. **O decoder continua produzindo o bloco completo treinado de oito**, mantendo as propostas originais; só a verificação é limitada. Para uma única saída restante, verifica o anchor com propostas vazias, evitando o draft inteiro. `verify_external` já suporta esse caso T1; não foi necessário modificar o kernel CUDA.

No C API, `cap` passa a limitar efetivamente a saída do DFlash. Cap1 usa o caso scalar; cap2..8 limita proposals. O MTP preserva a capacidade mínima antiga. `server/mk_server.py::Engine.step` recebe o orçamento restante apenas quando há draft DFlash. `MK_DFLASH_BUDGET=0` permite diagnosticar o comportamento anterior no CLI/servidor.

O limitador remove trabalho e contagem de aceitação referentes a tokens que seriam descartados após `max_tokens`. Uma melhora da taxa de aceitação por mudança do denominador não demonstra melhor qualidade do draft. O efeito de throughput diminui em respostas muito mais longas, pois atua no último ciclo.

`optimization-v2/budget-ab/`: três repetições por caso, mesmo binário e clocks padrão. Resultados: curto 93,463→93,491 tokens/s; longo 61,957→62,423; sampling 92,893→93,385. Os 128 IDs permaneceram iguais. Cycles 17/25/17; accepted 111→110, 106→102 e 114→110, eliminando aceitação fora da saída pedida.

O teste externo foi ampliado para depths0..7, todas as posições de rejeição (36 casos), para cobrir T1, T6 e T7 e replay. HTTP acrescenta budgets1/2/3/7/8/9 comparados ao prefixo greedy da resposta completa. Esses testes precisam passar antes da entrega final.

## Correção detectada na validação HTTP

Os 36 casos externos e os budgets HTTP passaram, mas o segundo turno encontrou `native draft invalid feature position/gap`. A primeira revisão salvava um novo checkpoint do draft no tail de uma saída, sobrescrevendo o checkpoint do prefill; o target ainda rebobinava ao prompt original. Removido o checkpoint do tail no CLI/C API: esse caminho não executa o draft nem escreve KV especulativo, logo não precisa de outro checkpoint. Ambos preservam o par de checkpoints do prefill. A suíte HTTP inclui agora a sequência budgets → sampling → segundo turno para reproduzir essa regressão.

A revisão corrigida passou a validação completa em `optimization-v2/validation-final/status.json`: 36 casos externos, qualidade 10/10 tarefas e 17/20 IDs iguais ao native anterior, regressão MTP4, 13 checks HTTP e dois memchecks filtrados com zero erros. HTTP 32K alocado terminou com 15.496/16.380 MiB de uso NVML; o prompt real atravessa SWA2048, mas não preenche 32K. O snapshot com o erro está em `budget-checkpoint-v1/`, e o corrigido em `selected-v2/`.
