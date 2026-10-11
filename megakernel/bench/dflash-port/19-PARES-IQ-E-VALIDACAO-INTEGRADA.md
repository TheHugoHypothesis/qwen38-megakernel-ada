# 19 — Seis IQ em pares e validação integrada do target

2026-10-11. Snapshot `optimization-v2/paired-six-v1/` com fontes, objetos, binários e hashes. Clocks padrão.

`rows_native_impl`, T>5/R2, passa a usar duas linhas simultâneas para IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S e IQ4_XS. A opção principal é `MK_IQ_PAIRED`, default1; `MK_IQ2_PAIRED` é alias. IQ2 mantém dois updates dot8; IQ3/IQ4 mantêm dot16. Guardas preservam o caminho T<=5.

`paired-six-ab/`: três repetições, alterando apenas a seleção paired no mesmo binário. Paired0→1: curto90,230→90,775; longo59,880→60,454; sampling89,577→90,232 tokens/s. Todos os128 IDs/ciclos/aceitações coincidiram com a fixture anterior. A ampliação trouxe ganho pequeno adicional; ainda não superou o llama.cpp.

`paired-six-validation/status.json` registra os hashes e comandos do target desta revisão, com o draft ainda anterior ao experimento MMQ tipado (arquivado antes da validação da revisão seguinte):

- Rejeição/replay: exit0,15 posições em depths1/4/7, continuação scalar,243 callbacks e prefill longo.
- Qualidade: suite20 executada; consultar `paired-six-quality/results.json` para acertos e identidade de texto/IDs. Não inferir acurácia ampla.
- MTP4 curto/longo: os128 IDs permaneceram iguais ao MTP anterior.
- HTTP: repetição greedy, streaming, seed, rejeição de samplers incompatíveis, segundo turno, prefill/SWA e contexto32K alocado.
- Compute Sanitizer: duas execuções filtradas `mk_cycle_wide`, exit0 e `ERROR SUMMARY: 0 errors`. Isso não sanitiza todos os kernels/contextos.

Relatório ptxas do wide:128 registradores,1328bytes de stack,612/1124bytes spill stores/loads estáticos. Isso não mede tráfego de spills em execução. Compilação dessa entry levou aproximadamente617s; ptxas chegou a aproximadamente21GiB de RAM host durante o build. Esses números não são reserva de VRAM. As alocações CUDA explícitas do modelo não mudaram.

Próximo experimento: especializar Q4_K/Q6_K antes do loop do MMQ nativo do draft e comparar tiles R2/R4, mantendo dot8, FMA, quantização e redução anteriores. O alvo inclui o head de vocabulário completo; nenhuma restrição de tokens, cópia expandida dos pesos ou mudança do selector.
