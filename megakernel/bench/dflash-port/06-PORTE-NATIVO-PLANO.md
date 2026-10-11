# 06 — Continuação autorizada: porte nativo completo

2026-10-10. Clocks padrão; baseline híbrido quatro propostas preservado
em `full-native-baseline/`, com fontes, binários e SHA-256.

Etapas em execução:

1. Target T=8 com queries attention divididos em dois grupos de três heads,
   sem ampliar shared memory; staging FFN dividido em até três slices.
   MTP interno mantém limite quatro. Remover pesos MTP quando desativado.
2. Draft CUDA nativo: encoder de features, KV sliding, cinco camadas,
   conv dinâmica em dois lados, RoPE/norms, FFN e selector top16/Markov.
   Reusar leitores quantizados; manter GGUF e precisão explícitos. A
   referência GGML permanece disponível para comparação, não como draft
   de produção do caminho nativo.
3. Captura/injeção GPU→GPU; compartilhar embedding/head sem duplicar target.
   Buffers por chunk, KV draft limitado à janela; rejeições não sobrescrevem
   histórico aceito com ruído. Checkpoint e reset coordenados.
4. CLI, C API e HTTP com sampling, streaming, repetição e multi-turn.
5. Oracles, sanitização, benchmarks stock, tuning de profundidade/cache e
   entradas especializadas quando os números justificarem. Registrar
   candidatos descartados e limites; não chamar ganho previsto de medido.

Esta etapa ainda está em execução. Resultados serão adicionados em novos
Markdowns, sem alterar os registros das medições anteriores.
