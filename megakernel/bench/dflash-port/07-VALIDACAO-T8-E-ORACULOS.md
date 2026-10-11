# 07 — T=8 e validação intermediária

- Corrigido ring de convolução DeltaNet: 8 → 16 posições; T=8 exige guardar T+3. Verificador passou profundidades 1/4/7, todas as posições de rejeição, replay até 64 tokens escalares, 243 callbacks, prefill longo e bytes compartilhados.
- Oracle CPU: 1.271.808 valores Q4_K/Q6_K idênticos; 2.548 dots F32, erro máximo 8,08e-6. Conv dinâmica, ambos os lados, passou.
- Atenção sliding independente em posição 2050 com oito tokens, janela 2048 e ring: erro máximo 1,26e-9.
- Decoder completo ainda diverge da referência GGML. Diagnóstico em andamento com dumps intermediários; não liberado como validado.
- Features do target para sequência longa fixa: comparação dos cinco taps em três posições registrada em features-long-comparison.json. Cosine mínimo ~0,9515 em posição intermediária, camada 48; divergência já no target, não prova equivalência numérica integral ao llama.cpp.
- Entradas de target T≤5 e T≥6 separadas para reduzir pressão de stack/spills do MTP; desempenho ainda precisa medir. Reserva explícita de VRAM externa para auto chunk antes do upload.
- Nenhuma alteração de clocks.
