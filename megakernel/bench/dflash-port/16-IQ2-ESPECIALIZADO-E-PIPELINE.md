# 16 — IQ2 especializado e investigação de pipeline IMMA

2026-10-10. Continuação sem mudanças de clocks.

## Mudanças incorporadas ao candidato

`mrows<T,IT,R,T_ANY>` agora especializa IQ2_XXS, IQ2_XS e IQ2_S quando T>5. Mantém o caminho genérico anterior para batches menores. O leitor tem tipo constante dentro do loopK, permitindo eliminar os testes de outros formatos e o cálculo do offset zero. `quant8(Q2_K)` lê duas palavras de32bits em lugar de oito bytes, preservando o payload e os valores.

Oráculo `narrow-micro/quant.out`:8.945.664 valores reais iguais à dequantização CPU llama.cpp. Testes dos produtos (`narrow-micro/oracle.out`):239.616 valores,36 pares de IQ para IMMA8/IMMA16/narrow e nove para fused DP4A; NRMSE máximo3,4340e-7 e erro absoluto máximo1,9104e-6. Leitor narrow de quatro valores passou, mas o seu IMMA direto ainda perdeu no microbenchmark; não habilitado no target.

Snapshot `typed-iq2-v1/` preserva fonte/objeto/binários e SHA256. `typed-iq2-ab/` contém os comandos, flags, temperaturas, clocks, potências, IDs e três repeats de cada combinação:

| Caso | Fused, inline0 | Fused, inline1 |
|---|---:|---:|
| Curto |85,967 tokens/s|86,321 tokens/s|
| Longo |57,075 tokens/s|57,500 tokens/s|
| Sampling |85,367 tokens/s|85,882 tokens/s|

Todos os128 IDs coincidiram com as fixtures anteriores. Ciclos/aceitação também permaneceram iguais: curto17/111, longo25/106, sampling17/114. Ainda requer comparação intercalada com o snapshot do início da etapa e llama.cpp na mesma sessão; não usar esta tabela para declarar que o llama foi superado.

## Variantes descartadas como default

`mixed-global-ab/`: fusão DP4A de formatos mistos chegou a82,876 tokens/s no caso curto. Leitura direta do input global no down caiu para66,266–66,841 tokens/s, mantendo IDs/aceitação. Down global continua desligado. O padding de SX para o IMMA não alterou os valores e praticamente não melhorou o tempo (`mma16-padding/`).

## Pipeline de pesos

`native_iq_pipeline.cuh`: oito warps produtores decodificam slices K128 em leituras contíguas por linha GGUF; oito consumidores executam IMMA16 sobre o slice anterior. Dois estágios de pesos e input ocupam84.800bytes de shared, dentro da região SM_MISC existente de87.040bytes. Nenhum buffer de pesos expandidos persistente em VRAM.

`pipeline-micro/`:313.344 valores passaram no oráculo CPU; todos os produtos das matrizes completas coincidiram bit a bit com o IMMA16 sem pipeline. O pipeline reduziu bastante o custo do IMMA direto, mas ainda perdeu para DP4A: IQ3_S/IQ3_S0,605ms contra0,312ms; IQ4_XS/IQ4_XS0,666ms contra0,361ms. Ainda não integrado ao target.

Próximo protótipo: cada warp decodifica e consome o próprio tile, usando barreiras de warp para os pesos e dois estágios compartilhados para o input. Dobra os tiles simultâneos e reduz à metade as barreiras CTA. Para gate/up com pesos de escala uniforme32 (IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_XS), o input `cB.qn` vem de `q32_store`; permite IMMA K32 sem requantização. Os formatos IQ2_XS/IQ2_S com escala16 continuam em K16. Essa condição não vale para o input Q16 do down, nem para uma hipotética mudança do quantizador de ativações.

Prototipar/oracular/medir antes de integrar; a redução de operações ou de barreiras não basta como evidência de ganho.
