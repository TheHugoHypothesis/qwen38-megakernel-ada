# 08 — Decoder completo e otimização DP4A

Corrigida a escrita do KV cache nativo: faltava somar o canal ao índice do head. Criado oracle independente norm/RoPE/escrita de todos os 128 canais, KV histórico e ruído; erro FP16 máximo 5,18e-4. Atenção sliding CPU passou, erro máximo 1,26e-9.

Decoder F32: selector frente à referência GGML FP16 teve 110–111/112 candidatos iguais, NRMSE 0,39–0,49% e propostas iguais na fixture curta. Benchmark três repetições: curto 66,22, longo 44,41, sampling 65,81 tokens/s. Draft ainda consome ~34 ms por ciclo; F32 não é recomendação final.

Adicionado caminho DP4A com ativação Q8 por 32 valores e escala FP16, preservando os pesos GGUF originais. Oracle usa quantização/dequantização Q8 CPU GGML e produto com pesos CPU em precisão double. Selector continua top16 sobre vocabulário completo. A verificação target decide cada token final. Validação completa e benchmark Q8 em execução.

Entradas especializadas: wide stack 1104 bytes, spills store/load 516/1124; MTP 2800 bytes, 1560/1928; entrada unificada T1..8 anterior 3072 bytes, 1676/2736. MTP A/B mesma sequência, ~57,6 tokens/s em ambos; nenhuma aceleração relevante nesse workload, preservação do baseline.

Features target: diferença numérica MK/llama permanece. Não reivindicar perplexidade/qualidade ampla preservada apenas pelos testes pequenos. Sem alteração de clocks.
