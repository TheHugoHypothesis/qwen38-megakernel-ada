# 15 — Variantes IQ, resultados negativos e próximos experimentos

2026-10-10. Todos os ensaios mantêm clocks padrão. Os binários e fontes medidos têm snapshots e SHA256 para distinguir versões.

## Integração e primeiro fatorial

`integration-mma/` em `optimization-v2/`: o target com `MK_FFN_IQ_FUSED=1`, `MK_IQ_INLINE=1`, `MK_FFN_IQ_MMA=1` passou em 15 posições de rejeição nas profundidades 1, 4 e 7; continuação em 64 tokens contra decode scalar, byte identity de embedding/head, propostas inválidas, 243 callbacks de features e prefill longo em chunks128. Demorou41,827s. Esse resultado valida casos específicos de execução, não qualidade ampla.

`first-factorial/`: dois repeats por variante, caso curto, 128 tokens. Todos os IDs, 17 ciclos e 111 drafts aceitos coincidiram com a fixture anterior.

| Variante | Tokens/s |
|---|---:|
| Fused DP4A v1, inline desligado | 81,202 |
| Fused DP4A v1, inline ligado | 81,718 |
| IMMA8, inline desligado | 41,733 |
| IMMA8, inline ligado | 41,898 |

IMMA8 rejeitado como default: é matematicamente correto, mas aproximadamente dobra o tempo de geração. Continua experimento explícito, desligado por padrão. Inlining tem ganho pequeno nesta primeira comparação e ainda requer ensaios longo/sampling.

`first-profile/summary.json` usa timers instrumentados, fora do benchmark de throughput. Gate/up:28,138ms com fused,28,024ms com fused+inline e115,375ms com IMMA8. Down:~19,9–20,0ms. Portanto o regresso do IMMA está no gate/up. Sem contadores NCU, não atribuir o regresso a uma causa única.

## Tile maior e formatos constantes

`iq_mma16_dual<GT,UT>` usa `m16n8k16`, mantendo exatamente um grupo Q16 por dot inteiro e separando as escalas das duas metades de linhas. Especializa os seis tipos IQ antes do loopK. É um protótipo de helper; ainda não selecionado pelo target.

Oráculo ampliado (`mma16-micro/oracle.out`): PASS em165.888 valores; IMMA8 e IMMA16 em36 pares de tipos, DP4A em9 pares, entradas nulas/não nulas. NRMSE máximo3,4340e-7; erro absoluto máximo1,9104e-6. A referência usa pesos reais dequantizados pela CPU GGML e dot em double.

Primeiro microbenchmark com matrizes completas FF17408×K5120 e shared92544bytes: IMMA16 sem padding perdeu para DP4A nos nove pares comparáveis. Exemplo IQ3_S/IQ3_S:2,058ms contra0,312ms; IQ4_XS/IQ4_XS:1,153ms contra0,362ms. Não extrapolar esses tempos para o ciclo completo: benchmark não inclui SwiGLU/Q16, todos os formatos/64 camadas, nem o restante do target.

O layout SX atual tem stride160 unidades por token. Um int4 por unidade ocupa16bytes: stride640 palavras de32bits, múltiplo dos32 bancos. O acesso por tokens do MMA aponta estruturalmente para os mesmos bancos. Protótipo de padding será comparado isoladamente antes de alterar o target. O efeito observado precisa de A/B; não há medição direta dos contadores de bank conflicts.

## DP4A misto e down global

Ampliação de `ffn_gu_iq<T,GT,UT>` para gate/up distintos entre IQ3_XXS, IQ3_S e IQ4_XS. Cobertura cresce de24 para41 das64 camadas. `iq_dp4a_dual` compartilha o helper de produção entre oráculo e benchmark, mantendo a ordem anterior de cada dot IQ16. Sem novos pesos nem requantização.

`MK_FFN_DOWN_GLOBAL=1` testa `resid32<T>`, T>5/IT17, com SX apontando para os buffers quantizados globais existentes. Remove três restagings/calls por grupo de32 outputs em T8, aproveitando reutilização do input pelo cache; o epílogo residual/Q32 permanece. Desligado por padrão até medir e validar. Pode perder pelo custo/latência de loads globais: a decisão será por geração completa e correção, não pela redução nominal de cópias.
