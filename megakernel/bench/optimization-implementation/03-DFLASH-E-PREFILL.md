# DFlash2 e prefill: decisões após medição

## DFlash2 Q4 no llama.cpp

O draft oficial carregou e gerou com o target GSQ-RCO real. Três repetições
de 128 tokens por caso/profundidade, contexto 4096/Q8, clocks padrão:

| Prompt | Draft 4 | Draft 7 |
|---|---:|---:|
| Curto | 73,95 tok/s | 91,35 tok/s |
| 1365 tokens | 59,47 tok/s | 62,29 tok/s |

Todos os IDs concordaram com a referência canônica do llama.cpp MTP 4,
incluindo o prompt longo. Os logs confirmam selector DFlash2, bloco 8 e
cinco camadas de extração. As mensagens de erro durante a sondagem de memória
do draft não abortaram o carregamento; a execução real funcionou. O loader
ignora os tensores MTP do target quando usa DFlash2.

`serve_ada.py` fornece uma configuração reproduzível com DFlash2 por padrão,
e alternativas `mtp` / `megakernel`. Confere tokenizer, dimensões, feature
taps, arquitetura e selector dos GGUF antes de iniciar DFlash2. É suporte
via llama.cpp, não um porte do draft para dentro de `mk_cycle`.

## GEMM IQ por tile: protótipo validado, rejeitado como padrão

Foi implementado um GEMM 64x64 com WMMA FP16 e acumulação FP32 que mantém
apenas 16 KiB de staging e decodifica pesos IQ16 por tile. O oráculo compara
pesos dequantizados pela CPU e cuBLAS em shapes reais, T=1/127/128/512.
Passou, mas ficou aproximadamente 2–4 vezes mais lento que cuBLAS nos casos
medidos, antes de contar a economia de dequantização global do fallback.
O caminho permanece experimental (`MK_NATIVE_GEMM=tile`), não é padrão.
Não há ganho presumido por eliminar o workspace global.

Para o caminho padrão, mudanças menores e reutilização:

- Loader IQ16 também na materialização FP16 das matrizes dominantes.
- cuBLAS grava diretamente no destino com seu leading dimension e beta para
  residual; elimina um kernel de cópia por matriz e o buffer FP32 temporário.
- Gate/up nativos compartilham a conversão de entrada, evitando seu segundo
  lançamento. Sem reuse quando o gate usou kernel k-quant ou modo tile.
- O preflight conserva margem conservadora de memória apesar da remoção do
  buffer; a seleção automática não aumenta contexto por causa dessa economia.

Foi acrescentado `--score-probe` para registrar top-1/top-2, margem e NLL
de uma posição em teacher forcing, com opção de salvar logits completos.
Isso serve para diagnosticar a divergência; não equivale a avaliação ampla
de perplexidade.

Os testes Python específicos passaram: 26/26 neste ambiente. Uma referência
sem especulação em vinte prompts, com dez respostas numéricas conhecidas,
está em andamento para ampliar a validação de DFlash2 e do megakernel.
