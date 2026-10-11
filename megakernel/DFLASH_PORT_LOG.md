# Porte DFlash2: registro de implementação

## Etapa 1 - target MK com draft GGML/CUDA

Autorizado em 2026-10-10: começar a implementar o porte. Clocks padrão;
target e draft GGUF originais. O checkout llama.cpp será usado como biblioteca
e referência sem alterações. Baseline IQ16 preservado com hashes em
`bench/dflash-port/baseline/`.

Primeira entrega: CLI experimental com até quatro propostas, captura das
entradas das camadas `[6,20,34,48,62]`, verificação externa e draft real.
MAXT e os layouts T=5 existentes permanecem nesta etapa. Sete propostas e
fusão dos kernels do draft dependem de validação posterior.

O embedding IQ2_S e LM head Q4_K serão migrados uma vez para buffers GGML
CUDA padrão, emprestados ao modelo draft. Os bytes serão copiados sem
requantização, os ponteiros MK atualizados e as alocações antigas liberadas.
O draft/contexto deve ser destruído antes do target que possui esses buffers.
Isso evita um segundo target e evita wrappers CUDA com layouts privados.

Features: buffer por chunk em F32, callback após prefill e após verificação.
Somente anchor + propostas aceitas são injetadas no KV draft; o bloco de
ruído é removido antes da injeção. Não armazenar cinco hiddens do contexto
inteiro. O bridge de modelo usa `llama-model.h` interno do checkout local;
essa dependência será explícita no build/relatório.

Validação prevista: proposals corretos/incorretos/rejeição por posição,
IDs sem especulação, captura por chunk, bytes/pointer sharing, geração
DFlash2 real, memória e throughput comparado a DFlash2 4 do llama.cpp.
Nenhum ganho será assumido antes da medição.

## Resultado executável

O build `make ARCH=89 all qwen38_dflash external_verify_test` terminou.
O CLI `qwen38_dflash` gera com o target MK e o draft GGUF real, com até
quatro propostas. Não há segundo target ou requantização. O draft GGML/CUDA
é reutilizado, com features F32 por chunk e embedding/head compartilhados.
Instruções e contratos de lifetime/reset: [DFLASH_PORT.md](DFLASH_PORT.md).

## Medições e checkpoints

1. [Verificador/features](bench/dflash-port/01-VERIFICADOR-E-FEATURES.md):
   rejeições nas posições 0–3, aceitação completa, continuação com IDs iguais
   ao decode escalar, bytes compartilhados e prefill por chunks passaram.
2. [Híbrido/memória](bench/dflash-port/02-HIBRIDO-E-MEMORIA.md): 67,05 tok/s
   curto, 55,65 longo, 64,41 sampling 0,7; pico NVML amostrado 14945 MiB.
3. [Qualidade/limites](bench/dflash-port/03-QUALIDADE-E-LIMITES.md): 10/10
   respostas numéricas corretas; 19/20 textos iguais ao MK anterior e 15/20
   iguais ao llama sem especulação. Não estabelece acurácia ampla.
4. [Regressão MK](bench/dflash-port/04-REGRESSAO-DO-MK.md): sem regressão
   observada; variação menor que 0,1% e IDs iguais ao baseline preservado.
5. [Checks finais](bench/dflash-port/05-CHECKS-FINAIS.md): teste externo
   repetido no build final e memcheck de quatro lançamentos MK, zero erros.

Baseline novo llama.cpp/DFlash2 4: 74,13 / 59,52 tok/s. O híbrido é ~10,5%
e ~11,2% mais rápido que MK MTP4 nos dois casos, mas ~9,6% e ~6,5% abaixo
do DFlash2 4 llama.cpp. Continuações longas diferentes limitam a comparação.
Gráfico: [PNG](bench/dflash-port/report/dflash-port.png),
[SVG](bench/dflash-port/report/dflash-port.svg),
[PDF](bench/dflash-port/report/dflash-port.pdf), fontes e hashes no mesmo
diretório. Dados brutos e comandos nos JSONs por ensaio.

Os números do híbrido foram coletados no primeiro build integrado. O build
final acrescentou prompts múltiplos, texto fora do intervalo cronometrado e
rollback seguro durante migração de pesos; os kernels numéricos de decode
permanecem iguais. A suíte de qualidade e o A/B MK usam o build final;
o teste externo completo também foi repetido e passou nesse build.

## Próxima etapa

Comparação independente de logits/features, seguida de estudo T=8/sete
propostas e redução do custo do verificador. O target/injeção consome perto
de 89% do decode; injeção sozinha ~1,4%. Fusão integral do draft não resolve
esse custo principal. Contextos maiores e integração HTTP/multi-turn ainda
não foram implementados/validados. A divergência numérica longa previamente
identificada no MK continua pendente. Launcher de produção e clocks padrão
preservados; llama.cpp sem alterações.


## Continuação: draft nativo e otimizações finais

Implementado o decoder completo em CUDA Graphs, target T=8, GPU features, DP4A/Q8, leituras packed, API32K/checkpoints e testes reais com clocks padrão. Ver [guia atual](DFLASH_NATIVE.md) e [resultados finais](bench/dflash-port/11-ENTREGA-NATIVA-RESULTADOS.md). A arquitetura não funde o draft em um único kernel persistente. Baseline llama.cpp continua mais rápido; limitações numéricas/qualidade estão explicitadas. Journals06–11 e snapshots mantidos.

## Continuação: IQ target e MMQ tipado — 2026-10-11

Journals13–21: perfil do FFN, fusão gate/up dos seis IQ, 36 pares de formatos, especialização/inlining e duas linhas IQ, protótipos IMMA/pipelines derrotados, MMQ Q4_K/Q6_K tipado com quatro linhas por warp e limitação do tail. O teste HTTP ampliado encontrou checkpoint incorreto no segundo turno; corrigido e novamente validado. Build/snapshot anterior e o estágio com falha foram preservados.

Selecionado e validado:93,21/62,28/93,36 tokens/s curto/longo/sampling, seis amostras, clocks padrão. Ganhos sobre o snapshot anterior15,18%/15,40%/15,88%; contra llama fresco+1,49% curto e−0,34% longo, sem vitória geral demonstrada. IDs das três fixtures128 preservados; qualidade10/10 tarefas,17/20 gerações com IDs iguais ao native anterior.36 casos externos,13 HTTP,34 CPU e memchecks filtrados passaram. Draft continua CUDA Graphs. Ver [22: resultados, código, limites e próximos passos](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md).

## Compatibilidade Unsloth — controles de raciocínio

O servidor Python passa a aceitar reasoning_effort no nível principal e os aliases high→xhigh/none→thinking desligado, preservando kwargs e padrão GGUF. O Studio já permite configurar o formato em Connections; não foram editadas sua instalação ou configurações salvas. Reiniciar MK para aplicar. 44 testes CPU/12 subtests e dez checks de renderização com template GGUF/serializer local passaram, sem inferência GPU nesta etapa. Ver [24: código, configuração e limites](bench/dflash-port/24-UNSLOTH-REASONING-EFFORT.md).

## Avaliação de estado da arte e próximos alvos

[25](bench/dflash-port/25-ESTADO-DA-ARTE-E-MARGEM.md) registra a pesquisa em fontes primárias e a comparação local: competitivo, ganho curto1,49% sobre llama.cpp/DFlash2, sem estado da arte geral demonstrado. Prioridade FFN/target e validação representativa; fusão isolada do draft é limitada por sua fração de tempo. Cenários de Amdahl foram calculados e salvos como hipóteses, sem novo benchmark, mudança CUDA, clocks ou interrupção do servidor ativo.


## Q4 target KV — implementation in progress

[26: Q4 target KV](bench/dflash-port/26-Q4-TARGET-KV.md) records the explicit
packed symmetric INT4 target cache, prefill/decode/T=8 integration, and
validation. Q8 remains the launcher default; DFlash cache and model weights
are unchanged. Intermediate results: CPU packing/dequantization oracle for
FP16/Q8/Q4 passed (9,472 values each), packing memcheck zero errors, existing
44 CPU tests/12 subtests passed. Ada compilation and integrated GPU checks
are still in progress; no final quality/performance claim at this stage.


## Q4 target KV — completed

[26](bench/dflash-port/26-Q4-TARGET-KV.md): packed target KV implemented in
prefill, decode and T=8 verification; native DFlash unchanged. At65,536
capacity/chunk128, cache saves1.0625 GiB versus Q8; measured CUDA use15.050
GiB/free0.597 GiB. Short93.23 tok/s versus preserved Q8 at38K93.36; sampling
93.38 versus93.47. The1,365-token fixture drops57.72 versus62.34 with a
changed continuation and27 versus25 cycles. Q8 remains default.

Validation: both formats10/10 small tasks, Q420/20 generations completed
and16/20 matching Q8 IDs; rebuilt Q820/20 matching prior IDs.36 verifier
cases per format, 9 attention cases including position65,535, packing
oracle, 13+13 HTTP checks, actual14,530-token secret retrieval, MTP smoke,
44 CPU tests/12 subtests, bounded CUDA memchecks all passed. No broad
perplexity/full65K-model quality guarantee. Raw/aggregate data and preserved
pre-change binaries: `bench/dflash-port/q4-target/`. No clocks changed.
