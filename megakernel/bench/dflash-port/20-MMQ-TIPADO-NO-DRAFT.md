# 20 — MMQ tipado e tiles R4 no draft nativo

2026-10-11. O target de [19](19-PARES-IQ-E-VALIDACAO-INTEGRADA.md) fica inalterado.

`dflash_native.cu::fragment_typed<TYPE>` especializa Q4_K ou Q6_K antes do loopK. O wrapper `fragment` continua disponível para embedding e referência. `mmq_typed<T,TYPE,ROWS>` usa duas ou quatro linhas por warp, compartilhando a leitura de input/escala; mantém o dot8, `fmaf` explícito, offset e redução XOR anteriores. `small_mm` escolhe o kernel na captura do CUDA Graph, sem novos buffers nem conversões de pesos.

Controles de experimento: `MK_DFLASH_MMQ_TYPED=0` usa o MMQ clássico; `MK_DFLASH_MMQ_ROWS=2|4` escolhe as matrizes normais; `MK_DFLASH_MMQ_HEAD_ROWS=2|4` escolhe separadamente o head. O candidato inicial usa typed/R2 no corpo e R4 no head. A escolha final depende do benchmark completo. Caminho F32 permanece disponível.

`draft-typed-oracle/math.out`:7.840 resultados R2/R4 emT1/4/7/8 coincidiram exatamente com o DP4A clássico, usando quatro linhas amostradas de cada matriz real do draft. Oráculo completo também passou:1.271.808 pesos,2.548 dots F32, convs, KV/RoPE, sliding attention2050+8; erro de peso0 e máximo matmul8,0724e-6.

`draft-typed-oracle/head.out`:10.429.440 scores do head de vocabulário completo, em três entradas incluindo zero, coincidiram exatamente com o MMQ clássico para ambos os tiles. Mais273 dots amostrados contra CPU GGML Q8/Q4_K e dot double: NRMSE máximo4,01674e-7; erro absoluto máximo1,64463e-6.

Microbenchmark headT7,30 repeats após cinco warmups: clássico4,379ms; typedR2 4,501ms; typedR4 4,325ms. Ganho R4 pequeno; typedR2 perdeu. Não extrapolar para tokens/s ou habilitar uma variante com base apenas nesse número. O teste do decoder completo, com IDs/aceitação, está em `draft-typed-ab/`.

## Decoder completo: escolha R4

Três repetições por variante/caso no mesmo binário, com target paired1 e clocks padrão. Todos os128 IDs, ciclos e aceitação coincidiram com as fixtures.

| Variante | Curto | Longo | Sampling |
|---|---:|---:|---:|
| MMQ clássico |90,951|60,227|90,161|
| TypedR2 |91,244|60,508|90,664|
| TypedR2 + headR4 |91,295|60,698|90,820|
| TypedR4 geral |93,172|61,984|92,839|

O default foi alterado para typedR4 no corpo e head. O ganho geral veio das outras matrizes também; o microbenchmark isolado do head não teria previsto isso. O curto superou nominalmente o llama atual (~92,19) por~1%; o longo ainda perdeu (~62,73). Isso não demonstra superioridade geral; ainda requer confirmação e qualidade/HTTP do build escolhido.

Validação da revisão selecionada, com a correção de checkpoint de [21](21-LIMITE-DE-SAIDA-NA-VERIFICACAO.md), em `optimization-v2/validation-final/`: 36 casos externos, 20 gerações de qualidade, regressão MTP4 curta/longa, 13 checks HTTP em contexto 32K alocado e memcheck filtrado do target/MMQ tipado passaram. O memcheck do draft cobre até 64 lançamentos selecionados por nome; não cobre todo o modelo em todos os contextos.
