# 22 — Otimização IQ do target e MMQ do draft nativo

2026-10-11. RTX 4060 Ti 16 GB, sm_89, 34 SMs. Clocks, power limit, driver e pesos preservados. Build selecionado em `optimization-v2/selected-v2/`, com fontes, objetos, executáveis, biblioteca e SHA256. O checkout llama.cpp permanece intacto em `aa94f2086147cd4bfbcff818ce998dffce5fcb39`.

O MK + DFlash2 nativo passou de aproximadamente 81 para 93 tokens/s no prompt curto nesta comparação controlada. Superou nominalmente o llama.cpp nesse caso. No longo, chegou próximo da média do llama, com faixas observadas sobrepostas. **Não foi demonstrada superioridade geral sobre o llama.cpp.** O target é um megakernel cooperativo; o decoder inteiro do draft continua em CUDA Graphs, sem computação GGML. A fusão do draft em um único kernel persistente permanece pendente.

## Medidas finais

Seis amostras por caso/engine, provenientes de duas execuções do CLI ou duas sessões do servidor. Mesmo target GSQ-RCO, draft DFlash2 Q4_K_M, prompt IDs, contexto 4096, target KV Q8 e profundidade sete. Foram emitidos 128 tokens; o intervalo mede os 127 posteriores ao primeiro, excluindo carregamento e prefill. Temperatura zero, exceto sampling 0,7, seed 123.

| Caso | MK anterior | MK selecionado | Ganho sobre MK | llama.cpp DFlash2 | MK vs llama |
|---|---:|---:|---:|---:|---:|
| Curto greedy | 80,923 | 93,208 | +15,18% | 91,837 | +1,49% |
| Longo greedy | 53,973 | 62,283 | +15,40% | 62,496 | −0,34% |
| Curto sampling | 80,566 | 93,361 | +15,88% | não repetido nesta rodada | — |

Valores em tokens/s. Curto: MK selecionado 93,043–93,424; llama 91,152–92,336. Longo: MK 62,216–62,369; llama 62,249–62,738. Mínimo–máximo observado não é intervalo de confiança; as amostras dentro de uma execução são correlacionadas. Boost de fábrica e condições do sistema variam entre sessões.

Todos os 128 IDs de cada uma das três fixtures MK coincidiram com os anteriores, nas seis amostras. O curto também coincide integralmente com o llama. **O longo diverge entre engines a partir do índice 41**, uma diferença já existente antes desta etapa; seus tempos comparam continuações distintas. Não usar esse caso para afirmar uma aceleração no mesmo trabalho entre MK e llama.

Os baselines escolhidos antes da consolidação são `before-b` + `before-c`, `after-a` + `after-b` e `llama-fresh-b` + `llama-fresh-c`. `final-bench-status.json` registra a sequência final: MK novo A, MK novo B, snapshot anterior C, llama C e perfil instrumentado. Os ensaios piloto, incluindo `before-a` e `llama-fresh-a`, também foram preservados.

Artefatos: [resumo JSON](optimization-v2/final-summary.json), [gráfico PNG](optimization-v2/report/iq-optimization.png), [SVG](optimization-v2/report/iq-optimization.svg), [PDF](optimization-v2/report/iq-optimization.pdf), [dados e hashes do gráfico](optimization-v2/report/chart-data.json). Cada diretório de execução contém comandos, flags, resultados, IDs, logs e amostras de VRAM/clocks.

## Código integrado

| Arquivo/função | Mudança selecionada |
|---|---|
| `qwen38_mk.cu::ffn_gu_iq`, `ffn_gu_iq_up`, `ffn_gu` | Gate/up em registradores, reutilizando a mesma ativação para os dois produtos; cobertura de 57/64 camadas nos seis formatos IQ, incluindo pares com tipos diferentes. As sete restantes usam o caminho existente. |
| `qwen38_mk.cu::rows_native_impl`, `native_rows`, `mrows` | Especialização de IQ2, inlining seletivo e processamento simultâneo de duas linhas nos seis IQ para T>5/R2. Guardas preservam o caminho MTP de T<=5. |
| `native_iq_mma.cuh::iq_dp4a_dual` | Helper DP4A reutilizado por gate/up e demais projeções. IQ2 mantém dois updates dot8; IQ3/IQ4 mantêm dot16. Offset e acumulação entre slices cobrem o down FFN. |
| `native_quant.cuh` | Leitores IQ16 dos seis formatos e carregamento packed/alinhado de Q2_K. Os valores de pesos coincidem com o oráculo CPU. |
| `dflash_native.cu::fragment_typed`, `mmq_typed`, `Impl::small_mm` | Q4_K/Q6_K escolhidos antes do loop K, quatro linhas por warp no corpo e no head. Reuso de input/escala com a mesma ordem dot8/FMA/redução do MMQ anterior. Seleção feita na captura do CUDA Graph. |
| `dflash_main.cpp`, `capi.cpp::mk_step` | Verificação limitada à saída restante. O draft padrão continua produzindo o bloco treinado de oito; limita-se apenas a lista passada ao target. Uma única saída restante usa propostas vazias e evita o draft. |
| `server/mk_server.py::Engine.step`, geração HTTP | Passa a capacidade restante ao C API. Mantém o par de checkpoints do prefill, streaming, sampling, reset e reutilização de prefixos. |
| `Makefile`, `src/megakernel_build.py` | Build sm_89, novos oráculos, dependências do helper IQ no objeto e na invalidação do cache. ELF da biblioteca contém apenas sm_89; cache local atual, evitando rebuild integral desnecessário. |

Os pesos GGUF permanecem quantizados em seus formatos originais, sem cópias expandidas persistentes ou requantização. Embedding IQ2_S e head Q4_K continuam compartilhados entre target e draft; vocabulário completo 248320, taps `[6,20,34,48,62]`, selector e parâmetros do modelo preservados. O draft aceita o schema validado localmente de 81 tensores, cinco blocos, Q4_K/Q6_K; não assumir suporte genérico a outros drafts.

Defaults: `MK_FFN_IQ_FUSED=1`, `MK_IQ_INLINE=1`, `MK_IQ_PAIRED=1`, `MK_DFLASH_MMQ_TYPED=1`, `MK_DFLASH_MMQ_ROWS=4`, `MK_DFLASH_MMQ_HEAD_ROWS=4`, `MK_DFLASH_BUDGET=1`. `MK_IQ2_PAIRED` é alias quando a opção principal não está definida.

IMMA8/16, pipelines produtores/consumidores e down via global direto foram testados e perderam desempenho. Permanecem experimentais; `MK_FFN_IQ_MMA=0` e `MK_FFN_DOWN_GLOBAL=0` são os defaults. O journal [13–21](13-FFN-PROFILING-E-BASELINE.md) registra as etapas e os ensaios rejeitados. `native_iq_pipeline.cuh` não está incluído no target de produção.

## Correção e qualidade

[Validação do build selecionado](optimization-v2/validation-final/status.json): todos os sete comandos concluíram com exit0, com hashes das fontes e executáveis.

- Verificação externa: 36 casos em depths0..7, todas as posições de rejeição, continuação escalar, bytes de pesos compartilhados, propostas inválidas e 616 callbacks de features/preenchimento por chunks.
- Qualidade: 10/10 tarefas objetivas corretas; 17/20 gerações com todos os IDs iguais ao native anterior, 16/20 textos iguais ao llama sem especulação e 18/20 iguais ao MTP anterior. As três mudanças frente ao native são `generation-1`, `generation-5` e `generation-8`, casos abertos sem nota objetiva nessa suíte. Identidade de texto não é uma medida de acurácia. Perplexidade e qualidade ampla não foram estabelecidas.
- MTP4 curto/longo: 128 IDs iguais às fixtures anteriores em ambos.
- HTTP: 13 checks, incluindo greedy repetido, streaming, seed, budgets1/2/3/7/8/9, samplers incompatíveis, segundo turno e prompt além de SWA2048, com contexto 32K alocado.
- Compute Sanitizer: zero erros nos filtros `mk_cycle_wide` (dois lançamentos) e `mmq_typed` (até 64 lançamentos). Sanitização limitada aos filtros; não equivale a cobrir todos os kernels/contextos.
- CPU: 34 testes de backend, GGUF e cache/build passaram. `make -q ARCH=89 all qwen38_dflash external_verify_test` confirmou artefatos atuais.

Oráculos dos mesmos helpers aritméticos: 8.945.664 valores de pesos iguais à CPU; 917.504 resultados IQ contra dot double, NRMSE máximo 3,5784e−7 e erro absoluto 1,9104e−6. No draft, 7.840 resultados MMQ R2/R4 iguais ao caminho anterior, mais pesos, convs, norm/RoPE/KV e atenção 2050+8. No head completo, **10.429.440 scores iguais ao MMQ anterior** para as entradas testadas. Esses resultados demonstram correção dos componentes testados, sem estabelecer equivalência integral do modelo.

O teste HTTP ampliado encontrou um erro na primeira versão do limitador: o tail de uma saída sobrescrevia o checkpoint do draft, sem alterar o checkpoint do target. Removido esse salvamento desnecessário; a sequência budgets → sampling → segundo turno passou na versão entregue. Logs com a falha e snapshot foram preservados em `validation-final-budget-checkpoint-failure/` e `budget-checkpoint-v1/`. Detalhes em [21](21-LIMITE-DE-SAIDA-NA-VERIFICACAO.md).

Cycles finais permanecem 17/25/17. Accepted passou a 110/102/110 e drafted a 118/168/115: o limitador deixa de contar/verificar tokens que seriam descartados. **Isso não demonstra melhor qualidade do draft**, e o ganho desse ajuste de tail diminui em respostas maiores.

## Memória e perfil restante

Pico NVML amostrado CLI 4K: 15.122 MiB. No snapshot anterior desses ensaios, 15.102 MiB. As otimizações não acrescentam alocações explícitas persistentes de pesos/ativações; uso total também inclui módulos CUDA, biblioteca e sistema.

HTTP 32K, após requests: **15.496/16.380 MiB**. O log após primeiro prefill reporta aproximadamente 0,56 GiB livres via CUDA; a diferença NVML total−usado não é toda necessariamente alocável pelo processo. Prompt real máximo testado: 2791 tokens; **não foi preenchido todo o contexto 32K**. O orçamento continua incluindo target sem pesos MTP, KV target Q8, DeltaNet e checkpoints, ring conv, features por chunk, draft/KV/checkpoint, staging do encoder e margem CUDA. Ver [orçamento detalhado](../../DFLASH_NATIVE.md).

Perfil instrumentado selecionado, fora das medidas de throughput:

| Fase, média por lançamento | Curto | Longo |
|---|---:|---:|
| Target, soma das fases | 66,738 ms | 66,822 ms |
| FFN gate/up | 23,806 ms | 23,987 ms |
| FFN down | 17,051 ms | 16,792 ms |
| Projeções DeltaNet | 10,808 ms | 10,671 ms |

As médias incluem prime/warmup e batches parciais do tail; não são comparação exclusiva T8. A soma FFN continua perto de 61% desse tempo. No benchmark sem instrumentação, verificação com injeção ocupa aproximadamente 84,7%/83,5% do decode curto/longo; o draft cerca de 15,3%/16,5%. Injeção já está incluída em verificação e não deve ser somada novamente.

O wide usa grade de 34 CTAs × 512 threads, 92.544 bytes de shared por CTA e 128 registradores no relatório ptxas. Stack1328 e spill stores/loads612/1124 são tamanhos estáticos, não tráfego medido. NCU não pôde coletar contadores por `ERR_NVGPUCTRPERM`; bandwidth, cache misses e stalls permanecem sem medição direta. Não foram alteradas permissões do driver.

## Executar e reproduzir

Da raiz do workspace:

```sh
make -C open-jet/megakernel ARCH=89 all qwen38_dflash
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 --ctx 4096
```

O launcher encontra/valida os GGUF locais. O build atual está pronto; o comando make só recompila se necessário. Para 32K alocado, substituir `--ctx 4096` por `--ctx 32768`, dentro dos limites de memória/teste acima. O default geral do launcher permanece llama.cpp/DFlash2; escolher explicitamente `megakernel-dflash2`.

Dentro de `open-jet/megakernel`, com os caminhos locais target/draft em variáveis próprias:

```sh
python3 bench/iq_cuda_checks.py --model "$QWEN_TARGET" --draft-model "$QWEN_DRAFT" --output bench/dflash-port/optimization-v2/validation-replay --sanitize
python3 bench/iq_optimization_bench.py --variants inlined --reps 3 --output bench/dflash-port/optimization-v2/replay
python3 bench/plot_iq_optimization.py
```

O runner IQ usa os prompts e comandos gravados no baseline native; aceita `--binary`, `--cases` e flags de variantes. O baseline anterior executa pelo snapshot, sem restaurar fontes sobre o build novo. O relatório JSON preserva comandos completos do llama e dos demais testes.

## Próximos passos por evidência

| Prioridade | Trabalho | Impacto esperado, complexidade e risco |
|---|---|---|
| 1 | Ampliar comparação de logits/features com teacher forcing idêntico, perplexidade e suíte objetiva; investigar as três gerações alteradas e a divergência longa preexistente. | Necessário para sustentar qualidade e comparar engines no mesmo trabalho; complexidade média/alta, risco de falsas conclusões se usar apenas texto livre. Nenhum ganho de throughput presumido. |
| 2 | Otimizar FFN target mantendo GGUF packed: reuso de metadados/tabelas, tiling e limite de acumuladores; medir microbench e ciclo completo antes de integrar. | Maior parte medida do target (~61%). Complexidade alta; risco de spills, i-cache, mudanças de arredondamento e sincronização. Ganho percentual ainda desconhecido. |
| 3 | Examinar down/projeções DeltaNet e reduzir código experimental no binário de produção; testar políticas de cache por unidade CUDA. | Fases medidas relevantes. Complexidade média/alta; remover variantes derrotadas pode reduzir build/módulo, sem garantia de tokens/s. Não mudar shared/grid sem rever scratch, reduções e barreiras. |
| 4 | Fundir kernels pequenos do draft dentro de blocos, depois estudar draft persistente com orçamento de registradores/shared próprio. | Draft ocupa apenas ~15–17% do decode; seu peso deve orientar prioridade. Complexidade alta, risco de perder occupancy ou alterar conv/norm/RoPE/SWA. Fusão monolítica não garante ganho. |
| 5 | Ampliar prompts/respostas e medir fim a fim, TTFT e contextos realmente preenchidos; repetir A/B por sessões. | Estabelece desempenho útil além do decode128 e efeito do tail. Complexidade baixa/média; aceitação e comportamento da saída podem dominar variação. |

Aceitar uma próxima otimização somente com ganho repetido em múltiplos casos, orçamento de VRAM real, oráculos e replay/HTTP aprovados, e avaliação de qualidade apropriada. Uma vitória geral ainda depende desse trabalho; a vitória curta e os ganhos sobre o próprio MK estão medidos nesta etapa.
