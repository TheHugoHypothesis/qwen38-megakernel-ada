# DFlash2 nativo no backend MK — RTX 4060 Ti 16 GB

O target e a verificação continuam no megakernel cooperativo; o decoder DFlash2 inteiro executa kernels CUDA nativos capturados em CUDA Graphs, sem `llama_decode`, `llama_process` ou execução de grafo GGML para o draft. O GGML permanece para os descritores/buffers dos pesos compartilhados, e libllama para o tokenizer. O caminho de referência GGML existe no CLI para os oráculos.

**O draft não foi fundido em um único kernel persistente.** Esta entrega é o porte funcional completo para o backend MK com orquestração por CUDA Graphs. Fusão monolítica/IMMA adicional continua uma otimização futura; não foi medida como realizada.

## Executar

Na raiz do workspace:

```sh
make -C open-jet/megakernel ARCH=89 all qwen38_dflash
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 --ctx 4096
```

O launcher encontra os dois GGUF locais e valida arquitetura, dimensões, taps, tokens, merges, BOS/EOS e block size. Native DFlash2 usa sete propostas por padrão; `--draft 4` permite comparar quatro. MTP regular continua `--engine megakernel`. O default geral `--engine dflash2` continua o baseline llama.cpp; os resultados atualizados do MK estão abaixo. Nenhum comando altera clocks.

HTTP direto: `server/mk_server.py --model TARGET --draft-model DRAFT --ctx 32768 --kv q8 --prefill-chunk auto`. `--draft` omitido escolhe sete com DFlash2 e quatro com MTP. Não há fallback silencioso para GGML no caminho nativo. Uma falha de callback após o target modificar estado requer reset coordenado antes de continuar.

OpenJet: selecionar backend megakernel e definir `OPENJET_MK_DFLASH_MODEL` com o caminho do draft; `OPENJET_MK_DRAFT=7` é opcional. Esses valores são passados ao servidor nativo. Build cache inclui as novas fontes.

CLI: `qwen38_dflash -m TARGET --draft-model DRAFT --prompt-ids IDS -c 4096 -d 7 -n 128 --reps 3 --prefill-chunk auto`. IDs não recebem chat template. Flags `--backend ggml` e `--oracle` escolhem referência e comparação. `MK_DFLASH_MATH=f32` desativa DP4A/Q8; `MK_DFLASH_GRAPH=0` desativa captura, ambos para diagnóstico. `MK_DFLASH_TRACE=DIRETORIO_EXISTENTE` salva intermediários e desativa graphs; não usar em benchmarks.

### Unsloth Studio e parâmetros de sampling

O preset Qwen3.8 do Unsloth instalado usa `presence_penalty=1.5`. O MK atual aceita somente o valor neutro zero para presence/frequency e um para repetition; um valor diferente retorna HTTP400. Na seção **Sampling** do chat, selecionar **Presence Penalty = 0 / Off**, **Repetition Penalty = 1 / Off**, **Top K = 20** e **Min P = 0**. Temperature0,7/TopP0,8 continuam compatíveis. Se houver Frequency Penalty, usar zero. Não reenviar o preset padrão que restaura Presence1,5.

[Parâmetros JSON compatíveis](server/unsloth-mk-sampling.json) para copiar aos campos do cliente ou incorporar ao corpo da requisição; o arquivo não é um formato de importação do Studio. Isso desativa a penalidade explicitamente, sem fingir implementá-la. Para manter Presence1,5, executar `serve_ada.py --engine dflash2 --ctx 4096`, que usa llama.cpp/DFlash2. A implementação nativa de penalidades ainda está pendente. Ver [diagnóstico local](bench/dflash-port/23-UNSLOTH-SAMPLING-COMPATIVEL.md).

### Unsloth Studio e esforço de raciocínio

O template do GGUF local habilita thinking e escolhe `xhigh` quando nenhum controle é enviado. `max_tokens` limita raciocínio + resposta juntos; o servidor não reserva uma parcela para a resposta nem impõe um orçamento separado de thinking.

O MK agora aceita `reasoning_effort` no nível principal, como enviado pelo Unsloth: `low`/`medium` seguem para o template, `high` vira `xhigh`, `xhigh` também é aceito diretamente, e `none` desativa thinking. `enable_thinking` e `preserve_thinking` também podem ser enviados no nível principal. Valores não nulos em `chat_template_kwargs` têm prioridade por campo. Após resolver os campos, `enable_thinking=false` ou effort `none` desativa raciocínio. Controles inválidos retornam HTTP400. Ausência de controles preserva o padrão original do GGUF.

Para conexão **Custom** do Studio, em **Connections**, editar a conexão e usar API **Chat Completions**. Marcar **Reasoning model**, habilitar **Enable reasoning controls**, selecionar **Reasoning parameter style = reasoning_effort** e salvar. No chat, escolher **Low** para thinking breve ou desligar Thinking para resposta direta. A opção `chat_template_kwargs.enable_thinking` é adequada para ligar/desligar, mas não encaminha Low/Medium/High no provider Custom instalado. As opções `reasoning` e `thinking` não são os formatos usados por esta compatibilidade MK.

Se a conexão usa o preset **llama.cpp**, o Studio já envia o esforço no nível principal e o toggle em `chat_template_kwargs`; o MK passa a interpretar essa combinação. Basta reiniciar o servidor MK após atualizar este arquivo Python e escolher o esforço no chat. A instalação do Unsloth não precisa ser editada nem reiniciada para o ajuste de servidor; salvar a configuração da conexão é uma ação na interface. Manter as penalidades neutras descritas acima.

O stderr do servidor confirma `reasoning: on, effort=low`, `reasoning: on, effort=xhigh` ou `reasoning: off`. `effort=template default` indica que nenhum esforço foi recebido; para o GGUF atual significa `xhigh`. Esta alteração Python não requer recompilar CUDA. Não reiniciamos o processo em uso nem modificamos configurações salvas do Studio durante a correção.

[Registro da compatibilidade e validação](bench/dflash-port/24-UNSLOTH-REASONING-EFFORT.md): 44 testes CPU/12 subtests e dez checks de renderização com template GGUF real, incluindo o serializer instalado do Studio. Esses checks não são inferência GPU ou teste ponta a ponta de uma conversa.

## Componentes portados e contratos

- `dflash_native.cu/.h`: encoder `fc`, norm, K/V dos cinco blocos; decoder completo com duas convs dinâmicas por subbloco, Q/K RMSNorm, RoPE completo 128, atenção non-causal SWA, FFN SwiGLU, lm_head de vocabulário completo, top16 e selector Markov rank256. `draft`, `inject_device`, `checkpoint`, `restore_checkpoint`, `truncate` e `reset`.
- `qwen38_mk.cu/.h`: MAXT=8, atenção em grupos de três heads para T>5 com a mesma shared memory; FFN down em até três slices; `mk_cycle_wide` separado de `mk_cycle` MTP, sem carregar pesos MTP quando desligado. Ring da convolução ampliado de 8 para 16, pois T=8 precisa T+3 posições. `enable_device_features` captura entradas antes da norm de `[6,20,34,48,62]`, aceitas apenas, GPU→GPU. `verify_external` reutiliza rejeição/replay DeltaNet. MTP interno mantém limite quatro.
- `capi.cpp`, `server/mk_server.py`: `mk_new_dflash`, lifetime draft antes do target, sampling do target, checkpoint, rewind, truncate, reset, streaming e reutilização de prefixos. Servidor serializa requests; uma sequência de cada vez.
- `Makefile`, `serve_ada.py`, `src/llama_server.py`, `src/megakernel_build.py`: build sm_89, launcher explícito, configuração OpenJet e invalidação do cache de build.

Pesos permanecem nos formatos GGUF. Draft aceita Q4_K e Q6_K original 210B; leitores packed preservam os valores dequantizados, com carregamento alinhado a 2 bytes em Q6_K. Embedding IQ2_S e head Q4_K são emprestados do target, sem duplicação ou restrição de vocabulário. Ativações dos pequenos GEMMs são Q8 por 32 valores, escala FP16, produto DP4A e acumulação F32. Encoder com batch>8 usa cuBLAS FP16 com acumulação F32 e workspace limitado, dequantizando oito pesos por thread. F32 direto fica como referência alternativa.

Schema draft validado: 81 tensores, 5 blocos, D5120, FF17408, Q32/KV8, HD128, conv kernel2/group16, selector rank256/top16, block8, taps acima, todas as camadas SWA2048, epsilon1e-6, RoPE base1e7 e sections `[64,0,0,0]`. Isso significa RoPE FULL128 no draft; não copiar o partial64 do target. Outros schemas/quantizações exigem implementação e validação próprias.

## Explicit Q4 target KV (experimental)

The megakernel now accepts `--kv q4` for the **target KV cache**. GGUF
weights keep their original mixed quantization. The native DFlash sliding
cache remains FP16. Q8 is still the default; `auto` chooses FP16 or Q8.

```sh
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 --kv q4 --ctx 65536 --prefill-chunk 128
```

This uses symmetric signed INT4 with one FP16 scale per 32 values and
dequantizes tiles into the existing FP16 attention kernels. It is a custom
cache layout, not GGML Q4_0. Target cache memory falls from 36,992 to 19,584
bytes per context position (17 slots): a 1.0625 GiB saving at 65,536 tokens.
The preflight still accounts for draft, workspaces and CUDA/library reserve.
Total context includes the prompt and generated answer. Actual fit depends
on free VRAM; start with 60,000 if other applications consume GPU memory.

Q4 adds quantization error and changes target logits. Speculative verification
uses that Q4 target distribution; it does not recover the Q8 distribution.
Validation and measured limits: [implementation journal](bench/dflash-port/26-Q4-TARGET-KV.md).

## VRAM e estado

| Componente | Orçamento/forma |
|---|---|
| Target sem pesos MTP | aproximadamente 11 GiB de pesos; economiza ~0,343 GiB |
| KV target Q8 | 17 slots × 4 heads × contexto × 544 bytes: ~144,5 MiB em 4K; ~1,129 GiB em 32K |
| Estado DeltaNet | ~144 MiB + checkpoint ~144 MiB |
| Conv ring target | 30 MiB + checkpoint 30 MiB |
| Features | 5 × 5120 × 4 bytes × prefill cap: 12,5–200 MiB |
| Draft total inicial | ~1,129 GiB incluindo pesos, caches e buffers pequenos |
| KV draft FP16 | ~40,16 MiB; checkpoint acrescenta ~40,16 MiB |
| Encoder staging | maior matriz FP16 250 MiB + input 25 MiB, alocados sob demanda |
| CUDA/library/graphs | margem adicional; medir uso real, não somente cudaMalloc |

Target reserva 1792 MiB externos antes de escolher o prefill cap automático. O KV draft é um ring de 2056 posições, independentemente do contexto target; ruído speculative fica separado. O ring preserva histórico para truncate de até oito posições e o checkpoint clona K/V. Rejeição só injeta anchor/rows aceitas. Contexto alocado 32K passou HTTP com prompt real 2791; isso não é uma avaliação preenchendo 32K.

## Métricas e limites

Resultados brutos, comandos, clocks amostrados e Markdown intermediários estão em `bench/dflash-port/`. Histórico anterior permanece nos diretórios `full-native-baseline`, `hybrid`, `native-f32-depth7`, `native-q8-depth7` e `native-q8-depth4`.

Benchmarks de decode excluem load, prefill, primeiro token e detokenização; 128 tokens, três repetições, mesmas fixtures short/long/sampling. A continuação longa difere entre engines, limitando comparação de throughput. Reportar median/mean e amostras; não chamar ganho estimado de medido.

`verify_with_injection_seconds` inclui injeção; não somar as duas medidas. `injection_prefill_seconds` é tempo host, incluindo espera por features do target no stream default. `native_injection_*_gpu_seconds` usa eventos depois dessa espera; não atribuir o tempo host completo ao encoder. Nsight confirmou encoder dequant/cuBLAS em dezenas de ms por prefill, não ~1,2s GPU.

Os testes de qualidade são pequenos. Não há garantia de identidade integral ao llama.cpp, perplexidade igual ou acurácia ampla preservada. Comparação teacher-forcing dos cinco taps tem diferenças já no target; em uma posição intermediária longa, cosine ~0,9515 na camada 48. A suíte numérica e os oráculos validam componentes e casos específicos. Manter os baselines para comparação.

## Reproduzir validação

Build dos oráculos: `make -C open-jet/megakernel ARCH=89 external_verify_test dflash_math_test target_features_test`. Dentro desse diretório:

- `external_verify_test TARGET short.ids long.ids`: rejeição em todas as posições para depths0..7 (36 casos), continuação escalar, bytes compartilhados e layout de features.
- `dflash_math_test DRAFT`: pesos CPU GGML, F32 e Q8/DP4A versus produto double, staging FP16, ambas as convs, norm/RoPE/cache e atenção sliding 2050+8.
- CLI `--oracle`: comparar candidates e scores alinhados por token com draft GGML FP16, não somente coincidência de propostas.
- `target_features_test mk|llama TARGET ids output.f32`: teacher-force os mesmos IDs e extrair cinco taps em três posições.
- `bench/dflash_port_bench.py`, `dflash_port_quality.py`, `tests/gpu_server_smoke.py --draft-model DRAFT --draft 7`: performance, saída, sampling, HTTP e SWA com o modelo real.
- Memcheck com `--error-exitcode 99`; comandos exatos e escopo dos filtros em `native-final-validation.json`. Filtros/launch counts não equivalem a sanitização de todos os contextos.

Ver [registro completo](DFLASH_PORT_LOG.md) e [perfil Nsight](bench/dflash-port/10-NSIGHT-E-LEITURA-PACKED.md). Checkout llama.cpp permanece intacto em `aa94f2086147cd4bfbcff818ce998dffce5fcb39`.

## Histórico da primeira entrega nativa

Native7 final: 80.08/53.48/79.92 tokens/s (curto/longo/sampling). Pico amostrado CLI 15089MiB; servidor32K após requests15463MiB. Suite10/10 tarefas; identidade texto17/20 llama,16/20 MK anterior. [Resultados completos](bench/dflash-port/11-ENTREGA-NATIVA-RESULTADOS.md), [gráfico](bench/dflash-port/native-report/dflash-native.png).

## Atualização IQ/MMQ — 2026-10-11

Build selecionado: fusão gate/up em 57/64 camadas, inlining/duas linhas IQ para T>5, MMQ tipado Q4_K/Q6_K com quatro linhas por warp e verificação limitada à saída restante. Defaults `MK_FFN_IQ_FUSED=1`, `MK_IQ_INLINE=1`, `MK_IQ_PAIRED=1`, `MK_DFLASH_MMQ_TYPED=1`, `MK_DFLASH_MMQ_ROWS=4`, `MK_DFLASH_MMQ_HEAD_ROWS=4`, `MK_DFLASH_BUDGET=1`. Protótipos IMMA e down direto em global ficam desativados. Sem novos pesos expandidos ou requantização.

Seis amostras finais por caso: **93,21/62,28/93,36 tokens/s** em curto/longo/sampling; ganhos sobre o snapshot anterior de **15,18%/15,40%/15,88%**. Baseline llama repetido: 91,84/62,50 curto/longo. MK supera nominalmente o curto por 1,49%; longo −0,34%, com faixas observadas sobrepostas e continuação diferente a partir do índice 41. Não foi demonstrada vantagem geral.

Qualidade: 10/10 tarefas; 17/20 gerações com IDs iguais ao native anterior e 16/20 textos iguais ao llama sem especulação. Não estabelece acurácia ampla ou perplexidade. Passaram 36 casos de rejeição/replay, regressão MTP4, 13 checks HTTP, 34 testes CPU e memchecks filtrados do wide/MMQ com zero erros. Pico CLI 4K: 15122 MiB; HTTP 32K alocado: 15496 MiB, prompt real 2791, sem preenchimento integral 32K. Corrigido o checkpoint do draft no tail de uma saída para preservar rewind do segundo turno.

[Relatório detalhado e próximos passos](bench/dflash-port/22-RESULTADO-IQ-E-DRAFT-NATIVO.md), [gráfico](bench/dflash-port/optimization-v2/report/iq-optimization.png), [dados/hashes](bench/dflash-port/optimization-v2/final-summary.json). Journals13–21 registram resultados intermediários, inclusive experimentos que perderam desempenho. O draft permanece em CUDA Graphs; um único kernel persistente continua pendente.
