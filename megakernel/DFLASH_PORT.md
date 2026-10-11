# DFlash2 com target MK — protótipo híbrido

Registro da etapa inicial. Para o código atual, com draft nativo CUDA,
sete propostas, C API e HTTP, consulte [DFLASH_NATIVE.md](DFLASH_NATIVE.md).
Os limites e pendências abaixo descrevem o protótipo preservado; o CLI
atual escolhe `--backend native` por padrão.

Esta etapa executa o target GSQ-RCO no megakernel e o draft real no
graph GGML/CUDA existente. Implementa captura das entradas das camadas
`[6,20,34,48,62]`, compartilhamento do embedding IQ2_S/head Q4_K e
verificação de propostas externas. O draft continua separado do kernel
persistente. A profundidade máxima nesta etapa é **4**, com MAXT=5.

## Build e execução

Requer o build local do llama.cpp compatível com os headers privados
`llama-model.h` e `llama-ext.h`. O checkout usado é
`aa94f2086147cd4bfbcff818ce998dffce5fcb39`; não foi modificado.

Na raiz deste workspace:

```sh
make -C open-jet/megakernel ARCH=89 qwen38_dflash

open-jet/megakernel/qwen38_dflash \
  -m models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --draft-model models-dflash2-qwen38/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --prompt-ids open-jet/megakernel/bench/optimization-analysis/llama-mtp4-short/prompt.ids \
  --backend ggml -c 4096 -d 4 -n 128 --reps 3
```

O CLI recebe IDs já tokenizados; não aplica chat template. Use os IDs
GGUF corretos do target. `--prompt-ids` pode ser repetido para executar
vários prompts com a mesma carga do modelo. Cada prompt limpa target e
draft, aquece com 16 tokens, e então executa as repetições medidas.
`--temp 0.7 --seed 123` habilita sampling do target; o draft permanece
determinístico. Default de prefill: chunk 512; KV Q8.

`IDS:` contém os tokens gerados, `TEXT_HEX:` contém o texto UTF-8 em hex,
e `RESULT` informa tempos, aceitação e índice do prompt. Load, prefill,
primeiro token e conversão de texto ficam fora do throughput de decode.
`verify_with_injection_seconds` já inclui `injection_decode_seconds`;
não some esses dois campos.

## Código e contratos

- `qwen38_mk.cu/.h`: captura, verificador externo e tensores emprestados.
  Apenas anchor + propostas aceitas são emitidos como features. O controle
  de rejeição e replay DeltaNet existente permanece responsável pelo target.
- `dflash_bridge.cpp/.h`: carrega draft, valida vocabulário/taps/dimensões,
  injeta features por chunks e extrai propostas do selector top16. Remove
  todo o ruído especulativo do KV draft antes de cada nova injeção.
- `dflash_main.cpp`: coordenação, reset por prompt, sampling e métricas.
- `tests/external_verify.cpp`: oracle de IDs escalar, rejeições por posição,
  estado após rejeição, bytes compartilhados, layout e prefill por chunks.
- `bench/dflash_port_bench.py` e `dflash_port_quality.py`: benchmarks e
  suíte de saída, com comandos e resultados brutos preservados.

O target possui os buffers de pesos. O bridge/contexto draft deve ser
destruído antes do target. `enable_features` é chamado uma única vez,
antes do primeiro prefill; `reset` mantém a alocação e o callback. O
controlador deve também limpar o draft quando limpa o target. Um callback
que lança exceção após o forward exige reset de ambos antes de continuar.
Checkpoint, truncate, multi-turn e paralelismo do servidor ainda não estão
integrados ao controlador DFlash2.

## Limites e próximos passos

Validado na RTX 4060 Ti em 4K Q8; contextos maiores, outras GPUs, modelos
draft e quantizações do embedding/head exigem nova validação. Features
são F32 e ocupam 50 MiB por chunk 512, sem histórico completo na CPU.
O caminho inicial usa cópia device→host→draft; seus custos são medidos.

Os benchmarks iniciais ainda ficam abaixo do llama.cpp/DFlash2 com quatro
propostas. Não foi alterado o launcher de produção. Sete propostas exigem
T=8 e revisão dos layouts de atenção, reductions e shared memory. Antes
de implementar isso, medir o custo do verificador T=8 e comparar logits e
features por camada com o llama.cpp, incluindo a divergência numérica longa
já conhecida do MK. Fusão adicional do draft deve depender do perfil.

Resultados, qualidade e checkpoints de trabalho:
[registro do porte](DFLASH_PORT_LOG.md) e
[journals intermediários](bench/dflash-port/01-VERIFICADOR-E-FEATURES.md).
