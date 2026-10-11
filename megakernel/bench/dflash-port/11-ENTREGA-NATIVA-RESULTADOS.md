# 11 — Entrega final do DFlash2 nativo no backend MK

2026-10-10. RTX4060Ti16GB sm89. Nenhuma alteração de clocks; baselines e etapas anteriores preservados.

## Comportamento entregue

Draft completo CUDA com CUDA Graphs, sem execução de grafo GGML no caminho nativo. Target cooperativo MK com sete propostas, CUDA feature injection, pesos compartilhados, KV sliding FP16 e checkpoint/rewind/truncate/reset. C API/HTTP e OpenJet integrados. F32 e GGML continuam disponíveis como referências explícitas. O draft não é um único kernel persistente; essa fusão permanece fora da implementação entregue.

## Desempenho medido

| Configuração | Curto | Longo | Sampling |
|---|---:|---:|---:|
| Native F32, sete | 66,22 | 44,41 | 65,81 |
| Native Q8 antes do packed, sete | 74,53 | 49,76 | 74,17 |
| Native final Q8/packed, sete | 80.08 | 53.48 | 79.92 |
| Native Q8, quatro | 58,81 | 48,68 | 56,50 |
| MK MTP4 baseline histórico | 60,67 | 50,03 | — |
| MK+GGML draft4 baseline histórico | 67,05 | 55,65 | 64,41 |
| llama.cpp DFlash2/7 baseline histórico | 91,35 | 62,29 | — |

128 tokens, três repetições, contexto4096 Q8, load/prefill/primeiro token fora do decode. Baselines históricos não foram todos reexecutados nesta etapa. Continuação longa muda entre engines. Native final versus MK MTP histórico: +32.0% curto, +6.9% longo. Versus híbrido: melhora curto; perde cerca de 3.9% longo. llama.cpp continua superior; não há superação do baseline mais rápido.

A/B imediato do MTP: ~57,6 tokens/s tanto no snapshot quanto na entrada especializada, mesmos IDs; variação <0,4%. Isso limita atribuir pequenas diferenças históricas exclusivamente ao código.

Native final F32/Q8 e versões Q8 antes/depois do packed tiveram os mesmos IDs nas fixtures curto/longo/sampling. Novo contador GPU do encoder: ~4,67ms curto e 22,75ms longo. Contador host de injeção inclui espera do target; não somar nem tratar como tempo exclusivo de encoder.

## Validação final

- 34 testes host passaram; build sm89 atual e diff --check passaram.
- Oracle pesos: 1.271.808 valores, erro0. Dots F32: 2548, erro máximo 8,08e-6. DP4A versus CPU GGML Q8/double passou. Staging FP16 e ambas as convs passaram.
- Norm/RoPE/cache full128: erro máximo FP16 5,18e-4; atenção sliding posição2050+8 erro máximo1,26e-9.
- External verifier: depths1/4/7, todas as rejeições, replay igual a 64 tokens escalares, bytes de embedding/head, 243 callbacks e prefill longo.
- Oracle GGML final curto: 11 blocos, mínimo110/112 candidatos; NRMSE máximo0,658%; propostas11/11 iguais. SWA com prompt2730: 16 blocos, mínimo109/112; NRMSE máximo0,410%; propostas16/16 iguais.
- Memcheck: 0 erros no oracle completo; 0 no decoder com filtros/120 launches; 0 no target com filtro/4 launches. Escopos exatos e comandos em native-final-validation.json; não generalizar a contextos/GPUs não testados.
- Suíte de saída: 10/10 tarefas corretas, 17/20 textos iguais ao llama sem especulação, 16/20 iguais ao MK MTP anterior. Não prova de acurácia ampla ou perplexidade igual.
- HTTP final: 7 checks, greedy, seed, streaming, erros de sampler, segundo turno e prompt2791 > janela2048 em contexto ALOCADO32768. Memória após requests 15463, 16380 MiB; não foi preenchido todo32K. Launcher real também passou.
- Pico NVML amostrado CLI4K: 15089/16380MiB; CLI e servidor terminados e GPU liberada.
- Cinco feature taps comparados sob teacher-forcing ao llama: diferenças numéricas pré-existentes no target permanecem. Cosine mínimo ~0,9515 em uma posição intermediária/camada48; equiv. numérica integral continua não estabelecida.

## Arquivos e execução

`DFLASH_NATIVE.md` documenta funções, formatos, VRAM, comandos e limitações. `native-report/dflash-native.png/.svg/.pdf` e `chart-data.json` mostram desempenho e evidência de qualidade. `native-profile.nsys-rep/.sqlite`, CSVs e native-profile-summary.json registram o perfil.

Na raiz:

```sh
make -C open-jet/megakernel ARCH=89 all qwen38_dflash
python3 open-jet/megakernel/serve_ada.py --engine megakernel-dflash2 --ctx 4096
```

CUDA Graphs com native7 é o default desse engine; MTP regular e launcher default llama permanecem opções. OpenJet: OPENJET_MK_DFLASH_MODEL e OPENJET_MK_DRAFT.

## Próximos limites técnicos

O target wide ocupa ~77–79ms/ciclo e domina o tempo; draft final ~15–16ms. CUDA Graphs já amortizam launches. Prioridades futuras são MMQ/IMMA/tiling do target e atenção wide com menos spills/sincronizações; medir antes de fundir o draft e reduzir sua occupancy. Não implementar heurística depth4/7 sem evidência: quatro perdeu nos três casos. Encoder FP16 em dezenas de ms é prioridade menor para decode.

Compatibilidade demonstrada apenas para os dois GGUF locais e GPU sm89. Full32K preenchido, outras GPUs, schemas, corpus grande/perplexidade e identidade universal das saídas exigem avaliação adicional. Nenhum desses pontos foi declarado concluído.
