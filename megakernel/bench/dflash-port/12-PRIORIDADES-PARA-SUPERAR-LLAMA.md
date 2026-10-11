# 12 — O que falta para superar o baseline llama.cpp/DFlash2

2026-10-10. Análise do código e duas novas medições na RTX 4060 Ti 16 GB. Clocks padrão preservados. Nenhuma mudança de implementação nesta etapa; somente os registros desta análise foram adicionados.

## Distância até o melhor resultado registrado

| Caso | Native final, média de 3 runs | llama.cpp/DFlash2/7 histórico | Aumento necessário de tokens/s | Redução necessária de tempo | Economia por ciclo atual |
|---|---:|---:|---:|---:|---:|
| Curto | 80,079 | 91,347 | 14,07% | 12,33% | 11,51 ms |
| Longo | 53,476 | 62,294 | 16,49% | 14,15% | 13,45 ms |

Fontes: `native-final/results.json` e `../optimization-implementation/dflash2-q4/results.json`. Cálculo: 127 tokens contabilizados no decode de uma saída de 128 tokens, primeiro token excluído; comparar `decode_seconds` com `127/tps_baseline` e dividir a economia por 17/25 ciclos. Manter essa convenção na próxima comparação.

O verificador com injeção representa 84,0%/83,0% do decode; draft custa 14,88/16,12 ms por ciclo e verificador com injeção 78,40/78,87 ms. Mesmo eliminar hipoteticamente TODO o custo do draft levaria apenas a 95,28/64,41 tokens/s, mantendo os mesmos ciclos e aceitação. Isso é um teto contrafactual calculado, não uma otimização executável ou um resultado medido.

Baseline llama é histórico e a continuação longa diverge entre engines. Reexecutar os dois na mesma sessão e comparar também teacher-forcing antes de atribuir pequenas diferenças ao kernel.

## Nova medição das fases internas

Reutilizado o executável atual, comandos de `native-final`, `--reps 1` e `MK_PROFILE=1`. Logs, comandos, resultados e comparação de IDs estão em `target-phase-analysis/`. Ambos terminaram com código 0; os 128 IDs de cada caso coincidiram com native-final. GPU voltou a 41 MiB após os testes.

| Fase do target | Curto, média por launch | Longo, média por launch |
|---|---:|---:|
| FFN gate/up | 29,289 ms | 29,398 ms |
| FFN down | 19,586 ms | 19,672 ms |
| Projeções DeltaNet | 11,865 ms | 11,879 ms |
| Projeção de saída DeltaNet | 4,784 ms | 4,800 ms |
| Projeções da atenção | 3,978 ms | 4,029 ms |
| LM head do target | 2,667 ms | 2,667 ms |
| Recorrência/conv/norm DeltaNet | 2,400 ms | 2,319 ms |
| Projeção de saída da atenção | 1,585 ms | 1,617 ms |
| Atenção core | 0,393 ms | 0,868 ms |
| Combinação da atenção | 0,088 ms | 0,130 ms |
| Total instrumentado | 76,648 ms | 77,392 ms |

Médias incluem launches de prime e decode, medindo intervalos entre barreiras; não são custo isolado de operações matemáticas. As somas de mínimos por fase de T=8 são 75,147/76,065 ms, mas combinam mínimos de launches diferentes: não tratá-las como uma execução real ou novo baseline de throughput.

FFN soma aproximadamente 49 ms e 63–64% do target. FFN + projeções DeltaNet somam aproximadamente 61 ms e 79% do target. Atenção core + combinação ficaram abaixo de 1 ms nestes prompts. A ordem de otimização deve refletir isso; atenção poderá mudar de prioridade em contextos preenchidos muito maiores, que ainda não foram medidos.

## Ordem proposta de implementação

1. **Profiling de contadores do FFN T=8.** Medir DRAM/L2, instruções de dequantização/DP4A, cargas de local memory, stalls e distribuição de trabalho entre SMs. Nsight Systems e timers já localizam o tempo, mas não distinguem esses limitadores. Não assumir que spills ou bandwidth são a causa dominante sem contadores.
2. **Especialização FFN T=8 no target.** Alterar `ffn_gu`, `rows_native`/`mrows` e `resid32<T,17>` em `qwen38_mk.cu`. Gate/up usa RG=1 para T>=3, duas chamadas de multiplicação por linha e acumulações locais; FFN down T=8 faz três slices de ativações (256+256+32 unidades). Experimentar tiles que processem oito tokens com menos chamadas/cópias de acumuladores e menos restaging, e ajustar a distribuição de grupos de saída aos 34 SMs. Medir cada variante separadamente.
3. **Comparar DP4A especializado com IMMA do llama.cpp.** Reutilizar loaders/técnicas de `ggml/src/ggml-cuda/mmq.cuh` e `mma.cuh` para IQ*/Q*_K, adaptando layout de ativações e epílogos de residual/norm/quantização. Não é substituição direta: o MK tem grid cooperativo fixo, layout próprio, quantização de ativações própria e restrição de shared memory. Um microbenchmark de matrizes reais deve escolher o caminho vencedor; Tensor Cores não garantem ganho em batch oito. Manter pesos GGUF originais e dequantizar apenas tiles transitórios.
4. **Projeções DeltaNet.** Aplicar o multiplicador que ganhar no FFN a `dn_proj` e projeções de saída, preservando ordem de conv, estados, captura de features e replay após rejeição. Não alterar a regra recorrente para justificar um ganho de velocidade.
5. **LM head do draft.** `dn::mmq<7>` sobre 248.320 linhas ocupa cerca de 4,5 ms por ciclo no perfil Nsight; `top_partial` + `top_finish` acrescentam cerca de 0,6 ms. Comparar IMMA/DP4A com melhor tiling e fusão da seleção top16 por tile. O top16 por tile precisa preservar a seleção global, empates e scores consumidos pelo selector. Manter vocabulário completo. Em `dflash_native.cu`, modificar `mmq`, `multiply`, `top_partial` e `top_finish` conforme o microbenchmark vencedor.
6. **Fusão adicional somente após medir.** CUDA Graphs já estão ativos. Um único draft persistente não elimina o trabalho de multiplicação e pode piorar utilização por limitar o grid e shared memory. FFN e head oferecem ganhos mensuráveis antes dessa fusão.

Uma meta de engenharia possível de testar é FFN de ~49 para ~39 ms (-10 ms), projeções DeltaNet de ~12 para ~10 ms (-2 ms), e draft de ~15–16 para ~13–14 ms (-2 ms). Somadas, essas metas economizariam ~14 ms por ciclo, aproximadamente o necessário para alcançar o baseline histórico em ambos os casos, se os ciclos e a aceitação permanecerem iguais. São metas orçamentárias, não previsão ou promessa de ganho; verificar o throughput integral após cada mudança.

## Restrições e qualidade

- O target reserva ~90,4 KiB de shared memory e lança 512 threads por CTA, uma CTA por SM. Isso permite no máximo 16 dos 48 warps residentes suportados pelo sm_89 com esta configuração: teto teórico de 33,3%, não medição de warps ativos. A capacidade Ada é 100 KiB por SM, com 99 KiB endereçáveis por bloco. Diminuir threads isoladamente não cria uma segunda CTA enquanto shared memory continuar desse tamanho; especializar recursos por fase ou separar kernels exige comparação completa com o custo de launches/sincronização.
- Perfil de chegada às barreiras mostra desequilíbrio em FFN down (curto: primeiro ~249 us, mediana ~302 us, último ~305 us). Isso não é 56 us de custo intrínseco de barreira: parte dos blocos termina trabalho antes dos outros. Avaliar distribuição de trabalho e granularidade antes de remover sincronizações que garantem dependências.
- Pico CLI4K registrado: 15.089/16.380 MiB. Servidor com contexto alocado32K: 15.463 MiB. O head inteiro FP16 acrescentaria ~2,37 GiB, excedendo essa margem. Evitar duplicar pesos ou desquantizar matrizes grandes permanentemente.
- Aceitação curta 111/119 = 93,28%; longa 106/175 = 60,57%. A queda aumenta os ciclos de 17 para 25 e explica a maior parte da perda de geração no caso longo, enquanto o tempo do target por ciclo muda pouco. Como as continuações diferem entre engines, isso não isola o efeito do backend sobre a aceitação.
- Já existe diferença de features do target sob teacher-forcing, com cosine mínimo ~0,9515 numa posição intermediária/camada48. Isolar prefill, decode escalar e verificação em lote e comparar layerwise com llama.cpp. Melhor alinhamento pode melhorar aceitação; o ganho ainda não está demonstrado.
- Oráculos e 10/10 tarefas passaram, mas 17/20 textos coincidem com llama sem especulação. Falta avaliação em corpus maior e de perplexidade/logits para concluir preservação ampla de qualidade. Manter comparações teacher-forcing, rejeição em todas as posições, replay e CPU GGML em cada alteração matemática.
- Depth4 perdeu para depth7 nos três casos testados. Heurística adaptativa e blocos maiores exigem novas evidências; o draft local tem block8 e sete propostas como configuração natural.

## Critérios para a próxima entrega

Reexecutar native e llama.cpp/DFlash2/7 com os mesmos prompts, parâmetros, limites de contexto e clocks padrão, pelo menos três repetições intercaladas. Avaliar também sequências fixas para separar tempo de execução e diferenças de saída/aceitação. Considerar sucesso superar ambos os casos históricos após confirmar o baseline atual, com margem de pelo menos 5% sobre a variabilidade medida, sem regressão nos oráculos e na qualidade ampliada, e sem exceder a VRAM. Adicionar 4K/8K/16K/32K efetivamente preenchidos conforme couber, separando decode, prefill e latência do primeiro token. Não confundir contexto alocado32K com contexto efetivamente preenchido.

Os arquivos necessários aos primeiros passos já estão disponíveis localmente. O próximo dado necessário é o perfil de contadores do kernel, não novos pesos. Um corpus mais representativo será necessário para afirmar qualidade ampla.

## Referências consultadas

- [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/): limites de occupancy, registradores e shared memory.
- [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/): contadores, memória, stalls e roofline.
- [llama.cpp MMQ](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh), confrontado com o checkout local `aa94f2086147cd4bfbcff818ce998dffce5fcb39`: loaders e caminhos MMA para os formatos quantizados.
