# 25 — Resultado competitivo, estado da arte e margem de otimização

2026-10-11. Análise do resultado registrado e pesquisa em fontes primárias. Não foi iniciado novo benchmark/inferência, modificado kernel ou reiniciado o servidor ativo. Clocks permanecem padrão.

## O que está demonstrado

O MK + DFlash2 atual é competitivo com o melhor baseline local medido. A comparação de [22](22-RESULTADO-IQ-E-DRAFT-NATIVO.md) registrou 93,208 versus 91,837 tokens/s no curto (+1,49%) e 62,283 versus 62,496 no prompt de 1365 tokens (−0,34%). O ganho da última otimização sobre o próprio MK foi +15,18%/+15,40%. Os aproximadamente +93% desde o MK adaptado/MTP4 inicial misturam etapas históricas e mudança de estratégia de draft; não são uma vantagem de 93% sobre llama.cpp/DFlash2.

Medições: contexto alocado4096, KV targetQ8, uma sequência, 128 tokens emitidos/127 cronometrados, load/prefill/primeiro token excluídos. Seis amostras de duas sessões por barra; não são intervalos de confiança. O longo diverge entre engines a partir do índice41. As modificações posteriores de parâmetros HTTP/effort não foram uma nova rodada de benchmark CUDA.

Não está demonstrado estado da arte global para a RTX4060Ti: faltam uma suíte representativa, saídas/qualidade comparáveis, outros runtimes elegíveis e avaliação de contextos efetivamente longos. Temos dez tarefas objetivas corretas e oráculos de componentes, não qualidade ampla/perplexidade estabelecida. 32K foi alocado, mas o maior prompt real testado foi2791.

## Pesquisa externa e comparabilidade

DFlash2 já é uma técnica publicada e disponível em vários runtimes. O nosso trabalho portou seu draft/selector real e otimizou o caminho target IQ na Ada; não treinou um novo draft nem estabelece novidade científica do algoritmo. Os ganhos publicados de DFlash2 sobre DFlash/autoregressivo têm baselines diferentes do nosso llama.cpp já com DFlash2. [Publicação dos autores, 2026-08-18](https://inco.ai/blog/dflash2/).

Foi encontrado um registro primário comunitário da mesma placa, RTX4060Ti16GB, com MTP e outro GGUF Q4-XYZ-v2, contexto32K e KVq4_0. Esse resultado não é comparável diretamente ao nosso GSQ-RCO misto/KVQ8/DFlash2/contexto4K. Não usei sua taxa como prova de superioridade ou recorde. [Registro e condições do benchmark](https://github.com/sudoingX/qwen38-mtp/blob/master/README.md).

ExLlamaV3 tem resultados publicados com outros formatos/placas, que demandam uma comparação própria, incluindo memória e qualidade; não houve instalação/conversão neste turno. O exemplo de NInfer SM89 pesquisado usa outro artifact e orçamento que supera16GB. Esses registros não certificam o melhor runtime para os dois GGUF locais. [Releases ExLlamaV3](https://github.com/turboderp-org/exllamav3/releases), [qualificação NInfer SM89](https://github.com/Don-Chad/ninfer-3090/blob/master/RELEASE_NOTES_0.6.0_RTX4090_EARLY1.md).

## Onde está a margem restante

Perfil separado do benchmark, incluindo prime/warmup e caudas: target66,738ms, FFN gate/up23,806ms e down17,051ms. O FFN soma aproximadamente61% do tempo do target. Nas medidas sem instrumentação, verificação com injeção soma cerca de84,7% do decode curto; draft15,3%. Combinar as frações para planejamento sugere que o FFN representa aproximadamente metade do decode. Isso não é uma medição de DRAM/spills ou perfil exclusivo de T8.

O wide atual usa512 threads/CTA,128 registradores/thread,92544B de shared e34 CTAs cooperativas. A Ada dispõe de64K registradores32-bit/SM,100KiB shared/SM e até48 warps/SM. A configuração atual reside com uma CTA/SM; reduzir registradores sozinho não cria uma segunda CTA, nem aumentar occupancy garante maior throughput. [Guia oficial Ada](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

NCU retornou ERR_NVGPUCTRPERM na investigação anterior. Timers e Nsight Systems localizam fases, mas não estabelecem saturação de bandwidth, cache misses ou tráfego dinâmico de spill. Não foram alteradas permissões do driver.

## Prioridades propostas

| Prioridade | Alvo e arquivos | Experimento e limite |
|---|---|---|
| 0 | Logits/features, score/teacher forcing, suíte de qualidade e benchmark HTTP/CLI | Comparar o mesmo trabalho e investigar divergência longa/gerações alteradas. Medir prosa/PT-BR, código, JSON/tools e thinking; saídas512/2048, prompts/contextos realmente preenchidos. É requisito para alegar qualidade e superioridade geral, sem ganho CUDA presumido. |
| 1 | FFN: qwen38_mk.cu::ffn_gu_iq/ffn_gu_iq_up/rows_native_impl, native_quant.cuh, native_iq_mma.cuh | Medir leitores/codebooks/metadados e reuso entre linhas/tokens. Tiling/menos acumuladores vivos e políticas de cache por acesso. Usar GGUF packed e preservar a ordem numérica onde possível; não acrescentar cópias FP16 persistentes. Maior alvo medido, complexidade alta. |
| 2 | qwen38_mk.cu::mk_cycle_wide, mrows, resid/FFN down | Especialização de T8 e caudas, redução de variáveis vivas e sincronizações comprovadamente dispensáveis. Caso a fusão limite GEMMs, testar separar somente fases pesadas e usar graphs. Cada variante precisa medir ciclo inteiro, shared/grid/barreiras/replay e memória; não baixar maxrregcount ou threads cegamente. |
| 3 | dflash_native.cu::small_mm/MMQ e operações norm/quant/GLU | Fundir operações pequenas por bloco antes de estudar um draft persistente completo. Draft é cerca de15–17% do decode; reduzir launches não elimina sua computação. Complexidade/benefício orientados por esse peso. |
| 4 | capi.cpp::mk_step/dflash_main.cpp, verificador e estatísticas por posição | Investigar comprimento de verificação adaptativo em tarefas de baixa aceitação, mantendo o bloco treinado8 do draft. Otimizar tokens úteis por tempo, não a taxa de aceitação isolada. Depth4 já perdeu nos três casos históricos; um gate ingênuo pode aumentar custo por token. |
| 5 | Prefill: qwen38_mk.cu::gemm/gemm_glu, prefill.cuh/native_prefill.cuh | Reutilizar caminhos GGML/CUDA adequados aos IQ, reduzir materialização/leituras de matriz FP16 e cópias. Priorizar TTFT/documentos longos; o protótipo tiled anterior foi mais lento que cuBLAS e não deve virar default. Mudanças na precisão das ativações exigem avaliação própria. |

IMMA8/16, pipelines produtores/consumidores e down via global foram experimentados e perderam; não são ganhos pendentes prontos para habilitar. TMA/cp.async.bulk e WGMMA de Hopper não são caminhos diretos para sm89. O pragma enable_smem_spilling é proibido com shared dinâmica, usada atualmente; não é um switch gratuito para esta configuração. [Experimentos locais rejeitados](15-VARIANTES-REJEITADAS-E-TILING.md), [restrições PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## Ideias recentes de especulação

DScale, publicado em2026-09-29, preserva o draft inteiro e escolhe prefixos de verificação com um predictor, tiles e workspaces de endereço fixo. Seu resultado foi medido em A10040GB, Qwen3-4B/8B e concorrência8–32; não há evidência de transferência direta dos ganhos para nosso caso de uma sequência na4060Ti. Ideia para um piloto, não port pronto. [Artigo](https://arxiv.org/abs/2609.37532).

AdaFlash, revisão2026-09-26, adapta diffusion drafters com distillation on-policy e um head de comprimento. A dependência global do draft influencia a escolha dos prefixos. Aplicar a ideia ao GSQ-RCO pode exigir treinamento/novos pesos do draft e validação, além de kernels. O nome não significa uma otimização específica da arquitetura Ada Lovelace. [Artigo](https://arxiv.org/abs/2607.19223).

## Quanto ganho seria suficiente

[Cálculos reproduzíveis](sota-assessment/scenarios.json): modelo ilustrativo de Amdahl, usando52% do decode no FFN e15% no draft, com aceitação e demais custos constantes. Base93,208tok/s do caso curto:

| Hipótese, ainda não medida | Resultado ilustrativo |
|---|---:|
| Tempo do FFN20% menor |104,0tok/s, +11,6% |
| Tempo do draft50% menor |100,8tok/s, +8,1% |
| Ambos |113,5tok/s, +21,8% |
| Draft com custo zero, limite hipotético desse alvo isolado |109,7tok/s, +17,6% |

São cenários para escolher o trabalho, não previsões, benchmarks ou limites completos do hardware. Tornar o draft monolítico não torna seu custo zero. Melhorar aceitação muda as hipóteses. Uma meta de100+ no curto é útil para experimentos; ganhos maiores ou uma vitória geral não foram estabelecidos.

Para a próxima integração, exigir ganho repetido no ciclo completo e numa suíte representativa, em múltiplas sessões, acima da variação de fábrica, com pesos/GPU/sampling/effort/contexto equivalentes. Oráculos, replay/HTTP, memória, qualidade apropriada e TTFT devem acompanhar o throughput. Reduzir thinking para Low diminui a quantidade de trabalho da resposta, mas não é uma otimização CUDA comparável com o mesmo esforço.
