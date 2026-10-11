# Resultado da implementação — RTX 4060 Ti 16 GB

Implementado e medido em 2026-10-10, CUDA 13.4, `sm_89`, clocks padrão.
O megakernel ganhou aproximadamente **25% no decode** e **29,5% no throughput
de prefill longo**. O caminho mais rápido medido foi **DFlash2 Q4 no llama.cpp:
91,35 tok/s greedy curto e 92,56 tok/s sampling**. O modelo GSQ-RCO foi mantido
sem requantização ou alteração de pesos. O checkout llama.cpp não foi alterado.

![Ganhos medidos e validação local](bench/optimization-implementation/report/ganhos_implementados.png)

## Como executar

Partindo da raiz deste workspace, com Python, numpy e jinja2 disponíveis:

```sh
# Configuração mais rápida medida: DFlash2 Q4, draft 7, ctx 4096/Q8.
python open-jet/megakernel/serve_ada.py

# Alternativas já compiladas e testadas:
python open-jet/megakernel/serve_ada.py --engine megakernel
python open-jet/megakernel/serve_ada.py --engine mtp

# Verifica arquivos/compatibilidade e imprime o comando sem carregar a GPU:
python open-jet/megakernel/serve_ada.py --check

# DFlash2 também passou no teste de alocação/geração com contexto 32K:
python open-jet/megakernel/serve_ada.py --ctx 32768
```

O launcher usa `127.0.0.1:8080`; a API de chat é `/v1/chat/completions`.
`--port`, `--model`, `--draft-model`, `--ctx` e `--draft` permitem overrides.
`--fit off`, offload completo e uma sequência tornam a configuração explícita;
não há ajuste silencioso do contexto para fazê-la caber. Execute um servidor
por vez nesta GPU. O template de chat e o modo de raciocínio do modelo são
preservados; os benchmarks abaixo desabilitaram thinking explicitamente.

**DFlash2 funciona através do llama.cpp.** O launcher valida os GGUF e inicia
seu `llama-server`; não executa DFlash2 dentro de `mk_cycle`. O backend `auto`
do aplicativo open-jet conserva sua seleção existente de megakernel/fallback.
A nova configuração DFlash2 é escolhida pelo launcher acima, não automaticamente
pelo seletor do aplicativo. Não foi acrescentado um segundo engine de draft
ao kernel persistente.

## O que mudou nesta continuação

| Arquivo/função | Mudança e finalidade |
|---|---|
| `native_quant.cuh::quant16<TYPE>` | Decodifica grupos de 16 valores IQ3_XXS, IQ3_S e IQ4_XS; compartilha escala e índices. Respeita o alinhamento de dois bytes dos blocos IQ3. Os demais dtypes conservam `quant8`. |
| `qwen38_mk.cu::rows_native` | Usa quatro DP4A antes da escala de cada grupo IQ16 e compartilha os pesos entre os tokens verificados. Reduz trabalho de dequantização/conversão no FFN e nas projeções dominantes. |
| `qwen38_mk.cu::Impl` | Chunk automático entre 2048/1024/512/256/128 conforme VRAM livre e preflight conservador. Preserva a preferência FP16 do KV auto e overrides explícitos. |
| `prefill.cuh::pk_native_weights16` | Reutiliza IQ16 na materialização FP16 dos pesos para cuBLAS. |
| `qwen38_mk.cu::gemm`, `gemm_glu`, `pf_init` | cuBLAS grava diretamente no destino e soma o residual com beta; remove cópia de saída e buffer FP32 temporário. Gate/up nativos reutilizam a conversão da entrada quando ambos seguem caminhos compatíveis. |
| `native_prefill.cuh::pf::native_mmh` | Protótipo WMMA 64×64×64, FP16/FP32, staging de 16 KiB, com máscara de tails. Somente `MK_NATIVE_GEMM=tile`; rejeitado como padrão por desempenho inferior no microbenchmark. |
| `main.cu`, `qwen38_mk.cu::pf_run` | CLI agora usa draft 4. `--score-probe` e `--score-probe-file` expõem top-2, margem, NLL e logits completos de uma posição de teacher forcing no prefill batched. |
| `server/mk_server.py` | `--prefill-chunk auto` como padrão. Mantém draft 4, KV selecionável e API existente. |
| `src/llama_server.py::_mtp_spec_args` | Fallback llama.cpp usa MTP 4 nas GPUs Ada elegíveis de 16 GB; conserva MTP 2 nas demais. Override `OPENJET_MTP_DRAFT_MAX=1..8`. |
| `serve_ada.py::check_draft`, `main` | Configuração DFlash2/MTP/megakernel reproduzível; compara tokenizer completo, dimensões, arquiteturas, camadas e selector antes de lançar DFlash2. |
| `tests/native_quant.cu`, `tests/native_gemm.cu`, `tests/test_megakernel_backend.py` | Oráculos independentes de pesos/dot products/GEMM e regressões da escolha MTP. |
| `bench/compare_optimization.py`, `llama_spec_bench.py`, `mk_quality.py`, `final_cuda_checks.py`, `launcher_smoke.py`, `plot_optimization_results.py` | A/B intercalado, benchmark especulativo, tarefas/IDs, memória/probe/sanitizer, API real e gráfico derivado dos registros. |

Esta tabela descreve **esta continuação**, não atribui a ela toda a adaptação
GSQ/Ada anterior, cujas alterações e validações estão em [ADA_GSQ.md](ADA_GSQ.md).

## Compatibilidade DFlash2 realmente conferida

Draft baixado: `models-dflash2-qwen38/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`,
1.143.006.816 bytes, arquitetura `dflash`, 81 tensores, cinco camadas,
hidden 5120, FFN 17408, 32 cabeças Q/8 KV, head dimension 128.
Bloco treinado 8; usamos sete propostas. Selector rank 256/top_k 16,
convoluções de dois taps/16 grupos, extração de camadas GGUF `[6,20,34,48,62]`.
Origem e model card foram preservados em `models-dflash2-qwen38/README.md`.

O target real tem arquitetura `qwen35`, 866 tensores e quantização mista;
seus nomes/dimensões/dtypes foram validados pela adaptação anterior. Comparou-se
agora também **toda** a lista de tokens e merges, tipo de tokenizer e IDs BOS/EOS
entre target e draft. Não se inferiu compatibilidade pelo nome GSQ-RCO.
Os logs confirmam o selector DFlash2 e cinco camadas de extração. Os tensores
MTP adicionais do target são ignorados nesse caminho do llama.cpp.

## Throughput observado

Contexto 4096, KV Q8, uma sequência, 128 tokens de saída, prompts idênticos por
IDs. Decode exclui load/prefill. Clocks/voltage offsets zero; boost dinâmico
de fábrica, fan curve e PowerMizer preservados.

| Caso | MK original, MTP 4 | MK final, MTP 4 | llama.cpp MTP 4 | llama.cpp DFlash2 7 |
|---|---:|---:|---:|---:|
| Greedy, entrada 24 tokens | 48,32 | **60,66** | 62,96 | **91,35** |
| Greedy, entrada 1365 tokens¹ | 40,07 | **50,02** | 45,51 | **62,29** |
| Sampling temp=1, top_k=20, top_p=.95, seed=123 | 47,46 | **59,35** | 61,93 | **92,56** |

Valores em tokens/s. Ganho MK final/original: **+25,54% / +24,81% / +25,05%**.
DFlash2 versus baseline llama.cpp MTP 4: **+45,10% / +36,88% / +49,45%**.
DFlash2 versus MK original: +89,03% / +55,45% / +95,01%.
No curto o MK final ainda fica aproximadamente 3,6% abaixo de llama.cpp MTP 4.

¹ No longo, a continuação MK difere de llama.cpp MTP 4/sem especulação;
isso afeta a aceitação. Não é uma comparação com continuação fixa.

Três repetições por configuração, exceto referências llama.cpp longo/sampling
com duas. O A/B MK foi intercalado, com chunk 512 no original e auto no final;
todos os IDs dos dois MK coincidiram, inclusive com sampling. O primeiro A/B
IQ16 isolado usou chunk 512 nos dois e também mostrou aproximadamente +25% com
MTP 4 e +10% sem MTP. A aceitação curta permaneceu 101/114 nesse ensaio; o
ganho veio de menor custo por ciclo, não de maior aceitação.

Prefill de 1365 tokens: **2,575 → 1,988 s**, 530,10 → 686,50 tokens/s;
−22,78% de latência/+29,50% de throughput. Prefill curto praticamente igual.
Não se atribui todo esse ganho a IQ16: inclui chunk maior e redução de cópias.

As execuções llama.cpp/DFlash2 ocorreram separadamente do A/B MK; não há
intervalo de confiança ou ensaio prolongado. As barras de erro do gráfico são
mínimo/máximo observado. O desempenho depende do texto: no conjunto ampliado,
DFlash2 apresentou cerca de 26–97 tok/s. Os 91–93 tok/s descrevem os prompts
de código medidos, não uma promessa para qualquer tarefa.

## Memória real e política de contexto

| Engine | Contexto alocado | Chunk | Uso observado | Fonte |
|---|---:|---:|---:|---|
| MK final/Q8 | 4096 | 2048 | 13,506 GiB | CUDA, 12,788 GiB de alocações do engine |
| MK final/Q8 | 32768 | 2048 | 14,502 GiB | CUDA, 1,153 GiB livres |
| MK final/Q8 | 65536 | 1024 | 15,289 GiB | CUDA, 0,366 GiB livres |
| DFlash2/Q8 | 4096 | llama.cpp | 14103–14395 MiB = 13,77–14,06 GiB | snapshots NVML |
| DFlash2/Q8, fit off | 32768 | llama.cpp | até 15491 MiB = 15,13 GiB | snapshots NVML |

CUDA e NVML usam contabilidade distinta; os valores não são uma medida
direta da diferença entre engines. Snapshots não são pico instrumentado.
32K DFlash2 passou com entradas de 24/1365 tokens e mediu 91,72/62,64 tok/s
em uma repetição por caso. Nenhum desses testes preencheu efetivamente 32K
ou 64K tokens de entrada. DFlash2 64K não foi validado.
MK 262144 foi rejeitado no preflight por memória insuficiente.

Os componentes fixos MK anteriores continuam: pesos/repack aproximadamente
11,343 GiB, cópia de output draft aproximadamente 0,099 GiB, DeltaNet F32
144 MiB, conv ring 15 MiB, replay 24,28 MiB e checkpoint 159,31 MiB.
KV Q8 custa 36992 bytes/token, FP16 69632; RoPE 256 bytes/token.
O fallback mantém uma matriz FP16 de 170 MiB e entrada de 68 MiB no chunk 2048;
removeu o buffer FP32 de saída, que consumiria 136 MiB nessa capacidade.
Os demais buffers de prefill crescem com o chunk. O preflight mantém a margem
conservadora anterior, inclusive o custo estimado do temporário removido.

## O que a validação demonstra

- **8.945.664 valores GGUF reais** passaram no oráculo CPU llama.cpp, incluindo
  os novos helpers IQ16 e seus dot products. Tolerância do dot: 2e-6 × max(1,L1).
- GEMM tiled passou contra CPU-dequantização → FP16 → cuBLAS em shapes FFN
  reais, T=1/127/128/512, relative RMS no máximo aproximadamente 2,8e-5.
- **34 testes Python** de backend/GGUF/build passaram.
- Memcheck: **quatro lançamentos `mk_cycle`, zero erros**, geração de 16 tokens.
  Não instrumentou exaustivamente todos os kernels/pedidos.
- Launcher DFlash2 e MK: health/model list, resposta numérica conhecida,
  repetição greedy, streaming equivalente, seed e segunda rodada passaram.
- A/B final: mesmas sequências entre MK original e final nos casos medidos.
- DFlash2: os doze ensaios greedy iniciais reproduziram os IDs canônicos MTP 4
  nos dois prompts; sampling repetiu a mesma seed nas três execuções.

Conjunto ampliado: dez problemas numéricos com resposta conhecida e dez
prompts de código/prosa em inglês/português. Mesmos IDs de entrada, greedy,
até 128 tokens, uma execução por prompt:

| Engine | Problemas corretos | Gerações idênticas ao llama.cpp sem especulação |
|---|---:|---:|
| Referência sem especulação | 10/10 | referência |
| DFlash2 7 | **10/10** | **17/20** |
| MK final | **10/10** | **16/20** |

**Isso não comprova preservação universal da qualidade.** Não foram medidas
perplexidade, equivalência estatística de sampling nem acurácia ampla.
Identidade de IDs e acurácia em tarefas são critérios diferentes.

A divergência teacher forcing longa do MK, conferida com chunk 512, continua
**127/128**, primeira
posição diferente 41. O novo probe registrou top1 8029/logit 22,7672,
top2/esperado 2144/logit 22,4890, margem 0,2782 e NLL do esperado 1,4265.
O caminho é prefill batched FP16; isso não isola um bug específico do MTP ou
prova que a diferença seja apenas ruído insignificante. A causa não foi
isolada nem corrigida. `final-checks/probe-logits.f32` contém os 248320 logits.

## Candidatos rejeitados e trabalho pendente

O GEMM tiled ficou **1,7–4,4 vezes mais lento que cuBLAS** no microbenchmark,
sem contar a materialização global de pesos do fallback. Isso não é um ensaio
completo de prefill; não há ganho end-to-end demonstrado para ele. O protótipo
permanece opt-in, mas o caminho padrão reutiliza cuBLAS e passou no A/B real.

IQ16 não reduziu a ocupação limitada por recursos: o kernel permanece com
512 threads/CTA, 128 registradores/thread e 92544 bytes de shared dinâmica,
34 CTAs/16 warps por SM. Ptxas registrou **mais** spill bytes (stores/loads
1288/1632 → 1624/1960) e stack 2560 → 2736. Ainda assim, o tempo real caiu.
Não se vende a mudança como redução comprovada de registradores ou spills.

Permanecem pesquisas posteriores: separar/verificar kernels FFN para reduzir
spills, GEMM IQ/INT8 especializado em pequenos batches, profundidade MTP
adaptativa além da reação existente a rejeições, qualidade/PPL ampla,
diagnóstico numérico por camada e contextos efetivamente longos. Portar
DFlash2 para o megakernel exigiria capturar feature taps e implementar suas
cinco camadas, convoluções e selector; **não foi feito**, pois o caminho local
reutilizado já demonstrou melhor throughput. Não há alegação de máximo
teórico atingido nesta GPU.

## Evidência e reprodução

Fontes e binários anteriores foram preservados em
`bench/optimization-implementation/baseline/`, com hashes. O A/B final registra
uma nota sobre o relink do CLI durante o ensaio: comandos sempre usaram `-d 4`
e o objeto CUDA/shared library não mudaram. O ensaio iniciado antes do build
foi descartado e está identificado em `discarded-prebuild-run/`.
A primeira invocação memcheck tinha filtro inválido; o filtro foi corrigido
e o resultado final limpo foi preservado, sem contar a falha como aprovação.

```sh
make -C open-jet/megakernel ARCH=89 LLAMA=../../llama.cpp
MODEL=models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python open-jet/megakernel/bench/compare_optimization.py --model "$MODEL" \
  --output /tmp/mk-final-ab --drafts 4 --chunk auto --reps 3
python open-jet/megakernel/bench/launcher_smoke.py --output /tmp/ada-api
MPLCONFIGDIR=/tmp/qwen38-matplotlib python open-jet/megakernel/bench/plot_optimization_results.py
```

Referência llama.cpp: `aa94f2086147cd4bfbcff818ce998dffce5fcb39`, build 11549.
Scripts e saídas usam este workspace/modelo, não constituem benchmarks portáveis
para qualquer Qwen. O script A/B usa `LD_LIBRARY_PATH` para o binário preservado,
cujo rpath relativo mudou de localização.

[Dados/artefatos](bench/optimization-implementation/README.md),
[JSON do gráfico](bench/optimization-implementation/report/chart_data.json),
[SVG](bench/optimization-implementation/report/ganhos_implementados.svg),
[PDF](bench/optimization-implementation/report/ganhos_implementados.pdf),
[registro contínuo](IMPLEMENTATION_LOG.md).
