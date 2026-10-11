# RTX 4060 Ti 16 GB e Qwen3.8 GSQ-RCO

**Implementação posterior em 2026-10-10:** IQ16, chunk auto e prefill com
menos cópias elevaram o megakernel para 60,66 tok/s greedy curto, 50,02 longo
e 59,35 sampling, aproximadamente +25% sobre seu original. DFlash2 Q4 foi
baixado, conferido e executado via llama.cpp: 91,35/62,29/92,56 tok/s.
O launcher atual é `python open-jet/megakernel/serve_ada.py`.
[Resultado completo e limitações](IMPLEMENTATION_RESULTS.md),
[diário](IMPLEMENTATION_LOG.md) e
[gráfico atual](bench/optimization-implementation/report/ganhos_implementados.png).
As tabelas históricas abaixo descrevem a adaptação inicial; o relatório novo
contém os A/B, a memória com chunk auto e a avaliação local de vinte prompts.

Esta adaptação mantém os pesos GGUF e o ciclo de decode cooperativo do projeto.
O carregador exige a arquitetura `qwen35`, 64 blocos principais + um bloco MTP,
D=5120, FF=17408, 24 cabeças Q, 4 KV e vocabulário de 248320 tokens. Valida
metadados, nomes, dimensões e formatos de todos os 866 tensores antes de fazer
upload. O nome do arquivo não autoriza o carregamento de um modelo incompatível.
O snapshot local validado é `d562806dbafae37109975e970aae91b43e73b440`.
O GGUF tem 12.120.016.960 bytes (11,288 GiB); o payload dos tensores soma
12.109.021.184 bytes. Embedding é IQ2_S, output é Q4_K e as oito matrizes do
MTP são Q6_K. Apenas 25 dos 65 pares FFN gate/up compartilham o mesmo dtype.

| Dtype | Tensores | Dtype | Tensores |
|---|---:|---|---:|
| F32 | 360 | BF16 | 96 |
| IQ3_S | 144 | IQ3_XXS | 78 |
| IQ4_XS | 96 | IQ2_S | 17 |
| IQ2_XS | 9 | IQ2_XXS | 5 |
| IQ1_M | 1 | Q2_K | 13 |
| Q4_K | 39 | Q6_K | 8 |

## Compilar neste workspace

Partindo da raiz que contém `open-jet/`, `llama.cpp/` e o cache do modelo:

```sh
cmake -S llama.cpp -B llama.cpp/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA_GRAPHS=ON
cmake --build llama.cpp/build --target llama-server llama-bench -j 6
make -C open-jet/megakernel ARCH=89 LLAMA=../../llama.cpp
```

O build reutiliza `ggml-common.h` do checkout local do llama.cpp para os layouts
GGML e as tabelas IQ. `libllama` fornece o tokenizer e o template; cuBLAS fornece
a multiplicação do fallback de prefill. O objeto CUDA deve ser reconstruído ao
mudar arquitetura, headers do llama.cpp ou fonte (`make -B`). `ARCH=89` e
`ARCH=sm_89` são equivalentes.

Exemplo de servidor de texto; não pode coexistir na GPU com outro servidor que
já ocupe os 13–14 GiB deste modelo:

```sh
MODEL=models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python open-jet/megakernel/server/mk_server.py -m "$MODEL" \
  --port 18080 -c 32768 --kv q8 --draft 4 --prefill-chunk auto
```

Para escolher o backend no open-jet: `llama_backend: megakernel` exige o kernel;
`auto` verifica o GGUF local e a GPU e conserva o fallback de inicialização para
llama-server. A elegibilidade de GPU aceita sm_89 com pelo menos 15360 MiB.
O carregador CUDA decide se a memória **livre** suporta o contexto solicitado.

## Alterações CUDA e quantizações

- `native_quant.cuh`: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
  IQ1_M e Q2_K, sem requantização. Decodifica oito inteiros com escala e offset
  para `dp4a`; compartilha esses pesos entre os tokens verificados. Usa cargas
  vetoriais/cache de leitura para as tabelas e consulta IQ4 em registradores.
- `qwen38_mk.cu`: distribui os 48 value heads DeltaNet com stride pelo número
  real de blocos, incluindo GPUs com menos de 48 SMs. Projeções, embedding,
  gate/up, output e MTP despacham pelos tipos reais de cada matriz; gate e up
  podem ter formatos e row strides diferentes. Alpha/beta BF16/F16 são ampliados
  para F32 antes do folding existente da norma.
- IQ3_S, IQ3_XXS e IQ4_XS têm helpers com tipo fixado em compilação, evitando
  despachar novamente o formato a cada fragmento. O caminho Q6_K original é
  reutilizado no MTP, onde todas as oito matrizes usam esse formato.
- A acumulação IQ fica em registradores dentro do helper; referências ao array
  do chamador só são atualizadas ao terminar a redução. Isso evita tráfego
  repetido para a memória local do kernel.
- A grade usa `cooperative_groups::this_grid().sync()`. A atenção inicializa o
  padding das matrizes e os fatores de softmax. Split-K limita os chunks ao
  número de SMs; NT=512 foi preservado porque o layout depende dos 16 warps.
- `prefill.cuh`: preserva os GEMMs especializados existentes. Para os novos
  formatos, dequantiza **uma matriz por vez** para FP16, multiplica com cuBLAS
  usando acumulação F32 e reutiliza os mesmos buffers. Gate/up mistos usam dois
  GEMMs e a mesma função SiLU. O workspace não cresce com o número de camadas.
- O down projection de cinco tokens usa dois slices de K e mantém somas
  parciais em registradores. Projeções residuais abortam ao detectar valores
  não finitos, impedindo que NaNs sejam transformados silenciosamente em IDs.
- `MK_PREFILL_CHUNK` / `--prefill-chunk`: capacidade entre 128 e 2048, múltipla
  de 128; padrão atual `auto` seleciona pela VRAM livre (antes era 512).
  O scoring limita o microbatch de logits à capacidade real
  de `y0`, evitando overflow quando o chunk é reduzido.

Q6_K e Q8_0 conservam o repack existente; seus valores não são alterados.
O dtype global `general.file_type` não substitui a inspeção por tensor.
`mmproj` não é carregado: este backend continua sendo de texto e uma sequência.

## Orçamento de VRAM

Valores para este GGUF, chunk 512 e draft vocab de 32768 + 4096 posições:

| Componente | Memória |
|---|---:|
| Pesos, incluindo conversão alpha/beta e repack Q6 | 11,343 GiB |
| Cópia restrita do output para draft | 0,099 GiB |
| Estado DeltaNet F32 | 144 MiB |
| Conv ring | 15 MiB |
| Replay DeltaNet | 24,28 MiB |
| Checkpoint | 159,31 MiB |
| Buffers de prefill | aproximadamente 161,7 MiB |
| Fallback IQ: pesos + input + output | 170 + 17 + 34 MiB |
| KV F16, 17 slots | 69632 bytes/token |
| KV Q8, incluindo escalas | 36992 bytes/token |
| RoPE | 256 bytes/token |

Com contexto 4096/Q8, o logger mediu **12,297 GiB em alocações do engine** e
aproximadamente **13,020 GiB de uso total visível ao CUDA**, incluindo runtime,
cuBLAS e demais ocupantes. Os valores variam com processos existentes.
Foram medidos **aproximadamente 14,01–14,03 GiB em 32768/Q8**, **15,016 GiB em 32768/F16** e
**aproximadamente 15,14–15,16 GiB em 65536/Q8**. O modo `auto` escolheu F16 em 32768 e Q8 em
65536. Com 65536, restaram aproximadamente 0,50 GiB livres: essa capacidade exige a
GPU praticamente livre. A execução HTTP final marcou 14350/16380 MiB no nvidia-smi. O teste de 65536 carregou o cache completo e gerou
a partir de um prompt curto; não valida qualidade em um prompt de 65536 tokens.
O pedido de 262144/Q8 foi rejeitado antes do upload, com orçamento de 21,75 GiB.

`--kv auto` compara a memória livre com o orçamento antes de carregar os pesos
para decidir F16 ou Q8. Solicitações que não cabem são rejeitadas antes do
upload; não ocorre offload implícito nem redução silenciosa do contexto. O
orçamento inclui reserva de runtime, checkpoint e fallback cuBLAS. Desligar MTP
pula o cálculo de draft e sua cópia do output, mas mantém os pesos do bloco MTP
carregados para conservar a organização do engine.

## Validação reproduzível

```sh
make -C open-jet/megakernel ARCH=89 native_quant_test gguf_parser_test
open-jet/megakernel/native_quant_test "$MODEL"
open-jet/megakernel/gguf_parser_test
python open-jet/megakernel/bench/ada_bench.py -m "$MODEL" \
  --llama-server llama.cpp/build/bin/llama-server \
  --output /tmp/ada-benchmark --ctx 4096 --tokens 128 --reps 2 \
  --drafts 0,1,2,3,4 \
  --draft-vocab open-jet/megakernel/draft_vocab_32k.bin
```

O teste CUDA atual compara 8.945.664 valores de linhas reais com os
dequantizadores CPU independentes do llama.cpp, incluindo IQ16/dot products
e o caminho Q6 após repack. A primeira versão conferia 5.038.080 valores.
Os testes de backend/GGUF/build atuais passaram em 34/34 casos.
`python open-jet/megakernel/tests/gpu_server_smoke.py --model "$MODEL" --output /tmp/ada-http` verifica
sete casos reais de HTTP/CUDA, incluindo streaming, seed, segunda rodada e
prompt de 1351 tokens. Compute Sanitizer memcheck passou sem erros nos três
ciclos de decode instrumentados do binário final (teste curto de 16 tokens;
isso não equivale a uma verificação exaustiva). O teste
host cobre arquivos truncados e conversão de subnormais F16/BF16.
O benchmark aquece o llama-server, entrega os mesmos IDs de prompt aos dois
engines, registra todas as sequências, repetições e tempos, e faz teacher forcing
da continuação de referência. `--temperature 1 --top-k 20 --top-p .95`
permite repetir o caso com amostragem. Tempos de load e prefill ficam fora do throughput
de decode. Aceitação MTP e custo de prefill devem ser comparados separadamente.

`MK_PROFILE=1` apresenta os tempos por fase; `MK_PF_PROFILE=1` mede prefill.
`MK_CHECK_FINITE=1` identifica a primeira fase que apresentar valores não
finitos; a instrumentação altera o timing e não deve ser usada para reportar
throughput. Compute Sanitizer também deve ser separado dos benchmarks.

## Qualidade e limites

Os padrões do servidor correspondem aos metadados deste modelo: temperatura
1, top_k 20, top_p 0,95 e min_p 0. O engine aceita top_k até 32. Penalidades não
suportadas e top_k fora desse limite retornam erro, em vez de mudar a distribuição
silenciosamente. Para comparação greedy, use temperatura 0 e penalidades neutras.

As dequantizações preservam os pesos, mas as ativações Q8 do decode e o prefill
FP16 do engine original continuam aproximados. Igualdade de uma continuação não
é prova geral de qualidade: corpora de perplexidade e prompts longos precisam
ser avaliados antes de usar a adaptação como substituto universal do llama.cpp.
Não houve treino nem alteração do GGUF.

Os testes locais cobriram greedy, amostragem com seed fixa, streaming, segunda
rodada e prefill atravessando os chunks. São testes de regressão numérica e
funcional; não substituem uma avaliação ampla de perplexidade ou qualidade.

## Resultados medidos na RTX 4060 Ti

**Revisão do baseline em 2026-10-10:** a tabela abaixo conserva as medições
iniciais, que usavam MTP 2 no llama.cpp. Novos testes com MTP 4 atingiram
62,96 tok/s greedy curto, 61,93 sampling e 45,51 greedy longo. O megakernel
fica respectivamente 23,49%, 23,73% e 12,27% abaixo dessas referências;
no caso longo, as continuações diferem. Portanto os ganhos iniciais de
4,6%/2,7% não demonstram vantagem sobre o llama.cpp ajustado.

A nova referência greedy longa com MTP 4 é idêntica à referência sem
especulação e diverge do megakernel/ref. MTP 2 a partir do índice 41.
Teacher forcing concordou em 127/128 argmax, não 128/128. Ainda faltam
isolamento da causa e avaliação ampla de qualidade. As afirmações de
identidade abaixo se limitam às referências iniciais.

O novo perfil atribui aproximadamente 59% do decode ao FFN e menos de 1%
à atenção nesses prompts. Chunk 2048 elevou prefill de 530,5 para 630,2 tok/s
com aproximadamente +0,629 GiB, mantendo decode em cerca de 40 tok/s.
Diagnóstico, dados e próximas mudanças estão em
[OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md) e
[gráfico atualizado](bench/ada-report/ganhos_e_correcao.png).

Referência: llama.cpp `aa94f2086` (build 11549), CUDA 13.4, sm_89, contexto
4096/Q8, uma sequência. Clocks e voltage boost sem offsets no LACT; fan curve
original preservada. Cada linha abaixo usa duas repetições de 128 tokens,
exceto o baseline greedy com MTP, que usa três. O throughput exclui load e
prefill; o primeiro token do megakernel é produzido pelo prefill.

| Caso | llama-server | Megakernel | Observação |
|---|---:|---:|---|
| Greedy, prompt de código de 24 tokens, sem MTP | 20,92 tok/s | 16,42 tok/s | decode simples ainda favorece llama.cpp |
| Mesmo prompt, MTP | 46,04 tok/s, draft 2 | 48,17 tok/s, draft 4 | ganho de aproximadamente 4,6% |
| Amostragem temp=1, top_k=20, top_p=0,95, seed=123 | 46,01 tok/s, draft 2 | 47,24 tok/s, draft 4 | ganho de aproximadamente 2,7% |
| Greedy, prompt de 1365 tokens, MTP | 40,91 tok/s, draft 2 | 39,93 tok/s, draft 4 | llama.cpp aproximadamente 2,5% mais rápido |
| Prefill do prompt de 1365 tokens | aproximadamente 740 tok/s | aproximadamente 529 tok/s | fallback IQ limitado por workspace/dequantização |

No prompt curto, drafts 0/1/2/3/4 atingiram respectivamente
16,42 / 28,67 / 39,33 / 46,64 / 48,17 tok/s. Todas as dez gerações greedy
reproduziram exatamente os 128 IDs da referência. No prompt longo, as quatro
gerações também reproduziram os 128 IDs. Teacher forcing concordou em 128/128
nos dois prompts, inclusive com prefill chunk 128 no prompt longo. Com amostragem, os geradores de números aleatórios diferem
entre motores; igualdade dos IDs não é um critério válido de equivalência de
distribuição. A aceitação dos drafts depende fortemente do texto: caiu de cerca
de 89% no prompt curto para 69% no longo.

O resumo com hashes das fontes/binários está em [bench/ada_summary.json](bench/ada_summary.json).
Os dados, IDs, prompts, comandos e logs de desempenho estão em
[bench/results-ada-validated](bench/results-ada-validated/results.json),
[baseline com MTP](bench/results-ada-stock-mtp/results.json),
[amostragem](bench/results-ada-sampling/results.json),
[prompt longo](bench/results-ada-long/results.json),
[HTTP](bench/results-ada-server/results.json) e
[capacidade de memória](bench/results-ada-memory/results.json).
`-d 4` é o padrão atual do servidor e do CLI; a profundidade cai para 3 após
uma rejeição, como no algoritmo original. O CLI usava padrão 2 na adaptação
inicial. Use `-d 4` explicitamente para reproduzir os benchmarks.

O kernel final mantém 512 threads/CTA, 128 registradores/thread e 92544 bytes
em shared memory dinâmica. São 34 CTAs cooperativas, uma por SM, com 16 warps
residentes por SM (aproximadamente 33% de occupancy). Esses limites explicam
por que aumentar CTAs ou reduzir NT sem alterar os layouts não é um ajuste
seguro. O perfil identifica as projeções e o FFN como custos dominantes; os
helpers com IQ fixado em compilação e o Q6_K especializado foram ganhos reais.
Ainda há spills na compilação. Reduzir registradores/spills ou escrever GEMMs
IQ de prefill sem workspace FP16 são próximos trabalhos, que exigem nova
validação e não têm ganho presumido.

## Clocks e estabilidade

O perfil inicial do LACT tinha offsets de +190 no núcleo, +1200 na memória,
voltage boost 8 e limites explícitos de clocks. Nesse perfil foram observados
acessos ilegais/misaligned e gerações corrompidas, inclusive no MTP do llama.cpp.
Ao zerar os offsets, as repetições ficaram estáveis. Um teste A/B com o mesmo
binário passou três vezes em padrão e voltou a falhar em uma das duas repetições
com o perfil original. Isso associa a falha à configuração de clocks; não
isola qual ajuste é responsável nem certifica estabilidade geral do hardware.

O usuário pediu **manter os clocks padrão**. A configuração foi aplicada pela
[API do LACT](https://github.com/ilya-zlobintsev/LACT/blob/master/docs/API.md),
confirmada e persistida; fan curve e PowerMizer Adaptive foram preservados.
O backup e a evidência A/B estão em [results-ada-clocks](bench/results-ada-clocks/README.md).
Os erros iniciais com o perfil anterior não constituem uma limitação comprovada
do MTP do llama.cpp: com clocks padrão, ele passou as três repetições.
