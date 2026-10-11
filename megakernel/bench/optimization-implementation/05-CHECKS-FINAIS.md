# A/B final, memória e diagnóstico numérico

O A/B final mediu três repetições intercaladas, agora com chunk auto no
candidato e 512 no original:

| Caso | Original | Final | Ganho de decode |
|---|---:|---:|---:|
| Curto | 48,32 tok/s | 60,66 tok/s | +25,54% |
| Longo | 40,07 tok/s | 50,02 tok/s | +24,81% |

Prefill longo: 2,575 s para 1,988 s, aproximadamente −22,8% de latência
e +29,5% de throughput. Prefill curto praticamente igual. Todos os IDs
das gerações finais coincidiram com os do original nesses casos.

O CLI passou a draft 4 por padrão. O relink desse ajuste ocorreu durante
o A/B final; todos os comandos medidos especificavam `-d 4`, e o objeto
CUDA/shared library permaneceram iguais. A nota e hashes estão no JSON.
O runtime open-jet agora escolhe MTP 4 em Ada de 16 GB quando usa llama.cpp,
preservando MTP 2 nas demais GPUs; há override `OPENJET_MTP_DRAFT_MAX=1..8`.
Após essa mudança, 34 testes Python passaram.

## Memória: política automática exercitada

| Contexto Q8 | Chunk escolhido | Uso total CUDA | Livre |
|---:|---:|---:|---:|
| 4096 | 2048 | 13,506 GiB | 2,149 GiB |
| 32768 | 2048 | 14,502 GiB | 1,153 GiB |
| 65536 | 1024 | 15,289 GiB | 0,366 GiB |

Foram testes de alocação e geração com prompt curto, não qualidade de
entradas efetivamente preenchidas de 32K/64K. Contexto 262144 foi rejeitado
no preflight por falta de VRAM. Nenhum clock foi alterado.

## Corretude

Compute Sanitizer memcheck: quatro lançamentos `mk_cycle`, zero erros.
A primeira tentativa usou sintaxe de filtro incompatível com o tool
instalado e foi corrigida para `kns=mk_cycle`; o erro de invocação foi
preservado e não contou como resultado de validação.

O probe da divergência longa retornou top1 id 8029/logit 22,7672, top2
id 2144/logit 22,4890, margem 0,2782. A referência esperava 2144; NLL
dessa posição 1,4265. Teacher forcing continua 127/128. A divergência não
foi corrigida nem atribuída de forma conclusiva a um único operador.

Os snapshots DFlash2 anteriores de 14103–14395 MiB equivalem a
13,77–14,06 GiB. A descrição em commentary que dizia 14,1–14,4 GiB confundia
essas unidades; o valor bruto dos JSONs está correto.

Evidência: `final-ab/results.json`, `final-checks/`, `host-tests.log`.
