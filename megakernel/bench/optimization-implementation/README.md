# Registros da implementação e benchmarks

Consolidação em [IMPLEMENTATION_RESULTS.md](../../IMPLEMENTATION_RESULTS.md).
Clocks padrão, target GGUF original, CUDA 13.4/sm_89, uma GPU usada serialmente.

| Registro | Evidência |
|---|---|
| [01 — IQ16](01-IQ16.md) | Implementação e preservação do baseline |
| [02 — A/B inicial](02-IQ16-RESULTADOS.md) | `iq16-ab/results.json`, `iq16-oracle.log` |
| [03 — DFlash2/prefill](03-DFLASH-E-PREFILL.md) | `dflash2-q4/results.json`, `native-gemm.log` |
| [04 — validação ampliada](04-VALIDACAO-AMPLIADA.md) | `quality-suite.json`, `quality-{nospec,dflash2,megakernel}/results.json` |
| [05 — checks finais](05-CHECKS-FINAIS.md) | `final-ab/results.json`, `final-checks/`, `host-tests.log` |
| [06 — API/sampling/32K](06-API-SAMPLING-E-32K.md) | `launcher-smoke/`, `dflash2-32k/`, `dflash2-sampling/`, `final-sampling/` |

Diário intermediário: estados "em andamento" e contagens nos primeiros
registros descrevem aquele momento; o resultado final tem 34 testes host.

`baseline/sha256.json` identifica o engine anterior. `final-checks/sha256.json`
identifica fontes e binários CUDA finais. `delivery-sha256.json` identifica
também launchers, scripts de benchmark e integração final.
`discarded-prebuild-run/` foi excluído das médias. O relink do CLI durante
`final-ab` está explicitado no JSON. O erro inicial de sintaxe memcheck está
preservado, e a execução corrigida registra zero erros.

`quality-nospec` é a referência; sua primeira versão do campo de resumo
`all_canonical_ids_equal=false` representa ausência de outra referência,
não divergência consigo mesma. Os IDs/respostas brutos são a referência real;
o script foi corrigido para retornar null quando não existe referência.

Gráfico: [PNG](report/ganhos_implementados.png),
[SVG](report/ganhos_implementados.svg), [PDF](report/ganhos_implementados.pdf),
[dados e hashes das fontes](report/chart_data.json).
Regeneração pela raiz do workspace:

```sh
MPLCONFIGDIR=/tmp/qwen38-matplotlib python open-jet/megakernel/bench/plot_optimization_results.py
```

Todos os comandos exatos de inferência, prompts, IDs e tempos foram registrados
nos JSONs. Clocks dinâmicos observados nos snapshots são boost de fábrica,
não offsets de overclock. Medidas de memória especificam CUDA ou NVML;
MiB → GiB divide por 1024.
