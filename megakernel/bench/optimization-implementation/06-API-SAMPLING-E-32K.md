# Conclusão: API, sampling e contexto alocado de 32K

## DFlash2 em 32K

`--fit off`, offload completo, KV Q8, uma sequência, sete propostas.
Carregou usando 15169 MiB e os snapshots após pedidos chegaram a 15491 MiB
(15,13 GiB). Uma repetição de cada prompt: 91,72 tok/s no curto e 62,64 no
longo, mesmos IDs canônicos. Entradas de 24/1365 tokens: não é teste de um
prompt efetivamente preenchido com 32K tokens. DFlash2 64K não foi testado.

## Sampling e regressão A/B

Temp 1/top_k 20/top_p .95/min_p 0/seed 123, 128 tokens, três repetições.
DFlash2: **92,56 tok/s**, mesmas sequências nas três repetições internas.
Não se compara igualdade da saída com outros engines que usam RNG distinto.

O A/B intercalado original/final MK produziu **47,46 → 59,35 tok/s (+25,05%)**.
Os IDs foram iguais entre os dois MK em todas as repetições.
`compare_optimization.py` agora aceita `--temperature` e sanitiza as flags de
instrumentação/modo experimental herdadas do ambiente.

## Launchers realmente exercitados

`launcher_smoke.py` inicia `serve_ada.py`, primeiro DFlash2 e depois megakernel,
e encerra apenas os processos que criou. Ambos passaram: health, models,
resposta JSON correta, repetição greedy, equivalência de streaming, repetição
com seed e segunda rodada com resposta conhecida. Evidência em
`launcher-smoke/results.json` e logs por engine.

Os servidores de benchmark/testes foram encerrados. A GPU voltou a 41 MiB,
0% de utilização, clocks dinâmicos idle 210/405 MHz. Não houve ajuste de
clocks ou voltage offsets.

## Entrega consolidada

O gráfico foi gerado dos JSONs brutos, com três casos de desempenho, ganho
A/B e painel separado de problemas corretos/identidade de saídas. Inclui
limites de qualidade e de comparabilidade do caso longo. PNG/SVG/PDF e hashes
dos dados de origem estão em `report/`.

Relatório final: `../../IMPLEMENTATION_RESULTS.md`.
Os registros 01–05 permanecem como diário intermediário; contagens e estados
"em andamento" neles descrevem aquele momento. Resultado final: 34 testes
host, oráculos CUDA, A/B, memcheck e API passaram; divergência numérica longa
conhecida e avaliação ampla de qualidade continuam pendentes.
