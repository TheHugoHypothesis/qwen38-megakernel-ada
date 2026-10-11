# 04 — Regressão A/B do MK existente

Binário otimizado IQ16 preservado antes do porte vs binário atual,
MTP 4, três repetições intercaladas, 128 tokens, contexto 4096 Q8,
mesmos IDs de prompt e chunk auto nos dois engines:

| Caso | Anterior | Atual | Variação |
|---|---:|---:|---:|
| Curto | 60,67 tok/s | 60,70 tok/s | +0,06% |
| Longo | 50,03 tok/s | 50,07 tok/s | +0,08% |

Todos os IDs gerados coincidiram entre as duas versões nas seis comparações.
Variações tão pequenas não são ganho demonstrado; não foi observada
regressão de throughput nesses casos. Captura e compartilhamento são
ativados explicitamente somente no híbrido, sem mudar a configuração do
MK normal ou o launcher de produção.

Fonte: `regular-mk-regression/results.json`, comandos e logs individuais
no mesmo diretório. `compare_optimization.py` passou a aceitar baseline e
chunk do baseline explícitos, mantendo os defaults dos ensaios anteriores.

O build sm_89 terminou com sucesso. O kernel mantém 128 registradores;
o report ptxas e hashes dos binários/fontes finais foram preservados em
`sha256-final.json` e `../../ptxas.log`. llama.cpp permanece sem alterações.
