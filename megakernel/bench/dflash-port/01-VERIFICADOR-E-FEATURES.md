# 01 — Verificador externo e captura de features

2026-10-10, RTX 4060 Ti sm_89, clocks padrão.

Implementados `MKEngine::verify_external`, `enable_features` e
`share_weights`. O target reutiliza o controle de aceitação, KV e replay
DeltaNet do megakernel existente, com até quatro propostas externas.
As entradas das camadas são capturadas antes da normalização, conforme
`t_layer_inp` do builder Qwen35 do llama.cpp. O prefill copia por chunk;
decode emite somente anchor e prefixo aceito.

`tests/external_verify.cpp` executado na GPU: todas as cinco situações
(rejeições 0, 1, 2, 3 e aceitação completa de 4 propostas) passaram.
Cada situação continuou até comparar 64 IDs com o decode escalar MK.
Também passaram propostas inválidas sem mutação, identidade dos tensores
compartilhados, amostras de bytes IQ2_S/Q4_K iguais ao GGUF, 89 callbacks
e prefill longo cruzando chunks de 128 tokens.

Um tap adicional da camada zero foi comparado com dequantização CPU GGML
do embedding do token correto em cada posição. Todos os valores de todos
os taps foram finitos. Isso valida layout e posições; ainda não constitui
comparação numérica independente de todas as cinco camadas profundas.

Saídas completas: `external-verify.out` e `external-verify.log`.
Baseline anterior preservado em `baseline/` com SHA-256.

O bridge usa o draft GGUF real, sem segundo target, e compartilha o
embedding IQ2_S/head Q4_K em buffers CUDA GGML padrão. O smoke test de
16 tokens completou 3 ciclos, aceitou 12/12 propostas e mediu 68,89 tok/s.
Esse teste curto confirma integração; benchmarks de 128 tokens seguem.
