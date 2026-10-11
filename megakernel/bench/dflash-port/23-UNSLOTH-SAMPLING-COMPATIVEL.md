# 23 — Unsloth Studio: presence_penalty e configuração compatível

2026-10-11. Usuário encontrou `presence_penalty is not supported by the megakernel (invalid_request_error)` ao conectar o Studio ao servidor MK.

## Causa confirmada no código local

`server/mk_server.py::Server.sampling` aceita temperature, top_p, top_k<=32, min_p e seed. Rejeita presence/frequency diferentes de zero e repetition/repeat diferentes de um. A penalidade não existe no sampler CUDA atual; aceitar o campo e ignorar seu valor mudaria a semântica da requisição.

Na instalação local `/home/elliot/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/studio/backend/assets/configs/inference_defaults.json`, o preset `families.qwen3.8` define temperature0,7, top_p0,8, top_k20, min_p0, repetition1 e **presence1,5**. O provider externo em `backend/core/inference/external_provider.py` inclui esse campo no corpo encaminhado. O bundle frontend `frontend/dist/assets/chat-Owi-soaD.js` confirma os controles **Sampling**, **Presence Penalty** (zero = Off) e **Repetition Penalty** (um = Off).

## Resolução imediata

No chat do Studio, seção Sampling, definir **Presence Penalty = 0 / Off**. Manter Repetition Penalty1, Top K20 e Min P0. Frequency Penalty, quando disponível, deve ser zero. Temperature0,7/TopP0,8 do preset instalado são aceitos. Se o cliente continuar enviando1,5, conferir o parâmetro salvo na conversa e evitar restaurar o preset padrão Qwen3.8.

[server/unsloth-mk-sampling.json](../../server/unsloth-mk-sampling.json) registra os parâmetros neutros compatíveis. É um objeto de sampling para copiar à requisição/campos; não uma importação do Studio. O launcher na porta padrão usa o endpoint MK `http://127.0.0.1:8080/v1`, com `serve_ada.py --engine megakernel-dflash2 --ctx 4096`.

Para usar presence1,5, selecionar o backend llama.cpp/DFlash2, `serve_ada.py --engine dflash2 --ctx 4096`. Usar somente um servidor/modelo na GPU16GB. Não assumir que os números de benchmark coletados sem penalidades se aplicam à configuração penalizada.

## Alterações desta correção

As mensagens HTTP400 agora indicam os valores neutros e o backend que suporta penalidades. O sampler mantém a rejeição explícita; nenhuma penalidade foi silenciosamente removida. Acrescentados perfil JSON e instruções no guia nativo. Não houve modificação do CUDA, recompilação do megakernel, mudança de clocks, alteração do GGUF ou edição da instalação/configuração do Unsloth.

Validação: perfil aceito diretamente por `Server.sampling` com engine mock; presence1,5 rejeitado com orientação para presence0. Os 26 testes CPU de `tests/test_megakernel_backend.py` passaram (0,16s), além de py_compile e diff-check. Esses checks verificam o contrato de parâmetros, sem inferência GPU nem validação ponta a ponta da conversa do usuário. A execução do cliente depende de aplicar os valores no Studio. Reiniciar o processo MK só é necessário para receber a mensagem HTTP atualizada; presence0 já era aceito pelo servidor anterior.
