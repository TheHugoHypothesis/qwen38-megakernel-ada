# 09 — API, memória e profundidade

- CLI, C API `mk_new_dflash` e HTTP usam draft CUDA nativo; referência GGML selecionável apenas no CLI.
- Default native DP4A/Q8, CUDA Graphs por profundidade, FP16 KV SWA 2056 posições; F32 disponível por MK_DFLASH_MATH=f32.
- Target sem MTP remove pesos desse bloco (~0,343 GiB). Reserva explícita 1792 MiB para draft/encoder/features antes do auto chunk; evita usar toda a VRAM para workspace do target.
- Checkpoint/rewind/reset/truncate de target e draft coordenados. HTTP passou 7 checks em contexto alocado 32768: greedy, stream, seed, erro de sampler, segundo turno e prompt real 2791 > janela 2048. Memória após requests 15463 MiB /16380. Não é benchmark de contexto preenchido 32K.
- Launcher `--engine megakernel-dflash2` passou saúde, modelos, 2+2, greedy, streaming, seed e segundo turno 4+4. Memória após requests 15133 MiB.
- Suíte native7: 10/10 tarefas numéricas corretas; texto igual llama sem especulação 17/20, igual MK MTP anterior 16/20. Não identidade integral nem avaliação ampla de acurácia/perplexidade.
- Depth4 não venceu depth7: 58,81/48,68/56,50 vs 74,53/49,76/74,17 tokens/s (curto/longo/sampling); manter 7 default, 4 disponível explicitamente. Não adicionar heurística adaptativa sem evidência de ganho.
- OpenJet aceita env OPENJET_MK_DFLASH_MODEL e OPENJET_MK_DRAFT. Cache de build inclui novas fontes. Upstream llama.cpp intacto.
- Otimização final em validação: leituras packed dos oito valores Q4_K/Q6_K; Q6_K exige alinhamento de 2 bytes por bloco original 210B. Encoder dequantiza oito pesos por thread, reduzindo trabalho repetido; testes CPU de staging FP16 adicionados.
