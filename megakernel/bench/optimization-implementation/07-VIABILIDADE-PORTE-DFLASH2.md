# Viabilidade do porte DFlash2 para MK

2026-10-10. Investigação adicional, sem modificar kernels ou executar novos
benchmarks. Decisão recomendada: prova de viabilidade híbrida antes de um
porte completo. O baseline a superar é llama.cpp + DFlash2, não MK + MTP.

## Evidência disponível

Greedy curto: MK MTP 4 = 60,66 tok/s; llama.cpp MTP 4 = 62,96;
llama.cpp DFlash2 7 = 91,35. No longo: 50,02 / 45,51 / 62,29,
com continuações diferentes entre MK e llama.cpp. Portanto não há evidência
de que MK seja sistematicamente um verificador mais rápido, nem que o ganho
IQ16 possa ser multiplicado diretamente pelo ganho DFlash2.

Logs curtos registram DFlash2 110/118 drafts aceitos (93,22%) e MK MTP
101/114 (88,6%). Estratégias/profundidades e contabilidade diferem; esses
números não isolam causalmente os ganhos. Falta perfil DFlash2 separando
encoder/injeção, decoder/selector, target verification e coordenação.

## Restrições concretas do código

- `qwen38_mk.cu`: `MAXT=5`; dispatch `run_main<1..5>`; `AR=6*MAXT` e layouts
  de atenção/shared com 32 linhas. Sete propostas exigem verificação de oito
  posições, revisão de layouts e medição de registradores/spills. Não basta
  editar MAXT.
- `MKEngine::step` gera drafts MTP internamente. A API não recebe proposals
  externos nem exporta as cinco features intermediárias exigidas pelo draft.
  `last_hidden()` é apenas o hidden final e não substitui esses taps.
- Features GGUF `[6,20,34,48,62]` são entradas das camadas extraídas pelo
  llama.cpp. O MK precisa reproduzir exatamente a convenção, posições e
  estados aceitos/rejeitados, no prefill e no decode. Copiar resíduos em uma
  posição presumida pode reduzir aceitação mesmo sem quebrar a geração.
- O GGUF DFlash2 real **não contém `token_embd.weight` nem `output.weight`**.
  `fc.weight` e `selector_hidden.weight` estão presentes. O graph local
  `dflash.cpp` toma embedding/LM head do target por `ctx_other` quando ausentes.
  Reutilizar o draft llama.cpp com target MK exige ponte de pesos/estado;
  não se pode resolver isso carregando uma segunda cópia completa do target.
- O draft tem cinco camadas, convoluções, selector e atenção diferente do
  target. Embutir tudo no kernel já limitado a 128 registradores/thread e
  92544 bytes shared cria risco de perda de desempenho.
- Memória llama.cpp/DFlash2 já observada: 14,06 GiB em 4K e 15,13 em 32K
  (NVML, snapshots). Isso prova viabilidade desse engine, não do novo porte.
  Features devem ser processadas por chunks e consumidas/injetadas; não
  armazenar cinco hiddens de todo o contexto sem necessidade.
- A divergência numérica MK longa continua pendente. Capturar features MK
  pode alterar a aceitação do draft treinado; é preciso comparar por camada.

## Experimento recomendado

1. Perfilar llama.cpp/DFlash2 e medir target verification em batch de oito.
   Medir um verificador MK equivalente com proposals fixos e rollback real.
   Os tempos têm de incluir manutenção/injeção das features para serem úteis.
2. Criar ponte explícita para compartilhar embedding/head e executar o draft
   com GGML/CUDA reutilizado, recebendo features MK e devolvendo proposals.
   Começar com quatro propostas cabe no limite atual; comparar com DFlash2 4
   (73,95/59,47 tok/s), depois ampliar para sete e comparar com DFlash2 7.
3. Validar logits/features, rejeições em cada posição, replay DeltaNet, KV,
   sampling e orçamento de memória. Aumentar prompts e duração dos ensaios.
4. Investir em fusão/porte adicional apenas onde o perfil demonstrar custo
   significativo e removível. CUDA Graphs/coordenação em GPU podem bastar;
   um único kernel não é critério de sucesso.

Critério econômico sugerido: ganho estável de pelo menos 10% sobre o DFlash2
atual em uma suíte representativa, sem piora de qualidade e sem offload nos
contextos desejados. Aproximadamente 100/69 tok/s nos dois casos atuais seriam
marcos desse critério, **não previsões de desempenho**. A decisão deve usar
também prosa/raciocínio e contextos preenchidos, não somente esses dois prompts.

Referências: código local `qwen38_mk.cu`, `qwen38_mk.h`,
`llama.cpp/common/speculative.cpp`, `llama.cpp/src/models/dflash.cpp`, GGUF
DFlash2 e registros finais existentes. Conferência externa:
[DFlash2 no llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp),
[implementação DFlash](https://github.com/z-lab/dflash).
