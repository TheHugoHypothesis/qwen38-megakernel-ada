# IQ16: comparação intercalada concluída

Três repetições por motor/caso/profundidade; ordem invertida a cada repetição.
Contexto 4096/Q8, chunk explícito 512, mesmos IDs de entrada, 128 tokens de
saída, clocks padrão. Os hashes dos binários estão no JSON da comparação.

| Caso | Antes | IQ16 | Resultado |
|---|---:|---:|---|
| Curto, MTP 4 | 48,18 tok/s | 60,66 tok/s | +25,91% |
| Longo, MTP 4 | 39,93 tok/s | 50,02 tok/s | +25,26% |

Todas as gerações foram idênticas às do binário anterior, inclusive sem MTP.
A aceitação curta continuou 101/114, 30 ciclos: o ganho vem do menor tempo
do ciclo, não de uma mudança de propostas ou aceitação nesse caso.

A divergência longa no índice 41 frente ao llama.cpp sem especulação/MTP 4
permanece. O IQ16 preserva o comportamento observado do binário anterior;
isso não demonstra identidade com todas as referências nem qualidade ampla.

O snapshot `iq16-snapshot/` conserva o candidato validado. A próxima etapa
inclui o decode IQ16 no prefill e uma saída de diagnóstico dos logits no
teacher forcing, além do teste DFlash2 no llama.cpp.

Evidência: `iq16-ab/results.json`, logs e IDs na mesma pasta.
