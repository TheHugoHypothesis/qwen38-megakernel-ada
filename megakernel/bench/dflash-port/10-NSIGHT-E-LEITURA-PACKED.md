# 10 — Leitura packed e perfil Nsight

Leitores Q4_K e Q6_K com oito valores packed passaram oracle CPU: nenhum erro em 1.271.808 pesos. Q6_K carrega uint16, respeitando alinhamento do bloco original 210 bytes. Dequantização FP16 oito valores/thread passou comparação CPU. Memcheck de todos os kernels dos oráculos: 0 erros.

Benchmark antes da instrumentação adicional de eventos: 80,47 /53,64 /80,07 tokens/s curto/longo/sampling, três repetições, native7. Mesmos IDs da versão Q8 anterior e da F32 nas três fixtures. Draft caiu para ~15–16ms/ciclo; target T8 continua ~77–79ms/ciclo. Default sete; quatro inferior nos três casos medidos.

Nsight nativo: native-profile.nsys-rep/.sqlite e CSVs. Total agregado de dez ciclos wide ~772ms. No stream nativo (13), mmq domina; lm_head usa ~4,5ms nas entradas head de sete rows. Conv/selector/controle têm custo pequeno. Encoder dequant ~17ms e cuBLAS ~23,55ms somados em DOIS prefills longos; não ~1,2s por prefill GPU!

O contador host injection_prefill_seconds inclui o tempo aguardando features produzidas pelo target em outro stream. Portanto sobrepõe parte do prefill target: não subtrair como custo exclusivo do encoder e não somar tempos host como GPU de fases exclusivas. Adicionados eventos depois do stream wait e ao fim da injeção para medir explicitamente native_injection_*_gpu_seconds. Profile contém warmup, duas prefills e decode curto; os números agregados não são tokens/s de benchmark.

Sem overclock. Baseline llama.cpp continua superior; avaliar Tensor Core MMQ/IMMA e atenção wide do target é trabalho de otimização futuro, não ganho já demonstrado.
