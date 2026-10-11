# 24 — Unsloth Studio: controle de reasoning_effort sem alterar o cliente

2026-10-11. Correção da compatibilidade HTTP/Python do servidor MK. Clocks, CUDA, GGUF e instalação do Unsloth preservados.

## Diagnóstico

O template de 8952 caracteres do target local escolhe `reasoning_effort|default('xhigh')` quando thinking está habilitado. Os esforços nativos aceitos são `low`, `medium` e `xhigh`. `enable_thinking=false` fecha o bloco think no próprio prefixo de geração. O limite de saída não aparece no template: `Server.generate` conta raciocínio e resposta no mesmo `max_new` e, quando o limite não é enviado, usa contexto menos prompt menos 16 posições.

Antes desta correção, `Server.render` copiava somente controles presentes em `chat_template_kwargs`. `reasoning_effort=low` no nível principal era ignorado, mantendo o padrão xhigh. A investigação de renderização anterior reproduziu exatamente esse caso.

Na instalação local, `backend/core/inference/external_provider.py::_apply_custom_reasoning_controls` só encaminha controles quando `reasoning_config.enabled=true`. O formato `reasoning_effort` encaminha low/medium/high e envia `none` quando Thinking está desligado. O formato `chat_template_kwargs.enable_thinking` só encaminha o booleano de thinking, deixando de fora o esforço.

O preset `llama_cpp` usa `supports_chat_template_kwargs`: encaminha esforço low/medium/high no nível principal e o toggle de thinking no objeto de template. Essa combinação também precisava ser entendida pelo MK.

O bundle local `frontend/dist/assets/connections-tab-6jW06C4u.js` confirma os campos **Enable reasoning controls**, **Reasoning parameter style**, a opção **reasoning_effort**, e **Reasoning model**. Não foi necessário editar ou consultar dados privados/configurações persistidas de conexões.

## Mudança no servidor

Arquivo `server/mk_server.py::Server.render`:

- Resolve enable_thinking, preserve_thinking e reasoning_effort a partir de kwargs e, quando o valor está ausente/nulo, do nível principal.
- Mantém prioridade dos kwargs não nulos por campo; não modifica o corpo da requisição.
- Aceita low/medium/xhigh; converte high para xhigh; none força thinking falso e é removido do contexto para não violar o contrato do template nativo.
- Thinking falso continua falso com um esforço positivo; os valores são resolvidos antes da aplicação dessa regra.
- Rejeita tipos inválidos de kwargs/booleanos e esforços desconhecidos com BadRequest/HTTP400.
- Sem controles, mantém o prompt/default anterior. Acrescenta uma linha no log com o thinking e esforço efetivos.

As mudanças alteram a interpretação dos controles HTTP e, quando solicitado, o prompt renderizado. Não há novo sampler, limite rígido de thinking, mudança no decoder, requantização ou alteração de alocações GPU. Esforço low é uma instrução do template para pensamento breve, não garantia de quantidade fixa de tokens. Os resultados históricos de qualidade/performance não foram repetidos com cada esforço.

## Configuração sem editar o Unsloth

1. Reiniciar o servidor MK com o mesmo comando e contexto em uso para carregar a alteração Python. Não é necessário make/recompilar CUDA.
2. Em **Connections**, editar a conexão **Custom** e usar **Chat Completions**.
3. Marcar **Reasoning model**, habilitar **Enable reasoning controls**, selecionar **Reasoning parameter style = reasoning_effort** e salvar.
4. No chat, escolher **Low** ou desligar Thinking. Continuar com Presence0/Frequency0/Repetition1 para o sampler MK.

Exemplo de campos encaminhados pelo Studio: `{"reasoning_effort":"low"}`. O servidor converte o campo para a variável do template. Thinking Off produz `{"reasoning_effort":"none"}`. O log confirma `reasoning: on, effort=low` ou `reasoning: off`. Ausência do controle continua em `effort=template default`, que é xhigh neste GGUF.

Na conexão com preset **llama.cpp**, não há necessidade de configurar o estilo de Custom; basta reiniciar MK e selecionar o esforço/toggle no chat. O código do cliente já envia a combinação reconhecida pela correção.

## Validação

`tests/test_megakernel_backend.py` ganhou dez testes de contrato: esforços no nível principal, aliases/off, corpo emitido pelo preset llama.cpp, disable explícito, prioridade dos kwargs, valores nulos/default, equivalência com o formato antigo, preservação do corpo/preserve_thinking, controles inválidos e log efetivo.

```sh
# Executado em open-jet, com o venv de validação na raiz do workspace:
../.venv-validation/bin/python -m pytest -q tests/test_megakernel_backend.py tests/test_megakernel_build.py tests/test_megakernel_gguf.py
python3 -m py_compile megakernel/server/mk_server.py tests/test_megakernel_backend.py
git diff --check
```

Resultado: **44 testes CPU e 12 subtests passaram em 0,57s**, py_compile e diff-check aprovados.

[reasoning-compat/validation.json](reasoning-compat/validation.json) registra hashes do servidor, template GGUF e provider instalado. Um script de verificação carregou o template real via GGUFReader e extraiu somente a função serializer do Unsloth por AST, sem inicializar o app. Dez checks passaram: seis formatos produzidos pelo serializer Custom, combinação do preset llama.cpp, prioridade de kwargs, toggle no nível principal e preservação do default original. Os prompts ficaram idênticos aos renderizados com kwargs canônicos, incluindo low/high e prefixo com thinking desligado.

Escopo: renderização CPU e contrato de parâmetros. Não houve inferência CUDA, benchmark, HTTP ponta a ponta, reinício do servidor ativo ou alteração da configuração salva do usuário. A aplicação em uma conversa requer reiniciar MK e selecionar/salvar os controles na interface.
