# B1.4 — preparação para Colab

Autorizado pelo usuário em 30 de setembro de 2026. Este documento registra o
protocolo antes da execução na GPU remota.

## Arquivos para abrir

- Notebook: colab/Track_B_B1_4_Colab.ipynb
- Dados e script: runs/track_b/colab_b1_4_bundle_v1/track_b_colab_package_v1.zip

Abra https://colab.research.google.com/ e escolha Arquivo > Fazer upload de
notebook. Envie o arquivo ipynb. Escolha GPU em Ambiente de execução > Alterar
tipo de ambiente de execução. Execute as células em ordem; a primeira solicita
o ZIP. A célula de teste curto é separada da que inicia as três sementes.

O pacote tem uma lista explícita de arquivos e campos: texto WMT24++ público,
classe, domínio, documento, segmento, variedade, divisões, manifesto e script.
Nenhum caminho é varrido automaticamente para procurar outros dados.

## Protocolo fixo

- Encoder e tokenizer BERTimbau base cased, revisão
  94d69c95f98f7d5b2a8700c420230ae10def0baa.
- Sementes 7, 42 e 2026; mesmos membros de treino e validação dos baselines.
- Dados: 1.877 textos; 1.479 de treino e 398 de validação por semente.
- Fine-tuning de todos os parâmetros e cabeça binária nova.
- Três épocas, AdamW, learning rate 2e-5, weight decay 0,01,
  warmup de 10%, máximo de 256 tokens.
- Batch físico 8, acumulação de 2 (batch efetivo 16), FP16,
  gradient checkpointing e preenchimento dinâmico.
- Avaliação e checkpoint a cada 50 passos; seleção pelo F1 macro de validação.
- Checkpoints incluem modelo, otimizador, scheduler e estados aleatórios.
- Teste curto de 10 passos, isolado; serve apenas para medir execução e memória.

As configurações são escolhas iniciais de experimento, não parâmetros ótimos
demonstrados. As três sementes medem variação de divisão e de treinamento.
Os resultados devem ser comparados com B1.1--B1.3 por domínio e classe.

## Persistência e retomada

USE_DRIVE=True solicita montar o Drive e salva apenas na pasta de resultados
MyDrive/pt-variant-track-b/b1_4_colab_v1. Espaço ocupado pode aproximar-se de
10 GB quando modelos e checkpoints das três sementes forem mantidos.
USE_DRIVE=False mantém tudo no disco temporário do Colab, sujeito a perda no
encerramento da sessão. A perda de GPU não implica cobrança automática: este
fluxo usa o runtime gratuito selecionado pelo usuário.

Após interrupção, repetir a preparação e a célula full reconhece sementes
concluídas e retoma a incompleta do último checkpoint disponível. Passos após
esse checkpoint podem ser reexecutados. A mesma configuração e versão do
pacote são exigidas; resultados podem variar ligeiramente entre hardwares.

A última célula exporta um ZIP pequeno de relatórios, configurações e predições.
Modelos pesados ficam no Drive/disco de saída para recuperação posterior.

## Limites metodológicos

É validação com documentos novos dentro dos mesmos quatro domínios, e não
leave-one-domain-out. O WMT24++ é tradução humana; rótulos de variedade de
origem não garantem que todo segmento contenha marcadores linguísticos.
Os textos idênticos com rótulos conflitantes foram removidos da visão de
modelagem, portanto esse subconjunto é mais discriminativo que o corpus bruto.
A escolha de checkpoint também usa validação. Comparações são exploratórias,
e o DSL-TL test permanece reservado para uma avaliação após a seleção.

## Fontes do mecanismo

Validação local: cinco testes passaram (integridade do pacote, cobertura das
divisões, bloqueio de vazamento documental, sintaxe das células e treinamento
com retomada de checkpoint em uma rede minúscula). O ZIP final tem 251.722 bytes.
Nenhum treinamento BERTimbau B1.4 foi executado localmente ou remotamente nesta
preparação; tempo, VRAM e compatibilidade do runtime CUDA serão medidos no Colab.

- Trainer e checkpoints: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
- Limites e persistência Colab: https://research.google.com/colaboratory/faq.html
- Modelo: https://huggingface.co/neuralmind/bert-base-portuguese-cased
