# Testes locais, reprodução e arquivos publicados

## Duas verificações diferentes

**Suíte de software:** testa regras, interfaces, integridade, checkpoints,
recuperação do gold e bloqueio de segunda inferência usando dados sintéticos e
redes minúsculas. Não mede F1 científico e não acessa o teste reservado.

**Experimentos científicos:** usam os corpora e artefatos reais, revisões
registradas e protocolos congelados. Seus resultados estão documentados, mas
pesos modernos, caches e arquivos detalhados em `runs/` não são publicados.

## Instalação e suíte

Python 3.12.10 e Windows 11 foram usados para a verificação local.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-track-b-cpu.txt
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O arquivo combinado mantém o PyPI para dependências gerais e adiciona o índice
oficial de wheels CPU apenas como fonte extra do PyTorch fixado em versão CPU.
A instalação requer internet; os testes, não. Os fixtures são criados em pastas
temporárias e não leem os pacotes/relatórios reais de `runs/`.

Pacotes auxiliares de testes e medição (Accelerate e psutil) já fazem parte dos
requisitos. A inferência real usa safetensors; a retomada de checkpoint dos
testes usa apenas objetos criados pelos próprios testes, nunca um pickle externo.

## O que publicar

- Código em `src/`, `scripts/` e testes sintéticos.
- Notebooks Colab sem outputs de execução, requisitos e documentação.
- Métricas agregadas, hashes e descrição dos filtros nas páginas de resultados.

Não publicar `.venv`, `.hf_cache`, `runs/`, novos corpora, pesos modernos,
ZIPs, configurações locais, `.env` ou credenciais. Os arquivos acadêmicos já
versionados em `models/`, `data/` e `notebooks/` são preservados como legado;
`.gitignore` não remove arquivos que já estavam rastreados.

## Reprodução científica

As fontes e revisões reais são descritas em
[TRACK_B_B0_RESULT_V1.md](TRACK_B_B0_RESULT_V1.md),
[TRACK_B_B1_4_RESULT_V1.md](TRACK_B_B1_4_RESULT_V1.md) e
[TRACK_B_B4_PROTOCOL_V1.md](TRACK_B_B4_PROTOCOL_V1.md).

Fluxo de referência, **não uma execução automática ou convite para reabrir testes**:

1. Adquirir a revisão registrada de WMT24++ e aplicar os filtros documentados.
2. Gerar as divisões por documento e os baselines; registrar os artefatos locais.
3. Preparar o pacote B1.4, executar a validação de três sementes no Colab e
   importar os relatórios retornados.
4. Preparar o pacote final de 318 passos, executar o ajuste final e importar os
   pesos/recibo, conforme [TRACK_B_FINAL_SELECTION_V1.md](TRACK_B_FINAL_SELECTION_V1.md).
5. Recuperar o gold DSL-TL, auditar sobreposição e registrar os IDs antes de uma
   avaliação única, conforme o protocolo B4.

Um novo checkout não contém os pesos necessários para `predict_track_b` nem
os pacotes reais de upload. A geração depende dos passos anteriores e de GPU
para o fine-tuning. Reexecutar software para reproduzir um resultado conhecido
não transforma o benchmark novamente em teste não observado. Nenhum resultado
reproduzido pode servir para ajuste posterior sobre esse mesmo teste.

O estado original deste ciclo está encerrado: arquivos de experimento existentes
não são sobrescritos pelos scripts de importação, auditoria e avaliação final.
