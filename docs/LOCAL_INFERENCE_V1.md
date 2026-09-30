# Inferência local - BERTimbau + LinearSVC v1

## Objetivo

Disponibilizar o candidato final do projeto para classificação local de texto em
pt-BR ou pt-PT. A interface é deliberadamente uma CLI pequena: ela é fácil de
testar, não abre portas de rede e não exige conta ou API externa.

## Artefatos esperados

Após `python -m scripts.train_final_bertimbau`, o diretório ignorado pelo Git
`runs/bertimbau_final_v1/` contém:

- `linear_svc.joblib`: classificador ajustado;
- `manifest.json`: revisão do encoder, hash do corpus, configuração, dimensão
  do embedding e tempo de ajuste.

O encoder BERTimbau não é copiado para o repositório. Ele é carregado do cache
local definido por `.hf_cache/bertimbau`, na revisão registrada no manifesto.

## Uso

```powershell
$env:HF_HOME = "$PWD\.hf_cache"
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m scripts.predict_local "O comboio parte às oito horas."
```

Saída esperada:

```json
{
  "label": "PT-PT",
  "margin": -0.42,
  "low_margin": false,
  "characters": 34
}
```

`margin` é a distância assinada da fronteira da `LinearSVC`; valores positivos
favorecem PT-BR e negativos favorecem PT-PT. Ela **não** é probabilidade e não
deve ser interpretada como confiança percentual. `low_margin` sinaliza uma
zona conservadora próxima da fronteira, definida inicialmente por
`abs(margin) < 0.25`.

## Limites

- A entrada é truncada em 256 tokens pelo tokenizer.
- O modelo foi validado em desenvolvimento e em avaliações externas já
  documentadas; não foi otimizado para todos os domínios, registros ou textos
  muito curtos.
- A CLI exige aproximadamente 0,5 GB para os pesos do encoder, além da memória
  do processo Python.
