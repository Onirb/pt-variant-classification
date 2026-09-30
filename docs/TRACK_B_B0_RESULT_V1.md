# Track B B0 — importação e auditoria WMT24++ v1

**Status:** concluído em 30 de setembro de 2026. Nenhum modelo foi treinado
nesta etapa e o teste final DSL-TL permanece não baixado e não observado.

## Artefatos congelados

- Importação bruta filtrada:
  data/track_b/wmt24pp_development_v1/development.parquet
- Manifesto da importação:
  data/track_b/wmt24pp_development_v1/manifest.json
- Visão preparada para os baselines:
  data/track_b/wmt24pp_development_v1/development_prepared.parquet
- Manifesto da preparação:
  data/track_b/wmt24pp_development_v1/preparation_manifest.json

Esses caminhos são locais e ignorados pelo Git. Os manifestos incluem hash
SHA-256 do respectivo arquivo de dados.

## Proveniência e filtros

| Item | Resultado |
| --- | ---: |
| Fonte | google/wmt24pp |
| Revisão | fd7405c06494bc66a57b25f55d217a72f96e60dc |
| Configurações | en-pt_BR, en-pt_PT |
| Linhas brutas | 998 por variedade |
| Exclusões por is_bad_source=true | 38 por variedade |
| Linhas elegíveis importadas | 1.920, equilibradas (960/960) |
| Documentos | 170 |
| Domínios | literary 412; news 298; social 988; speech 222 |

Foi usado o campo target, que representa a pós-edição humana indicada pela
fonte. Source_en e original_target foram preservados para auditoria, mas não
serão entradas do classificador.

## Rótulos contraditórios e visão preparada

A importação contém 20 grupos de texto final idêntico que aparecem com ambos
os rótulos. Eles correspondem a 43 linhas, em geral frases muito curtas,
nomes, hashtags ou trechos cuja tradução é a mesma nas duas variedades.

Esses registros são preservados na importação para rastreabilidade, mas
retirados da visão de modelagem: um texto idêntico com rótulos opostos não
oferece evidência para uma classificação binária e poderia atravessar uma
divisão por documento. A regra é determinística, anterior a qualquer modelo e
não usa métricas:

1. Excluir todo texto associado a mais de um rótulo.
2. Remover duplicatas exatas remanescentes.

Resultado: 1.877 textos, sendo 939 PT-PT e 938 PT-BR. Não houve duplicatas
adicionais de mesmo rótulo após a primeira exclusão.

## Auditoria de sobreposição exata normalizada

A comparação usa texto em minúsculas, espaços normalizados e igualdade exata.
O resultado foi zero sobreposição com:

- desenvolvimento histórico do Track A: 14.764 textos;
- teste público histórico: 2.570;
- DSL-TL PT_dev: 857;
- PtBrVId valid: 6.000;
- FRMT test congelado: 5.232.

Esta auditoria não mede paráfrases ou sobreposição semântica; ela impede o
vazamento textual direto antes dos baselines.

## Próxima etapa

B1.1: SVM char+word com três divisões por document_id. Em seguida, a mesma
divisão será usada para B1.2 (mascaramento) e B1.3 (embeddings congelados),
sem acessar o teste final DSL-TL.

Após o piloto B1.1, a divisão foi refinada: além de separar por document_id,
ela estratifica domínio e classe. A primeira versão preserva o resultado do
piloto, mas não é o baseline oficial porque uma de suas validações não continha
o domínio literary.

## Fontes

- [WMT24++ Dataset Card](https://huggingface.co/datasets/google/wmt24pp)
- [WMT24++ paper](https://aclanthology.org/2025.findings-acl.634.pdf)
