# Avaliação externa v1 - PtBrVId multidomínio

## Propósito e escopo

Esta é uma segunda avaliação externa, separada do DSL-TL. Ela usa as seis
partições `valid` do PtBrVId, uma por domínio: jornalístico, jurídico,
literatura, política, redes sociais e web. O arquivo foi obtido na revisão
`910745e06ee2a66e64c3cd958b56728c28abd5dc` e contém 6.000 textos balanceados.

PtBrVId é um recurso amplo, mas seus rótulos são *silver* (baseados na origem
dos textos), não uma nova anotação humana. Portanto, ele serve para medir
transferência entre domínios; o DSL-TL continua sendo a avaliação com rótulos
humanos.

## Integridade

| Verificação | Resultado |
| --- | ---: |
| Linhas baixadas | 6.000 |
| pt-PT / pt-BR após normalização | 3.000 / 3.000 |
| Sobreposição textual normalizada com treino | 0 |
| Sobreposição textual normalizada com teste público | 0 |
| SHA-256 do arquivo local | `41c00ea2ad895f38a67c1a4e35fb24bf6028e944cf4691e10b7af9ff951cef11` |

As seis partições são avaliação somente: não podem entrar em treino, escolha de
parâmetros ou seleção de arquitetura.

## Normalização de rótulos

Cinco domínios usam `0 = pt-PT` e `1 = pt-BR`. Na partição `web`, as amostras
oficiais mostram a orientação inversa (`0 = pt-BR`, `1 = pt-PT`). A avaliação
preserva `raw_label` e normaliza essa partição antes da métrica. Sem essa
correção, o resultado de web seria artificialmente quase invertido.

## Resultado do modelo final v1

| Domínio | Linhas | F1 macro |
| --- | ---: | ---: |
| Jornalístico | 1.000 | 0,962996 |
| Jurídico | 1.000 | 0,751263 |
| Literatura | 1.000 | 0,515963 |
| Política | 1.000 | 0,924937 |
| Redes sociais | 1.000 | 0,560336 |
| Web | 1.000 | 0,901912 |
| **Total** | **6.000** | **0,780887** |

O contraste entre jornalístico e literatura/redes sociais é o resultado mais
útil deste experimento: o classificador atual captura bem marcas presentes nos
domínios formais mais próximos do treino, mas não mantém a mesma robustez em
registro literário ou informal.

## Reprodução

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_ptbrvid_external
$env:HF_HOME = "$PWD\.hf_cache"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m scripts.evaluate_ptbrvid_external
```

O arquivo externo fica em `data/external/ptbrvid_valid_v1/` e o relatório
executável em `runs/ptbrvid_external_v1/`; ambos não entram no Git.
