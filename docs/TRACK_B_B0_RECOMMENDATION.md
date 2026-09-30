# Track B — recomendação de fontes e protocolo B0

**Status:** proposta para aprovação. Nenhum dado novo do Track B foi baixado
ou inspecionado localmente durante esta etapa.

## Decisão proposta

| Papel | Fonte | Justificativa |
| --- | --- | --- |
| Desenvolvimento | WMT24++: pares `en-pt_BR` e `en-pt_PT` | Traduções e pós-edições humanas, variedades explicitamente identificadas, quatro domínios e identificador de documento. |
| Avaliação final | DSL-TL: partição oficial `test` | Textos jornalísticos naturais com anotação humana e partição de teste ainda não utilizada neste projeto. |

O objetivo do Track B não é otimizar novamente contra o FRMT: ele permanece
congelado como avaliação externa do Track A.

## Desenvolvimento: WMT24++

O WMT24++ disponibiliza, por segmento, os campos `lp`, `domain`,
`document_id`, `segment_id`, `is_bad_source`, `source`, `target` e
`original_target`. Para a tarefa proposta, a base será formada apenas pelos
pares `en-pt_BR` e `en-pt_PT`.

O protocolo será:

1. Registrar a revisão exata do conjunto no manifesto.
2. Remover registros com `is_bad_source=true`.
3. Usar `target`, que é a pós-edição humana indicada pela documentação.
4. Remover duplicatas e auditar sobreposição exata ou próxima com o
   desenvolvimento do Track A.
5. Separar treino e validação por `document_id`, nunca por segmento aleatório.
6. Reportar desempenho por domínio e também por pares da mesma frase em
   português brasileiro e europeu.

É um bom conjunto de desenvolvimento porque permite investigar domínio,
contexto documental e diferenças entre variedades. A limitação é importante:
trata-se de texto traduzido, portanto não substitui uma avaliação final em
texto originalmente produzido em português.

## Avaliação final: DSL-TL test

O DSL-TL reúne textos jornalísticos naturais anotados por humanos. Para o
português, o artigo descreve 4.953 exemplos divididos em 3.467 de treino, 991
de desenvolvimento e 495 de teste. Há três rótulos possíveis no protocolo:
PT-BR, PT-PT e `both/neither`.

No Track A foi usado apenas o arquivo público de desenvolvimento PT_dev. A
partição oficial `test` permanece não observada neste repositório. Na
avaliação final do Track B serão excluídos exemplos `both/neither`, pois a
tarefa atual é binária.

Risco residual: ainda é a mesma família de benchmark da validação DSL já vista
no Track A. Portanto, ela será tratada como uma partição oficial independente,
mas não como uma coleta completamente independente. Uma avaliação ainda mais
forte exigiria uma nova amostra de texto natural com anotação humana, decisão
operacional que não faz parte deste ciclo.

## Protocolo pré-registrado do Track B

1. Baixar somente WMT24++ nas duas variedades propostas e registrar versão,
   contagens e filtros.
2. Fazer auditoria de duplicação e de vazamento contra todos os conjuntos já
   usados.
3. Criar três separações por documento e executar B1.1--B1.3 sem consultar
   DSL-TL test.
4. Considerar B1.4 (fine-tuning completo) apenas depois dos baselines e com
   decisão explícita sobre hardware.
5. Selecionar um único candidato com base no desenvolvimento.
6. Baixar DSL-TL test somente após essa seleção; auditar sobreposição.
7. Executar uma única inferência final e congelar o resultado.

## Fontes

- [WMT24++ Dataset Card](https://huggingface.co/datasets/google/wmt24pp)
- [WMT24++ paper](https://aclanthology.org/2025.findings-acl.634.pdf)
- [Repositório oficial DSL-TL](https://github.com/LanguageTechnologyLab/DSL-TL)
- [Artigo DSL-TL](https://aclanthology.org/2024.lrec-main.882.pdf)
