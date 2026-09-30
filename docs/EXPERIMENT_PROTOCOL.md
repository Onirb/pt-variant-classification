# Protocolo experimental

## Meta

Maximizar desempenho de generalização para classificação de textos como pt-PT
ou pt-BR sem usar repetidamente o teste público como sinal de ajuste.

## Divisões

- **Teste final congelado:** a divisão `test` de `cc4051/pt_vid` (2.570
  exemplos). A baseline v1 já registra uma única medida de referência; nenhum
  candidato é escolhido a partir dela.
- **Validação de desenvolvimento:** 20% estratificado do `train` público, com
  `random_state=42`. Ela preserva domínio e formato próximos do teste final.
- **Treino de desenvolvimento:** os 80% públicos restantes mais as linhas
  inequivocamente rotuladas de `PT_train.tsv`.

## Regras

1. O TSV local nunca entra na validação de desenvolvimento.
2. Linhas com rótulo `PT` continuam excluídas e contabilizadas.
3. A seleção prioriza F1 macro; F1 por variante e tempo são reportados.
4. Apenas o vencedor da validação pode receber uma avaliação final no teste
   público e um artefato de demonstração.
5. Uma etapa posterior deve incluir avaliação cross-domain, pois acurácia pode
   refletir fonte ou tópico do texto, não somente a variante linguística.
