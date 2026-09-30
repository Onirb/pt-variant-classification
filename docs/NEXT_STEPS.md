# Estado final e próximos passos

## Ciclos concluídos

- Track A: baselines, seleção interna, artefato e CLI; FRMT final com F1 macro
  0,608864. Encerrado em [TRACK_A_CLOSED.md](TRACK_A_CLOSED.md).
- Track B B0–B4: desenvolvimento WMT24++, três sementes, ajuste final de 318
  passos, importação verificada e avaliação externa única.
- Resultado externo B: F1 macro 0,762398, 436 textos; recall PT-PT 0,970803 e
  PT-BR 0,675585. A meta de equilíbrio entre classes não foi atingida.
- Protocolos, resultados, fontes, limites e custos registrados; 35 testes
  offline sintéticos independentes dos artefatos desta máquina.
- Código e documentação preparados para publicação. Modelos modernos, caches,
  dados externos, previsões detalhadas e credenciais permanecem fora do Git.

Consulte [TRACK_B_B4_RESULT_V1.md](TRACK_B_B4_RESULT_V1.md) e
[TESTING_AND_REPRODUCTION.md](TESTING_AND_REPRODUCTION.md).

## Manutenção e portfólio

1. Validar instalação em outros sistemas/versões antes de anunciar suporte.
2. Opcionalmente configurar integração contínua para a suíte sintética.
3. Decidir separadamente como distribuir os pesos modernos, com revisão das
   licenças, armazenamento e documentação de versão. Não estão no GitHub.
4. Se desejado, criar uma interface de demonstração sobre o CLI existente;
   nenhuma nova interface/API foi prometida ou implementada neste ciclo.
5. Manter exemplos de saída com avisos de truncamento e escores não calibrados.

## Nova pesquisa: exige outra decisão

O teste final DSL-TL, o FRMT e os demais conjuntos já observados não podem
orientar novas escolhas de modelo, limiar ou checkpoint. Qualquer novo ciclo
de melhoria requer outro protocolo de validação e **outro teste não observado**.
É permitido analisar erros descritivamente, sem reutilizar essas métricas para
seleção. A assimetria e a diferença entre tradução e texto natural são questões
abertas, não problemas resolvidos pela publicação.
