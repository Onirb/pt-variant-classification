# Ponto de controle — Track A encerrado

**Data de encerramento:** 30/09/2026

**Estado:** concluído e congelado.

## O que foi fechado

1. Auditoria do projeto histórico e reconstrução de um fluxo reproduzível.
2. Seleção interna de BERTimbau mean-pooling + LinearSVC, com média de F1
   macro 0,970314 em três sementes predefinidas.
3. Ajuste único do artefato final em 14.764 textos de desenvolvimento, com
   manifesto, hash do corpus e CLI local.
4. Avaliações externas documentadas em DSL-TL, PtBrVId e FRMT.
5. Avaliação final FRMT `test`, com 5.232 textos, zero sobreposição normalizada
   e F1 macro 0,608864.

## Decisões congeladas

- O FRMT não pode ser usado novamente para escolher modelo, limiar, técnicas de
  pré-processamento ou calibração.
- O artefato em `runs/bertimbau_final_v1/` não será sobrescrito.
- Resultados do Track A devem ser apresentados com os limites de generalização
  descritos em [`FRMT_EXTERNAL_FINAL_V1.md`](FRMT_EXTERNAL_FINAL_V1.md).
- Qualquer melhoria passa a pertencer ao Track B e deve ter outro protocolo,
  outros dados de desenvolvimento e uma avaliação final ainda não observada.

## Ponto de retomada

Para retomar desenvolvimento sem perder o contexto, comece por
[`TRACK_B_PLAN.md`](TRACK_B_PLAN.md). O Track A permanece uma linha de base
publicável e reprodutível, não uma fonte de ajuste adicional.
