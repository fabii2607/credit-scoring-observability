# ADR 0002 — Controle de performance, comparação pareada e gate

- Estado: adotada.
- Data: 21/10/2026.
- Implementação: `performance.py`, `comparison.py`, `gate.py`.

## Contexto

A referência de distribuição vem do treino e não é uma baseline independente
para performance. Além disso, meses disjuntos podem ter AUCs diferentes antes
de qualquer intervenção.

## Decisão

Usamos 30 mil registros reservados fora do ajuste e disjuntos dos meses como
controle de performance. Queda = AUC do controle - AUC do lote.
O limite é uma diferença absoluta de 0,05, não queda relativa de 5%.

Também comparamos os mesmos registros originais/alterados, verificando
identidade, ordem e colunas permitidas. Originais/referência e
alterados/referência identificam diferenças prévias. Não subtraímos PSIs para
atribuir causalidade: PSI não é aditivo.

Intervalos percentis de 95% usam 200 reamostragens e seed 42:
AUC atual estratificada; diferença contra controle com amostragens independentes;
diferença pareada com índices comuns e estratificação pelo rótulo original.
Não incluem incerteza de retreinamento nem correção de múltiplas comparações.

O gate usa métricas e cobertura, sem receber o cenário sintético. Exige pelo
menos 500 rótulos e 90% de cobertura tanto no controle quanto no lote para
avaliar performance. Recomenda avaliar retreinamento somente se a queda
observada e todo seu IC estiverem acima do limite, com pelo menos cem
reamostragens configuradas. Sem confirmação, orienta investigação.

Drift de entrada sem queda acima do limite recomenda reforçar monitoramento.
AUC degradada sem drift marginal sugere investigar concept drift, rótulos e
dependências conjuntas; não comprova o mecanismo. Falta de rótulos é falta de
evidência, não sucesso do modelo. O gate não treina nem promove modelos.

## Consequências e limites

Brier e aprovação podem mudar mesmo com AUC estável; acompanhar essas métricas.
O controle provém do teste já usado na avaliação inicial. É independente do
ajuste, mas não constitui validação externa nova. A simulação mistura safras
históricas; seus meses não representam uma avaliação temporal real.

Limites de rótulos/reamostragens são guardrails operacionais, não um cálculo
formal de poder. Bootstrap é condicional às amostras observadas. A política
não valida economicamente o threshold de crédito 0,20 nem substitui análise
de custo, fairness ou revisão humana.

## Evidências

Veja [política detalhada](../experiment-analysis.md),
[model card](../model-card.md) e [linha do tempo](../incident-timeline.md).
A mesma recomendação é usada no HTML, JSON, MLflow e métricas dos alertas.
