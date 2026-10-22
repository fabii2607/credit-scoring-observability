# ADR 0001 — Estatísticas de drift de entrada

- Estado: adotada.
- Data: 21/10/2026.
- Implementação: `stats.py`, `drift.py`, `drift_report.py` e `params.yaml`.

## Contexto

Precisamos comparar cada lote à mesma referência, preservar interpretação por
feature e apresentar resultados consistentes em JSON, métricas e Evidently.
Uma mudança de distribuição não implica perda de capacidade preditiva.

## Decisão

Para numéricas, usamos PSI em dez faixas definidas pelos quantis da referência,
com deduplicação de cortes, faixas externas ilimitadas e piso de 1e-6 seguido
de renormalização. Valores do lote não mudam os cortes. Referência constante
recebe tratamento específico para detectar deslocamentos.

O PSI recebe faixa OK abaixo de 0,10, WARN de 0,10 até 0,25 e ALERT acima de
0,25. Drift numérico confirmado exige simultaneamente PSI > 0,25 e KS D > 0,10.
O p-valor KS é informativo; não decide o gate.

Para categóricas, usamos divergência Jensen–Shannon com log natural e suporte
unificado. O limite de confirmação é JS >= 0,10. Não aplicamos KS a categorias.

A proporção de ausências é avaliada à parte (diferença absoluta >= 0,05).
Amostras indisponíveis não são substituídas por zero. `drift_share` divide
features confirmadas por features analisáveis; score, target, identificadores
e metadados ficam fora desse denominador. São 23 features na execução final.

O score recebe PSI separado. O target recebe comparação de prevalência com
estatística z, sem usar esse resultado isolado como prova de concept drift.
Evidently usa os mesmos métodos numéricos/categóricos do núcleo; a apresentação
pode indicar a existência de pelo menos uma feature confirmada.

## Alternativas e consequências

KS apenas por p-valor tende a sinalizar diferenças pequenas em amostras grandes;
PSI isolado perde a checagem adicional de magnitude por KS. Os limites escolhidos
são convenções operacionais do experimento, não garantias universais.

Faixas por quantis facilitam comparação em escalas distintas, mas dependem da
referência. As verificações são marginais: dependências conjuntas podem mudar
sem alertas por feature. Dois meses estáveis não estimam uma taxa geral de
falsos positivos. Categorias, nulos, bins repetidos e referência constante
têm testes específicos.

## Evidências

Veja [estatística](../drift-statistics.md), [resultados](../evidencias/README.md)
e [interpretação](../experiment-analysis.md). Mudanças de limiar exigem nova
avaliação; não ajustar limites após observar os meses finais para obter o
padrão desejado.
