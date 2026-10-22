# Evidências de observabilidade

Execução de 2026-10-21. Os oito lotes foram verificados pelas APIs locais.

- [Resumo dos lotes](run-summary.json).
- [Métricas, MLflow, dashboard, logs e traces](stack-check.json).
- [Recebimento de alertas pelo Alertmanager](alertmanager-check.json).
- [Manifesto](manifest.json), subordinado ao [manifesto principal](../manifest.json).

Foram verificados 6 alvos disponíveis e o provisionamento do
Grafana. Os valores de AUC/gate coincidem com os relatórios. Bloqueio não gerou
predições e não interrompeu os demais meses. Alertas têm atraso de avaliação;
os snapshots podem registrar estados em instantes diferentes.

Verificação pelas APIs, sem revisão visual final do dashboard.
MLflow foi acessado na porta 5001 porque 5000 estava ocupada.
Veja [reprodução e limites](../../observability-integration.md).
