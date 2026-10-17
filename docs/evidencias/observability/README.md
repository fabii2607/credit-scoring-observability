# Evidências de integração da observabilidade

Execução acadêmica local em 17/10/2026 com modelo congelado: sete lotes de
10 mil registros e um lote corrompido de 2.100 linhas. Nenhum retreinamento.

- [Resultado agregado dos lotes](run-summary.json): sete sucessos e um bloqueio.
- [Verificação da stack](stack-check.json): seis alvos disponíveis, fontes e
  dashboard provisionados, métricas coincidentes, runs MLflow, logs e traces
  entregues para todos os oito lotes.
- [Recebimento pelo Alertmanager](alertmanager-check.json): alertas de entrada
  em março/abril e degradação/recomendação em maio–julho, além da quarentena.
- [Manifesto](manifest.json): origem local, transformações e hashes das cópias.

O teste de regras em `monitoring/prometheus/alert_rules.test.yml` também passou
com promtool. A suíte Python passou com 193 testes; após os últimos ajustes
de escopo do consolidado e status de publicação, os oito testes da integração
de observabilidade passaram novamente. Há avisos de depreciação em dependências.

As APIs foram verificadas, incluindo dashboard e datasources do Grafana.
A aparência do dashboard não foi revisada visualmente. Logs brutos, contratos
com exemplos e parquets não fazem parte desta seleção pública.

O MLflow foi acessado em http://localhost:5001 devido à porta 5000 ocupada;
a porta é configurável com `MLFLOW_PORT`. Comandos para reproduzir e limitações
estão no [guia de integração](../../observability-integration.md).
