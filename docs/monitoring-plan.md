# Plano de Monitorização e Resposta (Runbooks)

Este documento centraliza as métricas de observabilidade do pipeline de *Credit Scoring* e os procedimentos de resposta a incidentes (Runbooks), garantindo a estabilidade da infraestrutura e a fiabilidade do modelo.

## 1. Catálogo de Métricas (Etapa 3)

| Grupo | Métrica | Descrição |
|---|---|---|
| **Pipeline** | `pipeline_stage_status` | Status de sucesso (1), bloqueio (2) ou falha (0) de cada estágio do pipeline. |
| **Pipeline** | `pipeline_stage_duration_seconds` | Tempo total de execução do estágio. |
| **Infraestrutura** | `stage_cpu_seconds` | Tempo de CPU consumido pelo processo durante o estágio. |
| **Infraestrutura** | `stage_memory_rss_observed_bytes` | Maior RSS observado no início/fim do estágio; não é pico contínuo. |
| **Pipeline** | `pipeline_rows_processed` | Volume de linhas processadas com sucesso. |
| **Geral** | `up` | Status de disponibilidade dos alvos de *scrape* do Prometheus. |

## 2. Runbooks de Infraestrutura e Pipeline

### ContractViolation
- **Significa:** Um lote de dados de entrada quebrou regras estritas do contrato de dados e foi isolado em quarentena.
- **Investigar:** Consultar o dashboard de Qualidade de Dados no Grafana para identificar a regra quebrada. Ler o ficheiro gerado em `reports/contracts/` correspondente ao lote para ver os exemplos afetados.
- **Ação:** NÃO ignorar a quarentena. Notificar a equipa de engenharia de dados (data sources) sobre a quebra de padrão. O pipeline continuará a processar os lotes seguintes normalmente.

### PipelineStageFailed
- **Significa:** Um estágio específico do pipeline (ingestão, validação, etc.) falhou e devolveu um erro fatal.
- **Investigar:** Aceder ao Grafana, navegar para o Loki (pesquisando por `{batch_id="<lote>"}`) e consultar o estágio e o tipo do erro. Mensagens e stack traces de exceções não são exportados. Clicar no botão do *Tempo* no log para visualizar a cascata do trace e identificar o ponto exato da falha de hardware ou código.
- **Ação:** Corrigir o *bug* no código-fonte, criar um *pull request* com testes unitários, aprovar no CI e voltar a processar o lote falhado de forma isolada.

### MonitoringTargetDown
- **Significa:** Um dos alvos de scrape configurados não responde ao Prometheus. No modo core, somente Prometheus e Pushgateway são cobertos; o modo completo adiciona Loki, Tempo, Alloy e Alertmanager.
- **Investigar:** Aceder ao terminal do servidor e correr `docker compose ps` para verificar o status dos contentores. Ler os logs nativos do contentor afetado com `docker logs <nome_do_contentor>`.
- **Ação:** Tentar reiniciar o serviço isoladamente (`docker compose restart <serviço>`). Se falhar, verificar limites de memória do host (OOM Killer) ou configurações de rede corrompidas.

### PipelineStale
- **Significa:** O pipeline de *scoring* não processa um lote com sucesso há mais de 35 dias.
- **Investigar:** Verificar quando `make monitor` foi executado e consultar o resumo em `reports/monitoring/`. Não há Airflow obrigatório. Avaliar se o servidor de ingestão parou de enviar ficheiros.
- **Ação:** Reativar a automação de *scheduling* se estiver em pausa. Se a ausência de dados for intencional (ex: paragem de negócio), silenciar o alerta temporariamente no Alertmanager.

### FeatureDriftCritical
- **Significa:** Ocorreu um desvio de distribuição severo numa funcionalidade (PSI > 0.25 e KS > 0.10).
- **Investigar:** Abrir o relatório HTML do Evidently (gerado na Etapa 2) para a funcionalidade afetada e comparar os histogramas do lote atual contra os da Referência.
- **Ação:** Investigar com o negócio se a mudança no perfil do cliente é real (ex: nova campanha de marketing) ou se é um problema de extração de dados. O alerta `FeatureDriftWarning` estará silenciado automaticamente pelo Alertmanager devido a inibição.

### ModelPerformanceDrop
- **Significa:** A AUC realizada caiu mais que o limite configurado contra o controle fora do treino.
- **Investigar:** Verificar os painéis de *Drift* e *Performance* no Grafana e cruzar com os relatórios analíticos gerados no MLflow.
- **Ação:** Investigar qualidade dos rótulos, intervalo da queda e recomendação do gate. Retreinamento depende de avaliação humana; esse alerta não o executa.

## Integração e métricas do modelo

O [guia de execução](observability-integration.md) documenta comandos, modos
local/Docker, endereços, estados e métricas. O [gate](experiment-analysis.md)
é a fonte da recomendação de retreinamento, com confirmação pelo IC e cobertura
supervisionada. Os alertas consomem os resultados publicados do mesmo lote.

### FeatureDriftWarning

PSI na faixa de atenção. Comparar histogramas e ausências; não indica
necessidade de retreino isoladamente.

### ConceptDriftSuspected

Gate aponta degradação sem drift marginal confirmado. Investigar rótulos,
mudanças conjuntas e comparação pareada; o diagnóstico não prova causa econômica.

### LabelDrift

Prevalência dos rótulos mudou além do limite z configurado. Verificar maturação
dos desfechos e composição da população; mudança de P(Y) não prova P(Y|X).

### RetrainRecommended

Gate recomenda investigar e avaliar retreinamento com queda confirmada.
Consultar HTML, ICs, cobertura e run MLflow. Não iniciar promoção automática.
