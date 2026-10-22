# Plano de monitoramento e resposta

Runbooks correspondentes às regras em `monitoring/prometheus/alert_rules.yml`.
Os títulos preservam os nomes dos alertas e suas âncoras em minúsculas.
A stack envia alertas a um receptor local; comunicações externas não são automáticas.

## Catálogo de métricas

| Métrica | Interpretação |
|---|---|
| pipeline_batch_status | -1 executando, 0 falha, 1 sucesso, 2 bloqueado |
| pipeline_stage_status | 0 falha, 1 sucesso, 2 bloqueado |
| pipeline_stage_duration_seconds | Duração monotônica por estágio |
| stage_cpu_seconds | CPU de processo por estágio |
| stage_memory_rss_observed_bytes | Maior RSS no início/fim, não pico contínuo |
| batch_rows / pipeline_rows_processed | Linhas lidas / pontuadas do lote mensal |
| pipeline_last_success_timestamp | Timestamp de sucesso por lote; freshness considera o maior |
| batch_quarantined / contract_violations | Bloqueio e contagens por regra/severidade/camada |
| feature_psi / feature_ks_d / feature_js | Distância por feature e método |
| feature_drift_confirmed / drift_share | Confirmação por feature e proporção entre analisáveis |
| feature_missing_rate_delta | Diferença da proporção de nulos |
| score_psi | Mudança na distribuição dos scores |
| model_roc_auc / model_reference_roc_auc | AUC do lote / controle |
| model_auc_drop / model_auc_drop_ci_low / model_auc_drop_ci_high | Queda e IC contra controle |
| model_auc_drop_limit | Limite absoluto configurado para a queda |
| model_pr_auc / model_brier / approval_rate | Métricas complementares |
| model_default_rate / label_z / label_z_limit | Prevalência e sinal de mudança |
| gate_status | 0 OK, 1 WARN, 2 ALERT |
| gate_retrain_recommended / gate_concept_suspected | Recomendação/suspeita calculada pelo gate |
| telemetry_delivery_status | 1 sem erros detectados até a publicação; 0 com erro detectado |
| up | Disponibilidade somente dos alvos configurados de scrape |

Métricas indisponíveis são omitidas. Labels são técnicos: lote, estágio, feature,
regra/severidade/camada. Nunca usar identificador de cliente como label.
Valores de modelo são snapshots por lote; não são contadores acumulados.
Lote bloqueado não publica AUC ou gate. Falha no envio remoto preserva .prom local,
mas precisa ser investigada. Verificar resumo de entrega, além do status da análise.

## Runbooks

### ContractViolation

- **Significa:** O contrato bloqueou o lote; batch_quarantined=1. Não houve predição.
- **Investigar:** Consultar o contrato local e o resumo agregado no MLflow; identificar regra e origem dos dados.
- **Ação:** Corrigir a origem e reprocessar o lote completo. Não remover o bloqueio nem usar dados de quarentena como treino.

### PipelineStageFailed

- **Significa:** Um estágio teve falha inesperada (status 0). Bloqueio esperado usa status 2.
- **Investigar:** Consultar reports/monitoring/<lote>/run.json, tipo do erro nos logs Loki e trace do estágio no Tempo.
- **Ação:** Corrigir o problema, validar com teste e reexecutar o lote. Mensagens/tracebacks com valores de clientes não são exportados.

### MonitoringTargetDown

- **Significa:** Um alvo de scrape configurado está indisponível. Core cobre Prometheus/Pushgateway; full acrescenta Loki, Tempo, Alloy e Alertmanager.
- **Investigar:** Consultar Targets no Prometheus, docker compose ps e logs do serviço. Isso não cobre automaticamente todos os processos, como MLflow.
- **Ação:** Recuperar o serviço e confirmar o scrape. Não interpretar ausência de série como valor zero.

### PipelineStale

- **Significa:** Não existe evidência de sucesso ou o maior timestamp de sucesso está há mais de 35 dias.
- **Investigar:** Verificar última execução de make monitor, relatórios e ambiente. Não há agendador/Airflow obrigatório.
- **Ação:** Executar ou corrigir o processamento conforme a periodicidade acordada. Se a ausência for intencional, registrar o motivo antes de silenciar.

### FeatureDriftCritical

- **Significa:** Uma feature tem drift confirmado: numérica PSI > 0,25 e KS D > 0,10; categórica JS >= 0,10.
- **Investigar:** Abrir o HTML Evidently do lote, verificar histograma, nulos, contrato e versões originais/alteradas.
- **Ação:** Investigar composição dos clientes e origem dos dados; comparar AUC, Brier e aprovação. Drift isolado não autoriza retreinamento.

### FeatureDriftWarning

- **Significa:** PSI da feature entre 0,10 e 0,25, inclusive.
- **Investigar:** Verificar tendência, tamanho da amostra e score PSI no HTML e nas métricas. O alerta crítico da mesma feature pode inibir o warning.
- **Ação:** Reforçar monitoramento e investigar persistência. Não classificar PSI em WARN como drift confirmado.

### ModelPerformanceDrop

- **Significa:** Queda absoluta de AUC contra o controle maior que o limite configurado, em lote analisado com sucesso.
- **Investigar:** Consultar AUC do controle/lote, IC da queda, cobertura de rótulos, maturação dos desfechos e comparação pareada.
- **Ação:** Investigar degradação e consultar o gate. Esse alerta observa a queda pontual; confirmação para avaliar retreinamento exige critérios adicionais.

### ConceptDriftSuspected

- **Significa:** Gate aponta degradação sem drift marginal confirmado; é suspeita operacional.
- **Investigar:** Verificar rótulos, dependências conjuntas, originais e análise de sensibilidade. PSI baixo não prova estabilidade de P(Y|X).
- **Ação:** Investigar mecanismos possíveis e revisão de dados/modelo. Não apresentar o diagnóstico como causalidade econômica.

### LabelDrift

- **Significa:** Mudança de prevalência ultrapassou o limite z configurado, em lote bem-sucedido.
- **Investigar:** Comparar taxas de default, maturação/cobertura dos rótulos e segmentos. Conferir se foi uma intervenção sintética conhecida.
- **Ação:** Investigar mudança de P(Y); não concluir mudança de P(Y|X) só com esse alerta.

### RetrainRecommended

- **Significa:** Gate recomenda avaliar retreinamento: queda e IC acima do limite, com quantidade/cobertura de rótulos e reamostragens suficientes.
- **Investigar:** Abrir a recomendação no HTML, JSON gate e run MLflow; revisar ICs, dados, Brier, aprovação e limites da simulação.
- **Ação:** Encaminhar à avaliação humana. Um eventual challenger precisa de validação própria; nenhum treino ou promoção é disparado automaticamente.

## Referências

[Execução e serviços](observability-integration.md),
[política do gate](experiment-analysis.md),
[linha do tempo](incident-timeline.md) e
[evidências](evidencias/README.md).
