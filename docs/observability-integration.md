# Execução com observabilidade

`make monitor` processa os lotes do manifesto com o mesmo núcleo de
`make analyze`. Cada lote tem uma execução MLflow e um trace independente.
São instrumentados preparação da referência/controle, ingestão, validação,
score, drift, performance, comparação pareada, gate e relatórios.

## Execução local

Após preparar o modelo e gerar os lotes, execute:

```bash
make monitor
uv run python -m credit_scoring_observability.run_batch --batch 2026-06
uv run python -m credit_scoring_observability.run_all --batches 2026-01 2026-04 2026-06
```

Os entrypoints carregam `.env` sem sobrescrever variáveis já exportadas.
Endpoints vazios usam MLflow SQLite em `mlruns/tracking.db`, artefatos locais,
métricas em `reports/metrics/<lote>.prom` e logs em `logs/pipeline.jsonl`.
Spans locais fornecem correlação de logs; sem endpoint não são exportados.
O comando de um lote atualiza seus artefatos individuais. O comando de vários
lotes também consolida a seleção em `reports/experiment/` e
`reports/performance/summary.*`.

Para forçar o modo local independentemente do `.env`:

```bash
PUSHGATEWAY_URL= MLFLOW_TRACKING_URI= OTEL_EXPORTER_OTLP_ENDPOINT= OTEL_EXPORTER_OTLP_TRACES_ENDPOINT= make monitor
```

## Stack Docker

```bash
make up-full
PUSHGATEWAY_URL=http://localhost:9091 MLFLOW_TRACKING_URI=http://localhost:5000 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_TRACES_ENDPOINT= make monitor
```

| Serviço | Endereço | Uso |
|---|---|---|
| Grafana | http://localhost:3000 | Dashboard “Observabilidade - Credit Scoring”, provisionado automaticamente |
| Prometheus | http://localhost:9090 | Consultas e estado dos alertas |
| MLflow | http://localhost:5000 | Experimento `credit_scoring_monitoring`, métricas e HTML/JSON |
| Alertmanager | http://localhost:9093 | Alertas agrupados |
| Pushgateway | http://localhost:9091 | Último snapshot de cada lote |
| Tempo | http://localhost:3200/ready | Readiness; traces consultados no Grafana |
| Loki | http://localhost:3100/ready | Readiness; logs consultados no Grafana |

`make up` inicia apenas o núcleo (métricas, Grafana e MLflow). `make up-full`
inclui logs, traces e alertas, com os alvos adicionais de scrape configurados em
`prometheus-full.yml`. A stack é um ambiente acadêmico local.

Se a porta 5000 estiver ocupada (comum no macOS), inicie a stack com
`MLFLOW_PORT=5001 make up-full` e use `MLFLOW_TRACKING_URI=http://localhost:5001`
na execução. O dashboard segue em http://localhost:3000.

O dashboard versionado é gerado por `uv run python scripts/generate_dashboard.py`.
As fontes Prometheus/Loki/Tempo são provisionadas; o campo `trace_id` nos logs
oferece navegação ao trace. Trace IDs são gerados pelo OpenTelemetry e associados
ao run MLflow por tag, sem forçar o run ID a virar trace ID.

O servidor MLflow recebe os artefatos pelo proxy HTTP; caminhos do container não
precisam existir no computador cliente. Experimentos antigos criados com outra
configuração de armazenamento não são migrados automaticamente.

## Resultados e falhas

- Código 0: lotes analisados e publicação sem erros detectados.
- Código 2: houve bloqueio por contrato, sem falhas inesperadas/de entrega.
- Código 1: houve falha inesperada ou erro detectado na entrega da telemetria.
- O processamento dos lotes seguintes continua após bloqueio ou falha.
- O lote bloqueado não recebe predições, drift, AUC ou gate.

`reports/monitoring/<lote>/run.json` registra estágios, run MLflow e resultado
de publicação; `reports/monitoring/summary.json` consolida o monitoramento.
“Publicação sem erros detectados” não garante disponibilidade posterior dos
serviços; o OpenTelemetry também pode registrar falhas assíncronas de exportação.

Para demonstrar quarentena, gere novamente os lotes com `make simulate` e execute
`uv run python -m credit_scoring_observability.run_all --include-corrupted`.
O arquivo corrompido é movido à quarentena; uma segunda execução exige recriá-lo.

## Semântica das métricas

`pipeline_batch_status`: -1 executando, 0 falha, 1 sucesso, 2 bloqueado.
`pipeline_stage_status`: 0 falha, 1 sucesso, 2 bloqueado.
`gate_status`: 0 OK, 1 WARN, 2 ALERT.
`gate_retrain_recommended`: 1 quando o gate recomenda avaliar retreinamento.

AUC atual, baseline, queda, ICs e limite são publicados separadamente como
`model_roc_auc`, `model_reference_roc_auc`, `model_auc_drop`,
`model_auc_drop_ci_low/high` e `model_auc_drop_limit`.
`feature_drift_confirmed` usa o resultado calculado para numéricas e categóricas.
Métricas indisponíveis são omitidas, não convertidas em zero.

`pipeline_rows_processed` conta as linhas pontuadas do lote mensal; não soma
referência, controle e comparação. `stage_memory_rss_observed_bytes` é o maior
RSS observado no início/fim do estágio, não o pico medido continuamente.
Duração usa relógio monotônico; CPU usa tempo de processo.

O publisher mantém registry próprio e grava o arquivo local antes do envio.
O PUT substitui o grupo do lote no Pushgateway, removendo séries antigas na
reexecução. No início, um snapshot de estado “executando” invalida resultados
anteriores; alertas de modelo exigem lote bem-sucedido.
Os snapshots de outros lotes permanecem disponíveis.

Contratos enviados ao MLflow contêm somente contagens/regras agregadas; exemplos
individuais e parquets não são enviados. Erros em logs/spans registram o tipo da
exceção, sem mensagens/tracebacks que possam conter valores de clientes.

## Referências

- [Análise e política do gate](experiment-analysis.md).
- [MLflow: servidor e proxy de artefatos](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
- [OpenTelemetry: exportação Python](https://opentelemetry.io/docs/languages/python/exporters/).


## Verificações reproduzíveis

Após terminar o monitoramento e aguardar o scrape/ingestão dos serviços:

```bash
uv run python scripts/check_observability.py --mlflow-uri http://localhost:5000
docker run --rm --entrypoint /bin/promtool   -v "$PWD/monitoring/prometheus:/etc/prometheus:ro"   prom/prometheus:v2.48.0 test rules /etc/prometheus/alert_rules.test.yml
```

O primeiro comando compara métricas com os JSONs, verifica runs MLflow,
dashboard/fontes provisionados, logs correlacionados e traces entregues.
Usa as portas locais padrão dos serviços; `--mlflow-uri` aceita a alternativa
5001. Grava um resumo agregado em `reports/monitoring/stack-check.json`.
O teste de regras usa séries controladas, incluindo métricas antigas de lote
bloqueado que devem ser ignoradas. Os alertas têm espera de um minuto;
a avaliação do Prometheus e o agrupamento do Alertmanager acrescentam latência.
