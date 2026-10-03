# Handoff — Etapa 3: Observabilidade de Pipelines e Modelos

> Documento autocontido. Cole o arquivo inteiro no início de uma conversa com uma LLM. O porquê de cada decisão está em [`docs/guias/etapa-3-observabilidade-pedro.md`](../guias/etapa-3-observabilidade-pedro.md) (escrito para outro dataset; aqui está a versão adaptada ao código real).

## 1. Onde o projeto está

Tech Challenge Fase 4 (FIAP MLET): sustentação de um modelo de credit scoring (Lending Club) em lotes mensais. A Etapa 1 está pronta (dados, modelo, contrato) e a Etapa 2 (drift) está em andamento. **Esta etapa junta tudo num pipeline por estágios observável:** logs estruturados, métricas, MLflow, dashboard e alertas. Ela vale 20% da nota.

O que já existe e você usa:

| Peça | Onde | Notas |
|---|---|---|
| Logger | `logger.py`: `setup_logging(level, json_logs, log_file)`, `get_logger(__name__)`, `build_processors(...)` | Já mascara `customer_id` (`redact_identifiers`). Estenda a cadeia de processadores (trace_id, JSONL para o Loki) **sem mudar as assinaturas** |
| Contrato | `validate.py`: `validate_batch(df, batch_id) -> ValidationResult`, `enforce(df, result, source_path)`, `ContractViolationError` | `ValidationResult.violations[]` tem `rule`, `column`, `severity`, `count`, `layer`, `examples` (ids mascarados) |
| Modelo | `registry.load_model()` → `ScoringModel` com `threshold`, `version`, `score(df)` | `score(df)` aplica o contrato das features e acrescenta `score`/`prediction` |
| Métricas de modelo | `evaluate.classification_metrics`, `approval_rate` | — |
| Lotes | `data/production/<lote>.parquet` (Etapa 2, `make simulate`) | Antes disso: `make demo-contract` gera `corrupted.parquet`; para teste, `synthetic.make_model_batch` |
| Makefile | alvo `monitor` imprime "TODO (Etapa 3)" | Ligue ao seu `run_all` |
| CI | `.github/workflows/ci.yml` (ruff + pytest) | Acrescente a validação do compose e o job de DAGs quando existirem |
| `.env.example` | `MLFLOW_TRACKING_URI`, `PUSHGATEWAY_URL`, `OTEL_EXPORTER_OTLP_ENDPOINT` | Vazios = roda offline |

```powershell
uv sync; make setup; make prepare; make train
make demo-contract      # exit 2: o mesmo exit code que o seu run_batch deve devolver num lote bloqueado
```

## 2. O que o enunciado pede

- Centralizar **logs** de sucesso e falha nas etapas do pipeline (ingestão, validação, transformação).
- Integrar os resultados de drift da Etapa 2 às métricas.
- **Entregável:** configuração do MLflow **e/ou** dashboard (Prometheus + Grafana).
- Definir e documentar **métricas de saúde**, incluindo a estabilidade da infraestrutura (CPU, memória) e **alertas** de degradação.

## 3. Prioridade: núcleo primeiro

| Núcleo (obrigatório) | Extra (diferencial) |
|---|---|
| Pipeline por estágios com logs estruturados de sucesso e falha | Alertmanager com roteamento e inibição (recomendado) |
| Resultados de drift/performance da Etapa 2 viram métricas | Loki + Alloy (logs consultáveis no Grafana) |
| MLflow (um run por lote) **e/ou** Prometheus + Grafana via Pushgateway | OpenTelemetry + Tempo (trace por lote) |
| Métricas de saúde: status, latência, volume, CPU e memória por estágio | Airflow 3 com DAG mensal (o mais caro: deixe por último) |
| Regras de alerta visíveis no dashboard | MLflow Registry atrás do `load_model` |
| Catálogo de métricas documentado; `make monitor` sem Docker | — |

## 4. Decisões já tomadas

| Tema | Decisão |
|---|---|
| Estágios | `ingest → validate → score → drift → performance → gate → publish`, idempotentes, com handoff em disco em `data/interim/<lote>/` (`model_input.parquet` depois do contrato; JSON de cada estágio) |
| Estágio validate | `validate_batch` → **publica as métricas do contrato** → `enforce` (que grava o relatório, põe o lote em quarentena e lança `ContractViolationError`). Lote bloqueado = exit 2 e **não para os lotes seguintes** no `run_all` |
| Métricas | Pushgateway com **`pushadd_to_gateway`** (POST), gauges agrupados por `batch_id` (+ `stage`). Sem `PUSHGATEWAY_URL`, grava `reports/metrics/<lote>_*.prom` |
| Logs | structlog JSON em `logs/pipeline.jsonl` com `batch_id`, `stage`, `mlflow_run_id` (+ `trace_id`); `customer_id` mascarado (já no logger) |
| Privacidade | Nenhum `customer_id` em logs, métricas, spans ou notificações; spans só com contagens e status. **Teste automatizado de ponta a ponta** (a Etapa 4 audita) |
| Retenção (LGPD) | Prometheus `--storage.tsdb.retention.time=90d`; Loki `retention_period: 2160h`; Tempo `block_retention: 2160h`. É o plano de retenção de 90 dias **aplicado** na configuração |
| Alertas | Regras com labels `severity`/`team` e annotations `summary`/`runbook_url` = `docs/monitoring-plan.md#<nomedoalerta>` (âncora em minúsculas) |
| Compose | Profiles `core` (MLflow, Pushgateway, Prometheus, Grafana), `alerting`, `logs-traces`, `orchestration`; versões de imagem fixas |
| Python | O projeto usa **3.11** (`.python-version`): a imagem do Airflow e qualquer container do projeto devem usar 3.11 |
| Telemetria | Já desligada em `src/credit_scoring_observability/__init__.py` (MLflow, Evidently, GE, NannyML) |

**Nomes de métricas combinados:**

| Grupo | Métricas | Origem |
|---|---|---|
| Contrato | `contract_violations{rule,severity,layer}`, `batch_quarantined`, `data_completeness_ratio{feature}`, `batch_rows` | `ValidationResult` (Etapa 1) |
| Drift | `feature_psi{feature}`, `feature_ks_d{feature}`, `drift_share`, `score_psi`, `label_drift_z`, `multivariate_mmd_pvalue` | `DriftResult` (Etapa 2) |
| Modelo | `model_roc_auc`, `model_estimated_roc_auc`, `performance_estimation_gap`, `approval_rate`, `drift_gate_status` | `PerformanceResult`/`GateResult` (Etapa 2) |
| Retreino | `retrain_runs{outcome}`, `retrain_guardrail_passed{guardrail}` | `RetrainDecision` (Etapa 2, extra) |
| Pipeline | `pipeline_stage_status{stage}`, `pipeline_stage_duration_seconds{stage}`, `stage_cpu_seconds{stage}`, `stage_memory_peak_bytes{stage}`, `pipeline_rows_processed{stage}`, `pipeline_last_success_timestamp` | seus estágios |

Alertas esperados no mínimo: `ContractViolation`, `ContractWarnings`, `FeatureDriftCritical` (PSI > 0,25 **and on(batch_id, feature)** KS D > 0,10), `FeatureDriftWarning`, `LabelDrift`, `ModelPerformanceDrop`, `ConceptDriftSuspected`, `RetrainRecommended`, `PipelineStageFailed`, `PipelineStale`, `MonitoringTargetDown`.

## 5. O que você implementa

Módulos em `src/credit_scoring_observability/` (ex.: `metrics.py`, `stages.py`, `run_batch.py`, `run_all.py`, `tracing.py`) e a stack na raiz (`docker-compose.yml`, `monitoring/`).

```python
# metrics.py
class MetricsPublisher:
    def __init__(self, batch_id: str, stage: str | None = None): ...
    def set(self, name: str, value: float, **labels) -> None: ...   # gauge criado sob demanda
    def publish(self) -> None: ...  # .prom local + pushadd (falha de rede só vira warning)

# stages.py
def run_stage(name: str, batch_id: str, fn) -> dict   # status, duração, CPU e memória, mesmo na falha (finally)

# run_batch.py — python -m credit_scoring_observability.run_batch --batch 2026-03  (exit 2 = contrato)
# run_all.py   — make monitor: todos os meses; lote bloqueado não interrompe os seguintes
```

- **Antes da Etapa 2 terminar:** use stubs de `detect_drift`/`evaluate_performance` devolvendo dicts com os campos combinados (`etapa-2-drift.md`, seção 5) e troque pela implementação real quando chegar o `pacote-2`.
- **MLflow:** um run por lote, com params (batch_id, versão do modelo, threshold), métricas, os artefatos (relatório do contrato, HTML do Evidently, gate) e a tag `trace_id`. Para registrar o modelo no Registry, faça isso **atrás** do `registry.load_model` (mesma assinatura), sem mudar quem chama.
- **Dashboard:** gere o JSON por script (fonte da verdade) e provisione; status sempre com cor + texto; `batch_id` no eixo x.

## 6. Entregas (Pacote 3, tags `pacote-3` e `execucao-referencia`)

- `make monitor` rodando todos os meses sem Docker (métricas em `.prom`, logs em JSONL).
- `make up` (core) e, se feitos os extras, `make up-full`, com os endereços no README.
- Dashboard sem painel em "No data"; alertas da seção 4 disparando nos lotes certos.
- **Execução de referência** com a tag `execucao-referencia`: números congelados e capturas em `docs/images/`. Toda a documentação da semana 4 usa estes números.
- Teste automatizado: nenhum `customer_id` em `logs/pipeline.jsonl`, nos `.prom` e nos relatórios.
- Semana 4: catálogo de métricas e runbooks de pipeline/infraestrutura (`PipelineStageFailed`, `PipelineStale`, `MonitoringTargetDown`...), entregues à Etapa 4 para o `docs/monitoring-plan.md`.

## 7. Armadilhas

- **`push_to_gateway` (PUT) apaga o grupo inteiro:** o fechamento do lote some com as métricas do contrato. Use `pushadd_to_gateway`. Counters não acumulam entre processos: use gauges por lote.
- **structlog:** não chame `.bind()` no import (congela a configuração padrão); limpe as contextvars entre lotes, senão o `mlflow_run_id` do lote anterior aparece nos logs seguintes.
- **Imagem oficial do MLflow** não traz o `prometheus_flask_exporter`: `--expose-prometheus` derruba o servidor. Use uma imagem derivada.
- **Grafana:** o painel `alertlist` só mostra os alertas gerenciados pelo próprio Grafana; use uma tabela da métrica `ALERTS` do Prometheus. `sort_by_label` é experimental e dá "No data".
- **Loki/Alloy:** `trace_id` como *structured metadata*, não como label (cardinalidade); `allow_structured_metadata: true`.
- **Airflow 3:** imports em `airflow.sdk`/`airflow.providers.standard`; porta 8080 pode estar ocupada (use 8088); DAG pausada não executa runs disparados; **não edite `src/` ou `params.yaml` com a DAG rodando** (volumes montados ao vivo causaram falha por corrida). No CI: `pytest.importorskip("airflow.sdk")` e `--confcutdir=tests/dags`.
- **Windows:** o Git Bash converte caminhos de container (use `MSYS_NO_PATHCONV=1` no `docker compose exec`); configs montadas em Linux precisam de LF (o `.gitattributes` já força `eol=lf`).
- **Documente o que não é produção:** Grafana anônimo, Airflow sem login, métricas de lotes antigos que nunca expiram no Pushgateway.

## 8. Checklist de pronto

- [ ] `make monitor` sem Docker; lote corrompido com exit 2 sem interromper os outros.
- [ ] Métricas de contrato publicadas **antes** do `enforce` lançar a exceção.
- [ ] MLflow com um run por lote e/ou dashboard Grafana provisionado.
- [ ] CPU, memória, duração e status por estágio no dashboard.
- [ ] Alertas com `runbook_url` e disparando; retenção de 90 dias na config.
- [ ] Teste de ausência de `customer_id` em logs, métricas e relatórios.
- [ ] Execução de referência com a tag e as capturas; catálogo de métricas e runbooks entregues à Etapa 4.

## 9. Prompt sugerido para a LLM

```
Você vai me ajudar na Etapa 3 (Observabilidade) do Tech Challenge descrito no documento
acima, no repositório credit-scoring-observability (Python 3.11, uv, pacote
src/credit_scoring_observability). Siga as "Decisões já tomadas" e os nomes de métricas
sem alterá-los, e não mude as assinaturas de logger.py, validate.py e registry.py. Comece
pelo MetricsPublisher e pelo run_stage com testes (dados de synthetic.make_model_batch),
depois o run_batch com o estágio validate. Antes de cada passo, diga quais armadilhas da
seção 7 se aplicam.
```
