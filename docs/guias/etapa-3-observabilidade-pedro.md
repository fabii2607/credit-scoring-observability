# Guia da Etapa 3 — Observabilidade de Pipelines e Modelos (Pedro)

> Documento de direcionamento e contexto para LLM. Cole o arquivo inteiro no início da conversa.

## 0. Posição no revezamento

**Você é o 3º: semana 3.** Recebe o **Pacote 1** (Fabi) e o **Pacote 2** (Helio).

- **Antes do bastão (semanas 1 e 2), adiante sem depender de ninguém:**
  - a stack Docker inteira (profiles, Prometheus, Alertmanager, Grafana, Loki, Tempo, Alloy) alimentada por um script que empurra **métricas falsas** com os nomes definidos;
  - logger, `MetricsPublisher` e tracing;
  - os 7 estágios chamando **stubs** que devolvem JSONs no formato combinado;
  - regras de alerta e o gerador do dashboard;
  - o Airflow rodando a DAG contra os stubs.

  Com os pacotes, você só troca os stubs pelas funções reais.
- **Não depende de quem vem depois:** retenção de 90 dias e "nenhum identificador em logs, spans e notificações" já estão na seção 3.
- **Fim da semana 3: execução de referência.** Com tudo integrado, rode a execução limpa (MLflow zerado, `make train`, a DAG com os 7 meses, os lotes de demo), tire as capturas para `docs/images/` e marque a tag `execucao-referencia`. **Os números dessa execução ficam congelados**: toda a documentação da semana 4 usa só eles.
- **Entrega o Pacote 3 para o Erick** (seção 6) junto com a execução de referência.
- **Semana 4 (depois de entregar):** documente o que construiu:
  - as ADRs de orquestração/Pushgateway e de Alloy/Loki/Tempo;
  - o **catálogo de métricas**;
  - os runbooks de pipeline e infraestrutura (formato em `docs/guias/README.md`).

  Faça também a revisão cruzada.

## 1. Contexto do projeto

Tech Challenge da Fase 4 do MLET (FIAP). Uma fintech tem um modelo de credit scoring em produção e suspeita de **degradação silenciosa** por mudanças na economia. O grupo constrói a **camada de sustentação**: contratos de dados, simulação e detecção de drift, observabilidade e governança LGPD.

- **Pipeline batch mensal**, sem API. Lotes 2026-01 a 2026-07, processados um por vez.
- **Stack da etapa:** structlog, prometheus-client + **Pushgateway**, Prometheus, **Alertmanager**, Grafana, **Loki + Grafana Alloy**, **OpenTelemetry + Tempo**, MLflow 3 (servidor), **Airflow 3**, Docker Compose.
- **Critérios de avaliação:** Validação 25% · Drift 25% · **Observabilidade 20%** · Governança 15% · Vídeo 15%.

**Sua etapa vale 20% e é a que integra tudo:** você transforma os módulos da Fabi e do Helio num pipeline por estágios, instrumentado, orquestrado e visível num dashboard com alertas.

## 2. O que o enunciado pede na Etapa 3

- **Centralizar logs e métricas:** integrar os logs de execução do pipeline (sucesso e falhas na ingestão e transformação) e os resultados dos testes de drift da Etapa 2.
- **Rastreamento/monitoramento:** MLflow para o ciclo de vida do modelo e dos experimentos, **ou** exportar as métricas de infraestrutura e pipeline para um dashboard interativo (Datadog, cloud nativa ou open source).
- **Definir e documentar as métricas** de saúde operacional (status das tarefas, latência, volume, quantidade de alertas de falha ou drift).
- **Entregável:** dashboard ou ambiente de rastreamento evidenciando a saúde completa do pipeline de dados, a **estabilidade da infraestrutura** e os **alertas de degradação** do modelo.

Opcional citado no enunciado: `prometheus-client` para métricas (integração com a Fase 3).

## 2.1 Prioridade: núcleo primeiro, extras depois

O objetivo é **cumprir o enunciado**. Faça o núcleo inteiro antes de qualquer extra. Se o prazo apertar, os extras são cortados sem prejudicar o requisito.

| Núcleo (obrigatório) | Requisito atendido |
|---|---|
| Pipeline por estágios com **logs estruturados** de sucesso e falha (ingestão, validação, transformação) | Etapa 3 ("centralizar logs") |
| Resultados de drift da Etapa 2 integrados às métricas | Etapa 3 |
| **MLflow** (um run por lote) **e/ou** dashboard (Prometheus + Grafana via Pushgateway) | Etapa 3 (entregável) |
| Métricas de saúde: status, latência, volume, alertas de falha/drift, **CPU e memória** | Etapa 3 ("estabilidade da infraestrutura") |
| **Alertas de degradação** visíveis (regras do Prometheus no dashboard) | critério Observabilidade 20% |
| Catálogo de métricas documentado | Etapa 3 ("definir e documentar métricas") |
| `make monitor` rodando todos os meses sem Docker | CI e fallback |

| Extra (diferencial) | Custo |
|---|---|
| Alertmanager com roteamento e inibição | baixo — recomendado |
| Loki + Alloy (logs consultáveis no Grafana) | médio |
| OpenTelemetry + Tempo (traces por lote) | médio |
| Airflow 3 com DAGs e retreino disparado | alto — o mais caro; deixe por último |

## 3. Decisões já tomadas (não reabrir sem o grupo)

| Tema | Decisão | Motivo |
|---|---|---|
| Estrutura do pipeline | 7 estágios idempotentes (`ingest → validate → score → drift → performance → gate → publish`), com handoff em disco em `data/interim/<lote>/` | O mesmo código roda em sequência (Makefile) ou como tasks separadas do Airflow |
| Métricas | **Pushgateway** com `pushadd` (POST), agrupado por `batch_id` (+ `stage`) | O job batch termina antes de qualquer scrape; `pushadd` evita que o fechamento do lote apague as métricas do contrato |
| Sem stack | Sem `PUSHGATEWAY_URL`, grava `reports/metrics/<lote>_*.prom`; sem `OTEL_EXPORTER_OTLP_ENDPOINT`, o tracing é no-op | O CI e quem não roda Docker executam o pipeline inteiro |
| Logs | structlog JSON em `logs/pipeline.jsonl` com `batch_id`, `stage`, `mlflow_run_id`, `trace_id`; **`customer_id` mascarado por processador** | Correlação + LGPD |
| Coletor | **Grafana Alloy** (logs → Loki, OTLP → Tempo) | O Promtail chegou ao fim de vida em março de 2026 |
| Alertas | Regras no Prometheus (15) e no Loki (1), com labels `severity`/`team` e annotations `summary`/`runbook_url`; Alertmanager com roteamento e **inibição** | Alerta sem runbook não é acionável |
| Orquestração | Airflow 3 com uma task por estágio, catchup mensal, `max_active_runs=1`, retreino como DAG disparada que é **esperada** | Status, retry e duração por tarefa; o mês seguinte já usa o modelo novo |
| Compose | Profiles `core`, `alerting`, `logs-traces`, `orchestration`; versões de imagem fixas | Cada um sobe só o necessário (a stack completa usa 6–8 GB de RAM) |
| Dashboard | Gerado por script Python (fonte da verdade) e provisionado | Consistência de cores e thresholds; fácil de alterar |
| Privacidade | `customer_id` mascarado nos logs; spans só com contagens e status; notificações só com labels agregados; telemetria de GE, NannyML, MLflow e Evidently desligada antes do import | LGPD; o Erick audita no fim |
| Retenção | Loki `retention_period: 2160h`, Tempo `block_retention: 2160h`, Prometheus `--storage.tsdb.retention.time=90d` | Plano de retenção de 90 dias aplicado na configuração |

## 4. O que precisa existir no fim (números de referência)

> As quantidades abaixo são do projeto anterior. O que importa é que cada tipo de problema (contrato, drift de feature, queda de performance, falha de estágio) gere um alerta visível.

- **Execução pelo Airflow:** 7 runs com sucesso; `credit_retrain` disparada só em junho.
- **Alertas disparando** depois de `make monitor` + lote corrompido + `joint_drift`:
  - `FeatureDriftCritical` ×3 (M4);
  - `FeatureDriftWarning` ×3 (M3);
  - `MultivariateDrift` (M3, M4, joint);
  - `LabelDrift` (M5–M7);
  - `ModelPerformanceDrop` e `ConceptDriftSuspected` (M6);
  - `ContractViolation` e `PipelineStageFailed` (corrupted).
- **Inibição comprovada:** `RetrainRecommended` do M6 suprimido por `ModelPerformanceDrop`.
- **Loki:** ~276 linhas de log do pipeline + as notificações de alerta, com `batch_id` como label.
- **Tempo:** um trace por lote (`batch <lote>` com 7 spans `stage.*`).
- **Um lote leva ~5 s**; os 7 meses + retreino, ~30 s fora do Airflow.

## 5. Passo a passo

1. **Logger** (`logger.py`):
   - structlog com `merge_contextvars`, nível, timestamp ISO;
   - processador `redact_identifiers` (mascara `customer_id`);
   - processador que injeta `trace_id`/`span_id` do span ativo;
   - "tee" para `logs/pipeline.jsonl`;
   - `get_logger` **lazy** (sem `.bind()` no import).
2. **`MetricsPublisher`** (`metrics.py`): um `CollectorRegistry` por grupo; `set(nome, valor, **labels)` cria gauges sob demanda; `publish()` grava o `.prom` local e faz `pushadd_to_gateway`, com falha de rede só como warning.
3. **Estágios** (`stages.py`) com `BatchContext` (workspace, `read_json`/`write_json`, `mlflow_run_id`). Cada estágio chama os módulos dos colegas:
   - `ingest`: lê o lote e abre o run do MLflow;
   - `validate`: GE raw + imputação + Pandera; publica as métricas de contrato **antes** de falhar;
   - `score`: modelo `@production`; pontua o lote e a Referência de teste;
   - `drift`, `performance`, `gate`: funções do Helio;
   - `publish`: métricas de drift e modelo + tags, métricas e artefatos no MLflow.
4. **`run_stage()`:**
   - span `stage.<nome>`;
   - `bind_contextvars(batch_id, stage, mlflow_run_id)`;
   - `psutil` para CPU e pico de memória;
   - publica `pipeline_stage_status`, `pipeline_stage_duration_seconds`, `stage_cpu_seconds`, `stage_memory_peak_bytes` e `pipeline_rows_processed`, **inclusive na falha** (`finally`).
5. **Orquestradores sem Docker:**
   - `run_batch.py`: um lote; exit 2 = contrato violado;
   - `run_all.py`: todos os meses; lote bloqueado não para os seguintes; retreino antes do mês seguinte.
6. **Tracing** (`tracing.py`):
   - `TracerProvider` + exportador OTLP/HTTP só se houver endpoint;
   - contexto salvo como W3C `traceparent` no workspace, para que tasks em processos separados continuem no mesmo trace;
   - `trace_id` vira tag do run do MLflow.
7. **docker-compose.yml:**
   - MLflow (imagem derivada com `prometheus-flask-exporter` e `--expose-prometheus`);
   - Pushgateway e Prometheus (retenção de 90 dias);
   - Grafana (datasources Prometheus, Loki com *derived field* do `trace_id` para o Tempo, Tempo com *trace to logs*, Alertmanager);
   - Alertmanager e o receiver de webhook (Python stdlib, grava em `logs/alerts.jsonl`);
   - Loki (retenção de 90 dias), Tempo e Alloy.
8. **Regras de alerta:**
   - dados: `ContractViolation`, `ContractWarnings`;
   - drift: `FeatureDriftCritical` (PSI > 0,25 **and on(batch_id, feature)** KS D > 0,10), `FeatureDriftWarning`, `MultivariateDrift`, `LabelDrift`;
   - modelo: `ModelPerformanceDrop`, `EstimatedPerformanceDrop`, `ConceptDriftSuspected`, `RetrainRecommended`, `ChallengerRejected`;
   - pipeline: `PipelineStageFailed`, `PipelineStale`, `PipelineStageSlow`;
   - infraestrutura: `MonitoringTargetDown`;
   - Loki: `PipelineErrorLogsSpike`.
9. **Alertmanager:**
   - agrupamento por `alertname` + `batch_id`;
   - critical → `oncall`, platform → `platform`, o resto → `ml-team`;
   - inibições: contrato bloqueado silencia o drift do mesmo lote; crítico silencia warning da mesma feature; `ModelPerformanceDrop` silencia `RetrainRecommended`; estágio falho silencia `PipelineStale`.
10. **Dashboard** (gerador `build_dashboard.py`), com 6 linhas:
    - visão geral: status por lote, modelo vigente, alertas, quarentena, tabela de alertas;
    - pipeline e infraestrutura;
    - qualidade de dados;
    - drift;
    - modelo: AUC realizada × estimada;
    - logs.

    Métricas por lote com consultas **instantâneas** e `batch_id` no eixo x, porque os meses rodam em segundos. Status sempre com **cor + texto**. Sem gráfico de dois eixos.
11. **Airflow:**
    - imagem `apache/airflow:3.1.0` + `libgomp1` + o projeto num venv próprio (`uv sync`) isolado das dependências do Airflow;
    - `src/`, `params.yaml` e `data/` montados como volumes;
    - DAG `credit_monthly`: `CronTriggerTimetable("0 0 1 * *")`, `catchup=True`, `max_active_runs=1`, `BashOperator` por estágio, `BranchPythonOperator` lendo o `gate.json`, `TriggerDagRunOperator(credit_retrain, wait_for_completion=True)`;
    - métricas StatsD → `statsd-exporter`.
12. **`runbook_url` nas regras:** use `docs/monitoring-plan.md#<nomedoalerta>` (âncora = nome do alerta em minúsculas). Na semana 4 você escreve o catálogo de métricas e os runbooks de pipeline e infraestrutura; os de dados vêm da Fabi e os de drift/modelo do Helio, e o Erick consolida tudo no `docs/monitoring-plan.md`.
13. **CI:**
    - lint + testes + `docker compose --profile ... config --quiet`;
    - job separado com o Airflow instalado (constraints oficiais) para a integridade das DAGs.

## 6. Contratos entre as etapas

**Você recebe:**
- da **Fabi**: `validate_raw`, `validate_model_input`, `enforce` (lança `ContractViolationError`), `build_model_input`, `load_model`, `train.py`;
- do **Helio**: `detect_drift`, `evaluate_performance`, `evaluate_gate`, `retrain`, `rollback` e o simulador (lotes em `data/production/`);
- nada do Erick: as regras de privacidade e retenção já estão na seção 3.

**Você entrega:**

| Artefato | Quem usa |
|---|---|
| `get_logger` / `setup_logging` | todos |
| `MetricsPublisher` e o catálogo de nomes de métricas | Fabi (contrato), Helio (drift, modelo, retreino) |
| Estágios, `run_batch`, `run_all` e os alvos do `Makefile` | todos |
| Stack (`make up`, `make up-full`) com os endereços | todos |
| Dashboard, alertas, catálogo de métricas e runbooks de pipeline/infraestrutura | Erick (consolida o plano de monitoramento; usa as evidências) |

**Pacote 3, entregue ao Erick com as tags `pacote-3` e `execucao-referencia`:** `make monitor` e `make up-full` funcionando, a DAG do Airflow com os 7 meses verdes, a execução de referência com os números congelados e as capturas em `docs/images/`.

## 7. Armadilhas que já encontramos

- **`push_to_gateway` (PUT)** apaga o grupo inteiro: o fechamento do lote sumia com as métricas do contrato. Use **`pushadd_to_gateway`**.
- **Counters no Pushgateway** não acumulam entre processos: use gauges por lote.
- **structlog: `get_logger(name).bind()` no import** congela a configuração padrão antes do `setup_logging`. Use valores iniciais: `structlog.get_logger(module=name)`. A chave `logger=` conflita com o argumento de `wrap_logger`.
- **Vazamento de contexto:** limpe as contextvars entre lotes, senão o `mlflow_run_id` do lote anterior aparece nos logs seguintes.
- **Imagem oficial do MLflow** não traz `prometheus_flask_exporter`: `--expose-prometheus` derruba o servidor. Crie uma imagem derivada.
- **Grafana:**
  - o painel `alertlist` só mostra alertas gerenciados pelo próprio Grafana; use uma **tabela da métrica `ALERTS`** do Prometheus;
  - `sort_by_label` é função **experimental** do Prometheus (desligada) e resulta em "No data";
  - stat/bargauge com consulta em formato tabela ignoram o `legendFormat`: use `format: time_series` + `reduceOptions.values=false`;
  - rotacione os rótulos do eixo x em painéis estreitos.
- **Alloy:** o `trace_id` vai como *structured metadata*, **não** como label (alta cardinalidade). O Loki precisa de `allow_structured_metadata: true`.
- **Airflow 3:**
  - imports em `airflow.providers.standard.*` e `airflow.sdk`;
  - faltava `libgomp1` na imagem (XGBoost/LightGBM);
  - `uv sync --python python3.12` (não o caminho absoluto);
  - a porta 8080 pode estar ocupada por outro projeto (usamos 8088);
  - DAG pausada não executa runs disparados: despause a `credit_retrain` antes da `credit_monthly`;
  - **não edite `src/` ou `params.yaml` com a DAG rodando**: os arquivos são montados ao vivo e uma edição no meio causou falha por condição de corrida.
- **Teste da DAG no CI:**
  - `pytest.importorskip("airflow.sdk")`, e não `"airflow"`: a pasta `airflow/` do repositório vira namespace package;
  - `--confcutdir=tests/dags`, para não carregar o conftest raiz que importa numpy;
  - `dagbag.dags[...]` em vez de `get_dag()`: no 3.1, `get_dag` consulta o banco de metadados, que não existe no CI.
- **Windows:**
  - o Git Bash converte caminhos de container: use `MSYS_NO_PATHCONV=1` no `docker compose exec`;
  - configs montadas em container Linux precisam de **LF**: use `.gitattributes` com `eol=lf`.
- **Documente o que não é produção:** Airflow standalone sem login, Grafana anônimo, métricas de lotes antigos que nunca expiram no Pushgateway.

## 8. Checklist de pronto

- [ ] `make monitor` roda os 7 meses sem Docker (métricas em `.prom`, logs em JSONL).
- [ ] `make up-full` sobe tudo; o painel "Serviços da stack no ar" mostra todos UP.
- [ ] Dashboard com as 6 linhas preenchidas e nenhum painel em "No data".
- [ ] Alertas da seção 4 disparando; inibição comprovada no Alertmanager; notificações em `logs/alerts.jsonl` e no Loki.
- [ ] Trace por lote no Tempo; clique do `trace_id` no log abre o trace.
- [ ] MLflow com um run por lote (relatórios, gate, trecho do log, tag `trace_id`).
- [ ] Airflow: 7 runs verdes, retreino só em junho.
- [ ] Execução de referência (tag `execucao-referencia`) com as capturas em `docs/images/`.
- [ ] Catálogo de métricas e runbooks de pipeline/infraestrutura (semana 4); `runbook_url` no formato `docs/monitoring-plan.md#<nomedoalerta>`.
- [ ] CI verde (principal + DAGs); nenhum `customer_id` em logs e métricas (teste automatizado).
- [ ] ADRs escritas:
  - orquestração e Pushgateway;
  - Alloy, Loki e Tempo.

## 9. Referência: projeto anterior

Implementação completa para consultar quando travar (repositório privado; peça acesso): https://github.com/eeyamazaki/credit-scoring-sustentacao. Use para **comparar**, não para copiar: o objetivo é o grupo construir e entender cada parte.

Arquivos equivalentes no projeto anterior:

`src/logger.py`, `src/monitoring/{metrics,tracing}.py`, `src/pipeline/{stages,run_batch,run_all}.py`, `docker-compose.yml`, `monitoring/` (prometheus, alertmanager, alert-receiver, grafana com `build_dashboard.py`, loki, tempo, alloy, statsd, mlflow), `airflow/` (Dockerfile e DAGs), `.github/workflows/ci.yml`, `tests/integration/test_pipeline_smoke.py`, `tests/dags/`, `docs/monitoring-plan.md`, `docs/decisions/00{3,5}-*.md`, `docs/images/`.

Aulas da disciplina de Monitoramento: Aula 1 (observabilidade de ML), Aula 2 (pipelines, Airflow e Datadog), Aula 3 (alertas de falha de ingestão e transformação, Dead Letter), Aula 5 (logs, métricas e traces), Aula 6 (MLflow), Aula 7 (infraestrutura open source), Aula 8 (automação e AIOps).

## 10. Prompt inicial sugerido para a LLM

```
Você vai me ajudar na Etapa 3 (Observabilidade de Pipelines e Modelos) do Tech
Challenge descrito no documento acima. Siga as "Decisões já tomadas" e os
"Contratos entre as etapas" sem alterá-los. Comece pelos passos 1 e 2: o logger
structlog (com mascaramento de customer_id) e o MetricsPublisher com pushadd e
fallback para arquivo .prom. Depois o esqueleto dos 7 estágios, com funções stub
para os módulos da Fabi e do Helio. Escreva Python 3.12 com type hints, docstrings
em português e testes pytest. Para cada config (compose, Prometheus, Alertmanager,
Alloy), fixe versões de imagem e explique cada bloco.
```
