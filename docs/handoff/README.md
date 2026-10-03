# Handoffs por etapa

Um handoff por etapa do Tech Challenge, para cada integrante **seguir sozinho** a partir do que já está no repositório. Cada arquivo é autocontido: dá para colar o arquivo inteiro numa LLM e trabalhar a partir dele.

> **Fonte de verdade:** estes handoffs. Os guias em [`docs/guias/`](../guias/) foram escritos para o dataset Give Me Some Credit e ficam como **referência conceitual** (o porquê das decisões, as aulas, as armadilhas). Onde um guia e um handoff divergem, vale o handoff: ele descreve o código real deste repositório, com o Lending Club.

| Ordem | Etapa | Responsável | Handoff | Estado |
|---|---|---|---|---|
| 1º | Validação de dados e contratos | Fabi | [etapa-1-dados.md](etapa-1-dados.md) | **Núcleo pronto** (notebooks + scripts); pendências de documentação |
| 2º | Simulação e detecção de drift | Helio | [etapa-2-drift.md](etapa-2-drift.md) | A fazer — pode começar |
| 3º | Observabilidade de pipelines e modelos | Pedro | [etapa-3-observabilidade.md](etapa-3-observabilidade.md) | A fazer — pode começar |
| 4º | Governança (LGPD, fairness, causalidade) e fechamento | Erick | [etapa-4-governanca.md](etapa-4-governanca.md) | **Núcleo pronto** (fairness, mitigação, retenção, causal, `lgpd.md`, README); semana 4 depende da execução de referência |

O vídeo STAR não está nestes handoffs.

## O que já existe (para todas as etapas)

```powershell
uv sync                 # ambiente (Python 3.11, uv.lock)
make setup              # + pre-commit e .env a partir do .env.example
make test               # testes com dados sintéticos (sem o CSV do Kaggle)
make prepare            # Referência × pool de produção (precisa do CSV em data/raw/)
make train              # modelo V2 + threshold em models/
make demo-contract      # lote corrompido bloqueado pelo contrato (exit 2)
make help               # todos os alvos; os das etapas 2–4 imprimem "TODO (Etapa N)"
```

| Peça | Onde | Para que serve |
|---|---|---|
| Caminhos e nomes de colunas | `src/credit_scoring_observability/config.py` | `customer_id`, `issue_date`, `region`, `target`, `score`, `prediction`, caminhos de `data/`, `models/`, `reports/` |
| Parâmetros | `params.yaml` + `parameters.py` | Uma seção por etapa, validada por pydantic (`extra="forbid"`: typo vira erro) |
| Logger | `logger.py` | `setup_logging()` uma vez, `get_logger(__name__)` em todo módulo; `customer_id` mascarado |
| Dados sintéticos | `synthetic.py` + `tests/conftest.py` | `make_raw_loans` (formato do CSV) e `make_model_batch` (formato pós-prepare). Use para adiantar trabalho e nos testes |
| Prepare | `prepare.py` | `reference.parquet`, `reference_sample.parquet`, `production_pool.parquet` |
| Governança | `fairness.py`, `mitigation.py`, `causal.py`, `retention.py` | `fairness_by_region`/`approval_disparity` (guardrail e monitoramento), `make fairness`, `make causal`, `make purge-quarantine` |
| Modelo | `train.py`, `registry.py`, `evaluate.py` | `load_model().score(lote)` devolve `score` e `prediction` com o threshold versionado |
| Contrato | `data_contract.py`, `validate.py` | `validate_batch` → `ValidationResult`; `enforce` (relatório + quarentena); `make_corrupted` |
| CI | `.github/workflows/ci.yml` | ruff + testes em todo PR e push na `main` |

## Regras do repositório

- **Commits semânticos** (Conventional Commits): `feat(drift): ...`, `fix(pipeline): ...`, `docs: ...`, `test: ...`, `ci: ...`, `build: ...`. Um assunto por commit.
- **PRs pequenos direto na `main`**, com o CI verde. O próprio autor faz o merge (não há proteção de branch); só não faça merge com o CI vermelho.
- **Dependências:** o núcleo das Etapas 2–4 já está no `pyproject.toml` (Evidently, MLflow, prometheus-client, psutil, fairlearn). Extras (NannyML, Great Expectations, OpenTelemetry) entram no PR de quem usar, junto com o `uv.lock` (`uv add <pacote>`).
- **Cada etapa é dona da sua seção do `params.yaml`** e dos seus módulos. Mudança numa interface de outra etapa vira um PR pequeno, avisando o dono.
- **Dados e modelos não vão para o git** (`data/`, `*.joblib`). Cada um gera os seus com `make prepare && make train`; os números batem (o split e o treino são determinísticos).
- **Tags de entrega:** `pacote-1a`, `pacote-1b`, `pacote-2`, `pacote-3`, `execucao-referencia`.
- **Runbooks** (cada dono escreve os da sua parte; a Etapa 4 junta no `docs/monitoring-plan.md`):

  ```markdown
  ### NomeDoAlerta
  - **Significa:** o que disparou, em uma frase.
  - **Investigar:** onde olhar (relatório, painel do Grafana, consulta no Loki, trace).
  - **Ação:** o que fazer, e o que NÃO fazer.
  ```

## Mapa de cobertura dos requisitos

| Requisito do enunciado | Onde está / quem entrega | Estado |
|---|---|---|
| Classificador base treinado na Referência | `train.py` (V2, notebooks 03–04) | Pronto |
| Contrato de dados com ≥ 3 regras rígidas | `data_contract.py` (Pandera, 23 features) + regras de lote em `validate.py` | Pronto |
| Lote com erros bloqueando a ingestão; script de validação operante | `make demo-contract` (exit 2, quarentena, relatório) | Pronto |
| Separação Referência × Produção | `prepare.py` | Pronto |
| Produção com ≥ 2 variáveis alteradas + predições | Etapa 2 | A fazer |
| Evidently Referência × Produção; PSI/KS por feature | Etapa 2 | A fazer |
| Logs e métricas centralizados; MLflow e dashboard | Etapa 3 | A fazer |
| Métricas de saúde documentadas; alertas | Etapa 3 (catálogo) + Etapa 4 (plano consolidado) | A fazer |
| Governança LGPD no README: PII, base legal, retenção | Etapa 4 (README seção 9, `docs/lgpd.md`) | Pronto; números finais com a execução de referência |
| Mitigação de vieses; análise causal | Etapa 4 (`fairness.py`, `mitigation.py`, `causal.py`, README seção 10) + Etapa 2 (guardrail) | Código pronto; causal por lote depende da Etapa 2 |
| Commits semânticos | todos | Contínuo |
