# Handoff — Etapa 4: Governança (Privacy by Design e by Default) e fechamento

> Documento autocontido. Cole o arquivo inteiro no início de uma conversa com uma LLM. O porquê das decisões está em [`docs/guias/etapa-4-governanca-erick.md`](../guias/etapa-4-governanca-erick.md) (escrito para outro dataset, com idade; aqui está a versão adaptada ao Lending Club, com região).

## 1. Onde o projeto está

Tech Challenge Fase 4 (FIAP MLET): sustentação de um modelo de credit scoring (Lending Club). Esta etapa vale 15%, **fecha o projeto** e escreve a documentação que o avaliador lê primeiro. O enunciado pede, **no próprio README**, a documentação de **Governança e Causalidade**: como a PII foi tratada, a base legal da decisão de crédito, o plano de retenção e a mitigação de vieses.

**Estado:** em andamento (dono: Erick). A Etapa 1 já entrega a base da governança (pseudonimização, região, telemetria desligada, ids mascarados). O trabalho que **não depende** das outras etapas é feito primeiro; a auditoria e a consolidação esperam a execução de referência da Etapa 3.

## 2. Decisões já tomadas (adaptadas ao Lending Club)

| Tema | Decisão |
|---|---|
| Papéis | Fintech = **controladora**; plataforma de ML = **operadora**; **encarregado (DPO)** nomeado (fictícios) |
| Identificador | **Pseudonimização** no `prepare`: SHA-256 de `"{sal}:{id}"`, 16 caracteres, sal no `.env` (fora do git). Não é anonimização: continua sendo dado pessoal (Art. 13, §4º) |
| Minimização | O `prepare` lê só as colunas de concessão + `id` + `addr_state`; `emp_title`, `desc`, `title`, `zip_code` e os campos pós-desfecho **não são lidos**. `addr_state` vira `region` e é descartado |
| Dados sensíveis | Nenhum do Art. 11 no Lending Club. **Região = atributo auditado** (proxy conhecido de raça e renda, *redlining*); **não é feature** |
| Base legal | **Art. 7º, X (proteção do crédito)** + Lei 12.414/2011; Art. 7º, V (procedimentos preliminares); **Art. 20** (revisão de decisão automatizada). Consentimento **não** é a base |
| Retenção | Quarentena 30 dias (`retention.quarantine_days`, com descarte automático); logs, traces e métricas 90 dias (**na configuração** da stack, Etapa 3); relatórios agregados mantidos |
| Telemetria | Opt-out de MLflow, Evidently, GE e NannyML em `src/credit_scoring_observability/__init__.py` |
| Fairness | Por **região**: aprovação, TPR, FPR e razão de impacto desigual (regra dos 4/5), com fairlearn `MetricFrame`; vigiada por lote e usada como **guardrail** no retreino (Etapa 2) |
| Mitigação | Thresholds por região com paridade de FPR **avaliados e registrados, não adotados** (`adopted_in_production: false`): usar a região na decisão é tratamento diferenciado por proxy de raça (ver a ADR 009 do projeto de referência) |
| Causalidade | **DAG de hipóteses** + **atribuição por intervenção** em grupos de variáveis ligadas (renda+dívida, rotativo, FICO), preservando a ordem no lote. Mede a sensibilidade do modelo, **não prova** a causa no mundo real |

## 3. Plano da etapa

| Item | Depende de | Estado |
|---|---|---|
| `fairness.py` (`approval_disparity`, `fairness_by_region`) + medição no pool | Etapa 1 | A fazer |
| `mitigation.py` (thresholds por região, avaliação) + `make fairness` | Etapa 1 | A fazer |
| `retention.py` + `make purge-quarantine` | Etapa 1 | A fazer |
| `causal.py` (`restore_group`, atribuição, diagnóstico) + `make causal` | Lotes da Etapa 2 (testável com sintéticos já) | A fazer |
| `docs/lgpd.md` (10 itens do guia) | — | A fazer |
| Seções "Governança e LGPD" e "Análise causal" no README | Números congelados | A fazer |
| Auditoria de privacidade registrada no `lgpd.md` | Pacote 3 (logs, métricas, stack) | Semana 4 |
| `docs/monitoring-plan.md` consolidado (catálogo, SLOs, um runbook por alerta) | Runbooks das Etapas 1–3 | Semana 4 |
| Post-mortem do incidente M3–M7 | Linha do tempo da Etapa 2 | Semana 4 |
| Revisão do model card (fairness, explicabilidade) | Model card da Etapa 1 | Semana 4 |
| Todos os números alinhados à tag `execucao-referencia` | Etapa 3 | Semana 4 |

## 4. Checklist de pronto

- [ ] README com Governança e Causalidade **no próprio README** (não só links).
- [ ] `docs/lgpd.md` completo, com a auditoria registrada.
- [ ] Retenção aplicada: quarentena com descarte; Loki, Tempo e Prometheus com 90 dias.
- [ ] Fairness por região medida e usada como guardrail no retreino.
- [ ] Mitigação avaliada, com o custo explícito e a decisão de não adotar justificada.
- [ ] Análise causal distinguindo covariate shift (M3–M4) de concept drift (M5–M7).
- [ ] Post-mortem, `monitoring-plan.md` consolidado e model card revisado.
- [ ] Todos os números batendo com a execução de referência.
