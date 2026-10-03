# Handoff — Etapa 4: Governança (Privacy by Design e by Default) e fechamento

> Documento autocontido. Cole o arquivo inteiro no início de uma conversa com uma LLM. O porquê das decisões está em [`docs/guias/etapa-4-governanca-erick.md`](../guias/etapa-4-governanca-erick.md) (escrito para outro dataset, com idade; aqui está a versão adaptada ao Lending Club, com região).

## 1. Onde o projeto está

Tech Challenge Fase 4 (FIAP MLET): sustentação de um modelo de credit scoring (Lending Club). Esta etapa vale 15%, **fecha o projeto** e escreve a documentação que o avaliador lê primeiro. O enunciado pede, **no próprio README**, a documentação de **Governança e Causalidade**: como a PII foi tratada, a base legal da decisão de crédito, o plano de retenção e a mitigação de vieses.

**Estado: o núcleo que não depende das outras etapas está pronto** (dono: Erick). Falta o que depende da execução de referência (Etapa 3) e dos lotes simulados (Etapa 2).

| Item | Onde | Estado |
|---|---|---|
| Fairness por região (`approval_disparity`, `fairness_by_region`) | `src/credit_scoring_observability/fairness.py` | ✅ (a Etapa 2 usa no guardrail; a Etapa 3, por lote) |
| Mitigação por thresholds com paridade de FPR, avaliada e não adotada | `mitigation.py`, `make fairness` → `reports/fairness/fairness_report.json` | ✅ |
| Descarte da quarentena (30 dias, `--dry-run`) | `retention.py`, `make purge-quarantine` | ✅ |
| Análise causal (`restore_group`, `analyze_batch`, diagnóstico) | `causal.py`, `make causal` → `reports/causal/causal_analysis.json` | ✅ código e testes; ⏳ rodar nos lotes da Etapa 2 |
| `docs/lgpd.md` (papéis, inventário, k-anonimato, base legal, 10 princípios, PbD, direitos, retenção, fairness, RIPD, causal, auditoria) | `docs/lgpd.md` | ✅ itens 1–9; ⏳ tabelas por lote (8 e 10) e auditoria da Etapa 3 (11) |
| README: seções "9. Governança e LGPD" e "10. Análise causal" com conteúdo próprio | `README.md` | ✅; ⏳ números por lote |
| Auditoria de privacidade da stack (logs/métricas/spans sem id; retenção 90 dias) | `docs/lgpd.md` item 11 | ⏳ depende do Pacote 3 |
| `docs/monitoring-plan.md` consolidado (catálogo, SLIs/SLOs, um runbook por alerta) | — | ⏳ semana 4 |
| Post-mortem do incidente M3–M7 | `docs/postmortem.md` | ⏳ semana 4 (linha do tempo da Etapa 2) |
| Revisão do model card (fairness, explicabilidade) | `docs/model-card.md` (Etapa 1) | ⏳ semana 4 |
| Todos os números alinhados à tag `execucao-referencia` | todos os docs | ⏳ semana 4 |

```powershell
make prepare; make train      # dados e modelo (Etapa 1)
make fairness                 # fairness por região + mitigação no pool de produção
make causal                   # precisa dos lotes de data/production (make simulate, Etapa 2)
make purge-quarantine         # descarte da quarentena (python -m ...retention --dry-run para só listar)
```

## 2. Decisões já tomadas (adaptadas ao Lending Club)

| Tema | Decisão |
|---|---|
| Papéis | Fintech = **controladora**; plataforma de ML = **operadora**; **encarregado (DPO)** nomeado (fictícios) |
| Identificador | **Pseudonimização** no `prepare`: SHA-256 de `"{sal}:{id}"`, 16 caracteres, sal no `.env` (fora do git). Não é anonimização: continua sendo dado pessoal (Art. 13, §4º) |
| Minimização | O `prepare` lê 27 das 151 colunas; `emp_title`, `desc`, `title`, `zip_code`, `member_id`, `url` e os campos pós-desfecho **não são lidos**. `addr_state` vira `region` e é descartado |
| Dados sensíveis | Nenhum do Art. 11 no Lending Club. **Região = atributo auditado** (proxy de raça e renda, *redlining*); **não é feature** |
| Base legal | **Art. 7º, X (proteção do crédito)** + Lei 12.414/2011; Art. 7º, V (procedimentos preliminares); **Art. 20** (revisão de decisão automatizada). Consentimento **não** é a base |
| Retenção | Quarentena 30 dias com descarte; logs, traces e métricas 90 dias **na configuração** da stack (Etapa 3); relatórios agregados mantidos |
| Telemetria | Opt-out de MLflow, Evidently, GE e NannyML em `src/credit_scoring_observability/__init__.py` |
| Fairness | Por região: aprovação, TPR, FPR, default observado e razão de impacto desigual (regra dos 4/5), com fairlearn `MetricFrame` |
| Mitigação | Thresholds por região com paridade de FPR, determinísticos: **avaliada e não adotada**. O ganho é desprezível (DI 0,980 → 0,984) e usar a região na decisão seria tratamento diferenciado por proxy de raça. `ThresholdOptimizer` descartado (sorteia a decisão) |
| Causalidade | **DAG de hipóteses** + **intervenção por grupo** (renda+dti, rotativo, FICO), preservando a ordem no lote. Mede a sensibilidade do modelo; **não prova** a causa no mundo real |

## 3. Números atuais (modelo V2, pool de produção)

- **Fairness:** aprovação de 58,5% (West) a 59,6% (South); FPR de 34,8% a 36,2%; **DI 0,983**.
- **Mitigação:** meta de FPR 35,5%; thresholds 0,198–0,203; diferença de FPR 1,4 → 0,5 p.p.; aprovação 59,1% → 59,2%; default entre aprovados 12,5% nos dois casos.
- **k-anonimato** (região + mês + valor exato): 6,4% de registros únicos.
- **Causal, validação do método** com lotes de teste do pool (não são os lotes oficiais):
  - deslocamento de X: recuperação de 99,6% do desvio do score;
  - rótulos trocados: AUC de 0,69 para 0,53 com PSI ≈ 0, diagnóstico de concept drift.

## 4. Passo a passo da semana 4

1. **Receber o Pacote 3** (tag `execucao-referencia`): os lotes M1–M7, as capturas, o catálogo de métricas e os runbooks.
2. **Rodar** `make fairness` e `make causal` sobre a execução de referência e preencher:
   - a tabela de razão de impacto por lote no `lgpd.md` (item 8);
   - as tabelas de intervenção de M4 (covariate) e M6 (concept) no `lgpd.md` (item 10) e no README (seção 10).
3. **Auditoria** (`lgpd.md`, item 11): conferir no código e nos artefatos da Etapa 3:
   - nenhum `customer_id` em `logs/pipeline.jsonl`, nos `.prom`, nos spans e nas notificações (há um teste automatizado?);
   - retenção de 90 dias em Loki, Tempo e Prometheus;
   - o guardrail de fairness no retreino (Etapa 2).

   Marque ✅ com a evidência, ou abra um PR pequeno para o dono da etapa.
4. **`docs/monitoring-plan.md`:**
   - catálogo de métricas (Etapa 3);
   - SLIs/SLOs: sucesso ≥ 95%, rejeição < 5%, queda de AUC ≤ 0,05, freshness < 35 dias;
   - um runbook por alerta, com o título igual ao nome do alerta (é a âncora do `runbook_url`);
   - incluir um alerta `DisparateImpactLow` (< 0,8), se a Etapa 3 publicar `fairness_disparate_impact`.
5. **Post-mortem sem culpados** do incidente M3–M7, a partir da linha do tempo da Etapa 2: detecção, causa (com a análise causal), mitigação, lições e ações.
6. **Model card** (Etapa 1): revisar a seção de fairness (com a tabela do item 8) e a explicabilidade (coeficientes da LogReg, Art. 20).
7. **Busca final** pelos números antigos em todos os `.md`. Use só os da execução de referência.

## 5. Armadilhas

- **Pseudonimização ≠ anonimização.** Não escreva "dados anonimizados" em lugar nenhum.
- **Consentimento como base legal** é um erro comum em crédito: a base é a proteção do crédito.
- **A amostra de referência foi tirada do treino:** a AUC "de referência" da análise causal vem do pool (metadados do modelo), não da amostra.
- **`restore_group` com uma coluna isolada** quebra a relação renda × dti. Restaure grupos.
- **Números na documentação:** só os da execução de referência; o split e o treino são determinísticos, mas os lotes simulados dependem dos parâmetros da Etapa 2.
- **Região na decisão:** não ligue a mitigação no pipeline; ela é registrada como avaliada e não adotada.

## 6. Prompt sugerido para a LLM

```
Você vai me ajudar a fechar a Etapa 4 (Governança) do Tech Challenge descrito no documento
acima, no repositório credit-scoring-observability. Siga as "Decisões já tomadas" sem
alterá-las. Com a execução de referência em mãos, vamos: (1) rodar make fairness e make
causal e preencher as tabelas pendentes do docs/lgpd.md e do README; (2) fazer a auditoria
do item 11 contra o código da Etapa 3; (3) consolidar o docs/monitoring-plan.md. Cite artigos
da LGPD só quando tiver certeza do número; na dúvida, sinalize para eu conferir.
```
