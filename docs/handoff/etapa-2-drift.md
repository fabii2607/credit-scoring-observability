# Handoff — Etapa 2: Simulação e Detecção de Drift

> Documento autocontido. Cole o arquivo inteiro no início de uma conversa com uma LLM. O porquê de cada decisão está em [`docs/guias/etapa-2-drift-helio.md`](../guias/etapa-2-drift-helio.md) (escrito para outro dataset; aqui está a versão adaptada ao Lending Club e ao código real).

## 1. Onde o projeto está

Tech Challenge Fase 4 (FIAP MLET): uma fintech tem um modelo de credit scoring em produção e suspeita de **degradação silenciosa** por mudanças na economia. A Etapa 1 está pronta: separação Referência × Produção, modelo V2 com threshold e contrato de dados. **Esta etapa simula o tempo passando (lotes mensais com drift) e detecta o drift.** Ela vale 25% da nota.

O que já existe e você usa (detalhes em [etapa-1-dados.md](etapa-1-dados.md)):

| Artefato / função | Como obter | Formato |
|---|---|---|
| `data/processed/production_pool.parquet` | `make prepare` | 269.620 linhas nunca vistas no treino: `customer_id`, `issue_date`, `region`, 23 features, `target` |
| `data/processed/reference_sample.parquet` | `make prepare` | 99.999 linhas da Referência (o Reference Dataset do notebook 02 sem a linha de renda nula), mesmas colunas: **a base de comparação do drift** |
| `models/credit_scoring_optimized.joblib` + `_metadata.json` | `make train` | V2, threshold 0,20 |
| `registry.load_model().score(lote)` | — | lote + `score` (probabilidade de default) + `prediction` (0/1) |
| `evaluate.classification_metrics(y, score, threshold)` | — | accuracy, precision, recall, f1, f2, roc_auc, pr_auc, ks, brier, approval_rate |
| `validate.make_corrupted(df, rng)` | — | lote corrompido para a demo do contrato (importe; não copie) |
| `synthetic.make_model_batch(n, seed)` | — | lote sintético no mesmo formato, para testes sem o Kaggle |

```powershell
uv sync; make setup          # ambiente + .env (peça ao grupo o PSEUDONYMIZATION_SALT)
make prepare; make train     # ~25 s + ~1 min com o CSV em data/raw/
make test                    # tudo verde antes de começar
```

## 2. O que o enunciado pede

- Gerar um dataset de "produção" com **pelo menos 2 variáveis importantes alteradas**, mais as predições do modelo nele.
- Relatório **Evidently** comparando Referência × Produção.
- Boa prática obrigatória: **PSI e KS** apontando as features com drift.
- O contexto cita **Data Drift e Concept Drift**, e o vídeo avalia a "demonstração da degradação".

## 3. Prioridade: núcleo primeiro

| Núcleo (obrigatório) | Extra (diferencial) |
|---|---|
| Lotes M1–M7 do pool: estável → covariate shift → concept drift | PSI por quantis registrado como stattest do Evidently (recomendado) |
| Predições do modelo em cada lote (`load_model().score`) | Teste z da taxa de default (label drift) |
| Evidently Referência × lote (HTML por lote) | Gate em camadas (nem todo drift pede retreino) |
| PSI e KS D por feature apontando as alteradas | MMD multivariado + lote `joint_drift` |
| AUC por lote caindo no concept drift | CBPE (AUC estimada sem rótulo) |
| — | Retreino champion/challenger com rollback (guardrail de fairness com o `fairness_by_region` da Etapa 4) |

## 4. Decisões já tomadas (adaptadas ao Lending Club)

| Tema | Decisão |
|---|---|
| Linha do tempo | `2026-01`–`2026-02` estáveis; `2026-03`–`2026-04` **covariate shift** (intensidade 0,6 → 1,0); `2026-05`–`2026-07` **concept drift**. Lotes de demo: `corrupted` (com `make_corrupted`) e, se fizer o MMD, `joint_drift`. |
| Tamanho do lote | **10.000 linhas**, disjuntos, embaralhados com seed fixa. Com 5 mil, a AUC oscila ±0,025 só por amostragem (ADR 010 do projeto de referência). |
| Covariate shift | Inflação corrói a renda real e aumenta o endividamento: `annual_inc` ↓, `dti` ↑, `revol_util` ↑, `fico_avg` ↓ (em degraus de 5 pontos, como o FICO real). **Rótulos originais mantidos.** São mais de 2 variáveis alteradas. |
| Concept drift | **Relabel por segmento, CONTRA o que o modelo aprendeu; X não muda.** Use os sinais que este modelo usa (não há `int_rate` nas features): inadimplentes de alto risco aparente "curam" (ex.: FICO baixo ou prazo 60 → 0) e bons pagadores de baixo risco aparente passam a inadimplir (ex.: FICO alto **e** `revol_util` > 70 → 1). Gerar rótulos a partir das predições do próprio modelo faz a AUC **subir**: não faça isso. |
| Base do drift | `reference_sample.parquet`, pontuada pelo mesmo modelo. |
| Triagem × confirmação | **PSI** para triagem (OK < 0,10 ≤ WARN ≤ 0,25 < ALERT); **estatística D do KS** > 0,10 para confirmar. O p-valor do KS com milhares de linhas acusa tudo: não use como gate. |
| PSI | Bins pelos **quantis da Referência**, deduplicados, com epsilon. O PSI nativo do Evidently usa bins de largura igual e apaga o drift em variáveis de cauda longa (`annual_inc`, `revol_bal`). |
| Categóricas | `home_ownership`, `verification_status`, `purpose`, `application_type`: drift pelo Evidently (qui-quadrado / Jensen-Shannon). PSI/KS só nas numéricas. |
| Label drift (extra) | Teste z da taxa de default contra a Referência (19,98%), alerta com z > 3. |
| Gate (extra) | **Só retreina com queda de AUC realizada > 0,05.** Covariate shift sem queda de AUC → monitoramento reforçado, não retreino. |
| Fairness | `region` é o atributo sensível (não há idade). Decisão 1 = negar; aprovação = 1 − taxa de seleção. |

**Comportamento esperado** (os números não precisam bater com nenhum projeto anterior):
- M1–M2: sem alerta.
- M3–M4: PSI > 0,25 em ≥ 2 features e **sem** queda relevante de AUC.
- M5–M7: PSI ≈ 0 e **queda** de AUC > 0,05, com a taxa de default subindo.

AUC de referência da V2 no pool: **0,694**. Com AUC-base perto de 0,69, calibre o concept drift para derrubar pelo menos 0,05; a intensidade máxima do projeto de referência derrubou ~0,10.

## 5. O que você implementa (interfaces combinadas)

Crie os módulos dentro de `src/credit_scoring_observability/` (ex.: `simulation.py`, `stats.py`, `drift.py`, `performance.py`, `gate.py`, `retrain.py`) e ligue o alvo `make simulate` no `Makefile` (hoje imprime "TODO"). As saídas são **funções puras com dataclass serializável em JSON**: a Etapa 3 chama cada uma num estágio do pipeline.

```python
# simulation.py — make simulate
def apply_data_drift(batch: pd.DataFrame, intensity: float, rng) -> pd.DataFrame: ...
def apply_concept_drift(batch: pd.DataFrame, intensity: float, rng) -> pd.DataFrame: ...
def generate() -> dict  # grava data/production/<lote>.parquet + manifest.json

# drift.py
def detect_drift(reference_scored: pd.DataFrame, batch_scored: pd.DataFrame, batch_id: str) -> DriftResult
# DriftResult: features[{feature, psi, ks_d, status, ks_confirmed}], drift_share,
#              drifted_features, score_psi, default_rate, label_z (extra), mmd2/mmd_p_value (extra),
#              report_paths (reports/drift/<lote>.html|json)

# performance.py
def evaluate_performance(reference_scored, batch_scored, threshold) -> PerformanceResult
# PerformanceResult: roc_auc (None sem rótulo), reference_roc_auc, brier, approval_rate,
#                    score_p50/p90/p99, estimated_roc_auc/estimation_gap (extra, CBPE)

# gate.py (extra)
def evaluate_gate(drift: DriftResult, perf: PerformanceResult) -> GateResult
# GateResult: status (OK/WARN/ALERT), diagnosis (none/covariate/concept/prior/joint),
#             recommended_action (observe/reinforced_monitoring/investigate/retrain), failures[]

# retrain.py (extra)
def retrain(trigger_batch_id: str) -> RetrainDecision
def rollback() -> None
```

- **Nomes de métricas** que a Etapa 3 vai publicar a partir dos seus resultados (não mude): `feature_psi{feature}`, `feature_ks_d{feature}`, `drift_share`, `score_psi`, `label_drift_z`, `multivariate_mmd_pvalue`, `model_roc_auc`, `model_estimated_roc_auc`, `performance_estimation_gap`, `drift_gate_status`, `approval_rate`.
- **Seção `params.yaml`:** crie `simulation`, `drift`, `performance` e `retrain`, com os modelos pydantic em `parameters.py` e os campos em `PipelineParams`.
- **Fairness:** `fairness.fairness_by_region(y_true, score, threshold, region)` e `fairness.approval_disparity(prediction, region)` são da **Etapa 4** (`src/credit_scoring_observability/fairness.py`). No retreino, use-os no guardrail: o challenger não pode piorar o impacto desigual por região em mais de 0,05.
- **Retreino:** o `registry.load_model` só conhece o alias `production`. Estenda o `registry.py` com `challenger`/`previous` (por exemplo, `models/<alias>/`) ou combine com a Etapa 3 o uso do MLflow Registry. O treino reaproveita `train.build_model` e `evaluate.select_threshold`.
- **Layout que a Etapa 3 usa:** o estágio `validate` grava `data/interim/<lote>/model_input.parquet`. O retreino lê dali os lotes **que passaram no contrato**.

## 6. Entregas (Pacote 2, tag `pacote-2`)

- `make simulate` gerando M1–M7 + `corrupted` (+ `joint_drift`) em `data/production/`, disjuntos e reprodutíveis.
- Um script ou notebook que roda score → drift → performance (→ gate → retreino) para M1–M7 e mostra a tabela do comportamento esperado.
- Testes: PSI e KS com resposta conhecida; simulador determinístico; concept drift que só muda rótulos, na direção contrária ao modelo; gate em cada caminho.
- Documentação (semana 3): `docs/retraining-policy.md` (se fizer o retreino), ADR das faixas de PSI/KS, runbooks dos alertas de drift e modelo (`FeatureDriftCritical`, `FeatureDriftWarning`, `LabelDrift`, `ModelPerformanceDrop`, `ConceptDriftSuspected`, `RetrainRecommended`) e a **linha do tempo do incidente** M3–M7, que vira o post-mortem da Etapa 4.

## 7. Armadilhas

- **Escala da renda antes da imputação:** o pool está **sem imputação**; aplique o drift nele (o pipeline do modelo imputa depois). Escalar depois da imputação desloca o pico inteiro e o PSI explode.
- **Lote derivado do pool passa pelo contrato:** o FICO deslocado tem de continuar entre 300 e 850, e `dti`/`revol_util` não podem ficar negativos. Corte nos limites do contrato para o drift não virar lote bloqueado.
- **Evidently 0.7:** `from evidently import Report`; stattest customizado pela API legacy (`evidently.legacy.calculations.stattests.registry`). O drift share do Evidently conta o score junto; o do gate, só as features. Documente a diferença.
- **Ruído de AUC entre lotes** (±0,02 em 10 mil linhas): olhe os lotes estáveis antes de concluir que houve queda.
- **CBPE (extra) precisa de probabilidade calibrada.** A V2 é uma LogReg **sem** `class_weight` (Brier 0,147), razoavelmente calibrada, mas confira a curva de calibração no pool antes. Para o NannyML, adicione a dependência com `uv add nannyml`; ele pode exigir fixar o `kaleido`.
- **Safras recentes censuradas:** o pool mistura 2007–2018. Use o embaralhamento com seed para os lotes; se quiser drift **real** por safra (extra), lembre que o default de 2018 é baixo por censura, não por melhora.
- **Retreino sem peso** dilui o conceito novo. O projeto de referência usou peso 10 nas linhas recentes.

## 8. Checklist de pronto

- [ ] `make simulate` reprodutível, lotes disjuntos de 10 mil linhas.
- [ ] HTML do Evidently por lote em `reports/drift/`, com PSI por quantis e KS D.
- [ ] M3–M4 com PSI > 0,25 em ≥ 2 features e AUC estável; M5–M7 com AUC caindo > 0,05.
- [ ] Testes verdes no CI; `params.yaml` com as seções da etapa.
- [ ] Runbooks e linha do tempo entregues para a Etapa 4.

## 9. Prompt sugerido para a LLM

```
Você vai me ajudar na Etapa 2 (Simulação e Detecção de Drift) do Tech Challenge descrito
no documento acima, no repositório credit-scoring-observability (Python 3.11, uv, pacote
src/credit_scoring_observability). Siga as "Decisões já tomadas" e as interfaces da seção 5
sem alterá-las. Comece pelo simulador (apply_data_drift, apply_concept_drift, generate) e
pelos testes com synthetic.make_model_batch. Depois, stats.py (PSI por quantis, KS D). Antes
de cada passo, diga quais armadilhas da seção 7 se aplicam. Não invente números: eu rodo e
colo as saídas reais.
```
