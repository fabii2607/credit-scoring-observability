# Guia da Etapa 2 — Simulação e Detecção de Drift (Helio)

> Documento de direcionamento e contexto para LLM. Cole o arquivo inteiro no início da conversa.

## 0. Posição no revezamento

**Você é o 2º: semana 2.** Recebe o **Pacote 1** da Fabi (Referência, pool, imputação, contratos, modelo `@production`) em duas partes: o **1a** (dados + baseline) no **dia 3 da semana 1**, e o **1b** (contratos) no fim da semana 1. Comece a calibrar o simulador assim que o 1a chegar.

- **Antes do bastão (semana 1), adiante sem depender dela:** `stats.py` (PSI, KS, MMD, teste z) testado em distribuições conhecidas, o simulador e o gate rodando sobre os dados sintéticos de `tests/conftest.py` com um modelo qualquer treinado neles. Com o Pacote 1, você só troca a fonte de dados e **recalibra** os parâmetros.
- **Não depende de quem vem depois:** o `fairness_by_age` que o guardrail do retreino usa é **seu** (o Erick o reaproveita). Os nomes das métricas já estão definidos na seção 6: o Pedro publica depois.
- **Seu código não depende do pipeline do Pedro.** O retreino lê os lotes validados de `data/interim/<lote>/model_input.parquet`, que o **seu** script de ponta a ponta grava com as funções da Fabi (`validate_model_input` + `build_model_input`); o estágio `validate` do Pedro grava depois no mesmo lugar. O retreino devolve a decisão e **não publica métricas**: o Pedro publica a partir dela.
- **Entrega o Pacote 2 para o Pedro** (seção 6) no fim da semana 2.
- **Semana 3 (depois de entregar):** documente o que construiu:
  - `docs/retraining-policy.md`;
  - as ADRs de faixas de PSI/KS e teste z, MMD, CBPE e champion/challenger;
  - os runbooks de drift e modelo (formato em `docs/guias/README.md`);
  - a **linha do tempo do incidente** (M1–M7: o que aconteceu, o que o sistema viu, a decisão), que o Erick usa no post-mortem.
- **Semana 4:** revisão cruzada.

## 1. Contexto do projeto

Tech Challenge da Fase 4 do MLET (FIAP). Uma fintech tem um modelo de credit scoring em produção e suspeita de **degradação silenciosa** por mudanças na economia (inflação, novos perfis de cliente). O grupo constrói a **camada de sustentação**: contratos de dados, simulação e detecção de drift, observabilidade e governança LGPD.

- **Dataset:** Give Me Some Credit (Kaggle), 150.000 clientes, 10 features, rótulo `default_2y` (6,7% de positivos).
- **Pipeline batch mensal**, sem API. Lotes simulados de 2026-01 a 2026-07.
- **Stack:** Python 3.12, `uv`, pandas, scikit-learn, XGBoost, Evidently 0.7 (API nova), NannyML 0.13, scipy, MLflow 3, fairlearn.
- **Critérios de avaliação:** Validação 25% · **Drift 25%** · Observabilidade 20% · Governança 15% · Vídeo 15%.

**Sua etapa vale 25%.** Ela transforma "a diretoria suspeita" em evidência: simula a passagem do tempo, mede o drift, diz se é problema de população ou de conceito e decide se retreina.

## 2. O que o enunciado pede na Etapa 2

- Criar um "Dataset de Produção" simulando a passagem do tempo, alterando intencionalmente a distribuição de **pelo menos duas variáveis importantes** (ex.: renda, desemprego).
- Gerar **predições** para esse novo dataset.
- Usar o **Evidently AI** para comparar o dataset de Referência com o de Produção.
- Identificar, com **testes estatísticos (PSI, KS etc.)**, quais features sofreram drift.
- **Entregável:** relatório de drift (HTML do Evidently ou painel customizado) apontando claramente as variáveis degradadas.

Boa prática obrigatória: **relatório estatístico** evidenciando a mudança de distribuição (PSI ou KS).

## 2.1 Prioridade: núcleo primeiro, extras depois

O objetivo é **cumprir o enunciado**. Faça o núcleo inteiro antes de qualquer extra. Se o prazo apertar, os extras são cortados sem prejudicar o requisito.

| Núcleo (obrigatório) | Requisito atendido |
|---|---|
| Lotes de produção simulando o tempo, com **≥ 2 variáveis importantes alteradas** (covariate shift) | Etapa 2 |
| **Concept drift** simulado (o contexto do enunciado cita "Data Drift e Concept Drift") | contexto do enunciado, critério Drift 25% |
| Predições do modelo nos lotes | Etapa 2 |
| Relatório **Evidently** Referência × Produção por lote (HTML) | Etapa 2 (entregável) |
| **PSI e KS** apontando as features com drift | boa prática obrigatória |
| AUC por lote com rótulo, mostrando a queda no concept drift | critério Vídeo ("demonstração da degradação") |

| Extra (diferencial) | Custo |
|---|---|
| PSI por quantis como stattest do Evidently | baixo — **recomendado**: o PSI nativo esconde o drift da renda |
| Teste z da taxa de default (label drift) | baixo |
| Gate com diagnóstico e política em camadas | baixo — recomendado, dá a narrativa "nem todo drift pede retreino" |
| MMD multivariado + lote `joint_drift` | médio |
| CBPE (NannyML): AUC estimada sem rótulo | médio (exige modelo calibrado) |
| Retreino champion/challenger com guardrails e rollback | alto |

## 3. Decisões já tomadas (não reabrir sem o grupo)

| Tema | Decisão | Motivo (aulas de Data Drift) |
|---|---|---|
| Linha do tempo | M1–M2 estáveis, M3–M4 **covariate shift** gradual, M5–M7 **concept drift**; mais os lotes de demo `corrupted` e `joint_drift` | Controle de falso alarme + os dois tipos de drift da taxonomia |
| Covariate shift | Renda real ↓, `debt_ratio` ↑, utilização ↑, idade ↓, com intensidade 0,6 → 1,0; rótulos originais mantidos | Mais de 2 variáveis alteradas, como o enunciado pede |
| Concept drift | **Relabel por segmento contra o que o modelo aprendeu**: inadimplentes com atraso "curam"; adimplentes sem atraso e com utilização > 0,7 passam a inadimplir. X não muda | Gerar rótulos a partir das predições do próprio modelo faz a AUC **subir** (erro já cometido) |
| Triagem × confirmação | **PSI** para triagem (OK < 0,10 ≤ WARN ≤ 0,25 < ALERT), **estatística D do KS** > 0,10 para confirmar | Com milhares de linhas o p-valor do KS acusa tudo: ele não é gate |
| PSI | Bins pelos **quantis da Referência**, deduplicados, com epsilon | O PSI nativo do Evidently usa bins de largura igual e apaga o drift em variáveis de cauda longa |
| Evidently | API 0.7 (`from evidently import Report`), com o PSI por quantis e o KS D registrados como **stattests customizados** | O HTML mostra os mesmos números que o gate usa |
| Label drift | **Teste z** da taxa de default contra a Referência, alerta com z > 3 | O PSI de uma taxa binária é minúsculo (0,066 → 0,088 dá ~0,007) |
| Multivariado | **MMD** com kernel RBF e p-valor por permutação, α = 0,001 | Pega correlação quebrada com marginais iguais; α calibrado para não gerar falso alarme |
| Sem rótulo | **CBPE** (NannyML) estima a AUC; o gap estimada − realizada é a assinatura do concept drift | Em crédito o rótulo chega meses depois |
| Gate | Política em camadas: **só retreina com queda de AUC realizada > 0,05** | "Nem todo drift pede retreino" |
| Retreino | Champion/challenger com 3 guardrails (ganho de AUC recente, sem esquecimento, fairness) e rollback por alias | Fecha o ciclo detectar → corrigir |

## 4. Resultados de referência (calibrados no dado real)

> **Não precisa bater os números.** O que importa é o **comportamento**: meses estáveis sem alerta; covariate shift com PSI > 0,25 em ≥ 2 features **sem** queda relevante de AUC; concept drift com PSI ≈ 0 **e** queda de AUC > 0,05. Se o seu resultado mostra isso, está certo.

| Lote | AUC (v1) | AUC estimada (CBPE) | PSI renda / dívida / idade | Gate esperado |
|---|---|---|---|---|
| M1 | 0,864 | 0,864 | ≈ 0 | OK |
| M2 | 0,867 | 0,863 | ≈ 0 | OK |
| M3 | 0,854 | 0,858 | 0,13 / 0,18 / 0,11 (WARN; o MMD já acusa) | ALERT, covariate, sem retreino |
| M4 | 0,823 | 0,850 | **0,48 / 0,36 / 0,27**, KS D 0,19–0,26 | ALERT, covariate, **sem retreino** |
| M5 | 0,843 | 0,865 | ≈ 0; taxa de default z = 3,2 | ALERT, prior drift |
| M6 | **0,798** | 0,864 | ≈ 0 | ALERT, **concept drift → retreino** |
| M7 | 0,854 (v2) | 0,853 | ≈ 0; z = 6,3 | ALERT, prior drift |
| joint_drift | 0,852 | — | ≈ 0; só o MMD acusa (p = 0,0005) | ALERT, joint drift |

Parâmetros que chegaram nesses números (`params.yaml`):
- **Covariate:** `income_real_drop` 0,40, `debt_ratio_increase` 0,80, `utilization_increase` 0,35, `age_shift_years` −8.
- **Concept:** `cure_fraction` 0,60, `new_default_fraction` 0,45, `utilization_threshold` 0,7; intensidade de M5 = 0,4.
- **Retreino:** peso 10 nas linhas recentes. O M6 promove a v2, com AUC no holdout de 0,811 → 0,868.

**Atenção ao ruído:** lotes de 5.000 linhas **sem drift nenhum** variaram de 0,833 a 0,867 de AUC só por amostragem. Não calibre limiares menores que isso.

## 5. Passo a passo

1. **Simulador** (`drift_simulator.py`):
   - embaralhe o pool com seed fixa e fatie lotes **disjuntos** de `batch_size` (5.000);
   - aplique o cenário de cada mês (`stable`, `data_drift`, `concept_drift`) com a intensidade;
   - gere os lotes de demo: `corrupted` (com o `make_corrupted` da Fabi, importado) e `joint_drift` (permute renda e dívida de forma independente, o que preserva as marginais);
   - grave `data/production/<lote>.parquet` e um `manifest.json`.
2. **Calibração:** rode uma varredura de parâmetros medindo AUC, PSI e taxa de default por mês com o modelo da Fabi. Metas:
   - M4 com PSI > 0,25 em pelo menos 2 features e **queda de AUC < 0,05**;
   - M6 com PSI ≈ 0 e **queda de AUC > 0,05**;
   - taxa de default subindo em M5–M7.
3. **`stats.py`:**
   - `psi(ref, cur)` por quantis da Referência;
   - `ks_d(ref, cur)` → (D, p);
   - `mmd_test(ref, cur)` com **permutações vetorizadas** (todas num único `K @ máscaras`, ~0,8 s por lote);
   - `proportion_z(taxa_ref, rótulos)`.
4. **`drift.py`:**
   - tabela por feature (PSI, KS D, status, confirmado);
   - MMD sobre as features com `log1p` nas caudas longas;
   - teste z;
   - relatório Evidently (`DataDriftPreset` com o stattest `quantile_psi` + `ValueDrift` com `ks_d` nas contínuas + drift do score + `ClassificationPreset` se houver rótulo);
   - saída `DriftResult` e `reports/drift/<lote>.html|json`.
5. **`performance.py`:**
   - métricas realizadas (AUC, PR-AUC, KS, Brier) quando há rótulo;
   - quantis do score e taxa de aprovação;
   - CBPE ajustado na Referência de teste pontuada e o lote estimado como um único chunk;
   - saída `PerformanceResult`, com a propriedade `estimation_gap`.
6. **`gate.py`:**
   - junta drift e performance em `status` (OK/WARN/ALERT), `diagnosis` (covariate, concept, joint, prior, none), `recommended_action` (observe, reinforced_monitoring, investigate_and_recalibrate, retrain) e a lista `failures`;
   - PSI em ALERT **sem** confirmação do KS D conta só como WARN.
7. **Retreino** (`retrain.py`):
   - janela = Referência + os últimos 2 lotes rotulados **que passaram no contrato** (lidos de `data/interim/<lote>/model_input.parquet`, gravado pelo seu script de ponta a ponta — o mesmo layout que o estágio `validate` do Pedro usa);
   - holdout de 30% do lote gatilho;
   - peso 10 nas linhas recentes;
   - mesmo algoritmo, calibrado;
   - guardrails:
     - ganho de AUC no holdout ≥ 0,02;
     - perda na Referência ≤ 0,03;
     - impacto desigual por idade sem piora > 0,05 (com o seu `fairness_by_age`, em `src/governance/fairness.py`: fairlearn `MetricFrame` por faixa etária; decisão 1 = negar, então aprovação = 1 − taxa de seleção);
   - aliases `challenger` / `production` / `previous`, rollback e decisão em `reports/retrain/<lote>.json`;
   - **sem publicar métricas**: devolva a `RetrainDecision`. O Pedro publica `retrain_runs{outcome}` e `retrain_guardrail_passed{guardrail}` a partir dela.
8. **Testes:**
   - PSI, KS e MMD em casos com resposta conhecida (MMD: correlação 0,8 contra 0,0, com as mesmas marginais);
   - simulador determinístico e concept drift que só muda rótulos, na direção contrária ao modelo;
   - gate em cada caminho da política;
   - guardrails.

## 6. Contratos entre as etapas

**Você recebe:**
- da **Fabi**: `production_pool.parquet` (sem imputação), `reference.parquet` (coluna `split`), `build_model_input`, `load_model()` (com `predict_proba` e `threshold`) e `evaluate.py`;
- só isso. Os estágios do Pedro vão chamar as suas funções depois; entregue-as como funções puras (DataFrame entra, dataclass serializável em JSON sai), sem depender de pipeline.

**Você entrega:**

| Artefato | Quem usa | Formato |
|---|---|---|
| `data/production/<lote>.parquet` | Pedro (ingest) | `customer_id`, 10 features, `default_2y` |
| `detect_drift(ref_pontuada, lote_pontuado, lote)` → `DriftResult` | Pedro (estágio drift) | JSON: `features[]` (feature, psi, ks_d, status, ks_confirmed), `drift_share`, `drifted_features`, `score_psi`, `mmd2`, `mmd_p_value`, `default_rate`, `label_z`, `report_paths` |
| `evaluate_performance(...)` → `PerformanceResult` | Pedro (estágio performance) | `roc_auc`, `estimated_roc_auc`, `reference_roc_auc`, `brier`, `approval_rate`, `score_p50/p90/p99`, `estimation_gap` |
| `evaluate_gate(drift, perf)` → `GateResult` | Pedro (estágio gate, ramo de retreino no Airflow) | `status`, `diagnosis`, `recommended_action`, `failures[]` |
| `retrain(lote)` → `RetrainDecision` e `rollback()` | Pedro (DAG de retreino) | JSON da decisão |
| Colunas de score no lote pontuado | Pedro, Erick | `score` e `prediction` |
| Simulador reutilizável (`apply_data_drift`, `apply_concept_drift`) | Erick (análise causal) | — |
| `fairness_by_age(y, score, threshold, idade)` | você (guardrail), Erick (análise e mitigação) | aprovação, TPR e FPR por faixa (18–29, 30–44, 45–59, 60+), `disparate_impact`, `tpr_gap` |

**Pacote 2, entregue ao Pedro com a tag `pacote-2`:** os artefatos acima, `make simulate` e um script (ou notebook) que roda drift → performance → gate → retreino para M1–M7 e reproduz a tabela da seção 4.

Nomes das **métricas** que o Pedro vai publicar a partir dos seus resultados (já combinados, não mude): `feature_psi{feature}`, `feature_ks_d{feature}`, `multivariate_mmd_pvalue`, `label_drift_z`, `model_roc_auc`, `model_estimated_roc_auc`, `performance_estimation_gap`, `drift_gate_status`.

## 7. Armadilhas que já encontramos

- **Gerar rótulos a partir das predições do modelo** reforça o que ele já sabe e a AUC sobe (0,87 → 0,91). O relabel precisa ir **contra** os sinais que o modelo usa (atrasos e utilização).
- **Renda imputada pela mediana:** ~20% dos valores ficam num único ponto. Escalar a renda **antes** da imputação (no pool bruto) dá um PSI proporcional; escalar depois desloca o pico inteiro e o PSI explode (> 1,0).
- **Evidently PSI nativo:** usa bins de largura igual sobre a faixa combinada. Em M4, a renda dá 0,02 no nativo contra 0,48 por quantis. Registre o seu stattest com `register_stattest` (API legacy: `evidently.legacy.calculations.stattests.registry`).
- **Contagem de drift share:** o Evidently inclui o score na conta (3/11); o gate usa só as features (3/10). Documente a diferença.
- **MMD com α = 0,01** deu falso alarme em 25% das sementes no M1, que é estável mas atípico. Com α = 0,001, 2.000 permutações e subamostra de 2.500, a separação ficou limpa. Um loop Python de permutações com `np.ix_` leva ~5 s; o vetorizado leva 0,8 s.
- **CBPE precisa de probabilidades calibradas.** O peso de classe do XGBoost distorce as probabilidades, e a Fabi calibra com isotônica.
- **NannyML e dependências:** exigiu fixar `kaleido==0.2.1` (o 0.2.1.post1 só tem wheel ARM) e fez o XGBoost resolver para 2.1.x. Desligue a telemetria com `NML_DISABLE_USAGE_LOGGING=1`.
- **Ruído de AUC entre lotes** (±0,02): um M4 "azarado" parece queda de performance. Olhe os lotes estáveis antes de concluir.
- **Duplicatas criadas pelo drift:** deslocar a idade e cortar em 18 cria perfis idênticos. O contrato da Fabi trata isso como warning; não bloqueie.
- **Retreino sem peso** dilui o conceito novo: 8.500 linhas recentes contra 71.585 da Referência. Peso 1 recupera o M7 só até 0,827; peso 10, até 0,854.
- **XGBoost multi-thread** não é bit a bit determinístico: a AUC do challenger varia na terceira casa entre execuções. Arredonde nos documentos.

## 8. Checklist de pronto

- [ ] `make simulate` gera M1–M7 + `corrupted` + `joint_drift`, disjuntos e reprodutíveis.
- [ ] Relatório Evidently por lote, com PSI por quantis e KS D, apontando renda, dívida e idade em M4.
- [ ] Gate reproduz a tabela da seção 4 (M4 sem retreino, M6 com retreino, `joint_drift` só pelo MMD).
- [ ] CBPE acompanha a AUC realizada em M3–M4 e descola em M6 (gap > 0,05).
- [ ] Retreino promove o challenger em M6 com os 3 guardrails; `make rollback` funciona nos dois sentidos.
- [ ] Testes de PSI, KS, MMD, simulador, gate e guardrails.
- [ ] ADRs escritas:
  - faixas de PSI e KS D e teste z;
  - MMD próprio (sem Alibi Detect: dependência pesada e licença BSL);
  - CBPE e o seu ponto cego;
  - champion/challenger.
- [ ] `docs/retraining-policy.md` com a política em camadas.
- [ ] Runbooks de drift e modelo e linha do tempo do incidente entregues ao Erick (semana 3).

## 9. Referência: projeto anterior

Implementação completa para consultar quando travar (repositório privado; peça acesso): https://github.com/eeyamazaki/credit-scoring-sustentacao. Use para **comparar**, não para copiar: o objetivo é o grupo construir e entender cada parte.

Arquivos equivalentes no projeto anterior:

`src/simulation/drift_simulator.py`, `src/monitoring/{stats,drift,performance,gate}.py`, `src/training/retrain.py`, `params.yaml` (seções `simulation`, `drift`, `performance`, `retrain`), `tests/unit/test_{simulator,stats,drift,gate,retrain}.py`, `docs/decisions/00{2,6,7,8}-*.md`, `docs/retraining-policy.md`.

Aulas da disciplina de Data Drift: Aula 1 (taxonomia: covariate, prior, concept), Aula 2 (PSI e KS; limites do p-valor), Aula 3 (nem todo drift pede retreino), Aula 4 (monitoramento e restauração de features), Aula 5 (label drift), Aula 7 (testes estatísticos na prática), Aula 8 (Evidently).

## 10. Prompt inicial sugerido para a LLM

```
Você vai me ajudar na Etapa 2 (Simulação e Detecção de Drift) do Tech Challenge
descrito no documento acima. Siga as "Decisões já tomadas" e os "Contratos entre as
etapas" sem alterá-los. Comece pelo passo 1: o simulador com seed fixa, lotes
disjuntos e os cenários stable, data_drift e concept_drift, com os parâmetros vindo
do params.yaml. Depois vamos calibrar contra as metas da seção 5, passo 2. Escreva
Python 3.12 com type hints, docstrings em português e testes pytest. Nunca invente
números de resultado: me peça para rodar e colar a saída.
```
