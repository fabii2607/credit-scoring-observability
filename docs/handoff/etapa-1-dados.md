# Handoff — Etapa 1: Validação de Dados e Contratos

> Documento autocontido. Cole o arquivo inteiro no início de uma conversa com uma LLM para continuar a Etapa 1 ou para consumir o que ela entrega.

## 1. Onde o projeto está

Tech Challenge Fase 4 (FIAP MLET): uma fintech tem um modelo de credit scoring em produção e suspeita de degradação silenciosa. O grupo constrói a camada de sustentação: contratos de dados (Etapa 1), drift (Etapa 2), observabilidade (Etapa 3) e governança LGPD (Etapa 4).

A Etapa 1 foi feita em **notebooks** pela Fabi (PR #1) e depois **consolidada em scripts**, para as outras etapas consumirem sem rodar notebook:

| O que | Onde |
|---|---|
| EDA, pré-processamento, baseline V1, otimização V2, contrato e bad batch | `notebooks/01_eda.ipynb` … `05_data_contract.ipynb` |
| Funções de dados (população, target, features derivadas, split) | `src/credit_scoring_observability/preprocessing.py` |
| Contrato Pandera das 23 features | `src/credit_scoring_observability/data_contract.py` |
| Referência × pool de produção, pseudonimização, região | `src/credit_scoring_observability/prepare.py` (`make prepare`) |
| Treino da V2 em script + threshold | `train.py`, `evaluate.py`, `registry.py` (`make train`) |
| Script de validação operante, quarentena, lote corrompido | `validate.py` (`make validate BATCH=...`, `make demo-contract`) |

## 2. Dataset e população

- **Fonte:** [All Lending Club Loan Data (Kaggle, wordsforthewise)](https://www.kaggle.com/datasets/wordsforthewise/lending-club), arquivo `accepted_2007_to_2018Q4.csv` (ou o `.csv.gz` baixado), em `data/raw/`. A pasta `data/` não vai para o git.
- **Target:** `1` = `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`; `0` = `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`. `Current`, `Late` e `In Grace Period` ficam fora (sem desfecho conhecido).
- **População:** 1.348.099 empréstimos, default de **19,98%**, originados de 2007-06 a 2018-12.
- **23 features** (`data_contract.MODEL_FEATURES`), todas conhecidas **na concessão**. A auditoria de vazamento do notebook 01 excluiu as colunas pós-desfecho:
  - numéricas: `loan_amnt`, `term`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `collections_12_mths_ex_med`, `acc_now_delinq`, `pub_rec_bankruptcies`, `tax_liens`;
  - derivadas: `emp_length_years` (−1 = ausente), `emp_length_missing`, `fico_avg` (média de `fico_range_low`/`high`), `credit_history_years`;
  - categóricas: `home_ownership`, `verification_status`, `purpose`, `application_type`.
- **Limitação:** as safras de 2016–2018 têm rótulo censurado, porque só entram os empréstimos já encerrados (2018: 15,7% de default, contra 23% em 2016–2017). Documente isso ao analisar o tempo.

## 3. Decisões já tomadas

| Tema | Decisão |
|---|---|
| Split | 80/20 estratificado, `random_state=42` (notebooks 02–04). `prepare` reproduz exatamente as mesmas linhas. |
| Referência × Produção | **Referência** = treino (1.078.479 linhas). **Pool de produção** = o teste (269.620), nunca visto no treino, de onde saem os lotes mensais. **Base do drift** = `reference_sample.parquet` (as 100 mil linhas do Reference Dataset do notebook 02). |
| Identificador | `customer_id` = SHA-256 de `"{sal}:{id}"`, 16 caracteres; o sal vem do `.env` (`PSEUDONYMIZATION_SALT`) e não vai para o git. O `id` original é descartado no prepare. **É pseudonimização, não anonimização.** |
| Atributo para fairness | `region` (Northeast, Midwest, South, West) derivada de `addr_state`. Não há idade no Lending Club. **Nunca é feature.** |
| Imputação | Feita **dentro** do pipeline do modelo (mediana nas numéricas, `"Unknown"` nas categóricas). Os parquet ficam sem imputação, e o contrato aceita nulos onde o pipeline imputa. |
| Modelo | **V2**: Regressão Logística (`saga`, sem `class_weight`) com RobustScaler e one-hot esparso, treinada numa amostra estratificada de 300 mil linhas da Referência. Threshold **0,20**, escolhido na validação: maior F2 com Precision ≥ 0,30. |
| Decisão | `score >= 0,20` → classe 1 (alto risco, nega). `pipeline.predict()` usa 0,5: **não use**. Use `load_model()`. |
| Contrato | Pandera na camada `model_input` (23 features, `strict`, `coerce`, `lazy`) + regras de lote em `validate.py`. Great Expectations na camada raw é um extra que não foi feito. |

## 4. Números da execução (dado real)

`make prepare` (≈ 25 s) e `make train` (≈ 1 min) reproduzem os notebooks:

| Item | Valor |
|---|---|
| Referência / pool / amostra de referência | 1.078.479 / 269.620 / 100.000 |
| Default na Referência e no pool | 19,98% / 19,98% |
| Linhas por região (população) | South 479.503 · West 361.139 · Northeast 272.229 · Midwest 235.228 |
| `customer_id` em comum entre Referência e pool | 0 (o `prepare` falha se houver) |
| V2 no pool: ROC-AUC / PR-AUC / KS / Brier | 0,6936 / 0,3569 / 0,278 / 0,147 |
| V2 no pool: precision / recall / F2 / aprovação | 0,308 / 0,632 / 0,522 / 59,0% |
| V1 (threshold 0,5), para comparar | recall 0,045, AUC 0,694 (`models/baseline_metrics.json`) |
| `make demo-contract` | exit 2, 8 regras blocker, lote em `data/quarantine/`, relatório em `reports/contracts/corrupted.json` |

## 5. Interfaces que a Etapa 1 entrega

```python
from credit_scoring_observability.config import (
    REFERENCE_SAMPLE_FILE, PRODUCTION_POOL_FILE, SCORE_COLUMN, PREDICTION_COLUMN,
)
from credit_scoring_observability.registry import load_model
from credit_scoring_observability.evaluate import classification_metrics, approval_rate, select_threshold
from credit_scoring_observability.validate import validate_batch, enforce, ContractViolationError, make_corrupted

model = load_model()                    # ScoringModel(pipeline, threshold=0.2, version, feature_columns)
scored = model.score(lote)              # cópia do lote + "score" (prob. de default) + "prediction" (0/1)
proba = model.predict_proba(lote)       # aplica o contrato antes; lote inválido levanta SchemaErrors

result = validate_batch(lote, "2026-03")  # ValidationResult: violations[rule, column, severity, count, layer, examples]
enforce(lote, result, source_path)        # grava reports/contracts/<lote>.json; blocker -> quarentena + ContractViolationError
```

**Colunas dos parquet** (`reference.parquet` tem também `split = "train"`): `customer_id`, `issue_date`, `region`, as 23 features e `target` (int8).

**Regras do contrato:**

| Camada | Regra (`rule`) | Severidade |
|---|---|---|
| model_input | `greater_than` (loan_amnt > 0), `isin` (term ∈ {36, 60}, application_type, emp_length_years, emp_length_missing), `in_range` (FICO 300–850), `not_nullable` (renda e categóricas), `greater_than_or_equal_to` (não negativos), `emp_length_consistency`, `column_in_dataframe` / `column_in_schema` (coluna ausente ou extra), `coerce_dtype` | blocker |
| batch | `min_volume` (`contract.min_rows`), `missing_customer_id`, `customer_id_not_null`, `duplicate_customer_id` (cliente reenviado), `unexpected_column` (ex.: `loan_status`, vazamento), `target_binary` | blocker |
| batch | `duplicate_content` (perfil idêntico com outro id), `unknown_region` | warning |

## 6. O que ainda falta da Etapa 1 (dona: Fabi)

| Item | Por quê | Prioridade |
|---|---|---|
| `docs/model-card.md` | Explicabilidade (coeficientes da LogReg) e uso pretendido; a Etapa 4 revisa a seção de fairness | Alta |
| Runbooks `ContractViolation` e `ContractWarnings` (formato no `README` dos handoffs) | Os alertas da Etapa 3 apontam para eles | Alta |
| ADR "Pandera na camada model_input" (e por que não GE na camada raw) | Decisão registrada | Média |
| Notebooks 02–05 importando `prepare`/`train`/`validate` em vez de repetir o código | Uma fonte só | Baixa |
| Great Expectations na camada raw com Data Docs | Extra (diferencial) | Opcional |

## 7. Armadilhas

- **`pipeline.predict()` usa 0,5.** O threshold da V2 é 0,20: use `load_model().predict`/`score`.
- **O Reference Dataset saiu do treino.** Não meça a generalização nele; para isso, use o pool.
- **Memória:** o CSV tem 2,26 milhões de linhas × 151 colunas. O `prepare` lê só as colunas usadas, em chunks; não carregue o CSV inteiro.
- **pandas 3:** texto vem como `str` (não `object`). O `train.py` converte as categóricas para `object` antes do `SimpleImputer`.
- **Sal de pseudonimização:** o grupo deve usar o **mesmo** sal (combinado por canal privado) para os `customer_id` baterem entre máquinas. Com o sal padrão, o `prepare` avisa no log.

## 8. Prompt sugerido para a LLM

```
Você vai me ajudar a continuar a Etapa 1 (Validação de Dados e Contratos) do Tech
Challenge descrito no documento acima, no repositório credit-scoring-observability
(Python 3.11, uv, pacote src/credit_scoring_observability). Não altere as "Decisões já
tomadas" nem as interfaces da seção 5. Comece pelo model card (docs/model-card.md) a
partir dos números da seção 4 e dos coeficientes do pipeline salvo em models/.
```
