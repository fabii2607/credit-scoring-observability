# Credit Scoring Observability

**Tech Challenge – Fase 04 | Pós-Tech em Machine Learning Engineering (FIAP)**

Projeto acadêmico de classificação binária de risco de crédito, qualidade de dados e preparação para monitoramento de modelos. A base é o **All Lending Club Loan Data**. A frente documentada aqui implementa EDA, engenharia e preparação de features, Regressão Logística baseline (V1), otimização experimental (V2), contrato de entrada com Pandera e demonstração do bloqueio de um lote inválido (*bad batch*).

> **Estado da integração:** os testes automatizados locais da frente de dados/modelagem passaram (16 testes, conforme última execução do grupo). As etapas de observabilidade, drift e demais integrações do Tech Challenge pertencem à sequência do projeto e não são declaradas prontas neste README.

## 1. Dataset e definição do problema

**Fonte:** [All Lending Club Loan Data, por wordsforthewise, no Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

Baixe o arquivo **`accepted_2007_to_2018Q4.csv`** e coloque-o em:

```text
data/raw/accepted_2007_to_2018Q4.csv
```

A pasta `data/` é ignorada pelo Git: nenhum CSV bruto ou gerado é distribuído diretamente neste repositório. Cada integrante deve obter sua cópia pela fonte indicada. Se o download vier compactado, extraia o `.csv` antes de rodar os notebooks.

O arquivo de empréstimos aceitos tem aproximadamente **2.260.701 registros e 151 colunas**. A população de modelagem tem **1.348.099 empréstimos** com desfecho conhecido e usa o seguinte target:

| Classe | Desfechos incluídos |
|---|---|
| `0`: não inadimplente | `Fully Paid`; `Does not meet the credit policy. Status:Fully Paid` |
| `1`: inadimplente | `Charged Off`; `Default`; `Does not meet the credit policy. Status:Charged Off` |

`Current`, `Late`, `In Grace Period` e registros sem `loan_status` ficam fora da modelagem porque não representam desfecho final conhecido. A prevalência da classe 1 é de aproximadamente **19,98%**.

**Limitações:** trata-se de dados históricos. A análise por ano de originação (`issue_d`) deve considerar o viés de maturação: empréstimos recentes ainda abertos não entram na população rotulada. O modelo é uma prova de conceito acadêmica, não um mecanismo validado para decisão real de concessão de crédito.

## 2. Como configurar o ambiente

**Recomendado:** Python 3.11 e [`uv`](https://docs.astral.sh/uv/). Execute os comandos na raiz do repositório:

```powershell
uv sync
uv run pytest -v
```

O `uv sync` utiliza `pyproject.toml` e `uv.lock`. Para notebooks no VS Code, selecione o interpretador/kernel **`.venv/Scripts/python.exe`**. Se necessário, abra a pasta `notebooks/` e confirme que `Path('../data/raw/accepted_2007_to_2018Q4.csv').exists()` retorna `True` antes da leitura.

**Importante:** o `uv.lock` versionado deve acompanhar qualquer mudança de dependências. Se o `pyproject.toml` for editado, execute `uv lock` (ou `uv sync`) e inclua ambos no commit.

## 3. Estrutura do projeto

```text
credit-scoring-observability/
├── data/                              # arquivos locais; não versionados
│   ├── raw/
│   ├── reference/
│   ├── processed/
│   └── invalid/
├── models/                            # artefatos produzidos localmente
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_optimization.ipynb
│   └── 05_data_contract.ipynb
├── src/credit_scoring_observability/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── data_contract.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_data_contract.py
├── docs/
│   └── HANDOFF_FABI.md
├── .gitignore
├── .python-version
├── pyproject.toml
└── uv.lock
```

As pastas locais sob `data/` são criadas pelos notebooks conforme necessário. Os diretórios `models/` e `docs/` podem conter também arquivos criados por outras frentes do projeto.

## 4. Ordem de execução dos notebooks

Abra os notebooks **em ordem**, com o kernel do ambiente do projeto. Eles pressupõem que `../data/` aponta para a pasta local de dados.

| Notebook | Objetivo e entregável |
|---|---|
| `01_eda.ipynb` | Análise global em chunks, definição de target, nulos, auditoria de features/leakage, análise numérica, categórica e temporal. |
| `02_preprocessing.ipynb` | Transformações determinísticas, split 80/20 estratificado, pré-processador e criação de `data/reference/reference_dataset.csv` (100.000 linhas do treino, com target e `issue_date`). |
| `03_baseline_model.ipynb` | Regressão Logística V1 em amostra estratificada de 300.000 linhas do treino, avaliada no teste completo de 269.620 linhas. Gera modelo e métricas V1. |
| `04_model_optimization.ipynb` | Validação interna para comparar configuração padrão e `class_weight='balanced'`, seleção experimental de threshold por F2 condicionada à Precision mínima e avaliação final da V2. Gera modelo e metadados V2. |
| `05_data_contract.ipynb` | Valida lote correto com Pandera, cria `data/invalid/bad_batch.csv`, demonstra falhas do schema, comprova bloqueio antes da inferência e testa predições opcionais com a V2. |

**Memória:** não execute os cinco notebooks simultaneamente. O carregamento dos notebooks de treinamento foi ajustado em chunks de 25.000 linhas, com redução de tipos, porque carregar a base inteira de uma vez pode causar `MemoryError`. A V1/V2 treinam em uma amostra de 300 mil registros; o teste estratificado de cerca de 270 mil é preservado integralmente. O Reference Dataset não substitui o treino.

## 5. Modelos e métricas

Os arquivos são gerados **localmente**, na pasta `models/`:

| Artefato | Origem | Papel |
|---|---|---|
| `credit_scoring_baseline.joblib` | Notebook 03 | Pipeline completo V1 (pré-processamento + Regressão Logística). |
| `baseline_metrics.json` | Notebook 03 | Métricas V1 no teste final. |
| `credit_scoring_optimized.joblib` | Notebook 04 | Pipeline completo V2. |
| `credit_scoring_optimized_metadata.json` | Notebook 04 | Configuração escolhida, threshold, ordem das features e métricas V2. |

A V1 utiliza threshold de classificação de 0,50. Na execução registrada da V2, o threshold selecionado na validação foi **0,20**. Para reproduzir a decisão V2, **não use diretamente `pipeline.predict()`**, que segue a regra padrão do estimador. Leia `decision_threshold` dos metadados e compare com `predict_proba(X)[:, 1]`.

A comparação V1 × V2 mostrou, aproximadamente, Recall de **4,5%** na V1 e **63,5%** na V2, acompanhado de queda de Precision. ROC-AUC permaneceu perto de **0,694** e Average Precision perto de **0,357**. São resultados experimentais; os valores exatos devem ser lidos dos JSONs e gráficos gerados nos notebooks. Seleção de modelo/threshold foi feita na validação, não no teste final.

**Compartilhamento importante:** `.joblib` e `.pkl` estão no `.gitignore` e **não aparecem ao clonar o GitHub**. Para executar a inferência sem retreinar, cada colega deve receber ambos os arquivos da V2 (`credit_scoring_optimized.joblib` e `credit_scoring_optimized_metadata.json`) por um canal aprovado pelo grupo, por exemplo uma Release do repositório ou armazenamento compartilhado. O JSON de metadados sozinho **não contém o modelo**. Como alternativa, execute os notebooks 03 e 04, usando o mesmo dataset, código e `uv.lock`. Carregue arquivos `joblib` apenas de origem confiável.

## 6. Contrato de entrada e inferência validada

O contrato fica em `src/credit_scoring_observability/data_contract.py` e valida as **23 features finais**, isto é, após `engineer_features()` e `split_features_target()`. Ele **não** recebe diretamente as 151 colunas do CSV bruto.

Valida, entre outros pontos: presença e ausência de colunas extras, empréstimo positivo, `term` em `{36, 60}`, renda não negativa/obrigatória, DTI não negativo quando presente, FICO em faixa aceita, categorias permitidas de `application_type` e coerência entre `emp_length_years == -1` e `emp_length_missing == 1`. Alguns nulos numéricos são permitidos porque o pipeline faz imputação. O contrato não aplica um teto artificial de 100% à utilização de crédito rotativo.

Depois de obter o modelo e o Reference Dataset, este é um exemplo de inferência **a partir da raiz do repositório**:

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from credit_scoring_observability.data_contract import (
    MODEL_FEATURES,
    predict_proba_validated,
)

batch = pd.read_csv("data/reference/reference_dataset.csv", nrows=5)
batch = batch.loc[:, MODEL_FEATURES]  # sem target e sem issue_date

model = joblib.load("models/credit_scoring_optimized.joblib")
meta = json.loads(
    Path("models/credit_scoring_optimized_metadata.json").read_text(encoding="utf-8")
)
probabilities = predict_proba_validated(model, batch)
classes = (probabilities >= float(meta["decision_threshold"])).astype(np.int8)

print(pd.DataFrame({"probabilidade_default": probabilities, "classe_v2": classes}))
```

O gate `predict_proba_validated()` valida uma cópia profunda do lote antes de invocar o modelo. Dados inválidos levantam exceção do Pandera e **não são pontuados**. Não use o conjunto de referência como se fosse um lote independente para medir performance de generalização: ele foi amostrado do treino.

## 7. Bad Batch e testes automatizados

O notebook 05 cria localmente `data/invalid/bad_batch.csv`, alterando campos como valor negativo de `loan_amnt`, prazo inválido, renda ausente, FICO fora da faixa e incoerência no indicador de tempo de emprego. Sua execução demonstra tanto as mensagens do Pandera quanto a ausência de chamada ao modelo quando a validação falha.

Na raiz do repositório:

```powershell
uv run pytest -v
# testes específicos, caso necessário:
uv run pytest tests/test_preprocessing.py -v
uv run pytest tests/test_data_contract.py -v
```

**Última execução informada pela equipe:** 16 testes aprovados (14 do contrato e 2 de pré-processamento). Reexecute após atualizar arquivos ou dependências.

## 8. Handoff para a frente de observabilidade

Consulte [docs/HANDOFF_FABI.md](docs/HANDOFF_FABI.md) para obter entradas, saídas, ordem de reprodução, contrato das 23 features, artefatos não versionados e cuidados para integrar drift/monitoramento.

**Checklist mínimo para os próximos responsáveis:** Python/`uv` configurados; dataset local quando necessário; `reference_dataset.csv` disponível; modelo V2 **e** metadados/threshold disponíveis; lote de produção com as mesmas 23 features; validação Pandera executada antes de toda inferência; `uv run pytest -v` aprovado.
