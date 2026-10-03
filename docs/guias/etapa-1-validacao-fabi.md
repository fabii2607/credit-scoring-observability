# Guia da Etapa 1 — Validação de Dados e Contratos (Fabi)

> Documento de direcionamento e contexto para LLM. Cole o arquivo inteiro no início da conversa.

## 0. Posição no revezamento

**Você é a 1ª: semana 1, e também monta o esqueleto que todos usam.** Ninguém entrega nada para você antes, e você não depende de ninguém: a pseudonimização, a retenção e o que não pode aparecer em relatórios já estão definidos na seção 3 (o Erick só audita no fim).

Você entrega em **três partes**, para destravar os colegas o quanto antes:

| Quando | Entrega (tag git) | Destrava |
|---|---|---|
| Dia 1–2 | **Pacote 0** (`pacote-0`): esqueleto + gerador de dados sintéticos (passo 0) | Helio, Pedro e Erick começam a adiantar |
| Dia 3 | **Pacote 1a** (`pacote-1a`): `prepare` + baseline XGBoost registrado como `@production` (passos 1–3 e 8) | Helio começa a calibrar o simulador com o dado real |
| Fim da semana 1 | **Pacote 1b** (`pacote-1b`): contratos GE + Pandera, quarentena, CLI de validação e lote corrompido (passos 4–7) | Pacote 1 completo |

- **Semanas 2 e 3 (depois de entregar):** documente o que construiu:
  - o **model card** (`docs/model-card.md`);
  - as ADRs de GE × Pandera e de sentinelas/renda ausente;
  - os runbooks de `ContractViolation` e `ContractWarnings` (formato em `docs/guias/README.md`).

  Corrija o que os colegas apontarem no seu pacote.
- **Semana 4:** revisão cruzada.

## 1. Contexto do projeto

Tech Challenge da Fase 4 do MLET (FIAP). Uma fintech tem um modelo de credit scoring em produção e suspeita de **degradação silenciosa** por mudanças na economia (inflação, novos perfis de cliente). O grupo constrói a **camada de sustentação**: contratos de dados, simulação e detecção de drift, observabilidade e governança LGPD.

- **Dataset:** Give Me Some Credit (Kaggle), arquivo `cs-training.csv`, 150.000 linhas, 10 features, rótulo `SeriousDlqin2yrs` (inadimplência grave em 2 anos, 6,7% de positivos).
- **Pipeline batch mensal**, sem API. Lotes simulados de 2026-01 a 2026-07.
- **Stack:** Python 3.12, `uv`, pandas, scikit-learn, XGBoost, Pandera, Great Expectations 1.x, Evidently 0.7, MLflow 3, structlog, pydantic.
- **Critérios de avaliação:** Validação de Dados 25% · Drift 25% · Observabilidade 20% · Governança 15% · Vídeo 15%.

**Sua etapa vale 25% e é a base das outras três:** todo mundo consome a Referência, o pool de produção, a imputação e o modelo registrado que você produz.

## 2. O que o enunciado pede na Etapa 1

- Treinar um classificador base com um dataset limpo (**Dataset de Referência**).
- Criar um contrato de dados (Great Expectations **ou** Pandera) com regras estritas, por exemplo idade > 18, renda não pode ser nula, sem duplicatas.
- Forçar a passagem de um lote com erros para demonstrar o contrato **falhando e bloqueando** a ingestão.
- **Entregável:** script de validação de dados operante e modelo baseline treinado.

Boas práticas obrigatórias que caem na sua etapa:
- separação clara entre o dataset de Referência (treino) e o de Produção (onde o drift ocorre);
- pelo menos **3 regras rígidas** de qualidade de dados.

## 2.1 Prioridade: núcleo primeiro, extras depois

O objetivo é **cumprir o enunciado**. Faça o núcleo inteiro antes de qualquer extra. Se o prazo apertar, os extras são cortados sem prejudicar o requisito.

| Núcleo (obrigatório) | Requisito atendido |
|---|---|
| Limpeza + split **Referência × Produção** sem vazamento | boa prática obrigatória |
| Classificador base (XGBoost ou regressão logística) treinado na Referência, com métricas | Etapa 1 |
| Contrato com **Pandera** (ou GE) e ≥ 3 regras rígidas: idade ≥ 18, renda não nula, sem duplicatas (+ faixas) | Etapa 1, critério Validação 25% |
| Lote com erros **bloqueado**: CLI de validação com exit ≠ 0 e relatório das regras violadas | Etapa 1 ("script de validação operante") |
| `customer_id` pseudonimizado | base para a governança (Etapa 4) |
| Gerador de dados sintéticos + testes | CI sem Kaggle; destrava os colegas |

| Extra (diferencial) | Custo |
|---|---|
| Segunda ferramenta: **GE na camada raw** com Data Docs | médio |
| Quarentena (Dead Letter Channel) com relatório JSON | baixo — recomendado |
| Calibração isotônica do modelo | baixo, mas **necessária se o Helio fizer o CBPE** |
| Registro no MLflow com alias `@production` | baixo — recomendado, porque o Pedro e o Helio usam |

## 3. Decisões já tomadas (não reabrir sem o grupo)

| Tema | Decisão | Motivo |
|---|---|---|
| Ferramentas de contrato | **GE na camada raw** (antes da imputação) e **Pandera na camada model_input** (depois) | GE gera Data Docs (evidência auditável); Pandera bloqueia dentro do código |
| Renda nula | Tolerada no bruto (até 25%); **proibida depois da imputação** | ~20% da renda vem nula e a ausência é informativa (MNAR) |
| Sentinelas 96/98 | Linhas com 96/98 nas colunas de atraso são **removidas** | São códigos de erro do bureau, não contagens |
| Idade | Contrato aceita **18 a 110** (≥ 18) | 18 é a maioridade civil; documentar a diferença para o "> 18" do enunciado |
| Duplicatas | Dedup por **hash SHA-256 do conteúdo** antes do split | O id é sequencial e não detecta duplicatas de conteúdo |
| Identificador | `customer_id` = SHA-256 **com sal secreto** (do `.env`, fora do git) do id original, truncado em 16 caracteres | Pseudonimização (LGPD Art. 13, §4º); o Erick audita no fim |
| Privacidade nos relatórios | Exemplos de falha com `customer_id` **mascarado** (4 caracteres + `***`); Data Docs sem valores de linha | Nenhum identificador em claro fora da base |
| Split | 60% Referência (80/20 treino/teste) e 40% pool de produção, estratificado | O pool vira os lotes do Helio |
| Modelo | XGBoost **calibrado** (isotônica) + LogReg de comparação | O Helio precisa de probabilidades calibradas para o CBPE |
| Registro | MLflow Model Registry `credit-scoring`, alias `@production`, threshold como **tag da versão** | O retreino troca modelo e threshold juntos |

## 4. Fatos do dado real (para conferir o seu resultado)

> Os fatos do dado (contagens de linhas, nulos, sentinelas) devem bater exatamente, porque vêm do CSV. As métricas do modelo só precisam ficar **próximas** (AUC entre 0,84 e 0,87 está ótimo).

- 269 linhas com 96/98 nas **três** colunas de atraso ao mesmo tempo (0,18%).
- 1 linha com `age = 0`.
- 592 duplicatas de conteúdo, contadas depois de remover as sentinelas.
- **149.138 linhas** depois da limpeza.
- Renda nula em ~19%. Default de **5,6% com renda nula contra 7,0%** com renda informada, o que justifica criar `income_missing_flag`.
- `debt_ratio > 100` em 24.380 linhas, 22.693 delas com renda nula: sem renda, o DebtRatio vira dívida absoluta. **Não** pôr limite superior no contrato.
- Mediana de imputação: renda 5.400, dependentes 0.
- Resultado esperado do XGBoost calibrado: **AUC ≈ 0,86**, Brier de 0,138 (sem calibração) para ≈ 0,050 (calibrado). Features mais importantes: `late_90`, `late_30_59`, `revolving_utilization`, `late_60_89`.

## 5. Passo a passo

0. **Repositório novo e esqueleto do projeto (Pacote 0, dia 1–2)**, que todos usam:
   - repositório no GitHub com os 4 integrantes como colaboradores, a pasta `docs/guias/` copiada do projeto anterior e a `main` protegida (merge só com CI verde);
   - `pyproject.toml` com todas as dependências (inclusive GE, NannyML, OpenTelemetry e psutil) e o `uv.lock`, `.python-version`, `.gitignore`, `.gitattributes` (`eol=lf`), `.env.example`, pre-commit com ruff, `Makefile` com os alvos já declarados e **CI** (lint + testes);
   - `src/__init__.py` com o **opt-out de telemetria** (GE, NannyML, MLflow, Evidently) e a saída padrão em UTF-8;
   - logger básico (`setup_logging`, `get_logger`) — o Pedro estende depois sem mudar a assinatura;
   - **`tests/conftest.py` com o gerador de dados sintéticos**: mesmo schema e mesmas anomalias do CSV real (renda nula, 96/98, idade 0, duplicatas). É com ele que os colegas adiantam o trabalho antes do dado real;
   - `download.py` (Kaggle CLI, com instrução de download manual em caso de falha).
1. **Constantes e parâmetros.** Crie o `config.py` com os caminhos, o mapa de colunas para snake_case (`RAW_COLUMN_MAP`) e as listas de features. Crie o `params.yaml`, validado por pydantic com `extra="forbid"`: typo em parâmetro deve falhar.
2. **`prepare.py`**, nesta ordem:
   1. renomear as colunas;
   2. pseudonimizar o id (`sha256(f"{sal}:{id}")[:16]`, com o sal lido do `.env`);
   3. remover sentinelas (96/98 e idade < 18);
   4. calcular o hash do conteúdo das 10 features e deduplicar;
   5. split estratificado Referência/pool, com uma coluna `split` = train/test dentro da Referência;
   6. **teste de vazamento:** interseção de hashes entre treino, teste e pool tem de ser vazia;
   7. ajustar as medianas **só no treino da Referência** e salvar em `imputation.json`;
   8. salvar o pool **sem imputação**, porque cada lote precisa passar pelo contrato raw antes de imputar;
   9. gravar um `prepare_report.json` com as contagens.
3. **`features/pipeline.py`:** `build_model_input(df, stats)` cria a `income_missing_flag` e imputa com as medianas salvas. O mesmo código roda no treino e na produção.
4. **Contrato model_input (Pandera):** `DataFrameSchema` montado por função (os limites vêm do `params.yaml`), `strict=True`, `coerce=True`, `validate(lazy=True)` para coletar tudo de uma vez. Regras:
   - idade 18–110;
   - renda não nula e ≥ 0;
   - utilização e dívida ≥ 0;
   - atrasos e dependentes entre 0 e 20;
   - `income_missing_flag` em {0, 1};
   - `customer_id` único;
   - registro reenviado (mesmo id + conteúdo);
   - volume mínimo;
   - rótulo opcional (em crédito ele chega atrasado).
5. **Contrato raw (GE 1.x):** suite definida em código, contexto file-based, checkpoint com `UpdateDataDocsAction`. Severidade no `meta` de cada expectativa:
   - **blocker:** completude da renda (`mostly = 0,75`);
   - **warning:** schema, volume, idade, sentinelas nas colunas de atraso, id único.
6. **`validate.py`:**
   - `ValidationResult` com a lista de violações (`rule`, `column`, `severity`, `count`, `examples`, `layer`);
   - `enforce()`: grava `reports/contracts/<lote>.json`; com blocker, move o lote para `data/quarantine/` (**Dead Letter Channel**) e lança `ContractViolationError`;
   - exemplos do `customer_id` **mascarados** (4 caracteres + `***`).
7. **Lote corrompido e CLI de validação** (o "script de validação operante" do enunciado, sem depender do pipeline do Pedro):
   - `make_corrupted(df, rng)`: ~40% de renda nula, idade 15, renda negativa, atrasos = 98, linhas duplicadas e uma coluna extra. O simulador do Helio **importa** esta função para gerar o lote `corrupted`;
   - CLI `python -m src.contracts.validate --batch <arquivo.parquet>`: GE raw → imputação → Pandera → `enforce`. Exit 0 se passou, **exit 2** se bloqueou;
   - `make demo-contract` chama essa CLI. O Pedro troca depois para o pipeline completo, com o mesmo exit code.

   Resultado esperado: exit 2, 9 regras blocker.
8. **`train.py`:**
   - divisão: treino da Referência → 80% ajuste / 20% validação; o teste da Referência fica para as métricas finais;
   - XGBoost com `scale_pos_weight` e LogReg com `class_weight="balanced"` (`log1p` nas caudas longas + `StandardScaler`), ambos em `CalibratedClassifierCV(method="isotonic", cv=3)`;
   - métricas: ROC-AUC, PR-AUC, KS, Gini, Brier (calibrado e sem calibração) e taxa de aprovação;
   - threshold de Youden na validação;
   - MLflow: `log_input` com digest do dataset, params, métricas, importância das features e `log_model`;
   - registro com o alias `production` e as tags da versão.
9. **Testes** com o gerador sintético do passo 0 (o CI não tem Kaggle). Cubra pseudonimização, sentinelas, dedup, vazamento, imputação, cada regra do contrato, quarentena e mascaramento.

## 6. Contratos entre as etapas

**Você entrega:**

| Artefato | Quem usa | Formato |
|---|---|---|
| `data/processed/reference.parquet` | Helio, Erick | colunas: `customer_id`, 10 features, `default_2y`, `row_hash`, `split` |
| `data/processed/production_pool.parquet` | Helio (simulador) | mesmas colunas, **sem imputação**, sem `split` |
| `data/processed/imputation.json` | todos | `{"medians": {"monthly_income": ..., "dependents": ...}}` |
| `build_model_input(df, stats)` | Helio, Pedro | adiciona a flag e imputa |
| `validate_raw(df, lote)` e `validate_model_input(df, lote)` → `ValidationResult` | Pedro (estágio validate) | ver o passo 6 |
| `enforce(df, resultado)` | Pedro | lança `ContractViolationError` |
| Modelo `credit-scoring@production` + tag `decision_threshold` | Helio (score, retreino), Erick (fairness) | MLflow |
| `registry.load_model(alias)` → objeto com `predict_proba` (coluna positiva) e `threshold` | Helio, Pedro | — |
| `evaluate.py` (`classification_metrics`, `select_threshold`, `approval_rate`) | Helio (retreino, performance) | — |

**Você recebe:** só o esqueleto do projeto, com o logger básico (`get_logger`, que o Pedro estende depois sem mudar a assinatura). Não espere nada dos colegas.

**Deixe pronto para o Pedro publicar depois:** o `ValidationResult` com `rule`, `severity`, `layer` e `count` por violação. É dele que saem as métricas `contract_violations{rule,severity,layer}`, `batch_quarantined` e `data_completeness_ratio{feature}`. Você não precisa publicar métricas.

**Entregas ao Helio:** o **Pacote 1a** (tag `pacote-1a`, dia 3) com `make data` e `make train` funcionando, e o **Pacote 1b** (tag `pacote-1b`, fim da semana 1) com os contratos, `make_corrupted` e `make demo-contract` bloqueando o lote.

## 7. Armadilhas que já encontramos

- **pandas 3 mantém NaN depois de `astype(str)`:** o hash de conteúdo quebra. Use `astype("string").fillna("<NA>")` antes de concatenar.
- **`unique=[...]` do Pandera quebra no pandas 3** com índice duplicado (erro no reshape das falhas). Use um `pa.Check` próprio de tabela, `~df.duplicated(cols, keep=False)`, e faça `reset_index` antes de validar.
- **Checks de tabela no Pandera** aparecem nas falhas repetidos por coluna. Normalize o nome da regra pela mensagem de erro do check.
- **Perfis idênticos com ids diferentes** acontecem naturalmente (0,4% do dataset), e o drift do Helio cria mais (idade cortada em 18). Deixe como **warning** (`duplicate_content`); o blocker é o reenvio do mesmo cliente.
- **O GE 1.x envia telemetria por padrão.** Defina `GX_ANALYTICS_ENABLED=False` antes do import.
- **Data Docs do GE:** use `result_format` BASIC com `partial_unexpected_count: 0` para nenhum valor de linha aparecer no HTML (LGPD).
- **MLflow 3.16 serializa com skops** e exige `skops_trusted_types`. Passe `skops.io.get_untrusted_types(data=skops.io.dumps(modelo))`: confia só no objeto que você mesmo treinou.
- **O Model Registry exige backend em banco:** sem servidor, use `sqlite:///mlflow.db`, não `./mlruns`.
- **O venv do uv não tem pip:** passe `pip_requirements` explícito no `log_model`.
- **Windows:** o MLflow imprime emojis que quebram o console cp1252 quando a saída é redirecionada. Reconfigure stdout/stderr para UTF-8.

## 8. Checklist de pronto

- [ ] Pacote 0 no dia 1–2: esqueleto, CI verde e gerador sintético usável pelos colegas.
- [ ] `make data` gera Referência e pool com 149.138 linhas no total, e o teste de vazamento passa.
- [ ] `make train` registra `credit-scoring@production` com AUC ≈ 0,86 e Brier calibrado menor que o não calibrado.
- [ ] Lote válido passa; lote corrompido sai com exit 2, JSON listando cada regra e o lote em `data/quarantine/`.
- [ ] Data Docs do GE gerados sem valores de linha.
- [ ] Nenhum `customer_id` em claro nos relatórios de contrato.
- [ ] Testes rodando sem o CSV real.
- [ ] Model card e runbooks de `ContractViolation`/`ContractWarnings` (semanas 2–3).
- [ ] ADRs escritas:
  - GE × Pandera por camada;
  - sentinelas e renda ausente (MNAR);
  - a regra de idade ≥ 18.

## 9. Referência: projeto anterior

Implementação completa para consultar quando travar (repositório privado; peça acesso): https://github.com/eeyamazaki/credit-scoring-sustentacao. Use para **comparar**, não para copiar: o objetivo é o grupo construir e entender cada parte.

Arquivos equivalentes no projeto anterior:

`src/config.py`, `src/parameters.py`, `params.yaml`, `src/data/prepare.py`, `src/features/pipeline.py`, `src/contracts/{schemas,validate,ge_raw}.py`, `src/training/{train,models,evaluate,registry}.py`, `tests/conftest.py`, `tests/unit/test_{prepare,contracts,ge_raw,evaluate}.py`, `docs/decisions/001-*.md`, `docs/decisions/004-*.md`.

Aulas da disciplina de Validação de Dados: Aula 2 (dimensões de qualidade, MNAR), Aula 3 (duplicatas e ausentes), Aula 5 (schema e contratos), Aula 6 (outliers e sentinelas), Aula 7 (frameworks: GE × Pandera, Dead Letter Channel).

## 10. Prompt inicial sugerido para a LLM

```
Você vai me ajudar na Etapa 1 (Validação de Dados e Contratos) do Tech Challenge
descrito no documento acima. Siga as "Decisões já tomadas" e os "Contratos entre as
etapas" sem alterá-los. Comece pelo passo 0 do "Passo a passo": o esqueleto do projeto e o gerador de
dados sintéticos (tests/conftest.py), que os colegas usam desde o dia 2. Depois o
config.py e o params.yaml (com validação pydantic). Escreva código Python 3.12 com type hints,
docstrings em português e testes pytest com dados sintéticos. Antes de cada passo,
diga quais armadilhas da seção 7 se aplicam.
```
