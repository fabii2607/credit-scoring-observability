# Predições e relatórios de monitoramento

O comando de análise valida os lotes mensais, gera probabilidades com o modelo
V2 congelado e produz relatórios de drift e performance. Funciona localmente,
sem Docker, MLflow remoto ou serviço de API. Não retreina nem promove modelos.

## Execução

Pré-requisitos: ambiente instalado, parquets preparados, V2 e lotes simulados.

```bash
uv sync --locked
make prepare
make train
make simulate
make analyze
```

Se os artefatos já existem, basta `make analyze`. Equivalente sem `make`:

```bash
uv run python -m credit_scoring_observability.analyze
```

Para selecionar meses ou incluir o teste de bloqueio:

```bash
uv run python -m credit_scoring_observability.analyze --batches 2026-03 2026-06
uv run python -m credit_scoring_observability.analyze --include-corrupted
```

Por padrão, são analisados os sete meses listados no manifesto da simulação.
Não são analisados automaticamente os originais, a reserva de calibração ou a
demo válida. Referência e controle são pontuados para a comparação.

O código de saída é 0 quando os lotes selecionados terminam com sucesso, 2 se
houver bloqueio por contrato e 1 se houver falha inesperada. Bloqueio ou falha de
um lote não interrompe os demais. Com `--include-corrupted`, exit 2 é esperado.
O lote corrompido é movido para quarentena; execute `make simulate` para recriá-lo
antes de repetir essa demonstração. Se apenas o arquivo estiver ausente, a
execução registra falha de leitura, sem apresentá-la como um novo bloqueio.

## Fluxo e separação de referências

```text
referência de distribuição → validação → score → comparação de distribuições
controle de performance   → validação → score → AUC de comparação
                                  ↑
lote mensal → validação → score → drift / performance → HTML, JSON e CSV
```

| Conjunto | Papel |
|---|---|
| `data/processed/reference_sample.parquet` | Comparação das distribuições. Vem do treino; sua AUC não autoriza decisões sobre retreinamento. A coluna auxiliar `split` é removida antes do contrato. |
| `data/production/reserves/control.parquet` | Comparação de performance, com 30 mil registros fora do ajuste do modelo e disjuntos dos meses. Já pertenciam ao teste histórico da modelagem. |
| `data/production/<mês>.parquet` | Lote atual, estável ou alterado, pontuado após aprovação no contrato. |
| `data/production/corrupted.parquet` | Demonstração de qualidade; deve ser bloqueado sem predições. |

`validate_batch` identifica violações e `enforce` grava o relatório do contrato,
interrompe o lote e o move para quarentena se houver blocker. Uma única linha
com violação bloqueante reprova o lote completo. Warnings não bloqueiam.
`registry.load_model().score()` aplica ainda o schema de features na inferência.

Controle e lotes têm seus hashes verificados contra o manifesto. O controle deve
ser disjunto da referência; lotes devem ser disjuntos de ambos. Não modifique
modelo, fontes ou parâmetros durante a execução.

## Detecção de drift

`drift.detect_drift(reference_scored, batch_scored, batch_id, params)` recebe
DataFrames já pontuados. Os argumentos significam referência de distribuição,
lote atual e identificador como `2026-03`.

As 23 features são selecionadas explicitamente. Identificadores, região, datas,
target e prediction não entram na análise das features. O score tem análise
separada; o target fornece somente prevalência e comparação de rótulos.

- **Numéricas:** PSI por quantis da referência, KS D, p-valor e mudança de nulos,
  reutilizando `stats.py`. `status` representa a faixa PSI; `drifted` exige
  simultaneamente PSI > 0,25 e KS D > 0,10.
- **Categóricas:** distância Jensen–Shannon, com log natural e categorias
  alinhadas pela união. `drifted` exige distância ≥ 0,10. Não recebem PSI ou KS
  fictícios: esses campos permanecem `null`.
- **Ausências:** diferença de taxa de nulos acompanhada à parte, inclusive se
  não houver observações suficientes para calcular a distribuição.
- **Score:** PSI separado, excluído do denominador de features.
- **Rótulos:** diferença de prevalência expressa também por z bilateral de duas
  proporções. O z é assinado; magnitude grande nos dois sentidos exige atenção.
  Não constitui prova de concept drift, nem dispara retreinamento.

`drift_share` é a quantidade de features com `drifted=true` dividida pelo número
de features com distribuição analisável. A lista de features indisponíveis e o
denominador são explícitos. PSI WARN e mudanças de ausência não entram nessa
fração; permanecem visíveis nos detalhes.

## Performance realizada

`performance.evaluate_performance(control_scored, batch_scored, threshold, ...)`
compara os rótulos conhecidos com os scores. O primeiro argumento da interface
se chama `reference_scored`, mas neste módulo representa o **controle de
performance**, não a referência de distribuições. O chamador garante sua origem.

O `threshold` é o limiar de decisão do modelo, lido dos metadados (V2: 0,20).
Scores maiores ou iguais ao limiar produzem prediction 1, negação simulada;
aprovação é a fração de scores abaixo dele. O limiar não altera a AUC.

O resultado inclui ROC-AUC, PR-AUC (Average Precision), Brier, precision, recall,
aprovação, prevalência observada, quantis 50/90/99 do score e contagem de rótulos.

```text
auc_drop = AUC do controle − AUC do lote
performance_drop_detected = auc_drop > performance.auc_drop
```

O limite padrão de queda é 0,05. Quedas negativas significam AUC atual maior.
O indicador é uma regra operacional, não um teste formal da diferença nem uma
recomendação automática de retreino. Drift de entrada pode ocorrer com AUC
estável; degradação sem drift marginal das features merece investigação de
concept drift, cobertura de rótulos e outros problemas.

O intervalo de AUC é percentil de 95%, com 200 reamostragens estratificadas pelo
target e seed fixa. Além desse intervalo da AUC atual, são calculados intervalos da queda contra o controle e da queda pareada antes/depois, descritos no [guia do experimento](experiment-analysis.md).
`bootstrap_samples: 0` desabilita esses cálculos.

Sem target ou sem duas classes observadas, AUC e comparação ficam indisponíveis,
com motivo explícito; não são substituídas por zero. Aprovação e quantis de score
continuam disponíveis. Brier e métricas de decisão podem ser calculados com uma
classe. Com rótulos parciais, métricas supervisionadas usam somente as linhas
rotuladas; `labeled_rows` e `rows` informam a cobertura. Isso não corrige possível
viés na disponibilidade dos desfechos. Em produção real, default exige maturação;
na demonstração, usamos desfechos históricos ou rótulos simulados conhecidos.

## Relatórios e arquivos

```text
data/interim/
├── _reference/scored.parquet
├── _control/scored.parquet
└── <mês>/
    ├── model_input.parquet
    ├── scored.parquet
    └── status.json

reports/
├── contracts/<lote>.json
├── drift/<mês>.html
├── drift/<mês>.json
└── performance/
    ├── <mês>.json
    ├── summary.json
    └── summary.csv
```

Abra `reports/drift/2026-03.html` diretamente no navegador. É um relatório HTML
interativo gerado pelo Evidently, com tabela agregada do projeto e gráficos de
distribuição; não é uma API ou painel atualizado continuamente. Os arquivos
podem ser anexados como artefatos ao MLflow em integração posterior.

O JSON de drift guarda as mesmas estatísticas do resumo HTML, parâmetros,
versão do Evidently e proveniência. Caminhos de relatórios são relativos ao
diretório de drift. O JSON de performance contém métricas, motivos de
indisponibilidade e proveniência. CSV/JSON consolidados contêm os lotes da
seleção executada, incluindo seus estados. Uma execução de um subconjunto
substitui o consolidado com esse subconjunto; execute sem `--batches` para obter
os sete meses novamente.

Os arquivos pontuados preservam metadados para investigação e acrescentam
`score`/`prediction`. Ficam fora do Git. O status informa success, blocked ou
failed; arquivos gerados conhecidos de uma execução anterior do mesmo lote são
invalidados antes de reprocessar. Em falha posterior à pontuação, podem existir
predições válidas com status failed: consulte o status antes de consumir saídas.
Não execute duas análises simultâneas no mesmo diretório de saída.

## Coerência com Evidently e privacidade

São registrados métodos estatísticos customizados no Evidently, usando os
mesmos cálculos de PSI/KS e JS dos módulos puros. O resumo Dataset Drift usa o
limite `1 / analyzed_features`, sinalizando pelo menos uma feature confirmada.
O score fica fora desse resumo. O status PSI WARN é mostrado na tabela do
projeto; não é contado como drift confirmado pelo Evidently.

A integração foi verificada com Evidently 0.7.23 do `uv.lock`. Ela usa a API
atual de Report/Dataset e o registro de stattests legado disponível nessa
versão. Uma atualização da biblioteca exige executar os testes de integração.

O DataFrame entregue ao Evidently contém somente features analisáveis e score,
sem id, região, datas, target ou prediction, com índices reiniciados. Gráficos
são agregados. Os relatórios não incluem amostras individuais adicionadas pelo
projeto, sal ou identificadores de empréstimos. Categorias raras e valores
extremos ainda merecem revisão antes de compartilhar; agregação não garante
anonimização por si só.

HTMLs, JSONs de monitoramento e consolidados são locais e ignorados pelo Git.
Evidências públicas devem ser selecionadas e revisadas antes de serem copiadas
para `docs/`. A [seleção pública de evidências](evidencias/README.md) preserva
relatórios representativos e o consolidado revisado desta execução. Logs e
relatórios do contrato seguem o mascaramento já existente.
O contexto registra hashes de modelo, metadados, referência, controle, lote,
manifesto e implementação, além de versões relevantes. Não contém registros.

## Verificação

```bash
uv run pytest tests/test_drift.py tests/test_performance.py tests/test_analysis.py
make lint
make test
```

Os testes usam dados sintéticos e verificam respostas conhecidas, feature drift
versus relabel, métricas sem rótulo, bootstrap determinístico, correspondência
entre cálculos e Evidently, exclusão de identificadores e bloqueio antes da
inferência. O teste integrado processa um lote bloqueado antes de meses válidos
para comprovar a continuidade. A integração de métricas ao dashboard, regras de
alerta da stack e política de intervenção são trabalhos separados.

## Consolidação e recomendação

O [guia do experimento e do gate](experiment-analysis.md) descreve os gráficos
temporais, as comparações pareadas e a recomendação exibida no topo dos HTMLs.
