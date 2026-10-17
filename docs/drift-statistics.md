# Estatística e parâmetros de detecção de drift

Este documento descreve os cálculos estatísticos, as configurações e os formatos
de resultados disponíveis para comparar dados de referência e de produção.
Destina-se a quem precisa executar, avaliar ou estender o monitoramento.
Os testes estatísticos usam dados sintéticos e não exigem treinamento do modelo.

O módulo atual calcula PSI, KS e variação de ausência em variáveis numéricas.
A geração dos lotes está disponível em [simulação de produção](production-simulation.md).
Os relatórios Evidently e o cálculo de performance por lote estão disponíveis
via `make analyze`; consulte [Monitoramento de drift e performance](drift-monitoring.md).

## Configuração

`parameters.py` valida `params.yaml` com Pydantic. As seções de monitoramento são:

| Seção | Responsabilidade |
|---|---|
| `simulation` | Seed 42; sete meses consecutivos, janeiro–julho de 2026; lotes de 10.000; controle de 30.000; calibração de 10.000; demo de 2.000; duas intensidades de data drift e três de concept drift. |
| `drift` | Quantidade de bins, epsilon, faixas PSI, limite KS D, limites futuros de distância categórica e label drift, variação de ausência. |
| `performance` | Queda relevante de AUC de 0,05; seed e número de reamostragens para intervalos futuros. |

O simulador cria reservas de controle e calibração disjuntas dos sete lotes
mensais. O controle serve para performance fora do treino; a referência de
distribuição continua sendo `reference_sample.parquet`. As intensidades
configuram intervenções; não substituem a medição do drift e da performance
em cada mês.
As seeds da simulação não modificam a separação treino/teste do modelo.

Intensidades devem ser finitas, em [0, 1] e não decrescentes. Meses precisam ser
válidos, distintos e consecutivos. Volumes devem atender ao mínimo do contrato.
Campos desconhecidos são rejeitados; YAMLs antigos sem as três seções recebem os
valores padrão declarados nos modelos. Não há configuração de retreinamento
automático implementada.

`get_params()` tem cache por processo. Reinicie o processo/kernel ou chame
`get_params.cache_clear()` depois de editar o YAML. `load_params()` relê sem cache.

## Linha do tempo da simulação

A população de modelagem contém empréstimos originados entre junho de 2007 e
dezembro de 2018, com desfecho conhecido. O treino e o pool de produção são
separados aleatoriamente, com estratificação pelo target; não há corte temporal
de treinamento em dezembro de 2025.

Janeiro–julho de 2026 identifica a sequência de lotes de uma simulação de
produção. Esses nomes não representam novos empréstimos observados em 2026.
A data histórica de originação (`issue_date`) deve ser preservada, separada da
identificação do mês simulado. O experimento permite estudar mudanças induzidas,
mas não comprova generalização temporal real nem efeitos econômicos de 2026.

## Cálculos numéricos

`stats.py` expõe `psi_quantile`, `ks_two_sample`, `psi_status` e `compare_numeric`.
Não lê arquivos nem emite métricas. Passe `get_params().drift` para aplicar o YAML;
sem parâmetros explícitos, `compare_numeric` usa os valores padrão de `DriftParams`.

### PSI

Para proporções de referência p e produção q em faixas fixas:

```text
PSI = soma((q - p) * ln(q / p))
```

As fronteiras vêm dos quantis da referência, sem recalcular na produção.
Quantis internos repetidos são deduplicados. A atribuição às faixas fecha os
intervalos à direita: valores exatamente iguais a um corte ficam na faixa
inferior. Grupos externos ilimitados incluem as duas caudas, sem descartar
valores fora dos extremos históricos.

Para referência constante, as faixas distinguem valores abaixo, iguais e acima
da constante. Isso evita PSI zero artificial ao surgir outro valor.

As proporções recebem piso `epsilon` (1e-6) e são renormalizadas antes do cálculo.
Consequentemente, as proporções continuam somando um; mudanças com faixas vazias
podem ser sensíveis à escolha do epsilon. O valor e a quantidade de bins devem
acompanhar a execução final. O número efetivo de faixas é retornado.

| PSI | Status |
|---|---|
| < 0,10 | OK |
| ≥ 0,10 e ≤ 0,25 | WARN |
| > 0,25 | ALERT |
| Sem observações em uma das amostras | UNAVAILABLE |

Essas faixas são convenções operacionais do projeto, sujeitas à avaliação dos
meses estáveis. PSI não indica sozinho direção, causa ou prejuízo preditivo.

### KS de duas amostras

O KS D é a maior diferença absoluta entre as distribuições acumuladas das duas
amostras. Retorna também p-valor bilateral, usando SciPy com método assintótico
explícito; amostras com uma observação usam o método exato para evitar a
aproximação degenerada. O p-valor tem limitações com empates e variáveis discretas
e pode acusar diferenças pequenas em bases grandes; não determina o alerta crítico.

`ks_confirmed` exige D > 0,10. `critical` exige simultaneamente PSI ALERT e
KS confirmado. O campo `status` descreve somente a faixa PSI. Esse KS compara
uma feature entre datasets; é diferente do KS de separação entre bons e maus
pagadores calculado em `evaluate.py`.

### Ausências e qualidade

- Nulos são excluídos somente dos cálculos de distribuição. Contagens totais e
  válidas, taxas de ausência e delta (produção − referência) são retornados.
- `missing_drift` compara o módulo do delta com o limite (≥ 0,05), inclusive
  quando houve redução de ausências. É independente de `status`/`critical`.
- Vetor vazio: taxa de ausência indefinida (`None`). Vetor inteiramente nulo:
  taxa de ausência 1, mas PSI/KS indisponíveis. Indisponível nunca significa zero.
- Infinity, texto e valores complexos são rejeitados com `ValueError`, sem
  descarte silencioso. As funções não substituem o contrato de entrada do modelo.
- PSI/KS são numéricos. O módulo `drift.py` implementa a distância categórica
  Jensen–Shannon, integrada ao Evidently com o limite `categorical_js`.

## Exemplo reproduzível

Executado em Python no ambiente do projeto:

```python
from credit_scoring_observability.parameters import get_params
from credit_scoring_observability.stats import compare_numeric

result = compare_numeric(
    reference=[0, 1, 2, 3],
    current=[2, 3, 4, 5],
    params=get_params().drift,
)
print(result.to_json())
```

Neste exemplo, KS D = 0,5. Um caso de referência nos testes usa duas faixas e
proporções p=(0,5; 0,5), q=(0,75; 0,25): PSI = ln(3)/4 ≈ 0,274653.

## Interfaces de resultados

`results.py` declara dataclasses `FeatureDriftResult`, `DriftResult` e
`PerformanceResult`. A serialização `to_dict()`/`to_json()` rejeita NaN e Infinity,
inclusive em campos aninhados. Valores não calculados devem ser `None` → `null`.
As anotações de tipo descrevem o contrato; as dataclasses não validam todos os
tipos e domínios de cada campo na construção. Os módulos que produzem os
resultados devem garantir essa validação.

`DriftResult.analyzed_features` documenta o denominador de `drift_share`, excluindo
score, target e metadados. Campos de método e distância permitem resultados
categóricos sem atribuir um KS numérico a categorias.

`PerformanceResult` prevê baseline e sua origem, AUC, delta AUC, PR-AUC, Brier,
aprovação, precision/recall, quantis de score e intervalo AUC. Sem target ou sem
duas classes, `performance.py` deixa AUC/delta indisponíveis e explica o motivo.
O avaliador usa um controle fora do treino; não usa AUC da referência de drift.

## Verificação

```bash
uv sync --locked
uv run pytest tests/test_stats.py tests/test_results.py tests/test_parameters.py
make lint
make test
```

Os testes usam exemplos com resposta conhecida, deslocamentos, caudas, constantes,
quantis repetidos, ausências, limites de status, entradas inválidas, JSON estrito
e incompatibilidade de parâmetros. Nenhum deles exige o CSV do Kaggle.
