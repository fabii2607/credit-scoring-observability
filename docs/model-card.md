# Model card — V2 de classificação de crédito

## Finalidade e versão

`V2_optimized`: Regressão Logística scikit-learn em pipeline com imputação,
codificação categórica e tratamento das features numéricas. Uso acadêmico:
demonstrar contrato de entrada, drift, avaliação supervisionada e observabilidade.
Não foi validada para decisões reais de crédito ou população brasileira.

O score estima a classe 1 definida pelo projeto. `score >= 0,20` produz
`prediction=1` (negação simulada); abaixo disso, aprovação simulada.
Utilize `registry.load_model()`: `pipeline.predict()` aplicaria o corte padrão
0,5 e mudaria a regra. AUC utiliza scores contínuos.

## Dados, target e avaliação

Lending Club: 1.348.099 registros com rótulos conhecidos segundo a convenção do
projeto, originações entre junho/2007 e dezembro/2018. Classe 0 inclui Fully Paid;
classe 1 inclui Charged Off e Default. Ambos os desfechos com prefixo
“Does not meet the credit policy” são mantidos conforme o sufixo, como explica
o [README](../README.md).

Split aleatório estratificado 80/20, seed 42: 1.078.479 na referência de treino
e 269.620 no teste/pool. O treinamento usa amostra de 300 mil e validação interna
de 20%; seleciona entre pesos padrão/balanceado e escolhe threshold por F2
sujeito a precision mínima 0,30. A configuração selecionada usa pesos padrão.

No teste histórico completo, os metadados locais registram AUC aproximadamente
0,6936, PR-AUC 0,3569 e Brier 0,1474. O controle reservado de 30 mil usado no
monitoramento tem AUC 0,6974. São amostras diferentes; não confundir as métricas.

## Entradas e qualidade

23 features de crédito/emprego, incluindo renda, DTI, utilização do rotativo,
FICO médio e histórico de crédito. A lista exata está em
`data_contract.MODEL_FEATURES` e no JSON de metadados. Identificador, região,
mês e target não são features do modelo.

Contrato Pandera e regras de lote precedem a inferência. Um blocker impede o
lote inteiro, gera relatório e quarentena. Nulos permitidos seguem a imputação
do pipeline. Não há predição parcial de linhas válidas de lote bloqueado.

## Limitações

- Split aleatório não garante generalização temporal; safra recente com poucos
  desfechos conhecidos pode sofrer censura/seleção.
- Pool já foi usado no teste histórico; os meses sintéticos não são teste
  externo independente nem observações de produção de 2026.
- AUC moderada e threshold sem validação econômica não estabelecem prontidão
  para produção. Probabilidades não têm calibrador separado validado.
- Os 10 mil registros chamados de “calibração” são usados para calibrar as
  intervenções da simulação, não probabilidades do classificador.
- A auditoria por região (seção Fairness) não demonstra ausência de
  discriminação em atributos ausentes do dataset, como idade, sexo e raça.
  CT, challenger e promoção estão fora da implementação.
- Coeficientes são associações de um modelo regularizado, não efeitos causais
  (seção Explicabilidade).
- Sensibilidade por restauração de marginais não identifica causas econômicas.

## Explicabilidade (Art. 20 da LGPD)

A V2 é uma regressão logística, então toda decisão se decompõe em termos
aditivos. Para um solicitante com features pré-processadas `z`:

```text
logit = intercepto + Σ coef_j × z_j
score = 1 / (1 + exp(−logit))
prediction = 1 (negar) se score ≥ 0,20; 0 (aprovar) caso contrário
```

O intercepto é −0,013 e o threshold 0,20 equivale a logit −1,386. Utilize
`registry.load_model()` para obter o par binário + metadados e o threshold
versionado; `pipeline.predict()` aplicaria 0,5 e mudaria a regra.

### Proveniência dos coeficientes

Os coeficientes desta seção foram lidos do binário local
`models/credit_scoring_optimized.joblib` (sha256 iniciado em `3fbae9e6`,
gerado por `make train` em 2026-10-13). Esse binário não é byte-idêntico ao
da execução de referência (`model_sha256` iniciado em `86aede00` no
[manifesto das evidências](evidencias/manifest.json)), mas usa os mesmos
metadados (`metadata_sha256` iniciado em `eb05a610`) e reproduz a AUC no pool
de 269.620 linhas até a sexta casa (0,6936). Trata-se de outra execução do
mesmo treino, não de outro modelo; ainda assim, os valores abaixo podem
diferir nas últimas casas dos do modelo de referência. Para conferir,
execute `make train` e compare o hash com o manifesto; o binário fica fora do
Git e não há cópia dele em `docs/evidencias/`.

### O que é z

O pré-processamento (`preprocessing.py`) aplica, às 19 features numéricas,
imputação pela mediana e `RobustScaler`: `z = (x − mediana) / IQR`, com
mediana e IQR estimados na amostra de treino. Um coeficiente numérico é,
portanto, a variação do log-odds por um IQR da feature, e `exp(coef)` é a
razão de chances por IQR: FICO médio +40 pontos multiplica a chance de
classe 1 por `exp(−0,443) ≈ 0,64`; valor do empréstimo +US$ 12.000 multiplica
por `exp(0,251) ≈ 1,28`.

Nove features têm IQR = 0 na amostra de treino (`term`, `delinq_2yrs`,
`inq_last_6mths`, `pub_rec`, `collections_12_mths_ex_med`, `acc_now_delinq`,
`pub_rec_bankruptcies`, `tax_liens`, `emp_length_missing`). Para elas o
`RobustScaler` usa escala 1 e o coeficiente vale por unidade da feature, não
por IQR. O caso relevante é `term`: a mediana é 36 meses, logo `z = 0` para 36
meses e `z = 24` para 60 meses. O coeficiente +0,037 por mês parece pequeno,
mas 60 meses adiciona 24 × 0,037 ≈ +0,88 ao logit (`exp(0,88) ≈ 2,4`), a
maior contribuição prática do modelo. Ler a tabela de coeficientes brutos
sem esse ajuste subestima o prazo.

As 4 features categóricas (`home_ownership`, `verification_status`,
`purpose`, `application_type`) recebem imputação constante `Unknown` e
one-hot **sem** descarte de nível (25 colunas). Como todos os níveis têm
coeficiente, o valor absoluto de um nível isolado não é interpretável:
`application_type_Individual` = −0,733 não significa que solicitar
individualmente reduz o risco. Compare sempre níveis da mesma variável,
como na tabela de diferenças abaixo.

### Importância prática por variável

Ordenar pelos coeficientes brutos engana (entre os positivos, o prazo fica em
décimo lugar; entre os negativos, dominam os níveis base das categóricas).
A medida abaixo é o desvio-padrão da contribuição `coef × z` de cada variável na amostra de referência
(`data/processed/reference_sample.parquet`, 99.999 linhas; colunas one-hot da
mesma variável somadas), isto é, quanto cada variável de fato move o logit
entre solicitantes.

| # | Variável | Desvio típico da contribuição no logit |
|---|---|---|
| 1 | `term` | 0,376 |
| 2 | `fico_avg` | 0,354 |
| 3 | `dti` | 0,187 |
| 4 | `loan_amnt` | 0,183 |
| 5 | `home_ownership` | 0,179 |
| 6 | `total_acc` | 0,175 |
| 7 | `open_acc` | 0,159 |
| 8 | `annual_inc` | 0,153 |
| 9 | `inq_last_6mths` | 0,142 |
| 10 | `purpose` | 0,141 |
| 11 | `emp_length_missing` | 0,115 |
| 12 | `revol_bal` | 0,112 |

As outras 11 variáveis ficam abaixo de 0,10; entre elas, `application_type`,
`pub_rec_bankruptcies`, `acc_now_delinq` e `tax_liens` ficam abaixo de 0,02.
`emp_length_missing` (+0,487 quando o tempo de emprego não é informado) é a
única feature de dado ausente com peso relevante: não informar o tempo de
emprego eleva o score, o que importa para o direito de correção
([lgpd.md, seção 6](lgpd.md#6-direitos-do-titular-art-18-e-20)).

### Diferenças entre níveis categóricos

Diferença de log-odds em relação ao nível mais frequente de cada variável
(frequências na amostra de referência):

| Variável | Nível de referência (freq.) | Nível comparado (freq.) | Diferença |
|---|---|---|---|
| `home_ownership` | MORTGAGE (49,5%) | RENT (39,8%) | +0,378 |
| `home_ownership` | MORTGAGE | OWN (10,7%) | +0,239 |
| `verification_status` | Source Verified (38,9%) | Not Verified (29,9%) | −0,185 |
| `verification_status` | Source Verified | Verified (31,1%) | +0,026 |
| `purpose` | debt_consolidation (57,9%) | credit_card (21,9%) | −0,210 |
| `purpose` | debt_consolidation | home_improvement (6,5%) | +0,103 |
| `purpose` | debt_consolidation | other (5,8%) | +0,196 |
| `purpose` | debt_consolidation | medical (1,2%) | +0,368 |
| `purpose` | debt_consolidation | small_business (1,1%) | +0,616 |
| `application_type` | Individual (98,1%) | Joint App (1,9%) | −0,084 |

Níveis com menos de 0,1% dos casos (`home_ownership` ANY/NONE/OTHER,
`purpose` educational/renewable_energy) têm coeficientes pouco confiáveis e
não devem fundamentar uma explicação individual.

### Exemplo de explicação individual

Caso real da amostra de referência (posição 28514 do parquet local, sem
`customer_id`), escolhido por ter score exatamente no threshold: score 0,2000,
decisão "negar", logit −1,386. O solicitante típico, definido pelas medianas
numéricas (`z = 0`) e pelos níveis mais frequentes (MORTGAGE, Source
Verified, debt_consolidation, Individual), tem logit −1,930 e score ≈ 0,127
(aprovado). O caso fica +0,54 acima do típico; as seis maiores contribuições
relativas a ele:

| Fator | Valor informado (típico) | Contribuição relativa ao típico |
|---|---|---|
| Moradia alugada | RENT (MORTGAGE) | +0,378 |
| Poucas contas no histórico | `total_acc` 5 (23) | +0,262 |
| Comprometimento de renda baixo | `dti` 4,89 (17,6) | −0,226 |
| Poucas contas abertas | `open_acc` 5 (11) | −0,174 |
| Consulta de crédito recente | `inq_last_6mths` 1 (0) | +0,148 |
| Valor pedido baixo | `loan_amnt` 6.125 (12.000) | −0,123 |
| Demais fatores (finalidade, renda, histórico, rotativo, emprego, verificação) | — | +0,28 |

Prazo (36 meses) e FICO (692) coincidem com a mediana e não contribuem. Esse é
o formato previsto para a resposta ao titular: fatores que afastaram o caso do
solicitante mediano, em linguagem da variável, sem expor outras linhas nem
identificadores. Não existe função de explicação individual no pacote; as
contribuições são `coef × z` calculadas a partir do pipeline carregado.

### Atendimento ao pedido de revisão (Art. 20, §1º)

1. Localizar o registro pelo `customer_id` recalculado com o sal
   ([lgpd.md, seção 6](lgpd.md#6-direitos-do-titular-art-18-e-20)) e
   reconstruir o score com o mesmo par binário + metadados da decisão.
2. Listar os fatores que afastaram o caso do solicitante típico, como no
   exemplo acima, e verificar com o titular se os valores de entrada estão
   corretos (direito de correção; dado ausente em tempo de emprego pesa
   contra o solicitante).
3. Encaminhar para revisão humana. Além do pedido do titular, a proposta de
   governança prevê revisão humana na faixa próxima do threshold de 0,20;
   a largura dessa faixa ainda não está definida em código nem em
   documento. Na amostra de referência, 14,0% dos scores caem em
   [0,18; 0,22) e 21,0% em [0,17; 0,23).

### Limitações da explicação

- Os coeficientes vêm de um modelo regularizado (L2, `C = 1`), ajustado em
  amostra de 300 mil linhas; não há erro-padrão, intervalo de confiança nem
  teste de significância por coeficiente.
- Há colinearidade entre `open_acc` e `total_acc` (sinais opostos) e entre
  `loan_amnt`, `annual_inc` e `dti`; o sinal de um coeficiente isolado é
  frágil e a explicação deve ser lida como conjunto.
- Os coeficientes descrevem associações no modelo, não efeitos causais; mudar
  um atributo não garante a mudança correspondente no risco real. Consulte
  [análise de sensibilidade](causal-analysis.md) e
  [lgpd.md, seção 10](lgpd.md#10-análise-causal-do-problema).
- Frequências, quantis e o exemplo individual vêm de arquivos locais
  regenerados por `make prepare`, que não são byte-idênticos aos da execução
  de referência.

## Fairness

O dataset não tem idade, sexo nem raça. O atributo auditado é a **região**
do Census dos EUA (Midwest, Northeast, South, West), derivada do estado no
`prepare` e descartada antes do treino: ela não é feature do modelo, mas é
proxy conhecido de raça e renda, por isso a auditoria do Art. 20, §2º é feita
sobre ela. Convenções de `fairness.py`: decisão 1 = negar (`score ≥ 0,20`);
aprovação = 1 − taxa de seleção; TPR = inadimplentes negados; FPR = bons
pagadores negados; razão de impacto desigual = menor aprovação / maior
aprovação, com a regra dos 4/5 (≥ 0,8).

### Baseline por região

Fonte: `reports/fairness/fairness_report.json`, gerado por `make fairness`
em 2026-10-03 (commit `469de0e`) sobre o pool de produção (269.620 linhas,
threshold 0,20). O relatório não registra `model_sha256`, não está em
`docs/evidencias/` e não foi regenerado na execução de referência de
2026-10-21; os números abaixo datam de 2026-10-03.

| Região | n | Aprovação | TPR | FPR | Default observado |
|---|---|---|---|---|---|
| Midwest | 47.209 | 59,1% | 63,8% | 35,2% | 19,9% |
| Northeast | 54.484 | 58,6% | 63,4% | 35,8% | 20,4% |
| South | 95.583 | 59,6% | 62,2% | 34,8% | 20,7% |
| West | 72.344 | 58,5% | 64,1% | 36,2% | 18,8% |

Razão de impacto desigual 0,983 (`passes_four_fifths: true`); diferença
máxima de TPR 1,9 p.p. e de FPR 1,5 p.p. Parte da diferença de aprovação
acompanha a inadimplência observada por região. Não há intervalo de
confiança para essas diferenças; o relatório traz apenas estimativas
pontuais.

### Mitigação avaliada e não adotada

`mitigation.py` estima thresholds por região com paridade de FPR,
determinísticos: a meta é o FPR global do threshold 0,20 na metade de ajuste
(35,5%), e o threshold de cada região é o quantil correspondente do score dos
seus bons pagadores. Ajuste em metade do pool (134.810 linhas, estratificado
por região e target) e avaliação na outra metade (134.810).

| Métrica (metade de avaliação) | Threshold único 0,20 | Por região (0,198–0,203) |
|---|---|---|
| Aprovação global | 59,1% | 59,2% |
| Default entre aprovados | 12,5% | 12,5% |
| Impacto desigual | 0,980 | 0,984 |
| Diferença máxima de FPR | 1,4 p.p. | 0,5 p.p. |

A diferença de FPR de 1,4 p.p. é da metade de avaliação; a de 1,5 p.p. da
tabela anterior é do pool inteiro. `adopted_in_production: false`, por três
razões:

1. O ganho é desprezível: os thresholds por região ficariam quase iguais ao
   global.
2. Usar a região na decisão é tratamento diferenciado por proxy de raça
   (Art. 6º, IX), mais difícil de justificar do que um modelo que não a usa;
   a adoção dependeria de avaliação jurídica.
3. O `ThresholdOptimizer` do fairlearn foi descartado por sortear a decisão
   dentro do grupo, o que é indefensável diante de um pedido de revisão do
   Art. 20.

### Fairness por lote

A medição por lote existe via `make fairness-batches`
(`reports/fairness/fairness_by_batch.json`): nos sete lotes sintéticos de
2026-01 a 2026-07, a razão de impacto desigual fica entre 0,875 (2026-04,
lote de covariate shift com aprovação de 24,5%) e 0,977 (2026-01), sempre
acima de 0,8. A tabela por lote está em
[lgpd.md, seção 8](lgpd.md#8-viés-e-fairness). Essa medição não está
integrada ao pipeline mensal: `approval_disparity` permite calcular a razão
sem rótulo, mas não há métrica de fairness publicada, alerta correspondente
nem guardrail de promoção implementado. O guardrail de não piorar a razão em
mais de 0,05 é política proposta para um eventual challenger, que também não
existe.

## Artefatos e rastreabilidade

Modelo binário e metadados precisam ser usados como par. O binário é local,
ignorado pelo Git: reproduza com `make prepare && make train` ou obtenha ambos
por canal autorizado do grupo. Nunca carregue joblib de origem não confiável.

Hashes do modelo, metadados, dados, parâmetros e fontes da execução estão no
[manifesto das evidências](evidencias/manifest.json).
Veja [reprodução](delivery.md), [governança](lgpd.md) (direitos do titular na
seção 6, fairness na seção 8 e análise causal na seção 10) e
[análise de sensibilidade](causal-analysis.md).
