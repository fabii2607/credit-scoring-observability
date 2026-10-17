# Evidências de monitoramento

Seleção pública dos resultados da execução de **17/10/2026**, organizada em
17/10/2026. Os relatórios completos são gerados localmente em `reports/`;
esta pasta preserva uma amostra revisada para avaliação no GitHub.

## Relatórios selecionados

| Cenário | Relatório visual | Estatísticas | Performance |
|---|---|---|---|
| Janeiro: estável | [HTML Evidently](drift/2026-01.html) | [JSON de drift](drift/2026-01.json) | [JSON de performance](performance/2026-01.json) |
| Abril: mudança nas features | [HTML Evidently](drift/2026-04.html) | [JSON de drift](drift/2026-04.json) | [JSON de performance](performance/2026-04.json) |
| Junho: relabel simulado | [HTML Evidently](drift/2026-06.html) | [JSON de drift](drift/2026-06.json) | [JSON de performance](performance/2026-06.json) |

**Como visualizar:** o GitHub normalmente mostra o código do arquivo HTML em vez
de executar o relatório. Baixe o HTML pelo botão de download do arquivo e abra
no navegador, ou clone o repositório e abra o arquivo local. Cada HTML tem cerca
de 5 MB e inclui o relatório Evidently. Não é necessário iniciar uma API ou Docker.
Os relatórios locais originais permanecem em `reports/drift/`.

## Comparação dos sete meses

Modelo V2 congelado, threshold de decisão **0,20** e AUC do controle de
performance **0,6974**. Queda AUC = AUC do controle − AUC do mês; valores negativos
representam AUC atual maior. O controle contém 30 mil registros fora do ajuste
do modelo, separados dos meses. A referência de distribuição tem 99.999 registros
provenientes do treino e não é usada como baseline de AUC.

| Mês simulado | AUC | Queda AUC | Features com drift | Features confirmadas |
|---|---:|---:|---:|---|
| 2026-01 | 0.7051 | -0.0077 | 0.0% | Nenhuma |
| 2026-02 | 0.7035 | -0.0061 | 0.0% | Nenhuma |
| 2026-03 | 0.6963 | +0.0011 | 13.0% | dti,revol_util,fico_avg |
| 2026-04 | 0.6995 | -0.0021 | 17.4% | annual_inc,dti,revol_util,fico_avg |
| 2026-05 | 0.6181 | +0.0793 | 0.0% | Nenhuma |
| 2026-06 | 0.5993 | +0.0981 | 0.0% | Nenhuma |
| 2026-07 | 0.5589 | +0.1385 | 0.0% | Nenhuma |

- Janeiro–fevereiro: nenhuma feature com drift confirmado e nenhuma queda de AUC acima de 0,05.
- Março–abril: drift nas features alteradas, com performance próxima à do controle.
- Maio–julho: queda de AUC acima de 0,05 sem drift confirmado nas distribuições marginais de X. Esse é o comportamento esperado do relabel sintético; o HTML de junho mostra estabilidade de X, e a degradação de performance no início do HTML e no JSON.

A tabela completa, incluindo intervalos de AUC, aprovação, prevalência e estado
do lote corrompido, está em [CSV](performance/summary.csv) e
[JSON](performance/summary.json). O CSV usa ponto como separador decimal.

`Drift share` considera features numéricas com PSI > 0,25 e KS D > 0,10 e
categóricas com Jensen–Shannon ≥ 0,10, divididas pelas features analisáveis
(23 nesta execução). PSI em WARN não entra nessa contagem; o score é separado.
Os métodos e as limitações estão no [guia de monitoramento](../drift-monitoring.md).

## Bloqueio de dados inválidos

O [resumo do contrato](contracts/corrupted-summary.json) registra o bloqueio do
lote `corrupted`, com 2.100 linhas e oito entradas de violações bloqueantes.
Uma entrada de violação pode afetar múltiplas linhas; essas contagens não devem
ser somadas como se fossem registros distintos. Exemplos de registros foram
removidos deste resumo público. O consolidado registra `blocked`; o teste de
integração no repositório verifica que o modelo não é chamado para lote inválido.

## Proveniência e privacidade

O [manifesto das evidências](manifest.json) registra arquivos selecionados,
hashes SHA-256, fontes locais, versões e hashes da implementação/modelo/dados.
JSONs de drift/performance são cópias dos resultados gerados. Nos HTMLs, foram
removidos somente espaços em linhas vazias, preservando scripts e dados. O
resumo do contrato foi reduzido para retirar exemplos individuais. O manifesto
distingue os hashes das fontes e os dos arquivos publicados.

Os relatórios foram verificados quanto à presença dos identificadores completos
de empréstimos usados na execução e do sal local; nenhum foi encontrado.
Os datasets individuais e as predições por registro não fazem parte desta pasta.
As features categóricas aqui apresentadas são categorias financeiras do dataset;
a agregação não é uma garantia geral de anonimização.

Os meses de 2026 identificam uma simulação com empréstimos históricos, não dados
observados em produção nesse período. As intervenções são artificiais; os
resultados não demonstram causalidade econômica nem validade para concessão real.
O intervalo de AUC é da métrica de cada lote, não da diferença entre AUCs.

## Reprodução e atualização

Na raiz do projeto, com dataset e ambiente configurados:

```bash
make prepare
make train
make simulate
uv run python -m credit_scoring_observability.analyze --include-corrupted
```

O último comando termina com exit 2 por causa do bloqueio esperado da demo,
mesmo com os sete meses processados com sucesso. Para analisar somente os meses,
use `make analyze` (exit 0). Uma nova execução não atualiza automaticamente esta
pasta: revise os resultados, substitua a seleção e atualize o manifesto juntos.
A reprodução numérica pode sofrer pequenas diferenças entre ambientes; os hashes
identificam precisamente os artefatos desta execução.


## Gate e comparação pareada

Os HTMLs selecionados exibem a recomendação no topo, antes das métricas e
dos gráficos Evidently. O [experimento consolidado](experiment/summary.md)
reúne os sete meses, gráficos de AUC/intervalos e comparações antes/depois.

| Meses | Gate | Recomendação |
|---|---|---|
| Janeiro–fevereiro | OK | Monitoramento regular |
| Março–abril | WARN | Investigar entradas e reforçar monitoramento |
| Maio–julho | ALERT | Investigar degradação e avaliar retreinamento |

Os JSONs de [janeiro](gate/2026-01.json), [abril](gate/2026-04.json) e
[junho](gate/2026-06.json) registram os motivos. As comparações pareadas de
[janeiro](intervention/2026-01.json), [abril](intervention/2026-04.json) e
[junho](intervention/2026-06.json) preservam os resultados agregados dos mesmos
registros antes/depois.

A queda pareada em maio/junho/julho foi de 0,0649 / 0,0906 / 0,1301.
A diferença contra o controle inclui também diferenças entre amostras; por isso,
não deve ser atribuída integralmente à intervenção. Os originais dos sete
meses não apresentaram features com drift confirmado nesta execução.
PSI não é aditivo. A intervenção demonstra um efeito sintético nesta amostra,
sem provar causa econômica. Nenhum modelo foi retreinado ou promovido.
Consulte a [política do gate](../experiment-analysis.md) para limites e critérios.
