# Evidências da execução de referência

Execução de **2026-10-21** (America/Sao_Paulo), modelo V2 congelado,
threshold 0,20. Sete lotes de 10 mil registros e uma demo de 2.100 linhas bloqueada.

| Mês | AUC | Queda vs. controle | Queda pareada | Features confirmadas | Gate |
|---|---:|---:|---:|---|---|
| 2026-01 | 0.7051 | -0.0077 | +0.0000 | Nenhuma | OK |
| 2026-02 | 0.7035 | -0.0061 | +0.0000 | Nenhuma | OK |
| 2026-03 | 0.6963 | +0.0011 | -0.0014 | dti,revol_util,fico_avg | WARN |
| 2026-04 | 0.6995 | -0.0021 | -0.0025 | annual_inc,dti,revol_util,fico_avg | WARN |
| 2026-05 | 0.6181 | +0.0793 | +0.0649 | Nenhuma | ALERT |
| 2026-06 | 0.5993 | +0.0981 | +0.0906 | Nenhuma | ALERT |
| 2026-07 | 0.5589 | +0.1385 | +0.1301 | Nenhuma | ALERT |

AUC do controle de 30 mil registros: **0.6974**.
A referência de distribuição vem do treino e não é baseline de performance.
Os meses são sintéticos; o pool já participou da avaliação histórica do modelo.
Queda pareada compara os mesmos registros antes/depois; PSI não é aditivo.

## Navegação

- HTMLs de [janeiro](drift/2026-01.html), [abril](drift/2026-04.html) e [junho](drift/2026-06.html).
- [Experimento consolidado e gráficos](experiment/summary.md).
- [CSV mensal](performance/summary.csv) e [JSON completo](performance/summary.json).
- [Sensibilidade à restauração de marginais](causal/causal_analysis.json).
- [Verificação da observabilidade](observability/README.md).
- [Bloqueio do contrato](contracts/corrupted-summary.json).
- [Manifesto de dados, modelo, fontes, parâmetros e dependências](manifest.json).
- [Linha do tempo e interpretação](../incident-timeline.md).
- [Post-mortem do incidente simulado](../postmortem.md).

Baixe os HTMLs e abra no navegador; o GitHub normalmente mostra o código.
Cada HTML contém Evidently e o resumo com recomendação, métricas e comparação
pareada. Nenhuma API é necessária para abrir os arquivos.

## Proveniência

Todos os resultados desta seleção foram reempacotados para esta execução.
Os JSONs são cópias agregadas; no contrato foram removidos exemplos individuais.
Nos HTMLs foram removidos somente espaços em linhas vazias.
Parquets, sal, modelos binários e logs brutos não são publicados.

HEAD identifica a base do código; arquivos efetivos do working tree são
identificados pelos hashes do manifesto, inclusive ajustes ainda não commitados.
Depois de alterar código, modelo, dados ou parâmetros, repita os comandos do
[guia de entrega](../delivery.md) e atualize as evidências.
