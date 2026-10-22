# Linha do tempo do incidente simulado

Execução de referência de 21/10/2026. Modelo V2 congelado, threshold 0,20;
AUC do controle separado de 30 mil registros: 0.6974.
Os meses são nomes de cenários, não datas de produção ou corte temporal do treino.

## Evolução observada

| Mês | AUC original | AUC atual | Queda vs. controle | Queda pareada | PSI do score | Aprovação | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 2026-01 | 0.7051 | 0.7051 | -0.0077 | +0.0000 | 0.0015 | 59.4% | OK |
| 2026-02 | 0.7035 | 0.7035 | -0.0061 | +0.0000 | 0.0017 | 59.5% | OK |
| 2026-03 | 0.6949 | 0.6963 | +0.0011 | -0.0014 | 0.3014 | 35.4% | WARN |
| 2026-04 | 0.6970 | 0.6995 | -0.0021 | -0.0025 | 0.7248 | 24.5% | WARN |
| 2026-05 | 0.6830 | 0.6181 | +0.0793 | +0.0649 | 0.0011 | 59.3% | ALERT |
| 2026-06 | 0.6899 | 0.5993 | +0.0981 | +0.0906 | 0.0010 | 59.2% | ALERT |
| 2026-07 | 0.6890 | 0.5589 | +0.1385 | +0.1301 | 0.0024 | 58.3% | ALERT |

Queda positiva indica AUC menor. O efeito pareado original/alterado é diferente
da diferença contra controle, que inclui diferenças entre amostras.
Valores arredondados; ICs e demais métricas estão no
[consolidado](evidencias/performance/summary.csv).

### Janeiro–fevereiro: condição estável

Nenhuma intervenção. Nenhuma feature com drift confirmado e nenhuma queda de
AUC acima de 0,05. Gate OK. Esses dois meses verificam comportamento nesta
amostra; não estimam uma taxa geral de falsos alertas.

### Março: deslocamento parcial de entradas

Intensidade 0,6: renda reduzida, DTI aumentado, utilização acrescida e FICO
reduzido. Detectados dti, revol_util e fico_avg (3/23). Target preservado.
AUC próxima do controle; gate WARN, recomendando investigar entradas e reforçar
monitoramento. A aprovação cai de forma expressiva, apesar da AUC estável.

### Abril: deslocamento mais intenso

Intensidade 1,0. Detectadas as quatro features: annual_inc, dti, revol_util e
fico_avg (4/23). Nenhuma queda relevante de ranking, mas alteração importante
na distribuição dos scores e na aprovação. Isso impede interpretar “AUC estável”
como “nenhum impacto nas decisões”.

### Maio–julho: alteração de rótulos por segmentos

Intensidades 0,35 / 0,55 / 0,75. X e scores de cada registro são preservados;
somente o target muda nos segmentos definidos. As quedas pareadas de AUC são
0,0649 / 0,0906 / 0,1301. Nenhuma feature marginal com drift confirmado.
A diferença de aprovação entre meses decorre das amostras, não do relabel.

O IC da queda contra controle fica inteiro acima de 0,05 nos três meses.
Gate ALERT recomenda investigar e avaliar retreinamento. Alertas de queda,
suspeita de concept drift, label drift e recomendação são publicados.
Nenhum treinamento ou promoção foi executado.

### Demo inválida: proteção antes da inferência

O lote separado corrupted tem 2.100 linhas e oito entradas de violações
bloqueantes. Foi processado antes dos sete meses, colocado em quarentena e
não recebeu predição; os meses seguintes terminaram normalmente. Código 2
indica o bloqueio esperado. Violações podem se sobrepor; suas contagens não
são registros únicos.

## Restauração de marginais: evidência complementar

Após restaurar conjuntamente os três grupos analisados:

| Mês | AUC após restauração | PSI do score após restauração |
|---|---:|---:|
| 2026-03 | 0.6948 | 0.0008 |
| 2026-04 | 0.6970 | 0.0009 |
| 2026-05 | 0.6184 | 0.0010 |
| 2026-06 | 0.5993 | 0.0011 |
| 2026-07 | 0.5589 | 0.0023 |

Em março/abril, a recuperação dos scores é compatível com sensibilidade às
entradas alteradas. Em maio–julho, a restauração dos grupos não recupera
suficientemente a AUC. O método não recupera necessariamente os valores
originais de cada registro e não prova causa econômica. Veja
[limites da análise](causal-analysis.md) e o
[JSON de sensibilidade](evidencias/causal/causal_analysis.json).

## Encaminhamento para governança e apresentação

O incidente é deliberadamente induzido, com originais preservados e intervenções
conhecidas. Não atribuir o resultado a inflação, renegociação ou campanha real:
esses são exemplos de hipóteses do DAG. Não extrapolar a população norte-americana
histórica para concessão real de crédito no Brasil.

Para o post-mortem, apresentar a sequência entrada → score/decisão → performance
→ recomendação, incluindo o bloqueio do contrato. Uma política futura de
retreino precisará de dados/validação próprios, análise econômica e fairness;
o experimento não promove nenhum challenger.

Evidências: [HTML estável](evidencias/drift/2026-01.html),
[HTML de entradas](evidencias/drift/2026-04.html),
[HTML de degradação](evidencias/drift/2026-06.html),
[observabilidade](evidencias/observability/README.md) e
[manifesto](evidencias/manifest.json).
