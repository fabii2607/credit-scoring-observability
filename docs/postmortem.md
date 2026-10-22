# Post-mortem do incidente simulado (2026-03 a 2026-07)

Execução de referência de 21/10/2026 (America/Sao_Paulo): modelo `V2_optimized`
congelado, threshold 0,20, controle reservado de 30 mil registros com AUC
0,6974. A base de código é o `source_revision` 72468ee (commit de 17/10/2026);
as evidências públicas foram commitadas em c5d4029 e o HEAD da entrega é
8c6c880 (merge do PR #15, 21/10/2026). Modelo, metadados, dados e fontes
efetivas são identificados por SHA-256 no
[manifesto](evidencias/manifest.json) (`model_sha256` 86aede00…,
`control_sha256` 9a3173db…). A tag `execucao-referencia` ainda aponta para
dd1dc11 (14/10/2026) e não identifica esta execução. Os meses 2026-01 a
2026-07 são cenários sintéticos com intervenções conhecidas, não datas de
produção.

Este post-mortem não atribui culpa: o incidente foi induzido de propósito para
verificar o que o pipeline detecta, o que recomenda e o que deixa passar.

## Resumo

Oito lotes em três fases. A demo `corrupted` foi barrada pelo contrato antes da
inferência. Janeiro e fevereiro ficaram estáveis: gate OK e nenhum alerta. Em
março e abril as entradas foram deslocadas: a AUC permaneceu igual à do
controle, mas o PSI do score subiu para 0,30 e 0,72 e a aprovação caiu de cerca
de 59% para 35,4% e 24,5% (gate WARN). Em maio–julho os rótulos foram alterados
por segmentos: X e scores ficaram iguais, a AUC caiu para 0,6181, 0,5993 e
0,5589 e o IC da queda ficou inteiro acima de 0,05 (gate ALERT, recomendação
de investigar e avaliar retreinamento). Nenhum retreino, challenger ou promoção
foi executado.

## Impacto

O impacto nas decisões ocorreu de duas formas distintas, e a AUC só captou a
segunda. Valores de `evidencias/performance/summary.json`; aprovação é a
fração de registros com score abaixo de 0,20.

| Mês | AUC do lote | Queda vs. controle | PSI do score | Aprovação | Default observado | Gate |
|---|---:|---:|---:|---:|---:|---|
| 2026-01 | 0,7051 | −0,0077 | 0,0015 | 59,36% | 20,58% | OK |
| 2026-02 | 0,7035 | −0,0061 | 0,0017 | 59,51% | 20,31% | OK |
| 2026-03 | 0,6963 | +0,0011 | 0,3014 | 35,44% | 19,18% | WARN |
| 2026-04 | 0,6995 | −0,0021 | 0,7248 | 24,48% | 20,21% | WARN |
| 2026-05 | 0,6181 | +0,0793 | 0,0011 | 59,26% | 21,66% | ALERT |
| 2026-06 | 0,5993 | +0,0981 | 0,0010 | 59,24% | 21,56% | ALERT |
| 2026-07 | 0,5589 | +0,1385 | 0,0024 | 58,26% | 22,50% | ALERT |

Março e abril: a aprovação caiu de cerca de 59% para 35,4% e 24,5%, isto é,
cerca de 24 e 35 pontos percentuais a menos, com AUC estável (queda +0,0011 e
−0,0021, ICs cruzando zero). O modelo passou a negar muito mais: recall de
0,848 e 0,910 com precision de 0,252 e 0,244, contra 0,639 e 0,324 em janeiro.
Em abril, a versão original do mesmo lote aprovava 59,16% e tinha Brier 0,1479;
a versão alterada tem Brier 0,1619. Para quem pede crédito, esse é o impacto
maior do incidente, e ele não aparece na AUC.

Maio a julho: a aprovação ficou estável (59,26%, 59,24% e 58,26%; em junho a
versão original aprovava os mesmos 59,24%), mas a AUC caiu para 0,6181, 0,5993
e 0,5589, com quedas pareadas de 0,0649, 0,0906 e 0,1301. O default observado
subiu para 21,66%, 21,56% e 22,50% (em junho, 19,59% na versão original), com
`label_z` 3,99, 3,75 e 5,98. As decisões continuaram as mesmas, mas passaram a
separar pior quem paga de quem não paga: recall 0,543, 0,530 e 0,499.

## Linha do tempo: detecção, sinal, decisão

A ordem abaixo é a ordem de processamento da CLI
(`run_all --batches corrupted 2026-01 ... 2026-07`), não uma sequência de
horários reais: `evidencias/observability/alertmanager-check.json` guarda
apenas nome, lote e estado de cada alerta (21 entradas, todas `active`), sem
timestamps. Os lotes foram processados em sequência na mesma execução.

| Lote | O que o pipeline mediu | Alertas (alertmanager-check.json) | Gate e ação recomendada | Runbook aplicado |
|---|---|---|---|---|
| corrupted | Contrato: 2.100 linhas, 8 violações bloqueantes, 0 avisos (coluna `loan_status` inesperada; 200 `customer_id` duplicados; 103 a 107 linhas por regra de `model_input`). Sem predição; CLI retorna 2 | ContractViolation ×1 | Sem gate (lote bloqueado) | ContractViolation: corrigir a origem e reprocessar o lote completo; não remover o bloqueio nem usar quarentena como treino |
| 2026-01 | 0/23 features com drift; PSI do score 0,0015; AUC 0,7051 (IC 0,6912–0,7168); queda −0,0077 (IC −0,0212; +0,0057) | nenhum | OK / `observe`: continuar monitoramento regular | nenhum |
| 2026-02 | 0/23; PSI do score 0,0017; AUC 0,7035 (IC 0,6919–0,7161); queda −0,0061 (IC −0,0211; +0,0081) | nenhum | OK / `observe` | nenhum |
| 2026-03 | 3/23 confirmadas: `dti` (PSI 0,602; KS 0,319), `revol_util` (0,330; 0,212), `fico_avg` (0,444; 0,333); `annual_inc` em faixa de aviso (PSI 0,204); PSI do score 0,3014; aprovação 35,44%; AUC 0,6963 (IC 0,6831–0,7095); queda +0,0011 (IC −0,0141; +0,0152) | FeatureDriftCritical ×3, FeatureDriftWarning ×1 | WARN / `reinforced_monitoring`: reforçar monitoramento e investigar mudanças nas entradas; "acompanhar Brier e aprovação" | FeatureDriftCritical: abrir o HTML Evidently, investigar composição e origem, comparar AUC, Brier e aprovação; drift isolado não autoriza retreinamento. FeatureDriftWarning: reforçar monitoramento |
| 2026-04 | 4/23 confirmadas: `annual_inc` (0,645; 0,350), `dti` (1,165; 0,449), `revol_util` (1,800; 0,350), `fico_avg` (0,897; 0,488); PSI do score 0,7248; aprovação 24,48%; Brier 0,1619; AUC 0,6995 (IC 0,6896–0,7108); queda −0,0021 (IC −0,0159; +0,0112) | FeatureDriftCritical ×4 | WARN / `reinforced_monitoring` | FeatureDriftCritical |
| 2026-05 | 0/23; PSI do score 0,0011; AUC 0,6181 (IC 0,6046–0,6300); queda +0,0793 (IC 0,0641; 0,0953); pareada +0,0649; default 21,66%; `label_z` 3,99 | ModelPerformanceDrop, ConceptDriftSuspected, LabelDrift, RetrainRecommended | ALERT / `retrain`: investigar degradação e avaliar retreinamento (`retrain_recommended` = true) | ModelPerformanceDrop, ConceptDriftSuspected, LabelDrift e RetrainRecommended: encaminhar à avaliação humana; nenhum treino automático |
| 2026-06 | 0/23; PSI do score 0,0010; AUC 0,5993 (IC 0,5849–0,6141); queda +0,0981 (IC 0,0812; 0,1139); pareada +0,0906; default 21,56%; `label_z` 3,75 | os mesmos quatro | ALERT / `retrain`; limitação registrada: "Queda de AUC e estabilidade marginal de X não provam concept drift ou causa econômica" | os mesmos quatro |
| 2026-07 | 0/23; PSI do score 0,0024; AUC 0,5589 (IC 0,5455–0,5711); queda +0,1385 (IC 0,1207; 0,1545); pareada +0,1301; default 22,50%; `label_z` 5,98 | os mesmos quatro | ALERT / `retrain` | os mesmos quatro |

`retrain` é o código de `recommended_action`; o texto da recomendação é
"Investigar degradação e avaliar retreinamento". O gate não recebe o nome do
cenário: decide somente com métricas e cobertura de rótulos (10 mil rótulos no
lote e 30 mil no controle em todos os meses). A linha do tempo completa, com
os HTMLs por mês, está em [incident-timeline.md](incident-timeline.md).

## Causa

A causa é conhecida porque foi configurada. As intervenções estão em
`params.yaml` (seção `simulation`) e são descritas em
[production-simulation.md](production-simulation.md); os originais de cada mês
são preservados.

Março e abril (`data_drift`, intensidades 0,6 e 1,0): quatro features de
entrada foram deslocadas, com o target preservado. As intervenções máximas são
renda −35% (`income_reduction`), DTI +80% (`dti_increase`), utilização do
rotativo +25 p.p. (`utilization_increase_pp`) e FICO −30 pontos
(`fico_drop_points`), cada uma multiplicada pela intensidade do mês. Em março
isso equivale a renda ×0,79, DTI ×1,48, utilização +15 p.p. e FICO −20 pontos;
em abril, renda ×0,65, DTI ×1,80, utilização +25 p.p. e FICO −30 pontos
(deslocamento de FICO arredondado ao múltiplo de 5). Como o modelo é uma
regressão logística congelada, os scores sobem para os mesmos clientes e a
aprovação cai, com a ordenação entre bons e maus pagadores praticamente
preservada (queda pareada de AUC de −0,0014 em março e −0,0025 em abril).

Maio a julho (`concept_drift`, intensidades 0,35, 0,55 e 0,75): somente o
target muda, em segmentos definidos pelos valores originais. Registros com
target 0, FICO ≥ 700 (`concept_high_fico`) e prazo de 36 meses passam a 1 com
probabilidade 0,25 × i (`concept_default_probability`); registros com target
1 e FICO ≤ 680 (`concept_low_fico`) ou prazo de 60 meses passam a 0 com
probabilidade 0,20 × i (`concept_cure_probability`). As probabilidades
efetivas são 0,0875/0,07, 0,1375/0,11 e 0,1875/0,15. O relabel não consulta o
score e altera desfechos na direção contrária às associações de risco usuais;
por isso X e scores ficam idênticos e a AUC cai. As contagens de trocas por
mês ficam no manifesto local da simulação, fora do Git.

A análise de sensibilidade (`make causal`, resultado em
[causal_analysis.json](evidencias/causal/causal_analysis.json)) é coerente
com esse mecanismo, sem prová-lo. Restaurar as marginais dos três
grupos (renda+dívida, rotativo e FICO) para a referência devolve o PSI do score
de 0,3014 para 0,0008 em março e de 0,7248 para 0,0009 em abril (recuperação
0,997 e 0,999), com aprovação voltando a 58,87% e 59,25% e AUC praticamente
estável (0,6948 e 0,6970, contra 0,6963 e 0,6995 nos lotes alterados).
Restaurar só renda+dívida recupera 0,698 e 0,702 do desvio; só FICO, 0,766 e 0,734; só rotativo, 0,043 e 0,042. Em maio–julho a mesma
restauração não recupera a AUC: 0,6184, 0,5993 e 0,5589, com
`recovery_roc_auc` de 0,0032, −0,0001 e −0,0004. O diagnóstico do método é
"covariate shift compatível" em março/abril e "concept drift suspeito" em
maio–julho. É sensibilidade do modelo congelado a X, não identificação de
causa econômica: inflação, renegociação ou campanha são hipóteses ilustrativas
do DAG, não resultados medidos.

## O que funcionou

- O contrato barrou a demo antes do score, colocou o lote em quarentena e não
  interrompeu os sete meses seguintes; o código de saída 2 distingue bloqueio
  esperado de falha inesperada.
- O drift de entrada foi confirmado por PSI e KS por feature, antes de qualquer
  queda de performance, e o PSI do score mostrou o deslocamento das decisões.
- O gate separou os dois tipos de problema sem conhecer o cenário: WARN com
  "reforçar monitoramento" em março/abril e ALERT com "avaliar retreinamento"
  em maio–julho. Nenhum alerta ou recomendação indevida em janeiro/fevereiro.
- A queda de AUC foi acompanhada de intervalo de confiança (bootstrap, 200
  reamostragens) e de comparação pareada; o ALERT exigiu o IC inteiro acima do
  limite 0,05, não só o valor pontual.
- Os 21 alertas chegaram ao Alertmanager com `runbook_url` apontando para o
  runbook correspondente em [monitoring-plan.md](monitoring-plan.md); AUC e
  estado do gate iguais entre as métricas do Prometheus, o JSON dos resultados
  e os runs MLflow (verificação em `evidencias/observability/stack-check.json`);
  o HTML de cada mês repete a recomendação do JSON.
- A recomendação ficou como recomendação: nenhum treino, challenger ou
  promoção foi disparado.

## O que não funcionou ou ficou fora

- A aprovação caiu cerca de 35 p.p. em abril sem um alerta específico de
  aprovação. O sinal existiu (`approval_rate`, `score_psi`, Brier), e o
  runbook FeatureDriftCritical manda compará-lo, mas a detecção dependeu de
  leitura humana do gate WARN, não de uma regra.
- O impacto desigual por região não é medido pelo pipeline. Uma medição feita
  fora dele (`make fairness-batches`, gravada em
  `reports/fairness/fairness_by_batch.json`, versionado fora de
  `docs/evidencias/`) sobre os
  parquets locais regenerados por `make simulate` dá razão de impacto desigual
  de 0,875 em abril, contra 0,983 no pool
  (`reports/fairness/fairness_report.json`, execução local de 03/10). Ainda
  passa na regra dos 4/5, mas é a menor razão da série e ninguém a veria em
  produção com a configuração atual.
- Não há política de retreino: a recomendação do gate termina na avaliação
  humana, sem critérios de dados, validação ou fairness para um challenger.
- Não há alerta nem métrica publicada de taxa de sucesso ou de rejeição de
  lotes; os SLIs de proporção definidos em
  [monitoring-plan.md](monitoring-plan.md) são expressões PromQL lidas no
  Grafana/Prometheus sobre o snapshot do Pushgateway, sem janela temporal.
- Os timestamps dos alertas não foram preservados: `alertmanager-check.json`
  guarda só nome, lote e estado, e `logs/alerts.jsonl` do receptor não faz parte
  das evidências. Não é possível reconstruir quanto tempo cada alerta levou.
- O Pushgateway não tem volume nem retenção (o PUT substitui o grupo do lote),
  e o MLflow guarda `./mlruns` sem prazo definido; só Prometheus, Loki e Tempo
  têm retenção de 90 dias configurada.
- Em produção real, a fase de maio–julho só seria detectável depois da
  maturação dos desfechos: a detecção depende de rótulos, e o lote sintético os
  tem todos no momento do processamento.

## Lições

- AUC estável não significa ausência de impacto. Em março/abril, quem detectou
  o problema foram o PSI do score e a aprovação; a AUC mede ordenação, não
  decisão. O monitoramento precisa de um sinal de decisão com regra própria.
- Degradação sem drift marginal exige rótulos. Em maio–julho nada em X mudou;
  sem desfechos, nem PSI, nem KS, nem a sensibilidade à restauração veriam
  algo. Qualquer política de monitoramento precisa declarar a latência dos
  rótulos que assume.
- Separar "quem chega" de "o que acontece com o mesmo perfil" muda a ação
  recomendada: reforçar monitoramento e investigar entradas em um caso;
  investigar rótulos e avaliar retreinamento no outro. O gate só consegue
  isso porque combina drift, performance com IC e cobertura de rótulos.
- Sensibilidade organiza hipóteses, não prova causa. O método recupera o PSI
  em março/abril e nada em maio–julho, o que é consistente com as intervenções
  configuradas, mas o mesmo resultado poderia vir de problemas nos rótulos.
- Fairness fora do pipeline não é governança operacional. Uma razão que só é
  calculada à mão depois do incidente não protege ninguém durante o incidente.
- Observabilidade precisa guardar quando, não só o quê. Sem timestamps, a
  linha do tempo é a ordem de processamento.

## Ações

Todas as ações são propostas; nenhuma está implementada. O dono é o papel da
etapa, não uma pessoa.

| Ação | Dono (papel) | Estado |
|---|---|---|
| Definir SLI de aprovação por lote (desvio de `approval_rate` contra o controle) e alerta correspondente com runbook | Etapa 3 (observabilidade) | proposta |
| Integrar `approval_disparity` por lote à CLI mensal, publicar a razão por região como métrica e incluir a medição por lote nas evidências congeladas (`docs/evidencias/`) | Etapa 4 (governança), com Etapa 2 | proposta |
| Escrever a política de retreino: dados e validação próprios, análise econômica, fairness como guardrail e revisão humana antes de qualquer promoção | Etapa 4, com Etapa 2 (modelo) | proposta |
| Implementar os SLIs de taxa de sucesso e de rejeição (hoje só expressões PromQL em `monitoring-plan.md`) como métrica com janela explícita e alerta | Etapa 3 | proposta |
| Preservar `startsAt` dos alertas no checker e incluir `logs/alerts.jsonl` (sem dados de clientes) nas evidências | Etapa 3 | proposta |
| Definir retenção para Pushgateway e MLflow e registrar na tabela de retenção de [lgpd.md](lgpd.md) | Etapa 3, com Etapa 4 | proposta |
| Publicar `intervention/*.json` de todos os meses alterados, para que o default original de maio e julho fique nas evidências | Etapa 2 | proposta |
| Mover a tag `execucao-referencia` para o commit das evidências de 21/10 ou documentar por que permanece em dd1dc11 | responsável pela entrega | proposta |

## Limites

- Os meses são sintéticos, com intervenções escolhidas pelo grupo; o incidente
  mostra o que o pipeline faz diante delas, não o que faria diante de uma
  mudança real. Dois meses estáveis não estimam taxa de falsos alertas.
- O pool de produção é o conjunto de teste histórico do modelo; controle e
  meses são disjuntos entre si e não participaram do ajuste, mas não são uma
  validação externa nem observações de 2026.
- Sensibilidade não é causa: restaurar marginais por quantis não recupera os
  valores originais de cada registro, não preserva relações conjuntas e não
  identifica mecanismo econômico. O DAG é ilustrativo.
- ICs são condicionais aos dados, ao modelo congelado e ao procedimento; PSIs
  e recuperações não são aditivos.
- A medição de fairness por lote citada acima vem de parquets locais
  regenerados em 21/10, não idênticos byte a byte aos da execução de
  referência; o JSON resultante está versionado em `reports/fairness/`, não
  em `docs/evidencias/`.
- A população é norte-americana e histórica; nada aqui autoriza extrapolação
  para concessão real de crédito no Brasil.

Veja a [linha do tempo](incident-timeline.md), a
[análise de sensibilidade](causal-analysis.md), os
[runbooks](monitoring-plan.md) e as [evidências](evidencias/README.md).
