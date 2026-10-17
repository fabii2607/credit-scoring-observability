# Análise do experimento e recomendações

O comando `make analyze` valida os lotes, aplica o modelo congelado, compara
distribuições e performance, gera os relatórios Evidently e consolida o
experimento. Os meses são cenários sintéticos, não uma avaliação temporal de
produção real. Para gerar os dados pela primeira vez, execute `make simulate`.

## Como consultar

- Abra `reports/experiment/summary.md`: tabela temporal, interpretação e três gráficos.
- Consulte `reports/experiment/monthly.csv` para AUC, intervalos, PR-AUC, Brier,
  aprovação, prevalência, features com drift, score PSI e recomendação.
- Consulte `reports/experiment/features.csv` para PSI/KS e Jensen–Shannon por feature.
- Abra `reports/drift/<mês>.html` no navegador: a recomendação aparece no topo,
  seguida das métricas, comparação antes/depois e gráficos do Evidently.
- Os JSONs em `reports/gate/`, `reports/intervention/`, `reports/performance/`
  e `reports/drift/` preservam os resultados estruturados. O gate do HTML e do
  JSON de drift usa o mesmo objeto do JSON de gate.

A [seleção pública de evidências](evidencias/README.md) permite consultar exemplos
sem gerar os dados. Os resultados locais completos permanecem ignorados pelo Git.
Uma execução com `--batches` consolida somente a seleção solicitada; arquivos
individuais de outros meses podem continuar no disco.

## Três comparações complementares

1. **Original contra referência:** identifica diferenças que já existiam na
   amostra antes da intervenção.
2. **Alterado contra referência:** reproduz o monitoramento das entradas que
   seriam recebidas em produção.
3. **Original contra alterado:** mantém os mesmos identificadores e ordem e
   mede o efeito da intervenção naquela amostra. Identificadores não são
   publicados nos relatórios.

O código verifica os hashes dos originais no manifesto e valida seu contrato.
No cenário estável, nenhuma coluna pode mudar; no cenário de entradas, somente
as quatro features previstas e suas predições; no relabel, somente o target.
PSIs não são aditivos: não se deve subtrair dois PSIs e chamar a diferença
de efeito causal.

A queda de AUC contra o controle utiliza bootstrap independente estratificado
em cada amostra. A queda pareada utiliza os mesmos índices sorteados nas duas
versões, estratificados pelo target original. Os intervalos são percentis de
95%, com seed fixa e 200 reamostragens por padrão. São condicionais aos dados,
ao modelo congelado e ao procedimento; não incluem a incerteza do treinamento
nem demonstram causalidade econômica.

## Política do gate

O gate roda após drift e performance de cada lote válido. Não recebe o nome do
cenário para decidir: utiliza somente métricas e cobertura dos dados.

| Sinal | Status | Ação |
|---|---|---|
| Rótulos insuficientes ou features indisponíveis | WARN | Investigar cobertura e qualidade |
| Queda de AUC maior que 0,05, sem confirmação suficiente pelo intervalo | ALERT | Investigar degradação |
| Queda maior que 0,05 e IC inteiro acima de 0,05 | ALERT | Investigar e avaliar retreinamento |
| Drift de entrada com AUC sem queda acima do limite | WARN | Reforçar monitoramento |
| Intervalo permite queda acima do limite, prevalência alterada ou sinais de atenção | WARN | Investigar ou reforçar monitoramento |
| Nenhum dos sinais anteriores | OK | Continuar monitoramento regular |

Os critérios são avaliados nessa ordem, com checagem de disponibilidade das
features após avaliar a degradação. Os limites vêm de `params.yaml`:
`performance.auc_drop`, `performance.bootstrap_samples`, `drift.*` e
`gate.min_labeled_rows` (500), `gate.min_label_coverage` (0,90),
`gate.min_bootstrap_samples` (100). Quantidade e cobertura de rótulos são
exigidas tanto no lote quanto no controle. Desabilitar bootstrap impede
recomendar retreinamento.

O gate registra justificativas e limitações. “Concept drift suspeito” é um
diagnóstico operacional: AUC pior com marginais estáveis também pode resultar
de problemas nos rótulos ou mudanças conjuntas nas features. Retreinamento é
uma recomendação para avaliação humana; nenhum modelo é treinado, promovido
ou substituído automaticamente. Airflow não é necessário para executar esta análise.

## Interpretação dos cenários

- Janeiro–fevereiro: procurar alertas inesperados e variação amostral. Dois
  meses não estimam uma taxa geral de falsos positivos.
- Março–abril: verificar pelo menos duas features alteradas detectadas e medir
  o efeito sobre AUC, Brier e aprovação. AUC estável não garante probabilidades
  calibradas nem decisões de crédito adequadas.
- Maio–julho: verificar perda de ranking com X e scores dos mesmos registros
  preservados. Esse comportamento é induzido pelo relabel sintético.

O controle de performance e os meses são disjuntos e não participaram do ajuste.
Entretanto, vêm do conjunto de teste já utilizado na avaliação inicial do modelo;
o experimento não substitui uma nova validação externa.
