# Roteiro do vídeo STAR (5 minutos)

Doze slides, quatro apresentadores, cerca de 25 segundos por slide. A ordem
segue o método STAR: Situação (Fabi), Tarefa (Fabi), Ação (Fabi, Helio,
Pedro, Erick) e Resultado (Pedro, Erick). O script abaixo é o mesmo das
notas do apresentador no deck; os números vêm de
[`docs/evidencias/README.md`](evidencias/README.md), de
[`lgpd.md`](lgpd.md) e do [post-mortem](postmortem.md).

## Divisão e tempo

| # | Slide | Quem | Tempo | Mensagem |
|---|---|---|---|---|
| 1 | Capa | Erick | 10 s | Quem somos e o que é o projeto |
| 2 | Situação | Fabi | 25 s | Degradação silenciosa; dataset e modelo |
| 3 | Tarefa | Fabi | 25 s | Quatro etapas, uma pergunta |
| 4 | Contrato de dados | Fabi | 25 s | Pandera, lote bloqueado, exit 2 |
| 5 | Simulação e detecção | Helio | 25 s | Sete meses, covariate × concept, PSI/KS, Evidently |
| 6 | Gate | Helio | 25 s | Tabela dos sete meses; WARN × ALERT |
| 7 | Observabilidade | Pedro | 25 s | Estágios, logs, métricas, MLflow, alertas |
| 8 | Resultado | Pedro | 25 s | AUC 0,70 → 0,56; 21 alertas |
| 9 | Governança | Erick | 25 s | PII, base legal, retenção, fairness |
| 10 | Análise causal | Erick | 25 s | DAG e sensibilidade, sem prova de causa |
| 11 | Lições | Erick | 25 s | Post-mortem e o que foi entregue |
| 12 | Encerramento | Erick | 10 s | Repositório e agradecimento |

Total: cerca de 4 min 50 s. Se o tempo apertar, corte a leitura das linhas da
tabela do slide 6 (cite só abril e julho) e a lista do slide 7.

## Regras de gravação

- Um apresentador por slide; a troca acontece na virada do slide, sem
  despedida ("a Fabi começa pela situação" já faz a passagem).
- Demonstração ao vivo só no slide 4 (`make demo-contract`, exit 2) e no
  slide 8 (HTML do Evidently de abril ou o dashboard, se a stack estiver de
  pé). O resto é slide.
- Nenhum número fora dos documentos versionados; não prometer retreino,
  fairness integrada ao pipeline ou causalidade provada.

## Script

### Erick — slide 1 (10 s)

Olá. Somos a Fabi, o Helio, o Pedro e o Erick. Este é o nosso Tech Challenge
da Fase 4: uma camada de sustentação para um modelo de credit scoring que
começa a errar sem avisar. A Fabi começa pela situação.

### Fabi — slide 2 (25 s)

A situação é a de uma fintech com um modelo de credit scoring em produção. A
economia muda, inflação, novos perfis de cliente, e a diretoria desconfia que
o modelo está tomando decisões erradas sem ninguém perceber: a degradação
silenciosa. O modelo é simples de propósito, uma regressão logística treinada
em 1,35 milhão de empréstimos do Lending Club, com 20% de inadimplência e
AUC de 0,69. O difícil não é o modelo: é saber quando ele parou de funcionar.

### Fabi — slide 3 (25 s)

A tarefa tinha quatro etapas, uma por integrante. Eu fiquei com a validação
de dados: treinar o baseline e impedir que lixo entre no modelo. O Helio
simula a passagem do tempo e detecta o drift. O Pedro dá visibilidade a tudo
isso com logs, métricas e alertas. E o Erick fecha com a governança: LGPD,
viés e a análise causal, documentadas no README. Tudo roda do zero com make,
com integração contínua e 199 testes.

### Fabi — slide 4 (25 s)

O contrato é um schema Pandera sobre as 23 features finais, mais regras de
lote. Renda obrigatória, FICO entre 300 e 850, prazo válido, cliente não
duplicado: tudo bloqueante. Para provar, forçamos um lote corrompido de 2.100
linhas: oito violações, exit 2, nenhuma predição, e o lote vai para a
quarentena, que é descartada em 30 dias. A separação entre Referência e
Produção também é do contrato: 270 mil linhas que o modelo nunca viu, sem
nenhuma interseção de conteúdo.

### Helio — slide 5 (25 s)

Para simular o tempo passando, geramos sete lotes mensais de 10 mil linhas do
pool. Janeiro e fevereiro ficam intactos. Em março e abril aplicamos um
covariate shift: renda cai, DTI e rotativo sobem, FICO cai, primeiro com
intensidade 0,6 e depois 1,0, sem mexer nos rótulos. De maio a julho é o
contrário: só os rótulos mudam, por segmento, contra o que o modelo aprendeu.
A detecção usa PSI por quantis para triagem e KS para confirmar, e o
Evidently gera o relatório HTML de cada mês apontando as features degradadas.

### Helio — slide 6 (25 s)

O gate decide com métricas, sem saber o cenário. Em março e abril ele confirma
drift em três e quatro features, o PSI do score dispara e a aprovação despenca
de 59% para 24%, mas a AUC continua igual à do controle: WARN, reforçar
monitoramento, não retreinar. De maio a julho é o inverso: nenhuma feature
muda, o PSI é zero, e a AUC cai de 0,70 para 0,56. Aí o gate exige que o
intervalo de confiança da queda, por bootstrap, fique inteiro acima de 0,05
antes de recomendar avaliar o retreinamento. Nada é retreinado
automaticamente.

### Pedro — slide 7 (25 s)

A observabilidade começa no pipeline: sete estágios por lote, cada um com log
estruturado de sucesso ou falha, duração, CPU e memória, e o customer_id
mascarado antes de gravar. Os logs vão para o Loki, os traces para o Tempo, as
métricas para o Prometheus via Pushgateway, e cada lote vira um run no MLflow
com o relatório do Evidently e o JSON do gate. No Grafana, um dashboard
provisionado e dez regras de alerta, cada uma apontando para o seu runbook. E
tudo roda sem Docker com make monitor, para o avaliador reproduzir.

### Pedro — slide 8 (25 s)

O resultado da execução de referência: a AUC fica em 0,70 até abril e cai
para 0,56 em julho, e o pipeline reagiu a cada fase com o alerta certo. Drift
crítico de features em março e abril; queda de performance, suspeita de
concept drift, label drift e recomendação de retreino de maio a julho; e a
violação de contrato na demo. Foram 21 alertas, todos entregues ao
Alertmanager com o runbook correspondente, e nenhum retreino automático. Tudo
isso está congelado com hashes na pasta de evidências do repositório.

### Erick — slide 9 (25 s)

A governança está no README e no lgpd.md, e cada afirmação tem um teste ou um
arquivo de configuração por trás. A PII é tratada no pipeline: o id vira um
hash com sal que fica fora do git, só 27 das 151 colunas são lidas, e nenhum
identificador aparece em log, métrica ou relatório. A base legal é a proteção
do crédito, artigo sétimo, inciso dez, com a Lei do Cadastro Positivo, e o
artigo vinte garante a explicação e a revisão humana. A retenção vai de 30
dias na quarentena a 90 dias na stack. E o viés é medido por região: 0,98 no
pool, nunca abaixo de 0,8 nos sete meses; a mitigação por thresholds foi
avaliada e rejeitada.

### Erick — slide 10 (25 s)

A diretoria pergunta se a economia causou o problema. Respondemos em duas
camadas. Primeiro, um DAG de hipóteses: inflação corrói renda e eleva o DTI,
novos perfis derrubam o FICO; isso muda quem chega. Renegociação muda o que
acontece com o mesmo perfil. Segundo, um teste de sensibilidade: restauramos
as marginais dessas variáveis para a referência. Em março e abril o PSI do
score volta a zero; de maio a julho a AUC não se recupera. É compatível com as
hipóteses, mas é sensibilidade do modelo, não prova de causa: para isso seriam
necessários dados de intervenção.

### Erick — slide 11 (25 s)

O post-mortem, sem culpados, deixa quatro lições. AUC estável não significa
ausência de impacto: em abril a aprovação caiu 35 pontos com a AUC intacta.
Degradação sem drift exige rótulos, então a política precisa declarar a
latência que assume. Fairness medida à mão depois do incidente não protege
ninguém durante o incidente. E sensibilidade organiza hipóteses, mas não prova
causa. Entregamos tudo no repositório: governança no README, evidências com
hashes, tags, CI verde com 199 testes e commits semânticos dos quatro.

### Erick — slide 12 (10 s)

O modelo continua simples; o que mudou é que agora sabemos quando ele erra,
por quê, e o que fazer. O código, as evidências e a documentação estão no
repositório. Obrigado.
