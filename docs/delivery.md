# Reprodução e entrega do monitoramento de crédito

## Escopo entregue

Contrato de entrada, modelo V2 congelado, sete lotes de 10 mil registros,
intervenções reproduzíveis, predições, drift, performance, comparação pareada,
gate, HTML Evidently, observabilidade e sensibilidade à restauração de marginais.
A aplicação é uma simulação acadêmica, não um serviço de concessão de crédito.

CT, Airflow, challenger/promoção automática, detecção multivariada e estimativa
de performance sem rótulos não estão implementados. Os extras citados nos
handovers não devem ser apresentados como funcionalidades concluídas.

## Reprodução a partir de um clone

Python 3.11, uv e Make. Obtenha o CSV do Kaggle indicado no README e coloque-o
em `data/raw/`; ele não é distribuído pelo repositório.

```bash
make setup
# Configure um sal do grupo em .env antes da preparação; nunca o commite.
make prepare
make train
make simulate
make analyze
make causal
make lint
make test
```

`make train` produz o binário e os metadados como par. Usar modelo de outra
execução com um JSON antigo compromete a rastreabilidade. Se receber os artefatos
do grupo em vez de treinar, confira ambos contra os hashes do manifesto.
Seeds e dependências estão em `params.yaml` e `uv.lock`. Para conferir resultados
numéricos, compare hashes de dados/modelo e versões, não apenas o nome V2.

`make analyze` permite consultar relatórios locais sem Docker. Os testes usam
dados sintéticos e não dependem do Kaggle. Diferenças de plataforma ou versões
podem introduzir diferenças numéricas; registre-as, não edite métricas para
reproduzir um número.

## Execução de referência com a stack

```bash
MLFLOW_PORT=5001 make up-full
# make simulate recria também a demo, que é movida à quarentena ao ser validada.
make simulate
PUSHGATEWAY_URL=http://localhost:9091 MLFLOW_TRACKING_URI=http://localhost:5001 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_TRACES_ENDPOINT= uv run python -m credit_scoring_observability.run_all   --batches corrupted 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07
make causal
```

A CLI retorna **2**, esperado pelo lote bloqueado. Os sete meses seguintes
devem terminar com sucesso. Falhas inesperadas ou de entrega detectadas retornam
1 e precisam ser investigadas. `make` pode apresentar seu próprio código de erro
quando um subprocesso termina com 2; consulte a CLI e os relatórios.

Após aguardar scrape e avaliação dos alertas, execute:

```bash
uv run python scripts/check_observability.py --mlflow-uri http://localhost:5001
docker run --rm --entrypoint /bin/promtool   -v "$PWD/monitoring/prometheus:/etc/prometheus:ro"   prom/prometheus:v2.48.0 test rules /etc/prometheus/alert_rules.test.yml
```

Consulte [integração](observability-integration.md) para endereços, modo local e
semântica. Os alertas são locais; o receptor escreve `logs/alerts.jsonl`.
A stack continua ativa até `docker compose down`; não use `down -v` se quiser
preservar o histórico dos volumes.

## Como consultar e avaliar

1. [Evidências e manifesto](evidencias/README.md): resumo e proveniência.
2. [Experimento consolidado](evidencias/experiment/summary.md): gráficos e tabela.
3. HTMLs de janeiro, abril e junho: recomendação no topo e evidência abaixo.
4. [Linha do tempo](incident-timeline.md): intervenções e efeitos medidos.
5. [Sensibilidade](causal-analysis.md), [model card](model-card.md) e
   [decisões técnicas](decisions/0001-drift-statistics.md).
6. [Runbooks](monitoring-plan.md): como investigar os alertas.

No GitHub, baixe cada HTML e abra no navegador. Em um clone, abra diretamente
`docs/evidencias/drift/<mês>.html`; API e Docker não são necessários para isso.

## Congelamento e pacote

A execução selecionada está em `docs/evidencias/`. O manifesto identifica
modelos/dados por hash, parâmetros, versões instaladas, HEAD base e fontes
efetivas do working tree por SHA-256. HEAD não é apresentado como commit limpo
quando ainda existem ajustes a commitar; o manifesto não inclui a si próprio
na lista de hashes.

Para atualizar as evidências de uma nova execução completa:

```bash
# O checker acima atualiza stack-check.json e alertmanager-check.json.
uv run python scripts/freeze_evidence.py --execution-date YYYY-MM-DD
```

A data deve ser a data local da execução. O congelamento exige sete meses com
sucesso, bloqueio da demo, baseline/hashes consistentes entre causal e análise
e verificação da stack correspondente aos mesmos runs MLflow.
O checker consulta `http://localhost:9093/api/v2/alerts` e salva a confirmação
em `reports/monitoring/alertmanager-check.json` somente com nome, lote e estado.

O comando `uv run python scripts/build_delivery_package.py` cria o pacote local
`reports/release/pacote-2.zip`, reunindo código/configuração e
documentação pública selecionada. Não contém `.env`, `agents/`, dados,
binários de modelo, logs brutos ou banco MLflow. Esses materiais não devem ser
adicionados por `git add .`. Artefatos de modelo/dados seguem o canal combinado
pelo grupo.

Os commits e a eventual tag `pacote-2` são criados pelo responsável após revisar
as alterações. A entrega técnica não cria tag, commit, push ou publicação
automaticamente.

## Verificação antes da entrega

- Lint/format e testes aprovados.
- Sete meses completos; demo bloqueada sem score; falha inesperada distinguida.
- Métricas/gate iguais entre JSON, HTML, MLflow e Prometheus.
- Logs correlacionados e traces entregues.
- Regras Prometheus válidas e runbooks com nomes/âncoras correspondentes.
- Manifestos, links e hashes conferidos; seleção pública sem IDs de clientes/sal.
- README descreve as funcionalidades existentes e limitações.
