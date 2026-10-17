# Simulação de produção para monitoramento de crédito

O simulador seleciona registros do teste reservado da modelagem e constrói uma
sequência de sete meses com intervenções controladas. Seu objetivo é permitir
comparar mudanças nas características dos clientes com mudanças na relação
entre características e inadimplência. A geração não treina nem pontua modelos.

## Execução

Após instalar o ambiente e preparar os dados:

```bash
uv sync --locked
make prepare
make simulate
```

Sem `make`, execute na raiz do repositório:

```bash
uv run python -m credit_scoring_observability.simulation
```

Os caminhos podem ser configurados pela CLI:

```bash
uv run python -m credit_scoring_observability.simulation --pool data/processed/production_pool.parquet --reference data/processed/reference.parquet --output-dir data/production
```

`--reference` deve apontar para a referência **completa**, para verificar a
separação dos identificadores; passar só sua amostra enfraquece essa checagem.
O simulador usa `params.yaml`, validado por `parameters.py`. Não precisa de
`.env`, pois consome identificadores já pseudonimizados pelo `prepare`.
Não altere arquivos de entrada ou execute consumidores enquanto gera os lotes.

## Origem e partições

Os 269.620 registros do pool foram separados do treino por split aleatório
estratificado e já foram usados na avaliação inicial do modelo. Não constituem
um teste externo novo. As datas históricas de originação vão de 2007 a 2018;
janeiro–julho de 2026 são identificadores da simulação, preservando `issue_date`.

Com a configuração padrão, uma permutação com seed 42 seleciona sem reposição:

| Destino | Registros originais distintos | Função |
|---|---:|---|
| Controle | 30.000 | Comparação de performance fora do treino, sem intervenções. |
| Calibração | 10.000 | Verificação das intensidades antes da avaliação dos meses. |
| Demo | 2.000 | Origem exclusiva do lote corrompido. |
| Sete meses | 70.000 | 10.000 registros por cenário mensal. |
| Não utilizados | 157.620 | Permanecem no pool. |

As partições são disjuntas entre si e da referência por id de empréstimo. Isso
não garante isolamento de pessoas com múltiplos empréstimos. Cópias originais e
versões alteradas do mesmo mês são pareadas de propósito, não novas partições.

A auditoria exige target binário completo, metadados, features e aprovação no
contrato. Não descarta silenciosamente registros inválidos; interrompe a geração
antes de publicar saídas. O modelo não precisa estar disponível para gerar dados.

## Cenários e intensidades

| Meses simulados | Cenário | Intensidades padrão |
|---|---|---|
| Janeiro–fevereiro | Estável: cópia sem intervenção. | 0 |
| Março–abril | Deslocamento de quatro features; target original mantido. | 0,6 e 1,0 |
| Maio–julho | Relabel em segmentos; todas as features e metadados preservados. | 0,35; 0,55; 0,75 |

### Alterações das features

Para intensidade `i`, a configuração padrão aplica:

```text
annual_inc' = annual_inc × (1 − 0,35 × i)
dti'        = dti × (1 + 0,80 × i)
revol_util' = revol_util + 25 × i
fico_avg'   = clip(fico_avg − deslocamento, 300, 850)
```

O deslocamento de FICO é `30 × i` arredondado para o múltiplo de 5 mais próximo
(empates para cima). Para as intensidades padrão, são 20 e 30 pontos. A média
original da faixa FICO não é arredondada. O corte no limite pode reduzir o
deslocamento nos extremos. Nulos permitidos continuam nulos; não se impõe teto
de 100 a `revol_util`. O RNG não é consumido nessa intervenção determinística.

Manter Y enquanto X muda não garante covariate shift puro nem AUC estável; ambos
devem ser avaliados. O cenário representa mudança artificial de características,
não um efeito econômico identificado nos dados reais.

### Mudanças de conceito

Os segmentos são definidos usando os valores originais:

- **Novos defaults:** target 0, FICO ≥ 700 e prazo 36 meses; probabilidade de
  trocar target para 1 = `0,25 × i`.
- **Curas:** target 1 e (FICO ≤ 680 ou prazo 60 meses); probabilidade de trocar
  target para 0 = `0,20 × i`.

Essas regras alteram desfechos em direção contrária às associações de risco
usuais. Não consultam o score nem produzem rótulos a partir das probabilidades
do modelo. Features, ordem e identificadores ficam exatamente iguais à cópia
original. O manifesto registra elegíveis e trocas 0→1/1→0; só parte dos registros
elegíveis muda. O target histórico é preservado na cópia original.

Uma observação elegível recebe um sorteio uniforme. Se o mesmo lote e estado
inicial do RNG forem usados, intensidades maiores incluem as alterações das
menores. Meses distintos usam outros registros e streams: monotonicidade de AUC,
prevalência ou contagens entre meses não é garantida e precisa ser medida.

### Lote corrompido

`validate.make_corrupted` é reaproveitado para introduzir erros na reserva da
demo: renda ausente, valor negativo, FICO inválido, prazo/categoria inválidos,
inconsistência de emprego, duplicação e coluna de desfecho indevida.

Com os valores padrão, o arquivo tem 2.100 linhas, pois a função acrescenta
duplicatas a 2.000 registros originais. Deve falhar no contrato, enquanto os
meses regulares devem passar. A geração apenas verifica o bloqueio esperado;
não move o arquivo para quarentena. A CLI de validação ou o pipeline consumidor
fazem esse movimento quando processam o lote. Regenerar a simulação recria o
arquivo depois de uma demonstração que o tenha movido.

## Artefatos e reprodução

```text
data/production/
├── 2026-01.parquet ... 2026-07.parquet  # versões finais dos sete meses
├── corrupted.parquet
├── manifest.json
├── originals/
│   └── 2026-01.parquet ... 2026-07.parquet
└── reserves/
    ├── control.parquet
    ├── calibration.parquet
    └── demo.parquet
```

O manifesto contém parâmetros, seed e estratégia de streams, contagens, auditoria,
prevalências, trocas de rótulos, caminhos relativos, hashes SHA-256 das fontes,
código e arquivos gerados e versões de Python/NumPy/pandas/PyArrow/Pandera.
Não contém identificadores individuais, exemplos de linhas ou o sal.

Um hash de conteúdo adicional considera schema, ordem e valores do DataFrame;
o hash do parquet verifica os bytes do arquivo. A reprodução exata depende das
mesmas fontes, implementação, parâmetros e versões. O modelo de scoring não
participa da geração, por isso não recebe hash neste manifesto.

Todos os lotes são auditados em memória e escritos primeiro em uma pasta
temporária. Cada arquivo é substituído e o manifesto publicado por último.
Isso não equivale a uma transação de todo o diretório em caso de interrupção:
antes de consumir, confira os hashes. Uma nova execução com os mesmos parâmetros
recria as saídas conhecidas; calendário diferente exige outro diretório para não
misturar meses antigos. Não há exclusão automática de arquivos desconhecidos.

Datasets, cópias originais e manifesto ficam em `data/`, fora do Git. A
documentação e o código permitem recriá-los; relatórios de monitoramento são
produzidos separadamente. O pipeline de observabilidade ainda precisa ser ligado
a esses lotes; `make simulate` não executa `make monitor`.

## Verificações

```bash
uv run pytest tests/test_simulation.py tests/test_parameters.py
make lint
make test
```

Os testes não exigem Kaggle. Verificam disjunção, determinismo, integridade das
entradas, colunas alteradas, bloqueio do lote corrompido, limites, ausência de
rótulos, hashes dos artefatos e degradação em um modelo treinado em dados
sintéticos. Consulte também [estatística de drift](drift-statistics.md) e
[execução dos relatórios](drift-monitoring.md).
