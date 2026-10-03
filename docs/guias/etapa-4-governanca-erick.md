# Guia da Etapa 4 — Governança: Privacy by Design e by Default (Erick)

> Documento de direcionamento e contexto para LLM. Cole o arquivo inteiro no início da conversa.

## 0. Posição no revezamento

**Você é o 4º e fecha o projeto: semana 4.** Recebe os **Pacotes 1, 2 e 3**.

- **Antes do bastão (semanas 1 a 3), adiante sem depender de ninguém:**
  - rascunho completo do `docs/lgpd.md` (papéis, inventário, base legal, princípios, Privacy by Design, direitos do titular, retenção, RIPD) — nada disso precisa de código;
  - `causal.py`, `mitigation.py` e `retention.py` testados com dados sintéticos;
  - a partir do Pacote 2: fairness, mitigação e análise causal com o modelo e os lotes reais.
- **Você não entrega regras para os colegas.** As decisões de privacidade e retenção já estão nos guias deles. Seu papel no fim é **auditar** se foram cumpridas.
- **No fechamento você também:**
  - **consolida** o `docs/monitoring-plan.md`: SLIs/SLOs + os runbooks escritos por quem construiu cada parte (Fabi: dados; Helio: drift e modelo; Pedro: pipeline e infraestrutura, mais o catálogo de métricas);
  - escreve o **post-mortem** a partir da linha do tempo do Helio;
  - revisa o model card da Fabi (seção de fairness);
  - alinha **todos** os números à **execução de referência** do Pedro (tag `execucao-referencia`);
  - consolida o **README**.

  Com a documentação técnica escrita por quem construiu, a sua semana 4 fica para auditoria, governança e consolidação.

## 1. Contexto do projeto

Tech Challenge da Fase 4 do MLET (FIAP). Uma fintech tem um modelo de credit scoring em produção e suspeita de **degradação silenciosa** por mudanças na economia. O grupo constrói a **camada de sustentação**: contratos de dados, simulação e detecção de drift, observabilidade e governança LGPD.

- **Dataset:** Give Me Some Credit (Kaggle), 150.000 clientes, sem nomes nem documentos. Colunas: utilização do rotativo, idade, atrasos (3 colunas), razão de endividamento, renda mensal, linhas de crédito, empréstimos imobiliários, dependentes e o rótulo de inadimplência em 2 anos.
- **Pipeline batch mensal.** Os lotes de "produção" são **dados sintéticos baseados em regras** derivados do dataset.
- **Stack da etapa:** Markdown (README e `docs/`), fairlearn, pandas; Python para fairness, mitigação, análise causal e retenção.
- **Critérios de avaliação:** Validação 25% · Drift 25% · Observabilidade 20% · **Governança 15%** · Vídeo 15%.

**Sua etapa vale 15% e fecha o projeto:** você audita se as regras de privacidade que os colegas implementaram foram cumpridas, completa a governança (viés, causalidade, retenção) e escreve a documentação que o avaliador lê primeiro.

## 2. O que o enunciado pede na Etapa 4

Tarefas **teóricas, no README.md**:
- **Governança e LGPD:** plano de proteção de dados explicando:
  - como os dados sensíveis (PII) foram tratados;
  - qual a **base legal** para a decisão de crédito;
  - o **plano de retenção**.
- **Entregável:** documentação completa de **Governança e Causalidade** no README.

Outros pontos do enunciado que caem na sua etapa:
- o contexto pede "uma **análise causal** do problema";
- o critério de Governança avalia "clareza na documentação sobre privacidade, base legal e **mitigação de vieses**";
- boa prática obrigatória: "documentação de compliance contendo plano de privacidade (LGPD)";
- requisito do repositório: "documentação consolidada de Governança no README.md". Não basta linkar: o conteúdo precisa estar no README.

## 2.1 Prioridade: núcleo primeiro, extras depois

O objetivo é **cumprir o enunciado**. Faça o núcleo inteiro antes de qualquer extra. Se o prazo apertar, os extras são cortados sem prejudicar o requisito.

| Núcleo (obrigatório) | Requisito atendido |
|---|---|
| README com **como a PII foi tratada** (pseudonimização, minimização, logs sem identificador) | Etapa 4 |
| **Base legal** da decisão de crédito (Art. 7º, X + Lei 12.414; Art. 20) | Etapa 4 |
| **Plano de retenção** em tabela | Etapa 4 |
| **Viés**: fairness por idade medida + mitigação documentada | critério Governança ("mitigação de vieses") |
| **Análise causal** do problema (DAG de hipóteses + evidência covariate × concept) | Etapa 4 (entregável "Governança e Causalidade") |
| Tudo isso **dentro do README**, não só em links | requisito do repositório |

| Extra (diferencial) | Custo |
|---|---|
| `docs/lgpd.md` completo (10 princípios, Privacy by Design, direitos, RIPD) | médio — recomendado |
| Atribuição causal por intervenção em código (`causal.py`) | médio |
| Mitigação de viés implementada (thresholds por faixa) | baixo |
| Descarte automático da quarentena | baixo |
| Model card, post-mortem, `monitoring-plan.md` consolidado com runbooks | médio |

## 3. Decisões já tomadas (não reabrir sem o grupo)

| Tema | Decisão | Motivo |
|---|---|---|
| Papéis | Fintech = **controladora**, plataforma de ML = **operadora**, **encarregado (DPO)** nomeado (fictícios) | Caso Telekall (aula): ausência de encarregado agravou a sanção |
| Identificador | **Pseudonimização**: SHA-256 com sal secreto fora do git, truncado em 16 caracteres. Não é anonimização | O controlador precisa refazer a associação para atender os direitos do titular (Art. 18) |
| Dados sensíveis | Nenhum do Art. 11; **idade = atributo protegido** na análise de viés; não coletar gênero, CEP nem estado civil | Minimização e prevenção de *proxies* |
| Base legal | **Art. 7º, X (proteção do crédito)** + **Lei 12.414/2011**; Art. 7º, V (procedimentos preliminares); **Art. 20** (revisão de decisão automatizada). Consentimento **não** | Consentimento é revogável e inviabilizaria a análise de risco |
| Retenção | Quarentena 30 dias (com descarte automático); logs, traces e métricas 90 dias (**aplicado na configuração** da stack); relatórios agregados mantidos | Retenção aplicada, não só documentada |
| Telemetria | Opt-out de GE, NannyML, MLflow e Evidently em `src/__init__.py` | *Privacy by default*: as bibliotecas enviam dados de uso sem configuração |
| Fairness | Impacto desigual por faixa etária (18–29, 30–44, 45–59, 60+), com fairlearn `MetricFrame`; **guardrail de promoção** no retreino | Não discriminação (Art. 6º, IX) entra na decisão, não só no relatório |
| Mitigação | **Thresholds por faixa com paridade de FPR**, determinísticos; avaliada e registrada, **não ligada no pipeline** (adoção depende de avaliação jurídica) | O `ThresholdOptimizer` do fairlearn sorteia a decisão dentro da faixa (indefensável diante do Art. 20) |
| Causalidade | **DAG de hipóteses** + **atribuição por intervenção** em grupos de variáveis causalmente ligadas, preservando a ordem no lote | Restaurar uma feature isolada quebra as correlações (Aula 4 de Drift) |

## 4. Resultados de referência (dado real)

> Números do projeto anterior, para comparação. Os seus serão parecidos, não iguais. Documente sempre os números da **sua** execução de referência.

**Viés por idade, com o threshold único (v1, teste da Referência):**

| Faixa | Aprovação | FPR (bons pagadores negados) | Default real |
|---|---|---|---|
| 18–29 | 47,5% | **47,8%** | 11,4% |
| 30–44 | 60,2% | 35,3% | 8,9% |
| 45–59 | 70,7% | 25,6% | 6,9% |
| 60+ | 89,1% | 9,0% | 3,5% |

Impacto desigual **0,53** (regra dos 4/5: 0,8). Em M4 os jovens passam de 5,8% para 16,7% da carteira e o impacto desigual cai para 0,50, **sem a AUC cair**: o drift de população também é problema de governança.

**Mitigação (thresholds por faixa):**

| Conjunto | Impacto desigual | Aprovação | Default entre aprovados |
|---|---|---|---|
| Referência (metade de avaliação) | 0,53 → **0,96** | 72,2% → 72,2% | 1,69% → 1,93% |
| M4 | 0,50 → **0,82** | 59,8% → 64,7% | 1,67% → 2,19% |

O FPR dos jovens cai de 48% para 22%; o dos acima de 60 sobe de 9% para 24%. A mitigação redistribui o erro.

**Análise causal (modelo v1):**
- **M4:** restaurar os três grupos recupera 97% do desvio do score (PSI 0,131 → 0,004; aprovação 59,8% → 72,4%). Isoladamente: renda+dívida 55%, idade 49%, utilização 35%. A soma passa de 100% porque os efeitos se sobrepõem. **Covariate shift.**
- **M6:** restaurar X não recupera nada da AUC (0,798 → 0,797). **Concept drift.**

## 5. Passo a passo

1. **Auditoria de privacidade (início da semana 4):** confira no código e nos artefatos se as decisões pré-definidas foram cumpridas e registre o resultado no `lgpd.md`:
   - pseudonimização com sal fora do git (Fabi);
   - `customer_id` mascarado nos relatórios de contrato e Data Docs sem linhas (Fabi);
   - logs, spans e notificações sem identificador; teste automatizado passando (Pedro);
   - retenção de 90 dias na config do Loki, Tempo e Prometheus (Pedro);
   - telemetria das bibliotecas desligada;
   - fairness como guardrail do retreino (Helio).
2. **Fairness:** use o `fairness_by_age` do Helio (`src/governance/fairness.py`) para medir o viés do modelo em produção e o efeito do drift de M4 na carteira.
3. **`mitigation.py`:**
   - meta = FPR global do threshold atual;
   - threshold por faixa = quantil (1 − meta) do score dos bons pagadores da faixa;
   - ajuste em metade do teste da Referência e avaliação na outra metade + M1, M4, M7;
   - saída em `reports/fairness/mitigation.json` e run no MLflow;
   - `adopted_in_production: false`.
4. **`causal.py`:**
   - `restore_group(lote, ref, colunas)`: mapeamento de quantis que **preserva a ordem** dentro do lote;
   - grupos: renda+dívida, utilização, idade;
   - todas as combinações;
   - efeitos: PSI do score, aprovação, score médio, AUC;
   - recuperação = (lote − intervenção) / (lote − Referência);
   - diagnóstico automático (covariate × concept);
   - use o modelo **que estava em produção durante o drift** (v1).
5. **`retention.py`:** descarte dos lotes da quarentena com mais de `retention.quarantine_days` (30), com `--dry-run`; alvo `make purge-quarantine`.
6. **`docs/lgpd.md` (versão completa):**
   1. papéis;
   2. inventário de dados (pessoal? sensível? uso), anonimização × pseudonimização;
   3. base legal;
   4. os **10 princípios do Art. 6º**, cada um ligado a uma decisão concreta;
   5. os **7 princípios de Privacy by Design** (Cavoukian);
   6. direitos do titular (Arts. 18 e 20);
   7. retenção e descarte, com prazos legais de referência (CDC Art. 43 §1º: 5 anos para informação negativa; Cadastro Positivo: 15 anos) e sanções (Art. 52: até 2% do faturamento, limitado a R$ 50 milhões por infração);
   8. viés e fairness, com a mitigação;
   9. resumo de RIPD (risco × probabilidade × impacto × medida);
   10. análise causal (DAG em mermaid + tabelas de intervenção + diagnóstico + **limite do método**).
7. **README:** seções "Governança e LGPD" e "Análise causal" **com conteúdo próprio**: papéis, PII, base legal, retenção em tabela, viés com números, DAG, intervenções. Link para a versão completa.
8. **Model card** (escrito pela Fabi): revise a seção de fairness e a explicabilidade (importância das features, que também atende o Art. 20).
9. **Post-mortem sem culpados** do incidente simulado (M3 a M7), a partir da linha do tempo do Helio: detecção, causa (use a análise causal), mitigação, lições e ações.
10. **Consolidar o `docs/monitoring-plan.md`:**
    - catálogo de métricas (Pedro);
    - SLIs/SLOs: sucesso ≥ 95%, rejeição < 5%, queda de AUC ≤ 0,05, freshness < 35 dias;
    - os runbooks recebidos (Fabi, Helio, Pedro), conferindo que existe **um por alerta** e que o título bate com o `runbook_url` das regras.
11. **Números congelados:** use só a execução de referência do Pedro (tag `execucao-referencia`). Rode `make causal` e `make fairness` sobre ela e busque em todos os documentos os valores antigos. Só peça uma nova execução se algum código mudar depois da tag.
12. **Testes:**
    - intervenção preserva ordem e correlação;
    - thresholds por faixa igualam a FPR e são determinísticos;
    - descarte só dos vencidos;
    - **nenhum `customer_id` em logs, contratos e métricas** (no teste de ponta a ponta do Pedro).

## 6. Contratos entre as etapas

**Você recebe:**
- **Pacote 1 (Fabi):** `reference.parquet`, `build_model_input`, `load_model`, as importâncias das features (MLflow) e a pseudonimização implementada;
- **Pacote 2 (Helio):** os lotes em `data/production/`, o simulador (`apply_data_drift`, `apply_concept_drift`), os resultados do gate, a decisão do retreino (`reports/retrain/<lote>.json`) e o `fairness_by_age`;
- **Pacote 3 (Pedro):** a stack, a **execução de referência** (tag `execucao-referencia`) com as capturas, o teste automatizado de ausência de PII e, na semana 4, o catálogo de métricas e os runbooks de pipeline/infraestrutura;
- **documentação de quem construiu:** model card e runbooks de dados (Fabi); política de retreino, runbooks de drift/modelo e linha do tempo do incidente (Helio).

**Você entrega (fechamento do projeto):**

| Artefato | Para quem |
|---|---|
| Auditoria de privacidade registrada no `lgpd.md` | avaliador; correções pontuais pedidas aos donos das etapas |
| `mitigation.py`, `causal.py`, `retention.py` + alvos `make fairness`, `make causal`, `make purge-quarantine` | grupo |
| `docs/lgpd.md`, `docs/postmortem.md`, `docs/monitoring-plan.md` (consolidado), revisão do model card | avaliador |
| README consolidado (governança, causalidade, resultados, evidências) e números alinhados à execução de referência | avaliador |

## 7. Armadilhas que já encontramos

- **Pseudonimização ≠ anonimização.** Dado pseudonimizado continua sendo dado pessoal (Art. 13, §4º). Não escreva "dados anonimizados".
- **Consentimento como base legal** é um erro comum em crédito: a base é a proteção do crédito.
- **Data Docs do GE e exemplos do Pandera** vazam valores de linha por padrão. Configure `partial_unexpected_count: 0` e mascare os ids.
- **Bibliotecas com telemetria ligada por padrão** (GE, NannyML, MLflow, Evidently): o opt-out precisa vir **antes** do import. Centralize num único lugar.
- **`ThresholdOptimizer` do fairlearn** sorteia entre dois thresholds dentro da mesma faixa (em 18–29: 60% das vezes 0,31, 40% das vezes 0,07). Use thresholds determinísticos.
- **Mitigação com idade explícita** é tratamento diferenciado: documente o trade-off e deixe a adoção para a avaliação jurídica. Não ligue a mitigação no pipeline por conta própria.
- **Paridade demográfica** ignora o risco real (aprova mais inadimplentes), e **equalized odds** derrubou a aprovação para 54%. Justifique a escolha pela paridade de FPR.
- **Análise causal:** restaurar uma feature isolada quebra a correlação renda × dívida. Restaure **grupos** e mantenha a ordem dentro do lote. Deixe claro que o método mede a **sensibilidade do modelo** sob as hipóteses do DAG e **não prova a causa** no mundo real.
- **Números na documentação:** o XGBoost multi-thread varia na terceira casa entre execuções. Use só os números da **execução de referência** (congelada pelo Pedro) e alinhe todos os documentos a ela, com uma busca pelos valores antigos.
- **Regra de idade:** o enunciado cita "idade > 18"; o contrato usa ≥ 18 (maioridade). Documente para não parecer descuido.

## 8. Checklist de pronto

- [ ] README com Governança e Causalidade **no próprio README** (não só links).
- [ ] `docs/lgpd.md` com os 10 itens do passo 6.
- [ ] Auditoria de privacidade registrada: pseudonimização (Fabi), nenhum id em claro em logs, contratos e métricas (teste automatizado do Pedro), telemetria desligada.
- [ ] Retenção aplicada: quarentena com descarte; Loki, Tempo e Prometheus com 90 dias.
- [ ] Fairness por idade medida (DI 0,53) e usada como guardrail no retreino.
- [ ] Mitigação avaliada (DI 0,53 → 0,96) com o custo explícito e a condição jurídica para adoção.
- [ ] Análise causal reproduzindo o diagnóstico de M4 (covariate) e M6 (concept).
- [ ] Post-mortem e ADR da mitigação; model card revisado.
- [ ] `docs/monitoring-plan.md` consolidado: catálogo, SLOs e um runbook por alerta (títulos batendo com o `runbook_url`).
- [ ] README consolidado e todos os números batendo com a execução de referência.

## 9. Referência: projeto anterior

Implementação completa para consultar quando travar (repositório privado; peça acesso): https://github.com/eeyamazaki/credit-scoring-sustentacao. Use para **comparar**, não para copiar: o objetivo é o grupo construir e entender cada parte.

Arquivos equivalentes no projeto anterior:

`src/__init__.py` (opt-out de telemetria), `src/governance/{fairness,mitigation,causal,retention}.py`, `src/data/prepare.py` (`pseudonymize`), `src/logger.py` (`redact_identifiers`), `README.md` (seções Governança e LGPD, Análise causal), `docs/lgpd.md`, `docs/model-card.md`, `docs/postmortem.md`, `docs/decisions/009-mitigacao-vies-idade.md`, `reports/causal/`, `reports/fairness/`, `tests/unit/test_{causal,retention,retrain}.py`, `tests/integration/test_pipeline_smoke.py` (teste de ausência de PII).

Aulas da disciplina de Privacidade: Aula 1 (Privacy by Design e by Default), Aula 2 (sociedade orientada a dados), Aula 3 (GDPR e LGPD: princípios, bases legais, papéis, caso Telekall), Aula 4 (privacidade em ciência de dados, anonimização × pseudonimização, dados sintéticos), Aula 5 (novas tecnologias e desafios de PbD). Da disciplina de Data Drift, a Aula 4 (restauração de features) fundamenta a análise causal.

## 10. Prompt inicial sugerido para a LLM

```
Você vai me ajudar na Etapa 4 (Governança: Privacy by Design e by Default) do Tech
Challenge descrito no documento acima. Siga as "Decisões já tomadas" sem alterá-las.
Enquanto não recebo os pacotes dos colegas, vamos adiantar o rascunho do
docs/lgpd.md (itens 1 a 7 e 9 do passo 6) e o causal.py com dados sintéticos
(Python 3.12, testes pytest). Na semana 4 faremos a auditoria de privacidade
(passo 1) contra o código real. Cite artigos da LGPD só
quando tiver certeza do número; se tiver dúvida, sinalize para eu conferir.
```
