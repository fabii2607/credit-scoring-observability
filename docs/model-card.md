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
- Região não entra no modelo; auditoria regional não demonstra ausência de
  discriminação em atributos não disponíveis. CT, challenger e promoção estão
  fora da implementação.
- Sensibilidade por restauração de marginais não identifica causas econômicas.

## Artefatos e rastreabilidade

Modelo binário e metadados precisam ser usados como par. O binário é local,
ignorado pelo Git: reproduza com `make prepare && make train` ou obtenha ambos
por canal autorizado do grupo. Nunca carregue joblib de origem não confiável.

Hashes do modelo, metadados, dados, parâmetros e fontes da execução estão no
[manifesto das evidências](evidencias/manifest.json).
Veja [reprodução](delivery.md) e [governança](lgpd.md).
