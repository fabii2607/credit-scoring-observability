# Análise de sensibilidade e hipóteses causais

`make causal` executa `causal.py` nos meses regulares do manifesto e grava
`reports/causal/causal_analysis.json`. O resultado público está em
[evidencias/causal/causal_analysis.json](evidencias/causal/causal_analysis.json).

## O que é calculado

O modelo permanece congelado. Para cada grupo — renda/DTI, saldo/utilização do
rotativo e FICO —, cada coluna é mapeada para valores da referência segundo sua
posição relativa no lote. São avaliadas as sete combinações não vazias dos
três grupos, mantendo target e demais colunas.

A restauração é monotônica por coluna e mantém nulos. Pode haver empates;
não preserva necessariamente correlação linear, distribuição conjunta ou
consistência estrutural de relações econômicas. Agrupar colunas relacionadas
ajuda a organizar as hipóteses, mas não elimina essas limitações.

Medimos PSI do score (mesma implementação do monitoramento), média dos scores,
aprovação e AUC. A baseline de AUC é o controle reservado, enquanto a referência
de distribuição vem da amostra do treino. O comando verifica contrato e hashes
dos arquivos de origem. Não avalia o lote corrompido.

“Recuperação” é a fração do desvio lote/referência removida pela perturbação.
Pode ser negativa ou maior que um; não é probabilidade e fica indisponível
quando o módulo do desvio no denominador é menor que 1e-9. Recuperações isoladas não são
aditivas. Os efeitos são arredondados no relatório para apresentação.

## Relação com a comparação pareada

A comparação original/alterado mantém os mesmos registros e conhece a
intervenção que o simulador aplicou. Ela mede seu efeito nesta amostra.
A restauração por quantis é outra perturbação: não recupera necessariamente
os valores originais de cada registro.

Portanto, uma não substitui a outra. O controle disjunto também pode diferir
do mês antes da intervenção. As três evidências devem ser lidas em conjunto.

## Conclusão permitida

O DAG no README representa hipóteses de mecanismos, não relações identificadas.
Score alterado com recuperação após restauração é compatível com sensibilidade
a X. AUC degradada sem recuperação suficiente motiva investigar rótulos,
dependências conjuntas e concept drift; não prova P(Y|X) ou causa econômica.

O diagnóstico usa heurísticas de recuperação de 50% e limites de PSI/queda;
não calcula ICs das recuperações. O diagnóstico não recomenda treinamento automaticamente. Para ação operacional,
consulte o gate, que considera queda, IC e cobertura de rótulos.
Veja a [linha do tempo](incident-timeline.md) para os resultados finais.
