# TechChallenge2_Otimização
## Usando a GenIA para a otimização de portfólio de investimentos com algoritmos genéticos
Grupo 18 da PósTech em IA para devs da FIAP


Fizemos o desafio em 3 partes buscando uma solução para nossa carteira de ativos.
1ª Parte) no google colab com o notebook TechChallenge_2_grupo18validado.ipynb.
  Nessa etapa desenvolvemos o primeiro código, subimos o primeiro video e validamos com o Profº Sérgio Polimante.

2ª Parte) As validações foram aplicadas no método de crossover de ponto único e constam do segundo video.
Trouxemos o código validado e as alterações aplicadas para o Github.

3ª Parte) Criamos nova parte do código com mais inputs para os usuários, especialmente o ajuste opcional das taxas de retorno.

LINK PARA APP NO STREAMLIT: https://techchallenge2otimizacaogit-grupo18.streamlit.app/
 ------------------------------------------------------

 Otimização de Portifólio de Investimentos com algoritmo genético
TechChallenge IA para Devs - PósTech FIAP >> Grupo 18

O primeiro trabalho foi criar um dataframe pertinente. Buscamos uma lista de ativos que representasse diferentes perfis de investidores, mas que fossem acessíveis ainda para amadores.

Enseguida, buscamos as tabelas de rentabilidades de bancos como Itaú, Banco do Brasil e Bradesco, as tabelas do tesouro direto SELIC, PréFixado, IPCA em pdf. Com a ajuda do ChatGPT consolidamos esses dados. Ademais, com a API da yfinance, buscamos os dados históricos para conseguirmos integrar o cálculo de rentabilidades de 12, 24 e 36 meses para o dólar, as principais criptos e 11 ações brasileiras iniciais.

Necesitamos calcular o ROI e a TAXA de RISCO ponderada para chegar no melhor SHARPE RATIO, métrica que elegemos para avaliar nosso portifólio.

Esses 34 ativos compuseram nosso GENOMA INICIAL com uma alocação de recursos hipotética.

O Sharpe Ratio mede o desempenho ajustado ao risco de um investimento ou portfólio. Ele indica quanto retorno adicional você está recebendo para cada unidade de risco que você está assumindo.

Interpretação de Valores do Sharpe Ratio:

🔴Sharpe Ratio baixo (< 1): Retorno baixo para o nível de risco. O investimento está assumindo muito risco para o retorno que gera.

🟡Sharpe Ratio de 1 a 2: Relacionamento razoável entre risco e retorno. O investimento está compensando o risco de forma adequada.

🟢Sharpe Ratio de 2 a 3: Excelente retorno ajustado ao risco.

🔵Sharpe Ratio > 3: Desempenho excepcional, onde o retorno para o risco assumido é muito alto.

PIPELINE:

Geração do dataframe

EAP - Análise dos Dados

Algorítimo Genético

População Inicial: Criamos uma população inicial de portfólios, genoma.
Função de Fitness: Avaliaremos cada portfólio usando a relação de Sharpe (retorno vs. risco).
Seleção: Selecionaremos os melhores portfólios com base na função de fitness.
Cruzamento e Mutação: Cruzamos os melhores portfólios para gerar novos e aplicamos mutações.
Critério de Parada: Rodaremos o algoritmo por um número fixo de gerações ou até que não haja mais melhoria significativa.
OBS: Aplicaremos soluções para avaliar e restringir que nenhuma população criada (portfólio) seja inferior ou superior a 100%
OBS: Aplicaremos o elitismo, para selecionar os melhores portfólios a cada geração.
OBS: Aplicaremos o Streamlit
