# TechChallenge2_Otimização
## Usando a GenIA para a otimização de portfólio de investimentos com algoritmos genéticos
Grupo 18 da PósTech em IA para devs da FIAP


Fizemos o desafio em 3 partes buscando uma solução para nossa carteira de ativos.
1ª Parte) no google colab com o notebook TechChallenge_2_grupo18validado.ipynb.
  Nessa etapa desenvolvemos o primeiro código, subimos o primeiro video e validamos com o Profº Sérgio Polimante.
  Link 1º video: https://youtu.be/nidQdAVXoVY 

2ª Parte) As validações foram aplicadas no método de crossover de ponto único e constam do segundo video.
Trouxemos o código validado e as alterações aplicadas para o Github.

3ª Parte) STREAMLIT: Criamos nova parte do código com mais inputs para os usuários, especialmente o ajuste opcional das taxas de retorno.
  
  Link 2º video: https://youtu.be/FWv9QrDMo6g 

LINK PARA APP NO STREAMLIT: https://techchallenge2otimizacaogit-grupo18.streamlit.app/
Veja o pdf 
 ------------------------------------------------------

### Otimização de Portifólio de Investimentos com algoritmo genético
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

✅População Inicial: Criamos uma população inicial de portfólios, genoma.

✅Função de Fitness: Avaliaremos cada portfólio usando a relação de Sharpe (retorno vs. risco).

✅Seleção: Selecionaremos os melhores portfólios com base na função de fitness.

✅Cruzamento e Mutação: Cruzamos os melhores portfólios para gerar novos e aplicamos mutações.

✅Critério de Parada: Rodaremos o algoritmo por um número fixo de gerações ou até que não haja mais melhoria significativa.

  *OBS: Aplicaremos soluções para avaliar e restringir que nenhuma população criada (portfólio) seja inferior ou superior a 100%
  
  *OBS: Aplicaremos o elitismo, para selecionar os melhores portfólios a cada geração
  
  *OBS: Usaremos o Streamlit

### Perguntas de referência
Qual é a representação da solução (genoma)?

Qual é a função de fitness?

Qual é o método de seleção?

Qual método de crossover você vai implementar?

Qual será o método de inicialização?

Qual o critério de parada?

Representação da solução (genoma): O genoma é representado por um array de números reais (frações) que somam 1.0 (ou 100%), indicando a alocação percentual de cada ativo no portfólio. Fizemos um array manual para os 34 ativos. No primeiro teste com todos os ativos alocados e no segundo com alguns somente.

Função de fitness: A função de fitness é o Sharpe Ratio, que mede o retorno ajustado ao risco do portfólio. Quanto maior o Sharpe Ratio, melhor o portfólio. O objetivo é maximizar o Sharpe Ratio, o que significa encontrar um portfólio com alto retorno esperado e baixo risco (desvio padrão). def calcular_sharpe

Método de seleção: O método de seleção utilizado no algoritmo é a seleção por torneio. Neste método, um número fixo de indivíduos (portfólios) é escolhido aleatoriamente da população, e o indivíduo com o melhor fitness (Sharpe Ratio) entre eles é selecionado para reprodução. Esse processo é repetido para gerar a nova população. def selecao_torneio

Método de crossover: O método de crossover (cruzamento) implementado é o crossover de ponto único. Um ponto de corte aleatório é escolhido, e as frações do genoma de dois portfólios são trocadas a partir desse ponto para gerar dois novos portfólios. def cruzamento

Método de inicialização: A inicialização do algoritmo é feita com uma população de portfólios, onde o primeiro portfólio é o genoma inicial fixo (um portfólio sugerido previamente), e os outros são gerados de forma aleatória usando a distribuição de Dirichlet, que garante que as frações somam 1.0 (100%). def gerar_portfolios_com_genoma_inicial

O critério de parada: é o número fixo de gerações. O algoritmo é configurado para rodar por um número determinado de gerações (por exemplo, 100 gerações), e o melhor portfólio ao final dessas gerações é considerado a solução. def algoritmo_genetico_com_genoma_inicial

### Aplicando as validações no Crossover de Um Ponto:

  Restringir as alocações para um máximo de 20% em cada ativo

  Normalizar e garantir portfólios de 100% após crossover e mutação

  Aplicar penalidade no fitness

  Aplicar o elitismo
