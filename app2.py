import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Título da aplicação
st.title("Otimização de Portfólio")

# Entrada do usuário: valor total do investimento
valor_total = st.number_input("Digite o valor total do investimento", value=100000)

# Adicionar controle para selecionar a taxa de mutação
taxa_mutacao = st.slider("Taxa de Mutação", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

# Adicionar controle para selecionar a taxa livre de risco (exemplo: taxa SELIC)
taxa_livre_risco = st.number_input("Taxa Livre de Risco (Ex: SELIC, POUPANÇA)", value=0.1075)

# Pergunta sobre o uso do elitismo (Sim ou Não)
usar_elitismo = st.selectbox("Deseja usar elitismo?", options=["Sim", "Não"])

# Convertendo a resposta para um valor booleano
usar_elitismo = True if usar_elitismo == "Sim" else False

# Adicionar controle para selecionar qual tipo de retorno usar
tipo_retorno = st.selectbox("Deseja usar retornos ajustados ou reais?", options=["Ajustados", "Reais"])

# Definir qual conjunto de retornos será utilizado com base na escolha do usuário
if tipo_retorno == "Ajustados":
    retornos_usados = retornos_ajustados
else:
    retornos_usados = retornos_reais

# Carregar os dados do CSV atualizado diretamente do GitHub
csv_url = 'https://raw.githubusercontent.com/beatrizcardc/TechChallenge2_Otimizacao/main/Pool_Investimentos.csv'
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")
    st.stop()

# Exibir o valor total de investimento escolhido
st.write(f"Você deseja investir: R$ {valor_total}")

# Extrair retornos do CSV para os 34 ativos
retornos_12m = df['Rentabilidade 12 meses'].values
retornos_24m = df['Rentabilidade 24 meses'].values
retornos_36m = df['Rentabilidade 36 meses'].values

# Lista de tickers das 15 ações, criptomoedas e dólar
tickers_acoes_cripto_dolar = ['VALE3.SA', 'PETR4.SA', 'JBSS3.SA', 'MGLU3.SA', 'RENT3.SA',
                              'B3SA3.SA', 'WEGE3.SA', 'EMBR3.SA', 'GOLL4.SA', 'ITUB4.SA',
                              'BTC-USD', 'ADA-USD', 'ETH-USD', 'LTC-USD', 'BRL=X']

# Baixar dados históricos de preços para as 15 ações e criptos
dados_historicos_completos = yf.download(tickers_acoes_cripto_dolar, start='2021-01-01', end='2024-01-01')['Adj Close']

# Preencher valores NaN nos dados históricos com a média da coluna correspondente
dados_historicos_completos.fillna(dados_historicos_completos.mean(), inplace=True)

# Verificar se ainda existem valores NaN
#st.write("Verificando valores NaN após preenchimento:")
#st.write(dados_historicos_completos.isna().sum())

# Calcular os retornos diários e o desvio padrão (volatilidade) anualizado para as 15 ações, criptos e dólar
retornos_diarios_completos = dados_historicos_completos.pct_change().dropna()
riscos_acoes_cripto_dolar = retornos_diarios_completos.std() * np.sqrt(252)  # Riscos anualizados (15 ativos)

# Ajustar riscos para criptomoedas e ativos mais arriscados
risco_cripto = riscos_acoes_cripto_dolar[10:14] * 1.5  # Ponderar mais para os criptoativos (Bitcoin, Cardano, Ethereum, Litecoin)

# Atualizar os riscos das criptomoedas com o novo valor ponderado
riscos_acoes_cripto_dolar[10:14] = risco_cripto

# Definir riscos assumidos para os ativos de renda fixa e tesouro (totalizando 19 ativos)
riscos_fixa_tesouro = np.array([0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.05, 0.05, 0.06, 0.04, 0.05, 0.03, 0.04, 0.06, 0.04, 0.05, 0.03, 0.04, 0.03])

# Combinar os riscos de ações, criptomoedas e renda fixa/tesouro para totalizar 34 ativos
riscos_completos_final = np.concatenate((riscos_acoes_cripto_dolar.values, riscos_fixa_tesouro))

# Função para calcular o Sharpe Ratio com penalização e normalização
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)  # Retorno ponderado
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))  # Risco ponderado

    # Evitar divisões por zero ou risco muito baixo
    if risco_portfolio < 0.01:
        risco_portfolio = 0.01

    # Calcular o Sharpe Ratio
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / risco_portfolio

    # Penalizar Sharpe Ratios muito baixos ou altos
    if sharpe_ratio < 1.0:  # Exemplo: penalizar Sharpe Ratios muito baixos
        sharpe_ratio = sharpe_ratio * 0.8  # Penalidade adicional
    elif sharpe_ratio > 3:  # Permitir mais exploração de ativos com Sharpe Ratio maior
        sharpe_ratio = sharpe_ratio * 0.2  # Recompensa para maiores Sharpe Ratios

    return sharpe_ratio

# Função para gerar a população inicial com o genoma inicial fixo
def gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, num_ativos):
    populacao = [genoma_inicial]  # Começar com o genoma inicial fixo
    for _ in range(num_portfolios - 1):  # Gerar o restante aleatoriamente
        populacao.append(np.random.dirichlet(np.ones(num_ativos)))
    return populacao

# Ajustar os limites de alocação para permitir uma maior concentração em ativos de alto retorno
def ajustar_alocacao(portfolio, limite_max=0.25):
    # Aumentamos o limite de alocação de 20% para 25%
    portfolio = np.clip(portfolio, 0, limite_max)  # Limitar entre 0 e 25%
    portfolio /= portfolio.sum()  # Normalizar para garantir que a soma seja 1
    return portfolio

# Função de mutação com taxa ajustável
def mutacao(portfolio, taxa_mutacao, limite_max=0.25):
    if np.random.random() < taxa_mutacao:
        i = np.random.randint(0, len(portfolio))
        portfolio[i] += np.random.uniform(-0.1, 0.1)
        portfolio = ajustar_alocacao(portfolio, limite_max)
    return portfolio

# Função de Crossover de ponto único ajustada com verificação de índices
def cruzamento(pai1, pai2):
    # Vamos variar a quantidade de pontos de corte entre 1 e 3 para mais diversidade
    num_pontos_corte = np.random.randint(1, 4)  # Gerar de 1 a 3 pontos de corte
    pontos_corte = sorted(np.random.choice(range(1, len(pai1)), num_pontos_corte, replace=False))
    filho1, filho2 = pai1.copy(), pai2.copy()

    # Verificar se o número de pontos de corte é válido
    if len(pontos_corte) % 2 != 0:
        pontos_corte.append(len(pai1))  # Garantir que temos pares de índices
      
    for i in range(0, len(pontos_corte) - 1, 2):
        # Realizar a troca entre os segmentos dos pais nos filhos
        filho1[pontos_corte[i]:pontos_corte[i+1]] = pai2[pontos_corte[i]:pontos_corte[i+1]]
        filho2[pontos_corte[i]:pontos_corte[i+1]] = pai1[pontos_corte[i]:pontos_corte[i+1]]

    # Ajustar e normalizar os filhos
    filho1 = ajustar_alocacao(filho1)  # Limitar a alocação por ativo e normalizar
    filho2 = ajustar_alocacao(filho2)  # Limitar a alocação por ativo e normalizar

    return filho1, filho2
    
# Função para gerar o genoma inicial de portfólios com 34 ativos
genoma_inicial = np.array([
    0.00, 0.00, 0.20, 0.00, 0.05, 0.00, 0.03, 0.00, 0.00, 0.03,
    0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.06,
    0.10, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, 0.05,
    0.05, 0.03, 0.05, 0.00
])



# Função para rodar o algoritmo genético com ajustes de penalidade e variabilidade
def algoritmo_genetico_com_genoma_inicial(retornos, riscos, genoma_inicial, taxa_livre_risco=0.1075, num_portfolios=100, geracoes=100):
    populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, len(retornos))
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)

    for geracao in range(geracoes):
        # Calcular o Sharpe Ratio (fitness) para cada portfólio
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])

        # Verificar se algum portfólio é inválido (alocação fora do intervalo permitido)
        for portfolio in populacao:
            assert np.isclose(portfolio.sum(), 1.0), "Portfólio inválido: soma das alocações não é 100%"

        # Identificar o melhor portfólio
        indice_melhor_portfolio = np.argmax(fitness_scores)
        if fitness_scores[indice_melhor_portfolio] > melhor_sharpe:
            melhor_sharpe = fitness_scores[indice_melhor_portfolio]
            melhor_portfolio = populacao[indice_melhor_portfolio]

        # Seleção, cruzamento e mutação
        populacao = selecao_torneio(populacao, fitness_scores)
        nova_populacao = []
        for i in range(0, len(populacao), 2):
            pai1, pai2 = populacao[i], populacao[i+1]
            filho1, filho2 = cruzamento(pai1, pai2)

            # Garantir que os filhos estejam dentro dos limites
            filho1 = ajustar_alocacao(filho1)  # Limitar a alocação por ativo e normalizar
            filho2 = ajustar_alocacao(filho2)  # Limitar a alocação por ativo e normalizar
            
            nova_populacao.append(mutacao(filho1, taxa_mutacao))  # Adicionar a taxa de mutação ajustável
            nova_populacao.append(mutacao(filho2, taxa_mutacao))  # Adicionar a taxa de mutação ajustável

        # Inserir o elitismo: garantir que o melhor portfólio da geração anterior permaneça
        if usar_elitismo:
            nova_populacao[0] = melhor_portfolio  # Garantir que o melhor portfólio da geração anterior permaneça

        # Atualizar a população
        populacao = nova_populacao

        # Exibir o melhor Sharpe Ratio da geração atual no Streamlit
        st.write(f"Geracao {geracao + 1}, Melhor Sharpe Ratio: {melhor_sharpe:.2f}")

    return melhor_portfolio

# Função auxiliar: seleção por torneio
def selecao_torneio(populacao, fitness_scores, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(fitness_scores[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

# Exemplo de dados reais para retornos e riscos 
# Limitar retornos para garantir que não sejam excessivamente elevados
retornos_reais = np.random.rand(34) * 0.4  # Retornos simulados entre 0% e 40%
# Aumentar retorno esperado das criptomoedas e ações
retornos_ajustados = retornos_reais.copy()
retornos_ajustados[10:14] *= 1.2  # Aumentar em 20% os retornos das criptos
retornos_ajustados[:10] *= 1.15   # Aumentar em 15% os retornos das ações
riscos_reais = riscos_completos_final  # Riscos combinados para os 34 ativos

# Rodar o algoritmo com os parâmetros selecionados
melhor_portfolio = algoritmo_genetico_com_genoma_inicial(
    retornos_usados,  # Usar a variável de retornos baseada na escolha do usuário
    riscos_completos_final,  # Usar a variável de riscos correta
    genoma_inicial,  # Genoma inicial
    taxa_livre_risco,  # Taxa livre de risco
    num_portfolios=100,  # Número de portfólios
    geracoes=100,  # Número de gerações
    usar_elitismo=usar_elitismo,  # Definido pelo usuário
    taxa_mutacao=taxa_mutacao  # Definido pelo usuário
)

# Distribuir o valor total de investimento entre os ativos com base na melhor alocação
total_investido = valor_total  # Usando o valor definido pelo usuário no Streamlit
distribuicao_investimento = melhor_portfolio * total_investido

# Criar um DataFrame para exibir a distribuição de investimento
ativos = df['Ativo'].values  # Lista dos ativos
distribuicao_df = pd.DataFrame({
    'Ativo': ativos,
    'Alocacao (%)': melhor_portfolio * 100,
    'Valor Investido (R$)': distribuicao_investimento
})

# Exibir a distribuição ideal do investimento no Streamlit
st.write("Distribuição ideal de investimento:")
st.dataframe(distribuicao_df.style.format({'Alocacao (%)': '{:.2f}', 'Valor Investido (R$)': '{:.2f}'}))

# Função para salvar o DataFrame em um novo CSV para download
csv = distribuicao_df.to_csv(index=False)

# Botão para download do CSV atualizado
st.download_button(label="Baixar CSV Atualizado", data=csv, file_name='Pool_Investimentos_Atualizacao2.csv', mime='text/csv')

# Calcular os retornos esperados com base nas alocações
retorno_12m = np.dot(melhor_portfolio, retornos_12m)
retorno_24m = np.dot(melhor_portfolio, retornos_24m)
retorno_36m = np.dot(melhor_portfolio, retornos_36m)

# Exibir os retornos esperados no Streamlit
st.write(f"Retorno esperado em 12 meses: {retorno_12m:.2f}%")
st.write(f"Retorno esperado em 24 meses: {retorno_24m:.2f}%")
st.write(f"Retorno esperado em 36 meses: {retorno_36m:.2f}%")



