import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Título da aplicação
st.title("Otimização de Portfólio")

# Entrada do usuário: valor total do investimento
valor_total = st.number_input("Digite o valor total do investimento", value=100000)

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
st.write("Verificando valores NaN após preenchimento:")
st.write(dados_historicos_completos.isna().sum())

# Calcular os retornos diários e o desvio padrão (volatilidade) anualizado para as 15 ações, criptos e dólar
retornos_diarios_completos = dados_historicos_completos.pct_change().dropna()
riscos_acoes_cripto_dolar = retornos_diarios_completos.std() * np.sqrt(252)  # Riscos anualizados (15 ativos)

# Definir riscos assumidos para os ativos de renda fixa e tesouro (totalizando 19 ativos)
riscos_fixa_tesouro = np.array([0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.05, 0.05, 0.06, 0.04, 0.05, 0.03, 0.04, 0.06, 0.04, 0.05, 0.03, 0.04, 0.03])

# Combinar os riscos de ações, criptomoedas e renda fixa/tesouro para totalizar 34 ativos
riscos_completos_final = np.concatenate((riscos_acoes_cripto_dolar.values, riscos_fixa_tesouro))

# Função para calcular o Sharpe Ratio com penalização e normalização
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)  # Retorno ponderado
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))  # Risco ponderado
    
   # Ajustar penalidade: limite superior e inferior para valores do Sharpe Ratio
    if sharpe_ratio > 3:
        sharpe_ratio = 3 + np.log(sharpe_ratio - 1)  # Penalizar Sharpe muito alto
    elif sharpe_ratio < 0.5:  # Penalizar Sharpe muito baixo
        sharpe_ratio *= 1.5  # Aumentar o valor baixo suavemente

return sharpe_ratio

# Função para garantir que não há alocações negativas ou acima de 20%
def ajustar_alocacao(portfolio):
    portfolio = np.clip(portfolio, 0, 0.2)  # Limitar entre 0 e 20%
    portfolio /= portfolio.sum()  # Normalizar para garantir que a soma seja 1
    return portfolio

# Função de mutação ajustada para evitar valores negativos e respeitar limite de 20%
def mutacao(portfolio, taxa_mutacao=0.01):
    if np.random.random() < taxa_mutacao:
        i = np.random.randint(0, len(portfolio))
        portfolio[i] += np.random.uniform(-0.1, 0.1)
        portfolio = ajustar_alocacao(portfolio)  # Garantir que os valores estejam entre 0 e 20% e normalizar
    return portfolio

# Função de cruzamento de ponto único ajustada
def cruzamento(pai1, pai2):
    ponto_corte = np.random.randint(1, len(pai1) - 1)
    filho1 = np.concatenate((pai1[:ponto_corte], pai2[ponto_corte:]))
    filho2 = np.concatenate((pai2[:ponto_corte:], pai1[ponto_corte:]))

    # Ajustar e normalizar os filhos
    filho1 = ajustar_alocacao(filho1)
    filho2 = ajustar_alocacao(filho2)

    return filho1, filho2

# Função para gerar o genoma inicial de portfólios com 34 ativos
genoma_inicial = np.array([
    0.00, 0.00, 0.20, 0.00, 0.05, 0.00, 0.03, 0.00, 0.00, 0.03,
    0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.06,
    0.10, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, 0.05,
    0.05, 0.03, 0.05, 0.00
])

# Verificando se a soma das alocações é 100%
assert np.isclose(genoma_inicial.sum(), 1.0), "As alocações devem somar 100% (ou 1.0 em fração)"

# Função para rodar o algoritmo genético com genoma inicial fixo
def algoritmo_genetico_com_genoma_inicial(retornos, riscos, genoma_inicial, taxa_livre_risco=0.1075, num_portfolios=100, geracoes=100):
    populacao = [genoma_inicial]
    for _ in range(num_portfolios - 1):
        populacao.append(np.random.dirichlet(np.ones(len(genoma_inicial))))
    
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)

    for geracao in range(geracoes):
        # Calcular o Sharpe Ratio (fitness) para cada portfólio
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])

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
            nova_populacao.append(mutacao(filho1))
            nova_populacao.append(mutacao(filho2))
           
        # Inserir o elitismo: garantir que o melhor portfólio da geração anterior permaneça
        nova_populacao[0] = melhor_portfolio

        populacao = nova_populacao

        # Exibir o melhor Sharpe Ratio da geração atual no Streamlit
        st.write(f"Geracao {geracao + 1}, Melhor Sharpe Ratio: {melhor_sharpe}")

    return melhor_portfolio

# Funções auxiliares: seleção por torneio
def selecao_torneio(populacao, fitness_scores, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(fitness_scores[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

# Exemplo de dados reais para retornos e riscos 
# Limitar retornos para garantir que não sejam excessivamente elevados
retornos_reais = np.clip(np.random.rand(34) * 0.2, 0, 0.4)  # Limitar retornos entre 0% e 40%
riscos_reais = riscos_completos_final  # Riscos combinados para os 34 ativos

# Rodar o algoritmo genético com o genoma inicial fixo
melhor_portfolio = algoritmo_genetico_com_genoma_inicial(retornos_reais, riscos_reais, genoma_inicial)

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

