import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
st.title('Otimize seu Portfólio')

# Carregar os dados do CSV atualizado (deve conter 34 ativos)
# Mudando o endereço  para o github usando o raw
df = pd.read_csv('https://raw.githubusercontent.com/beatrizcardc/TechChallenge2_Otimizacao/main/Pool_Investimentos.csv')


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
print("Verificando valores NaN após preenchimento:")
print(dados_historicos_completos.isna().sum())

# Calcular os retornos diários e o desvio padrão (volatilidade) anualizado para as 15 ações, criptos e dólar
retornos_diarios_completos = dados_historicos_completos.pct_change().dropna()
riscos_acoes_cripto_dolar = retornos_diarios_completos.std() * np.sqrt(252)  # Riscos anualizados (15 ativos)

# Definir riscos assumidos para os ativos de renda fixa e tesouro (totalizando 19 ativos)
riscos_fixa_tesouro = np.array([0.05, 0.06, 0.04, 0.03, 0.04, 0.05, 0.05, 0.05, 0.06, 0.04, 0.05, 0.03, 0.04, 0.06, 0.04, 0.05, 0.03, 0.04, 0.03])

# Combinar os riscos de ações, criptomoedas e renda fixa/tesouro para totalizar 34 ativos
riscos_completos_final = np.concatenate((riscos_acoes_cripto_dolar.values, riscos_fixa_tesouro))

# Função para calcular o Sharpe Ratio
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)  # Retorno ponderado
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))  # Risco ponderado
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / risco_portfolio
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
    0.00,  # Tesouro Prefixado (sem alocação)
    0.00,  # Tesouro RendA (sem alocação)
    0.20,  # Tesouro Selic (20% do portfólio)
    0.00,  # Tesouro IPCA (sem alocação)
    0.05,  # Bitcoin (5% do portfólio)
    0.00,  # Cardano (sem alocação)
    0.03,  # Ethereum (5% do portfólio)
    0.00,  # Litecoin (sem alocação)
    0.00,  # Dólar (sem alocação)
    0.03,  # VALE3.SA (5% do portfólio)
    0.05,  # PETR4.SA (5% do portfólio)
    0.00,  # JBSS3.SA (sem alocação)
    0.00,  # MGLU3.SA (sem alocação)
    0.00,  # RENT3.SA (sem alocação)
    0.00,  # B3SA3.SA (sem alocação)
    0.00,  # WEGE3.SA (sem alocação)
    0.00,  # EMBR3.SA (sem alocação)
    0.05,  # GOLL4.SA (5% do portfólio)
    0.05,  # ITUB4.SA (5% do portfólio)
    0.06,  # Renda Fixa BB 1 (10% do portfólio)
    0.10,  # Renda Fixa BB 2 (10% do portfólio)
    0.00,  # Renda Fixa BB 3 (sem alocação)
    0.00,  # Renda Fixa BB 4 (sem alocação)
    0.00,  # Renda Fixa BB 5 (sem alocação)
    0.05,  # Renda Fixa Bradesco 1 (5% do portfólio)
    0.05,  # Renda Fixa Bradesco 2 (5% do portfólio)
    0.05,  # Renda Fixa Bradesco 3 (5% do portfólio)
    0.05,  # Renda Fixa Bradesco 4 (5% do portfólio)
    0.00,  # Renda Fixa Bradesco 5 (sem alocação)
    0.05,  # Renda Fixa Itaú 1 (5% do portfólio)
    0.05,  # Renda Fixa Itaú 2 (5% do portfólio)
    0.03,  # Renda Fixa Itaú 3 (5% do portfólio)
    0.05,  # Renda Fixa Itaú 4 (5% do portfólio)
    0.00   # Renda Fixa Itaú 5 (sem alocação)
])

# Verificando se a soma das alocações é 100%
assert np.isclose(genoma_inicial.sum(), 1.0), "As alocações devem somar 100% (ou 1.0 em fração)"

# Função para gerar a população inicial com o genoma inicial fixo
def gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, num_ativos):
    populacao = [genoma_inicial]  # Começar com o genoma inicial fixo
    for _ in range(num_portfolios - 1):  # Gerar o restante aleatoriamente
        populacao.append(np.random.dirichlet(np.ones(num_ativos)))
    return populacao

# Função para rodar o algoritmo genético com genoma inicial fixo
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

        # Seleção e cruzamento (crossover) e mutação
        populacao = selecao_torneio(populacao, fitness_scores)
        nova_populacao = []
        for i in range(0, len(populacao), 2):
            pai1, pai2 = populacao[i], populacao[i+1]
            filho1, filho2 = cruzamento(pai1, pai2)

            # Garantir que os filhos estejam dentro dos limites
            filho1 = ajustar_alocacao(filho1)  # Limitar a alocação por ativo e normalizar
            filho2 = ajustar_alocacao(filho2)  # Limitar a alocação por ativo e normalizar

            nova_populacao.append(mutacao(filho1))
            nova_populacao.append(mutacao(filho2))

        # Inserir o elitismo: garantir que o melhor portfólio da geração anterior permaneça
        nova_populacao[0] = melhor_portfolio

        populacao = nova_populacao

        # Exibir o melhor Sharpe Ratio da geração atual
        print(f"Geracao {geracao + 1}, Melhor Sharpe Ratio: {melhor_sharpe}")

    return melhor_portfolio

# Funções auxiliares: seleção por torneio, cruzamento e mutação
def selecao_torneio(populacao, fitness_scores, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(fitness_scores[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

# Exemplo de dados reais para retornos e riscos
retornos_reais = np.random.rand(34) * 0.2  # Retornos simulados entre 0 e 20%
riscos_reais = np.random.rand(34) * 0.1    # Riscos simulados entre 0 e 10%

# Rodar o algoritmo genético com o genoma inicial fixo
melhor_portfolio = algoritmo_genetico_com_genoma_inicial(retornos_reais, riscos_reais, genoma_inicial)

# Distribuir 100 mil reais entre os ativos com base na melhor alocação
total_investido = 100000  # 100 mil reais
distribuicao_investimento = melhor_portfolio * total_investido

# Criar um DataFrame para exibir a distribuição
ativos = df['Ativo'].values  # Lista dos ativos
distribuicao_df = pd.DataFrame({
    'Ativo': ativos,
    'Alocacao (%)': melhor_portfolio * 100,
    'Valor Investido (R$)': distribuicao_investimento
})

# Função para salvar o DataFrame em um novo CSV
#distribuicao_df.to_csv('https://raw.githubusercontent.com/beatrizcardc/TechChallenge2_Otimizacao/main/Pool_Investimentos_Atualizacao2.csv', index=False)
# Precisamos da API do Github para salvar diretamente nesse diretório
import requests
import base64
# Configurações do GitHub
repo = "beatrizcardc/TechChallenge2_Otimizacao"
path = "https://github.com/beatrizcardc/TechChallenge2_Otimizacao/blob/main/Pool_Investimentos_Atualizado%20(2).csv"
token = "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"

# Codificar o arquivo CSV
with open("Pool_Investimentos_Atualizacao2.csv", "rb") as file:
    content = file.read()

content_encoded = base64.b64encode(content).decode("utf-8")

# Configurar a URL da API
url = f"https://api.github.com/repos/{repo}/contents/{path}"

# Configurar o payload para a solicitação
data = {
    "message": "Atualizando Pool de Investimentos",
    "content": content_encoded
}

# Cabeçalhos de autorização
headers = {
    "Authorization": f"token {token}",
    "Content-Type": "application/json"
}

# Fazer a solicitação PUT para enviar o arquivo
response = requests.put(url, json=data, headers=headers)

# Verificar o status da resposta
if response.status_code == 201:
    st.write("Arquivo enviado com sucesso para o GitHub!")
else:
    st.write(f"Erro ao enviar o arquivo para o GitHub: {response.json()}")



# Exibir a distribuição ideal do investimento
print(distribuicao_df)

# Calcular os retornos com base nas alocações
retorno_12m = np.dot(melhor_portfolio, retornos_12m)
retorno_24m = np.dot(melhor_portfolio, retornos_24m)
retorno_36m = np.dot(melhor_portfolio, retornos_36m)

print(f"Melhor portfólio: {melhor_portfolio}")
print(f"Retorno em 12 meses: {retorno_12m}")
print(f"Retorno em 24 meses: {retorno_24m}")
print(f"Retorno em 36 meses: {retorno_36m}")

