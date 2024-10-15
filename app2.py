import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Customização de CSS para fundo verde escuro e letras brancas
st.markdown(
    """
    <style>
    body {
        background-color: #D3D3D3; /* Fundo cinza claro */
        color: black;
    }
    .stApp {
        background-color: #D3D3D3;
        color: black;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black;
    }
    .stButton button {
        background-color: #004d00;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Título da aplicação
st.title("Otimização de Investimentos - Realize seus Objetivos")
# Caixas de texto explicativas
st.write("### Conceitos Importantes")
st.write("**Mutação**: A mutação é uma forma de introduzir variações em uma população de soluções.")
st.write("**Elitismo**: O elitismo preserva as melhores soluções encontradas em uma geração.")
st.write("**Sharpe Ratio**: Uma medida que compara o retorno de um investimento com seu risco.")

# Entrada do usuário: valor total do investimento
valor_total = st.number_input("Digite o valor total do investimento", value=100000)

# Adicionar controle para selecionar a taxa de mutação com explicação
taxa_mutacao = st.slider(
    "Taxa de Mutação",  min_value=0.01, max_value=0.2, value=0.05, step=0.01, 
    help="A taxa de mutação é um mecanismo essencial para garantir a exploração de novas soluções em algoritmos genéticos, ajudando a balancear exploração (testar soluções novas) e aproveitamento (melhorar soluções existentes)."
)

# Adicionar controle para selecionar a taxa livre de risco (exemplo: taxa SELIC)
taxa_livre_risco = st.number_input("Taxa Livre de Risco (Ex: SELIC, POUPANÇA)", value=0.1075,
                                  help="Insira uma taxa que melhor de ajuste aos seus objetivos. A taxa livre de risco padrão no Brasil é a SELIC")

# Pergunta sobre o uso do elitismo (Sim ou Não)
usar_elitismo = st.selectbox("Deseja usar elitismo?", options=["Sim", "Não"])

# Convertendo a resposta para um valor booleano
usar_elitismo = True if usar_elitismo == "Sim" else False

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

# Exemplo de dados reais para retornos e riscos 
retornos_reais = np.random.rand(34) * 0.4  # Retornos simulados entre 0% e 40%

# Aumentar retorno esperado das criptomoedas e ações
retornos_ajustados = retornos_reais.copy()
retornos_ajustados[10:14] *= 1.2  # Aumentar em 20% os retornos das criptos
retornos_ajustados[:10] *= 1.15   # Aumentar em 15% os retornos das ações

# Adicionar controle para selecionar qual tipo de retorno usar
tipo_retorno = st.selectbox("Deseja usar retornos ajustados ou reais?", options=["Ajustados", "Reais"])

# Definir qual conjunto de retornos será utilizado com base na escolha do usuário
if tipo_retorno == "Ajustados":
    retornos_usados = retornos_ajustados
else:
    retornos_usados = retornos_reais

# Função para calcular o Sharpe Ratio
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))
    if risco_portfolio < 0.01:
        risco_portfolio = 0.01
    return (retorno_portfolio - taxa_livre_risco) / risco_portfolio

# Algoritmo genético com critério de parada
def algoritmo_genetico(retornos, riscos, genoma_inicial, taxa_livre_risco, num_portfolios=100, geracoes=100, usar_elitismo=True, taxa_mutacao=0.05, crit_parada=20):
    populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, len(retornos))
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)
    contador_sem_melhoria = 0
    historico_sharpe = []

    for geracao in range(geracoes):
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])
        indice_melhor_portfolio = np.argmax(fitness_scores)
        if fitness_scores[indice_melhor_portfolio] > melhor_sharpe:
            melhor_sharpe = fitness_scores[indice_melhor_portfolio]
            melhor_portfolio = populacao[indice_melhor_portfolio]
            contador_sem_melhoria = 0
        else:
            contador_sem_melhoria += 1

        historico_sharpe.append(melhor_sharpe)

        populacao = selecao_torneio(populacao, fitness_scores)
        nova_populacao = []
        for i in range(0, len(populacao), 2):
            pai1, pai2 = populacao[i], populacao[i+1]
            filho1, filho2 = cruzamento(pai1, pai2)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        if usar_elitismo:
            nova_populacao[0] = melhor_portfolio

        populacao = nova_populacao

        if contador_sem_melhoria >= crit_parada:
            break

    return melhor_portfolio, historico_sharpe

# Funções auxiliares
def gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, num_ativos):
    populacao = [genoma_inicial]
    for _ in range(num_portfolios - 1):
        populacao.append(np.random.dirichlet(np.ones(num_ativos)))
    return populacao

def selecao_torneio(populacao, fitness_scores, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(fitness_scores[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

def cruzamento(pai1, pai2):
    num_pontos_corte = np.random.randint(1, 4)
    pontos_corte = sorted(np.random.choice(range(1, len(pai1)), num_pontos_corte, replace=False))
    filho1, filho2 = pai1.copy(), pai2.copy()
    for i in range(0, len(pontos_corte) - 1, 2):
        filho1[pontos_corte[i]:pontos_corte[i+1]] = pai2[pontos_corte[i]:pontos_corte[i+1]]
        filho2[pontos_corte[i]:pontos_corte[i+1]] = pai1[pontos_corte[i]:pontos_corte[i+1]]
    return ajustar_alocacao(filho1), ajustar_alocacao(filho2)

def ajustar_alocacao(portfolio, limite_max=0.25):
    portfolio = np.clip(portfolio, 0, limite_max)
    portfolio /= portfolio.sum()
    return portfolio

def mutacao(portfolio, taxa_mutacao, limite_max=0.25):
    if np.random.random() < taxa_mutacao:
        i = np.random.randint(0, len(portfolio))
        portfolio[i] += np.random.uniform(-0.1, 0.1)
        portfolio = ajustar_alocacao(portfolio, limite_max)
    return portfolio

# Gerar o portfólio otimizado
genoma_inicial = np.random.dirichlet(np.ones(34))
melhor_portfolio, historico_sharpe = algoritmo_genetico(
    retornos=retornos_usados, 
    riscos=riscos_completos_final, 
    genoma_inicial=genoma_inicial, 
    taxa_livre_risco=taxa_livre_risco,
    usar_elitismo=usar_elitismo,
    taxa_mutacao=taxa_mutacao,
    crit_parada=20
)

# Distribuir o valor total de investimento
distribuicao_investimento = melhor_portfolio * valor_total
distribuicao_df = pd.DataFrame({
    'Ativo': df['Ativo'],
    'Alocacao (%)': melhor_portfolio * 100,
    'Valor Investido (R$)': distribuicao_investimento
})

# Ordenar tabela pela alocação percentual
distribuicao_df = distribuicao_df.sort_values(by='Alocacao (%)', ascending=False)

# Mostrar a tabela e gráfico final
st.write("Distribuição ideal de investimento:")
st.dataframe(distribuicao_df)

# Exibir gráfico de barras da alocação
distribuicao_df.plot(kind='bar', x='Ativo', y='Alocacao (%)', legend=False)
plt.title('Distribuição Percentual por Ativo')
plt.ylabel('Alocacao (%)')
st.pyplot(plt)

# Exibir gráfico da evolução do Sharpe Ratio depois da tabela
fig, ax = plt.subplots(figsize=(6, 3))  # Gráfico menor
ax.plot(historico_sharpe, marker='o')
ax.set_title('Evolução do Sharpe Ratio')
ax.set_xlabel('Gerações')
ax.set_ylabel('Melhor Sharpe Ratio')
st.pyplot(fig)

# Download do CSV atualizado
csv = distribuicao_df.to_csv(index=False)
st.download_button(label="Baixar CSV Atualizado", data=csv, file_name='Pool_Investimentos_Atualizacao2.csv', mime='text/csv')





