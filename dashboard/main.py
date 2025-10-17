import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(
    layout="wide",
    page_title="Monitoramento de Arboviroses no município do Rio de Janeiro",
    page_icon="🦟"
)

# Simulando dados para o protótipo
@st.cache_data
def simular_dados_clinicas():
    """
    Cria um dataset simulado de casos de arboviroses distribuídos por Clínicas da Família,
    baseado em fatores de risco realistas.
    """
    # Selecionando uma Clínica da Família representativa por Área de Planejamento (AP)
    clinicas_info = {
        'CF Victor Valla (AP 3.1)': {'lat': -22.8837, 'lon': -43.24, 'fator_risco': 1.5},
        'CF Dr. Felipe Cardoso (AP 3.1)': {'lat': -22.8465, 'lon': -43.2842, 'fator_risco': 1.4},
        'CF Rodrigo Yamawaki Aguilar Roig (AP 3.1)': {'lat': -22.8553, 'lon': -43.267, 'fator_risco': 1.7},
        'CF Zilda Arns (AP 3.1)': {'lat': -22.8546, 'lon': -43.2689, 'fator_risco': 1.7},
        'CF Augusto Boal (AP 3.1)': {'lat': -22.86, 'lon': -43.2562, 'fator_risco': 1.6},
        'CF Aloysio Augusto Novis (AP 3.1)': {'lat': -22.8407, 'lon': -43.2954, 'fator_risco': 1.3},
        'CF Heitor dos Prazeres (AP 3.1)': {'lat': -22.829, 'lon': -43.2941, 'fator_risco': 1.4},
        'CMS Esperança (AP 3.1)': {'lat': -22.8475, 'lon': -43.2725, 'fator_risco': 1.2},
        'CMS Alemão (AP 3.1)': {'lat': -22.858, 'lon': -43.27, 'fator_risco': 1.8},
        'CMS Vila do João (AP 3.1)': {'lat': -22.8654, 'lon': -43.2452, 'fator_risco': 1.8}
    }
    
    datas = pd.to_datetime(pd.date_range(start='2024-01-01', periods=52, freq='W')) # Dados semanais por um ano
    
    df_list = []
    
    for clinica, info in clinicas_info.items():
        # Fatores de risco simulados para cada clínica
        temperatura_base = np.random.uniform(25, 30)
        precipitacao_base = np.random.uniform(80, 150)
        densidade_pop_base = np.random.uniform(5000, 15000) * info['fator_risco']
        
        casos_semana_anterior = 0
        
        for data in datas:
            # Sazonalidade (mais casos no verão)
            sazonalidade = 1 + 0.8 * np.sin(2 * np.pi * (data.dayofyear - 80) / 365.25)
            
            temp_semanal = temperatura_base + np.random.normal(0, 2) * sazonalidade
            precip_semanal = max(0, precipitacao_base + np.random.normal(0, 40)) * sazonalidade
            
            # Fórmula de simulação de casos
            casos = int(
                (temp_semanal * 1.5) + \
                (precip_semanal * 0.5) + \
                (densidade_pop_base * 0.002) + \
                (casos_semana_anterior * 0.3) + \
                np.random.randint(0, 20)
            ) * info['fator_risco'] * sazonalidade
            casos = max(0, int(casos))
            
            df_list.append({
                'semana': data,
                'clinica_da_familia': clinica,
                'casos_registrados': casos,
                'temperatura_media_c': temp_semanal,
                'precipitacao_mm': precip_semanal,
                'densidade_populacional': int(densidade_pop_base),
                'casos_semana_anterior': casos_semana_anterior,
                'lat': info['lat'],
                'lon': info['lon']
            })
            casos_semana_anterior = casos
            
    return pd.DataFrame(df_list)

# Treinando o modelo e fazendo previsões
@st.cache_resource
def treinar_e_prever(df):
    """
    Treina um modelo RandomForest para prever os casos da próxima semana e retorna
    as previsões, o modelo e a importância das features.
    """
    X = df[['temperatura_media_c', 'precipitacao_mm', 'densidade_populacional', 'casos_semana_anterior']]
    y = df['casos_registrados']
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X, y) # Treinando com todos os dados históricos para prever o futuro
    
    # Criar dados da "próxima semana" para fazer a previsão
    df_previsao = df.groupby('clinica_da_familia').last().reset_index()
    # Simular pequenas variações climáticas para a próxima semana
    df_previsao['temperatura_media_c'] += np.random.normal(0, 1, len(df_previsao))
    df_previsao['precipitacao_mm'] += np.random.normal(0, 5, len(df_previsao))
    df_previsao['casos_semana_anterior'] = df_previsao['casos_registrados'] # Os casos desta semana são os "anteriores" da próxima
    
    X_futuro = df_previsao[['temperatura_media_c', 'precipitacao_mm', 'densidade_populacional', 'casos_semana_anterior']]
    df_previsao['casos_previstos'] = modelo.predict(X_futuro).astype(int)
    
    importancia_features = pd.DataFrame({
        'Fator de Risco': X.columns,
        'Importância': modelo.feature_importances_
    }).sort_values('Importância', ascending=False)
    
    return df_previsao[['clinica_da_familia', 'casos_previstos', 'lat', 'lon']], importancia_features

# Carrega, prepara os dados e treina o modelo
df_historico = simular_dados_clinicas()
df_previsao, df_importancia = treinar_e_prever(df_historico)

# Cabeçalho do dashboard
st.warning("AVISO: Todos os dados apresentados neste painel são simulados para fins de demonstração do protótipo de pesquisa.")
st.title("Monitoramento de doenças infecciosas no município do Rio de Janeiro")
st.caption("Um protótipo do projeto 'Predição de Surtos de Doenças Infecciosas com Machine Learning para a Saúde Pública do Rio de Janeiro'")

# Métricas principais
total_previsto = df_previsao['casos_previstos'].sum()
clinica_maior_risco = df_previsao.sort_values('casos_previstos', ascending=False).iloc[0]
fator_principal = df_importancia.iloc[0]['Fator de Risco'].replace("_", " ").title()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total de casos previstos (Próx. semana)", value=f"{total_previsto}")
with col2:
    st.metric(label="Clínica com maior risco", value=clinica_maior_risco['clinica_da_familia'])
with col3:
    st.metric(label="Principal fator de influência", value=fator_principal)

st.markdown("---")

# Principais métodos visuais para análise rápida
col_mapa, col_ranking = st.columns([0.45, 0.55])

with col_mapa:
        st.subheader("Mapa de risco")
        fig_mapa = px.scatter_mapbox(df_previsao, 
                                     lat="lat", lon="lon", 
                                     size="casos_previstos", 
                                     color="casos_previstos",
                                     hover_name="clinica_da_familia",
                                     hover_data={
                                         "casos_previstos": ":,d",
                                         "clinica_da_familia": False,
                                         "lat": False, 
                                         "lon": False
                                     },
                                     color_continuous_scale=px.colors.sequential.YlOrRd,
                                     size_max=40, zoom=9.5,
                                     mapbox_style="carto-positron")
        fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_mapa, use_container_width=True)

with col_ranking:
    st.subheader("📊 Ranking de clínicas por casos previstos")
    df_ranking = df_previsao.sort_values('casos_previstos', ascending=True)
    fig_ranking = px.bar(df_ranking, 
                         x='casos_previstos', 
                         y='clinica_da_familia', 
                         orientation='h',
                         text='casos_previstos',
                         labels={'casos_previstos': 'Nº de Casos Previstos para a Próxima Semana', 'clinica_da_familia': 'Clínica da Família'},
                         color='casos_previstos',
                         color_continuous_scale=px.colors.sequential.Reds)
    fig_ranking.update_traces(textposition='outside')
    st.plotly_chart(fig_ranking, use_container_width=True)

st.markdown("---")

# Análise detalhada do modelo de machine learning
st.subheader("🔍 Análise detalhada do modelo preditivo")
col_fatores, col_historico = st.columns(2)

with col_fatores:
    st.markdown("**Quais fatores mais impactam as previsões?**")
    fig_importancia = px.bar(df_importancia,
                             x='Importância',
                             y='Fator de Risco',
                             orientation='h',
                             text_auto='.2f',
                             color='Importância',
                             color_continuous_scale=px.colors.sequential.Blues_r)
    st.plotly_chart(fig_importancia, use_container_width=True)

with col_historico:
    st.markdown("**Análise histórica por clínica**")
    clinica_selecionada = st.selectbox(
        'Selecione uma clínica para ver o histórico de casos:',
        df_historico['clinica_da_familia'].unique()
    )
    df_filtrado = df_historico[df_historico['clinica_da_familia'] == clinica_selecionada]
    fig_historico = px.line(df_filtrado, 
                            x='semana', 
                            y='casos_registrados',
                            title=f'Série Histórica de Casos para {clinica_selecionada}',
                            labels={'semana': 'Semana', 'casos_registrados': 'Nº de Casos Registrados'})
    fig_historico.update_traces(line_color='#0072c6', line_width=3)
    st.plotly_chart(fig_historico, use_container_width=True)


# Barra lateral com informações do projeto e link do repositório
st.sidebar.header("Sobre o Projeto")
st.sidebar.info(
    "Este dashboard é uma demonstração do projeto de pesquisa **'Predição de Surtos de "
    "Doenças Infecciosas com Machine Learning para a Saúde Pública do Rio de Janeiro'**."
    " O objetivo é ilustrar como a IA pode fornecer ferramentas de suporte à decisão "
    "para a gestão da saúde pública. "
)
st.sidebar.success("O código fonte está disponível no [GitHub](https://github.com/danilofroes/machine-learning-saude-publica-jcc).")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/11/Gov.br_logo.svg", use_container_width=True)
