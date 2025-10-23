import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    layout="wide",
    page_title="SALVE | Sistema de Alerta e Vigilância Epidemiológica",
    page_icon="⚕️"  # Ícone de saúde
)

@st.cache_data
def simular_dados_clinicas():
    """
    Cria um dataset simulado de casos para MÚLTIPLAS DOENÇAS, distribuídos 
    por Clínicas da Família, com fatores de risco distintos.
    """
    clinicas_info = {
        # CENTROS MUNICIPAIS DE SAÚDE
        'CMS Madre Teresa de Calcutá (Bancários)': {'lat': -22.78659, 'lon': -43.18582, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Newton Alves Cardozo (Cacuia)': {'lat': -22.80908, 'lon': -43.18927, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Parque Royal (Portuguesa)': {'lat': -22.79464, 'lon': -43.20959, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Vila do João (Complexo da Maré)': {'lat': -22.87350, 'lon': -43.23976, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS João Cândido (Penha)': {'lat': -22.82191, 'lon': -43.27115, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS São Godoredo (Penha)': {'lat': -22.84146, 'lon': -43.27071, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS José Breves dos Santos (Cordovil)': {'lat': -22.81689, 'lon': -43.28955, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Américo Veloso (Ramos)': {'lat': -22.84181, 'lon': -43.24957, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Maria Cristina Roma Paugartten (Ramos)': {'lat': -22.85186, 'lon': -43.25192, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Iraci Lopes (Vigário Geral)': {'lat': -22.80730, 'lon': -43.30394, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Nagib Jorge Farah (Jardim América)': {'lat': -22.80780, 'lon': -43.32143, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS José Paranhos Fontenelle (Olaria)': {'lat': -22.84183, 'lon': -43.26965, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CMS Alemão (Complexo do Alemão)': {'lat': -22.86504, 'lon': -43.26845, 'fator_risco': np.random.uniform(1.0, 1.8)},
        
        # CLÍNICAS DA FAMÍLIA
        'CF Victor Valla (AP 3.1)': {'lat': -22.8860155, 'lon': -43.2511681, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Assis Valente (Galeão)': {'lat': -22.8107054, 'lon': -43.2296404, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Maria Sebastiana de Oliveira (Tauá)': {'lat': -22.7969615, 'lon': -43.1939483, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Wilma Costa (Cocotá)': {'lat': -22.803715, 'lon': -43.1813638, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Adib Jatene (Complexo da Maré)': {'lat': -22.8656456, 'lon': -43.2419067, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Diniz Batista dos Santos (Complexo da Maré)': {'lat': -22.8470668, 'lon': -43.2471472, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Jeremias Moraes da Silva (Complexo da Maré)': {'lat': -22.8545612, 'lon': -43.2421543, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Felippe Cardoso (Penha)': {'lat': -22.8427502, 'lon': -43.2815405, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Rodrigo Y. Aguilar Roig (Cordovil)': {'lat': -22.8604152, 'lon': -43.2705142, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Nilda Campos de Lima (Cordovil)': {'lat': -22.8284469, 'lon': -43.3044761, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Valter Felisbino de Souza (Ramos)': {'lat': -22.8546931, 'lon': -43.2671231, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Jorgina Tuta (Vigário Geral)': {'lat': -22.8518103, 'lon': -43.2543167, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Klebel de Oliveira Rocha (Vigário Geral)': {'lat': -22.850761, 'lon': -43.27306, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Heitor dos Prazeres (Brás de Pina)': {'lat': -22.824787, 'lon': -43.283083, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Augusto Boal (Bonsucesso)': {'lat': -22.8657052, 'lon': -43.244519, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Zilda Arns (Complexo do Alemão)': {'lat': -22.865377, 'lon': -43.269168, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Eidimir Thiago de Souza (Parada de Lucas)': {'lat': -22.8232363, 'lon': -43.3120509, 'fator_risco': np.random.uniform(1.0, 1.8)},
        'CF Aloysio Augusto Novis (Penha Circular)': {'lat': -22.8379708, 'lon': -43.2926308, 'fator_risco': np.random.uniform(1.0, 1.8)},
    }
    
    datas = pd.to_datetime(pd.date_range(start='2024-01-01', periods=52, freq='W'))
    doencas = ['Dengue', 'Chikungunya', 'Influenza']
    df_list = []

    for clinica, info in clinicas_info.items():
        for doenca in doencas:
            # Fatores base
            densidade_pop_base = np.random.uniform(5000, 15000) * info['fator_risco']
            
            base_casos_semanal = (densidade_pop_base * 0.0005) * info['fator_risco']
            
            casos_semana_anterior = 0
            
            for data in datas:
                temp_semanal = np.random.uniform(25, 30) + np.random.normal(0, 2)
                precip_semanal = max(0, np.random.uniform(80, 150) + np.random.normal(0, 40))
                
                # --- CORREÇÃO 2: Fatores agora modulam a base ---
                fator_climatico = 1.0
                fator_sazonal = 1.0
                # A autocorrelação agora é um valor pequeno somado à base, não um multiplicador
                fator_autocorrelacao = (casos_semana_anterior * 0.05) # 5% da semana anterior

                if doenca in ['Dengue', 'Chikungunya']:
                    fator_sazonal = 1 + 0.8 * np.sin(2 * np.pi * (data.dayofyear - 80) / 365.25) # Pico no verão
                    # Fatores climáticos modulam (aumentam/diminuem) a base
                    fator_climatico = (temp_semanal / 28.0) + (precip_semanal / 150.0) # ~1.0 + ~0.8 = ~1.8
                    if doenca == 'Chikungunya':
                        base_casos_semanal *= 0.6 # Menos casos que Dengue
                        
                elif doenca == 'Influenza':
                    fator_sazonal = 1 + 0.9 * np.sin(2 * np.pi * (data.dayofyear - 170) / 365.25) # Pico no inverno
                    # Frio aumenta casos (inverso da temperatura)
                    fator_climatico = (28.0 / temp_semanal) # ~1.0
                
                # --- CORREÇÃO 3: Nova fórmula de cálculo de casos (controlada) ---
                casos = (base_casos_semanal + fator_autocorrelacao) * (fator_sazonal * fator_climatico) + np.random.normal(0, 2)
                
                casos = max(0, int(casos))
                
                df_list.append({
                    'semana': data, 'clinica_da_familia': clinica, 'doenca': doenca,
                    'casos_registrados': casos, 'temperatura_media_c': temp_semanal,
                    'precipitacao_mm': precip_semanal, 'densidade_populacional': int(densidade_pop_base),
                    'casos_semana_anterior': casos_semana_anterior,
                    'lat': info['lat'], 'lon': info['lon']
                })
                casos_semana_anterior = casos
            
    return pd.DataFrame(df_list)

@st.cache_resource
def treinar_e_prever(df):
    """
    Treina um modelo de ML para CADA DOENÇA e retorna as previsões.
    """
    doencas = df['doenca'].unique()
    lista_previsoes = []
    importancias = {}

    for doenca in doencas:
        df_doenca = df[df['doenca'] == doenca].copy()
        
        # Features podem ser diferentes por doença
        if doenca in ['Dengue', 'Chikungunya']:
            features = ['temperatura_media_c', 'precipitacao_mm', 'densidade_populacional', 'casos_semana_anterior']
        else: # Influenza
            features = ['temperatura_media_c', 'densidade_populacional', 'casos_semana_anterior']
            
        X = df_doenca[features]
        y = df_doenca['casos_registrados']
        
        # Corrigido o nome da variável de 'modelos' para 'modelo'
        modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        modelo.fit(X, y)
        
        # Criar dados da "próxima semana" para prever
        df_previsao_doenca = df_doenca.groupby('clinica_da_familia').last().reset_index()
        df_previsao_doenca['temperatura_media_c'] += np.random.normal(0, 1, len(df_previsao_doenca))
        df_previsao_doenca['precipitacao_mm'] += np.random.normal(0, 5, len(df_previsao_doenca))
        df_previsao_doenca['casos_semana_anterior'] = df_previsao_doenca['casos_registrados']
        
        X_futuro = df_previsao_doenca[features]
        df_previsao_doenca['casos_previstos'] = modelo.predict(X_futuro).astype(int)
        
        lista_previsoes.append(df_previsao_doenca)
        
        # Salva a importância das features para esta doença
        importancias[doenca] = pd.DataFrame({
            'Fator de Risco': X.columns.str.replace("_", " ").str.title(),
            'Importância': modelo.feature_importances_
        }).sort_values('Importância', ascending=False)

    df_previsao_final = pd.concat(lista_previsoes, ignore_index=True)
    return df_previsao_final, importancias

# Carrega, prepara os dados e treina os modelos
df_historico = simular_dados_clinicas()
df_previsao, dict_importancias = treinar_e_prever(df_historico)

# --- CABEÇALHO ---
st.warning("AVISO: Todos os dados apresentados neste painel são simulados para fins de demonstração do protótipo de pesquisa.")
st.image("https://raw.githubusercontent.com/danilofroes/machine-learning-saude-publica-jcc/main/assets/logo.png", width=200) # Usando a logo do seu repositório
st.title("SALVE - Sistema de Alerta e Vigilância Epidemiológica")
st.caption("Um protótipo do projeto 'Predição de Surtos de Doenças Infecciosas com Machine Learning para a Saúde Pública do Rio de Janeiro'")

# Agrupar previsões por doença para encontrar a de maior risco
df_risco_doenca = df_previsao.groupby('doenca')['casos_previstos'].sum().sort_values(ascending=False)
doenca_maior_risco = df_risco_doenca.idxmax()
total_previsto_maior_risco = df_risco_doenca.max()

# Encontrar a clínica de maior risco geral
idx_clinica_maior_risco = df_previsao['casos_previstos'].idxmax()
clinica_maior_risco = df_previsao.loc[idx_clinica_maior_risco, 'clinica_da_familia']
doenca_clinica_maior_risco = df_previsao.loc[idx_clinica_maior_risco, 'doenca']

col1, col2, col3 = st.columns(3)
col1.metric(label="Doença em Maior Alerta", value=f"{doenca_maior_risco}",
            delta=f"{total_previsto_maior_risco} casos previstos", delta_color="inverse")
col2.metric(label="Clínica com Maior Risco", value=clinica_maior_risco,
            help=f"Previsão de {doenca_clinica_maior_risco}")
col3.metric(label="Total de Clínicas Monitoradas", value=df_previsao['clinica_da_familia'].nunique())

st.markdown("---")

tab_doenca, tab_clinica, tab_modelo = st.tabs([
    "Visão Geral por Doença", 
    "Análise por Clínica", 
    "Análise dos Modelos"
])

with tab_doenca:
    st.header("Visão Geral por Doença")
    
    # Filtro para selecionar a doença
    doenca_selecionada = st.selectbox(
        'Selecione uma doença para análise detalhada:',
        df_previsao['doenca'].unique(),
        index=list(df_previsao['doenca'].unique()).index(doenca_maior_risco) # Começa pela de maior risco
    )
    
    # Filtra os dataframes com base na doença selecionada
    df_previsao_filtrada = df_previsao[df_previsao['doenca'] == doenca_selecionada]
    
    col_mapa, col_ranking = st.columns([0.45, 0.55])
    
    with col_mapa:
        st.subheader(f"Mapa de Risco para {doenca_selecionada}")
        fig_mapa = px.scatter_mapbox(df_previsao_filtrada, 
                                     lat="lat", lon="lon", 
                                     size="casos_previstos", 
                                     color="casos_previstos",
                                     hover_name="clinica_da_familia",
                                     hover_data={"casos_previstos": ":,d", "lat": False, "lon": False},
                                     color_continuous_scale=px.colors.sequential.YlOrRd,
                                     size_max=40, zoom=11, # Zoom mais próximo na AP 3.1
                                     mapbox_style="carto-positron")
        fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_mapa, use_container_width=True)

    with col_ranking:
        st.subheader(f"Ranking de Clínicas para {doenca_selecionada}")
        df_ranking = df_previsao_filtrada.sort_values('casos_previstos', ascending=True)
        fig_ranking = px.bar(df_ranking, 
                             x='casos_previstos', 
                             y='clinica_da_familia', 
                             orientation='h', text='casos_previstos',
                             labels={'casos_previstos': f'Nº de Casos Previstos ({doenca_selecionada})', 'clinica_da_familia': ''},
                             color='casos_previstos',
                             color_continuous_scale=px.colors.sequential.Reds)
        fig_ranking.update_traces(textposition='outside')
        fig_ranking.update_layout(yaxis_title=None)
        st.plotly_chart(fig_ranking, use_container_width=True)

with tab_clinica:
    st.header("Análise Histórica por Clínica")
    
    clinica_selecionada_hist = st.selectbox(
        'Selecione uma clínica para ver o histórico de todas as doenças:',
        df_historico['clinica_da_familia'].unique()
    )
    
    df_filtrado_hist = df_historico[df_historico['clinica_da_familia'] == clinica_selecionada_hist]
    
    fig_historico = px.line(df_filtrado_hist, 
                            x='semana', 
                            y='casos_registrados',
                            color='doenca', # Mostra uma linha para cada doença
                            title=f'Série Histórica de Casos para {clinica_selecionada_hist}',
                            labels={'semana': 'Semana', 'casos_registrados': 'Nº de Casos Registrados'},
                            markers=True)
    st.plotly_chart(fig_historico, use_container_width=True)

with tab_modelo:
    st.header("Análise dos Fatores de Risco por Modelo")
    
    doenca_modelo_selecionada = st.selectbox(
        'Selecione um modelo (doença) para analisar os fatores:',
        dict_importancias.keys()
    )
    
    df_importancia = dict_importancias[doenca_modelo_selecionada]
    
    st.subheader(f"Principais Fatores de Risco para {doenca_modelo_selecionada}")
    fig_importancia = px.bar(df_importancia,
                             x='Importância',
                             y='Fator de Risco',
                             orientation='h', text_auto='.2f',
                             color='Importância',
                             color_continuous_scale=px.colors.sequential.Blues_r)
    fig_importancia.update_layout(yaxis_title=None)
    st.plotly_chart(fig_importancia, use_container_width=True)

st.sidebar.image("https://raw.githubusercontent.com/danilofroes/machine-learning-saude-publica-jcc/main/assets/logo.png", width=100)
st.sidebar.header("Sobre o SALVE")
st.sidebar.info(
    "Este dashboard é uma demonstração do projeto de pesquisa **'Predição de Surtos de "
    "Doenças Infecciosas com Machine Learning para a Saúde Pública do Rio de Janeiro'**."
    " O objetivo é ilustrar como a IA pode fornecer ferramentas de suporte à decisão "
    "para a gestão da saúde pública. "
)
st.sidebar.success("O código fonte está disponível no [GitHub](https://github.com/danilofroes/machine-learning-saude-publica-jcc).")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/11/Gov.br_logo.svg", use_column_width=True)