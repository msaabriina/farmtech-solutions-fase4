"""
FarmTech Solutions - Fase 4
Dashboard interativo (Streamlit) - PARTE 1 + IR ALEM 2.

Conecta os modelos de regressao treinados (Scikit-Learn) a uma interface
interativa para o gestor agricola: metricas de desempenho, graficos de
correlacao, previsoes em tempo real e recomendacoes de manejo.

Executar:
    streamlit run dashboard/app.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ----------------------------------------------------------------------------
# Caminhos e import da camada de recomendacao
# ----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(RAIZ, "ml"))

from recomendacoes import gerar_recomendacoes  # noqa: E402

CSV = os.path.join(RAIZ, "data", "dataset_agricola.csv")
PASTA_MODELOS = os.path.join(RAIZ, "ml", "modelos")
FEATURES = ["umidade_solo", "temperatura_ar", "ph_solo",
            "nitrogenio_n", "fosforo_p", "potassio_k"]

ROTULOS = {
    "umidade_solo": "Umidade do solo (%)",
    "temperatura_ar": "Temperatura do ar (°C)",
    "ph_solo": "pH do solo",
    "nitrogenio_n": "Nitrogênio (N)",
    "fosforo_p": "Fósforo (P)",
    "potassio_k": "Potássio (K)",
    "produtividade_t_ha": "Produtividade (t/ha)",
    "volume_irrigacao_mm": "Volume de irrigação (mm)",
}

st.set_page_config(page_title="FarmTech Solutions - Fase 4",
                   page_icon="🌽", layout="wide")


# ----------------------------------------------------------------------------
# Carregamento (com cache)
# ----------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    return pd.read_csv(CSV)


@st.cache_resource
def carregar_modelos():
    modelos = {}
    for alvo in ["produtividade_t_ha", "volume_irrigacao_mm"]:
        caminho = os.path.join(PASTA_MODELOS, f"modelo_{alvo}.pkl")
        if os.path.exists(caminho):
            modelos[alvo] = joblib.load(caminho)
    return modelos


@st.cache_data
def carregar_metricas():
    caminho = os.path.join(PASTA_MODELOS, "metricas.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    return None


def prever(modelos, entradas):
    """Recebe um dict de features e devolve as previsoes dos dois alvos."""
    X = pd.DataFrame([entradas])[FEATURES]
    previsoes = {}
    for alvo, pacote in modelos.items():
        previsoes[alvo] = float(pacote["modelo"].predict(X)[0])
    return previsoes


# ----------------------------------------------------------------------------
# Verificacao de pre-requisitos
# ----------------------------------------------------------------------------
if not os.path.exists(CSV):
    st.error("Dataset nao encontrado. Rode primeiro: `python data/gerar_dataset.py`")
    st.stop()

df = carregar_dados()
modelos = carregar_modelos()
metricas = carregar_metricas()

if not modelos:
    st.error("Modelos nao encontrados. Rode primeiro: `python ml/treinar_modelos.py`")
    st.stop()


# ----------------------------------------------------------------------------
# Cabecalho
# ----------------------------------------------------------------------------
st.title("🌽 FarmTech Solutions — Assistente Agrícola Inteligente")
st.caption("Fase 4 · Machine Learning aplicado ao agronegócio · Cultura: Milho")

with st.sidebar:
    st.header("Sobre o projeto")
    st.markdown(
        "Dashboard que integra **modelos de regressão (Scikit-Learn)** aos "
        "dados dos sensores IoT (ESP32/Wokwi) para prever **produtividade** e "
        "**volume de irrigação** e sugerir ações de manejo."
    )
    st.metric("Leituras no dataset", len(df))
    st.metric("Leituras reais (Fase 3)", int((df["origem"] == "real").sum()))
    st.divider()
    st.caption("Use a aba **Previsão & Recomendações** para simular cenários.")


# ----------------------------------------------------------------------------
# Abas
# ----------------------------------------------------------------------------
aba_visao, aba_corr, aba_prev, aba_tend = st.tabs(
    ["📊 Visão Geral", "🔗 Correlações", "🤖 Previsão & Recomendações", "📈 Tendências"]
)

# ===== Aba 1: Visao Geral ====================================================
with aba_visao:
    st.subheader("Desempenho dos modelos de regressão")
    if metricas:
        for alvo, info in metricas["alvos"].items():
            st.markdown(f"**{info['rotulo']}** — melhor modelo: "
                        f"`{info['melhor_modelo']}`")
            linhas = []
            for nome, m in info["metricas_por_modelo"].items():
                linhas.append({
                    "Modelo": nome, "R²": m["R2"], "RMSE": m["RMSE"],
                    "MAE": m["MAE"], "MSE": m["MSE"],
                    "R² (val. cruzada)": m.get("R2_cv_medio", "-"),
                })
            tabela = pd.DataFrame(linhas)
            melhor = info["melhor_modelo"]
            st.dataframe(
                tabela.style.apply(
                    lambda r: ["background-color: #1b5e20; color: white"
                               if r["Modelo"] == melhor else "" for _ in r],
                    axis=1),
                hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        r2_prod = metricas["alvos"]["produtividade_t_ha"][
            "metricas_por_modelo"][
            metricas["alvos"]["produtividade_t_ha"]["melhor_modelo"]]["R2"]
        r2_vol = metricas["alvos"]["volume_irrigacao_mm"][
            "metricas_por_modelo"][
            metricas["alvos"]["volume_irrigacao_mm"]["melhor_modelo"]]["R2"]
        c1.metric("R² Produtividade", f"{r2_prod:.3f}")
        c2.metric("R² Volume irrigação", f"{r2_vol:.3f}")
        c3.metric("Produtividade média", f"{df['produtividade_t_ha'].mean():.1f} t/ha")

    st.divider()
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(15), hide_index=True, use_container_width=True)
    st.caption("Coluna `origem`: 'real' = leituras da Fase 3 (Wokwi/Oracle); "
               "'simulado' = leituras geradas para ampliar a base.")


# ===== Aba 2: Correlacoes ====================================================
with aba_corr:
    st.subheader("Matriz de correlação")
    cols = FEATURES + ["produtividade_t_ha", "volume_irrigacao_mm"]
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                    labels=dict(color="Correlação"))
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dispersão")
    eixo_x = st.selectbox("Variável (eixo X)", FEATURES, index=0,
                          format_func=lambda c: ROTULOS[c])
    alvo_y = st.selectbox("Alvo (eixo Y)",
                          ["produtividade_t_ha", "volume_irrigacao_mm"],
                          format_func=lambda c: ROTULOS[c])
    fig2 = px.scatter(df, x=eixo_x, y=alvo_y, color="origem",
                      opacity=0.6,
                      labels={eixo_x: ROTULOS[eixo_x], alvo_y: ROTULOS[alvo_y]},
                      color_discrete_map={"real": "#d84315", "simulado": "#2e7d32"})
    st.plotly_chart(fig2, use_container_width=True)


# ===== Aba 3: Previsao & Recomendacoes =======================================
with aba_prev:
    st.subheader("Simulador de cenário agrícola")
    st.caption("Ajuste as leituras dos sensores e veja a previsão e as ações sugeridas.")

    c1, c2, c3 = st.columns(3)
    umidade = c1.slider("Umidade do solo (%)", 0.0, 100.0, 35.0, 0.5)
    temperatura = c2.slider("Temperatura do ar (°C)", 10.0, 40.0, 28.0, 0.5)
    ph = c3.slider("pH do solo", 4.0, 8.5, 5.4, 0.1)
    n = c1.slider("Nitrogênio (N) [índice 0-100]", 0.0, 100.0, 25.0, 1.0)
    p = c2.slider("Fósforo (P) [índice 0-100]", 0.0, 100.0, 30.0, 1.0)
    k = c3.slider("Potássio (K) [índice 0-100]", 0.0, 100.0, 55.0, 1.0)

    entradas = {
        "umidade_solo": umidade, "temperatura_ar": temperatura, "ph_solo": ph,
        "nitrogenio_n": n, "fosforo_p": p, "potassio_k": k,
    }
    previsoes = prever(modelos, entradas)
    prod = previsoes.get("produtividade_t_ha", 0.0)
    volume = previsoes.get("volume_irrigacao_mm", 0.0)

    m1, m2 = st.columns(2)
    m1.metric("Produtividade prevista", f"{prod:.1f} t/ha")
    m2.metric("Volume de irrigação previsto", f"{volume:.1f} mm")

    # Recomendacoes (camada de regras de manejo)
    resultado = gerar_recomendacoes(umidade, temperatura, ph, n, p, k, prod, volume)
    st.markdown("### Recomendações de manejo")
    cores = {"Alta": "🔴", "Media": "🟡", "Baixa": "🟢"}
    for r in resultado.recomendacoes:
        with st.container(border=True):
            st.markdown(f"{cores.get(r.urgencia, '⚪')} **{r.categoria}** "
                        f"— {r.acao}  \n*{r.justificativa}*")


# ===== Aba 4: Tendencias =====================================================
with aba_tend:
    st.subheader("Tendências de produtividade")

    df_plot = df.copy()
    df_plot["faixa_umidade"] = pd.cut(
        df_plot["umidade_solo"],
        bins=[0, 25, 40, 60, 85, 100],
        labels=["Crítico (<25)", "Seco (25-40)", "Moderado (40-60)",
                "Ideal (60-85)", "Encharcado (>85)"])
    media_faixa = (df_plot.groupby("faixa_umidade", observed=True)
                   ["produtividade_t_ha"].mean().reset_index())
    fig3 = px.bar(media_faixa, x="faixa_umidade", y="produtividade_t_ha",
                  labels={"faixa_umidade": "Faixa de umidade do solo",
                          "produtividade_t_ha": "Produtividade média (t/ha)"},
                  color="produtividade_t_ha", color_continuous_scale="Greens")
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Distribuição da produtividade**")
        fig4 = px.histogram(df, x="produtividade_t_ha", nbins=30,
                            color_discrete_sequence=["#2e7d32"])
        st.plotly_chart(fig4, use_container_width=True)
    with col_b:
        st.markdown("**pH do solo x Produtividade**")
        fig5 = px.scatter(df, x="ph_solo", y="produtividade_t_ha",
                          opacity=0.5, color="produtividade_t_ha",
                          color_continuous_scale="Viridis")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**Série temporal das leituras (produtividade)**")
    df_ts = df.copy()
    df_ts["datahora"] = pd.to_datetime(df_ts["datahora"])
    df_ts = df_ts.sort_values("datahora").head(300)
    fig6 = px.line(df_ts, x="datahora", y="produtividade_t_ha",
                   labels={"datahora": "Data/hora",
                           "produtividade_t_ha": "Produtividade (t/ha)"})
    st.plotly_chart(fig6, use_container_width=True)

st.divider()
st.caption("FarmTech Solutions · Fase 4 · FIAP — Da coleta à predição: "
           "sensores IoT + banco de dados + Machine Learning.")
