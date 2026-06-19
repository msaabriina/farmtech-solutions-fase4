"""
FarmTech Solutions - Fase 4
Pipeline de Machine Learning (Scikit-Learn) - PARTE 2.

Etapas:
  1. Carrega o dataset agricola (data/dataset_agricola.csv).
  2. Faz o tratamento/selecao de features.
  3. Treina e compara dois modelos de regressao para cada alvo:
        - Regressao Linear (multipla)
        - Random Forest (nao linear)
     Alvos:
        - produtividade_t_ha   (rendimento esperado)
        - volume_irrigacao_mm  (lamina de irrigacao recomendada)
  4. Avalia com MAE, MSE, RMSE e R2 (conjunto de teste) + R2 de validacao cruzada.
  5. Salva o melhor modelo de cada alvo (.pkl), as metricas (metricas.json)
     e graficos de apoio (assets/*.png).

Uso:
    python ml/treinar_modelos.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica (gera arquivos PNG)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
PASTA_ML = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(PASTA_ML)
CSV = os.path.join(RAIZ, "data", "dataset_agricola.csv")
PASTA_MODELOS = os.path.join(PASTA_ML, "modelos")
PASTA_ASSETS = os.path.join(RAIZ, "assets")
os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_ASSETS, exist_ok=True)

SEED = 42
FEATURES = [
    "umidade_solo", "temperatura_ar", "ph_solo",
    "nitrogenio_n", "fosforo_p", "potassio_k",
]
ALVOS = {
    "produtividade_t_ha": "Produtividade (t/ha)",
    "volume_irrigacao_mm": "Volume de irrigação (mm)",
}


# ----------------------------------------------------------------------------
# Funcoes auxiliares
# ----------------------------------------------------------------------------
def carregar_dados():
    df = pd.read_csv(CSV)
    # Tratamento: remove eventuais nulos e garante tipos numericos
    df = df.dropna(subset=FEATURES + list(ALVOS.keys()))
    for col in FEATURES + list(ALVOS.keys()):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + list(ALVOS.keys()))
    return df


def avaliar(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "MSE": round(mse, 3),
            "RMSE": round(rmse, 3), "R2": round(r2, 4)}


def treinar_alvo(df, alvo):
    X = df[FEATURES]
    y = df[alvo]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    modelos = {
        "Regressão Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1
        ),
    }

    resultados = {}
    treinados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        metricas = avaliar(y_test, y_pred)
        # R2 medio por validacao cruzada (5 folds) no conjunto de treino
        cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring="r2")
        metricas["R2_cv_medio"] = round(float(cv.mean()), 4)
        resultados[nome] = metricas
        treinados[nome] = modelo

    # Escolhe o melhor modelo pelo maior R2 no conjunto de teste
    melhor_nome = max(resultados, key=lambda n: resultados[n]["R2"])
    melhor_modelo = treinados[melhor_nome]

    return {
        "X_test": X_test, "y_test": y_test,
        "resultados": resultados,
        "treinados": treinados,
        "melhor_nome": melhor_nome,
        "melhor_modelo": melhor_modelo,
    }


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------
def grafico_correlacao(df):
    cols = FEATURES + list(ALVOS.keys())
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    ax.set_title("Matriz de correlação", fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    caminho = os.path.join(PASTA_ASSETS, "matriz_correlacao.png")
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_dispersao(df):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(df["umidade_solo"], df["produtividade_t_ha"],
                   alpha=0.4, s=12, color="#2e7d32")
    axs[0].set_xlabel("Umidade do solo (%)")
    axs[0].set_ylabel("Produtividade (t/ha)")
    axs[0].set_title("Umidade x Produtividade")

    axs[1].scatter(df["umidade_solo"], df["volume_irrigacao_mm"],
                   alpha=0.4, s=12, color="#1565c0")
    axs[1].set_xlabel("Umidade do solo (%)")
    axs[1].set_ylabel("Volume de irrigação (mm)")
    axs[1].set_title("Umidade x Volume de irrigação")
    fig.tight_layout()
    caminho = os.path.join(PASTA_ASSETS, "dispersao_variaveis.png")
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_real_vs_previsto(info, alvo):
    """Compara os dois modelos (Regressao Linear x Random Forest) lado a lado,
    mostrando real vs. previsto e o R2 de cada um. A diferenca de dispersao em
    relacao a diagonal justifica visualmente a escolha do melhor modelo."""
    treinados = info["treinados"]
    resultados = info["resultados"]
    y_test = info["y_test"]
    X_test = info["X_test"]
    nomes = list(treinados.keys())
    cores = {"Regressão Linear": "#1565c0", "Random Forest": "#ef6c00"}

    fig, axs = plt.subplots(1, len(nomes), figsize=(11, 5.2),
                            sharex=True, sharey=True)
    if len(nomes) == 1:
        axs = [axs]
    for ax, nome in zip(axs, nomes):
        y_pred = treinados[nome].predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.4, s=14, color=cores.get(nome, "#6a1b9a"))
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="gray", label="Previsão perfeita")
        marca = "  (melhor)" if nome == info["melhor_nome"] else ""
        ax.set_title(f"{nome}{marca}\nR² = {resultados[nome]['R2']:.2f}")
        ax.set_xlabel(f"Real - {ALVOS[alvo]}")
        ax.legend(loc="upper left", fontsize=8)
    axs[0].set_ylabel(f"Previsto - {ALVOS[alvo]}")
    fig.suptitle(f"Real vs. Previsto - {ALVOS[alvo]}", fontsize=12, weight="bold")
    fig.tight_layout()
    caminho = os.path.join(PASTA_ASSETS, f"real_vs_previsto_{alvo}.png")
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_importancia(info, alvo):
    modelo = info["melhor_modelo"]
    if not hasattr(modelo, "feature_importances_"):
        return None
    importancias = modelo.feature_importances_
    ordem = np.argsort(importancias)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURES)[ordem], importancias[ordem], color="#6a1b9a")
    ax.set_title(f"Importância das variáveis - {ALVOS[alvo]}")
    ax.set_xlabel("Importância relativa")
    fig.tight_layout()
    caminho = os.path.join(PASTA_ASSETS, f"importancia_{alvo}.png")
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FarmTech Solutions - Fase 4 | Pipeline de Machine Learning")
    print("=" * 70)
    df = carregar_dados()
    print(f"Leituras carregadas: {len(df)}  "
          f"(reais: {(df['origem'] == 'real').sum()}, "
          f"simuladas: {(df['origem'] == 'simulado').sum()})")
    print(f"Features: {FEATURES}")
    print()

    metricas_geral = {"features": FEATURES, "alvos": {}}

    for alvo, rotulo in ALVOS.items():
        print("-" * 70)
        print(f"ALVO: {rotulo}")
        info = treinar_alvo(df, alvo)
        for nome, m in info["resultados"].items():
            marcador = "  <== melhor" if nome == info["melhor_nome"] else ""
            print(f"  {nome:18} | R2={m['R2']:.3f} | RMSE={m['RMSE']:.3f} | "
                  f"MAE={m['MAE']:.3f} | MSE={m['MSE']:.3f}{marcador}")

        # Salva o melhor modelo
        caminho_modelo = os.path.join(PASTA_MODELOS, f"modelo_{alvo}.pkl")
        joblib.dump(
            {"modelo": info["melhor_modelo"], "features": FEATURES,
             "nome": info["melhor_nome"], "alvo": alvo},
            caminho_modelo,
        )

        # Graficos especificos do alvo
        grafico_real_vs_previsto(info, alvo)
        grafico_importancia(info, alvo)

        metricas_geral["alvos"][alvo] = {
            "rotulo": rotulo,
            "melhor_modelo": info["melhor_nome"],
            "metricas_por_modelo": info["resultados"],
            "arquivo_modelo": os.path.relpath(caminho_modelo, RAIZ),
        }
        print(f"  Modelo salvo em: {os.path.relpath(caminho_modelo, RAIZ)}")
        print()

    # Graficos gerais
    grafico_correlacao(df)
    grafico_dispersao(df)

    # Salva metricas
    caminho_metricas = os.path.join(PASTA_MODELOS, "metricas.json")
    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas_geral, f, indent=2, ensure_ascii=False)
    print("=" * 70)
    print(f"Metricas salvas em: {os.path.relpath(caminho_metricas, RAIZ)}")
    print("Graficos salvos na pasta assets/.")
    print("Pipeline concluido com sucesso.")


if __name__ == "__main__":
    main()
