"""
FarmTech Solutions - Fase 4
Gerador do dataset agricola usado para treinar os modelos de regressao.

Estrategia:
  1. Carrega as 55 leituras REAIS geradas no Wokwi/ESP32 na Fase 3
     (data/dados_fase3_reais.csv) e as converte para o schema da Fase 4.
  2. Gera leituras SIMULADAS de forma agronomicamente coerente para a
     cultura do milho, adicionando o alvo de produtividade (t/ha) e o
     volume de irrigacao recomendado (mm) que nao existiam na Fase 3.
  3. Salva tudo em data/dataset_agricola.csv com uma coluna 'origem'
     ('real' ou 'simulado') para manter a rastreabilidade.

O processo e 100% reprodutivel (seed fixa).
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------------
SEED = 42
N_SIMULADO = 1200            # quantidade de leituras simuladas
CULTURA = "Milho"
PASTA = os.path.dirname(os.path.abspath(__file__))
CSV_REAL = os.path.join(PASTA, "dados_fase3_reais.csv")
CSV_SAIDA = os.path.join(PASTA, "dataset_agricola.csv")

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Funcoes agronomicas (relacoes "verdadeiras" usadas para gerar os alvos)
# ----------------------------------------------------------------------------
def fator_umidade(umidade):
    """Produtividade do milho e maxima perto de 65% de umidade do solo."""
    return np.exp(-((umidade - 65.0) / 22.0) ** 2)


def fator_ph(ph):
    """Faixa ideal de pH para o milho fica em torno de 6.2."""
    return np.exp(-((ph - 6.2) / 1.0) ** 2)


def fator_temperatura(temp):
    """Temperatura otima do ar perto de 27 C."""
    return np.exp(-((temp - 27.0) / 8.0) ** 2)


def fator_nutrientes(n, p, k):
    """Indice ponderado de nutrientes (N tem maior peso para o milho)."""
    indice = (0.5 * n + 0.25 * p + 0.25 * k) / 100.0
    return 0.35 + 0.65 * indice


def calcular_produtividade(umidade, ph, temp, n, p, k, ruido=True):
    """Produtividade estimada em toneladas por hectare (t/ha).

    Modelo ADITIVO com um termo de interacao (umidade x nutrientes).
    A parte aditiva representa os efeitos principais (boa para a regressao
    linear) e o termo de interacao introduz nao-linearidade (capturada
    melhor por modelos como o Random Forest). Faixa tipica do milho.
    """
    f_umi = fator_umidade(umidade)
    f_ph = fator_ph(ph)
    f_temp = fator_temperatura(temp)
    indice_nutri = (0.5 * n + 0.25 * p + 0.25 * k) / 100.0  # 0..1

    base = (
        1.5
        + 6.5 * f_umi          # efeito da umidade (0..6.5)
        + 3.0 * f_ph           # efeito do pH (0..3)
        + 2.0 * f_temp         # efeito da temperatura (0..2)
        + 4.0 * indice_nutri   # efeito dos nutrientes (0..4)
        + 1.5 * f_umi * indice_nutri  # interacao nao-linear (0..1.5)
    )
    if ruido:
        base = base + rng.normal(0.0, 0.6, size=np.shape(base))
    return np.clip(base, 1.0, 16.0)


def calcular_volume_irrigacao(umidade, temp, ruido=True):
    """Volume de irrigacao recomendado em mm.

    Considera a meta de umidade do solo de 70% (capacidade de campo).
    Quanto mais seco e quente, maior a lamina de irrigacao recomendada.
    """
    deficit = np.clip(70.0 - umidade, 0.0, None)
    volume = deficit * 0.6 * (1.0 + (temp - 25.0) / 40.0)
    if ruido:
        volume = volume + rng.normal(0.0, 1.2, size=np.shape(volume))
    return np.clip(volume, 0.0, None)


def ldr_para_ph(ldr):
    """Converte o valor bruto do LDR (Fase 3) em um pH continuo plausivel."""
    return np.round(4.8 + (ldr / 4095.0) * 3.2, 2)


def ph_para_ldr(ph):
    """Operacao inversa, para manter a coluna valor_ldr nos dados simulados."""
    return np.clip(((ph - 4.8) / 3.2) * 4095.0, 0, 4095).astype(int)


# ----------------------------------------------------------------------------
# 1) Leituras REAIS da Fase 3 convertidas para o novo schema
# ----------------------------------------------------------------------------
def carregar_dados_reais():
    df = pd.read_csv(CSV_REAL, sep=";", encoding="utf-8-sig")
    n_linhas = len(df)

    umidade = df["Umidade do solo (%)"].astype(float).to_numpy()
    ldr = df["Valor do LDR"].astype(float).to_numpy()
    ph = ldr_para_ph(ldr)

    # A Fase 3 nao mede temperatura; simulamos em torno de 26 C.
    temp = np.round(rng.normal(26.0, 3.0, n_linhas), 1)

    # Na Fase 3 os nutrientes aparecem como "Bom"/"Baixo".
    def nivel(coluna, base_bom, base_baixo):
        vals = []
        for status in df[coluna].astype(str):
            if status.strip().lower() == "bom":
                vals.append(rng.normal(base_bom, 8))
            else:
                vals.append(rng.normal(base_baixo, 6))
        return np.clip(np.round(vals, 1), 0, 100)

    n = nivel("Nitrogenio (N)", 75, 18)
    p = nivel("Fosforo (P)", 75, 18)
    k = nivel("Potassio (K)", 75, 18)

    prod = np.round(calcular_produtividade(umidade, ph, temp, n, p, k), 2)
    volume = np.round(calcular_volume_irrigacao(umidade, temp), 1)

    out = pd.DataFrame({
        "cultura": CULTURA,
        "umidade_solo": np.round(umidade, 1),
        "temperatura_ar": temp,
        "ph_solo": ph,
        "nitrogenio_n": n,
        "fosforo_p": p,
        "potassio_k": k,
        "valor_ldr": ldr.astype(int),
        "produtividade_t_ha": prod,
        "volume_irrigacao_mm": volume,
        "origem": "real",
    })
    return out


# ----------------------------------------------------------------------------
# 2) Leituras SIMULADAS
# ----------------------------------------------------------------------------
def gerar_dados_simulados(n_amostras):
    umidade = np.clip(rng.normal(55, 22, n_amostras), 2, 100)
    temp = np.clip(rng.normal(26, 5, n_amostras), 10, 40)
    ph = np.clip(rng.normal(6.1, 0.8, n_amostras), 4.2, 8.5)
    n = np.clip(rng.normal(50, 25, n_amostras), 0, 100)
    p = np.clip(rng.normal(50, 25, n_amostras), 0, 100)
    k = np.clip(rng.normal(50, 25, n_amostras), 0, 100)

    prod = calcular_produtividade(umidade, ph, temp, n, p, k)
    volume = calcular_volume_irrigacao(umidade, temp)

    out = pd.DataFrame({
        "cultura": CULTURA,
        "umidade_solo": np.round(umidade, 1),
        "temperatura_ar": np.round(temp, 1),
        "ph_solo": np.round(ph, 2),
        "nitrogenio_n": np.round(n, 1),
        "fosforo_p": np.round(p, 1),
        "potassio_k": np.round(k, 1),
        "valor_ldr": ph_para_ldr(ph),
        "produtividade_t_ha": np.round(prod, 2),
        "volume_irrigacao_mm": np.round(volume, 1),
        "origem": "simulado",
    })
    return out


# ----------------------------------------------------------------------------
# Montagem final
# ----------------------------------------------------------------------------
def main():
    reais = carregar_dados_reais()
    simulados = gerar_dados_simulados(N_SIMULADO)

    df = pd.concat([reais, simulados], ignore_index=True)

    # Embaralha mantendo reprodutibilidade e cria um timestamp sequencial
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id_leitura", range(1, len(df) + 1))
    df.insert(
        1,
        "datahora",
        pd.date_range("2026-01-01 06:00:00", periods=len(df), freq="30min").astype(str),
    )

    df.to_csv(CSV_SAIDA, index=False, encoding="utf-8")

    print(f"Dataset gerado: {CSV_SAIDA}")
    print(f"  Total de leituras : {len(df)}")
    print(f"  Reais (Fase 3)    : {(df['origem'] == 'real').sum()}")
    print(f"  Simuladas         : {(df['origem'] == 'simulado').sum()}")
    print()
    print("Estatisticas dos alvos:")
    print(df[["produtividade_t_ha", "volume_irrigacao_mm"]].describe().round(2))


if __name__ == "__main__":
    main()
