"""
FarmTech Solutions - Fase 4
Camada de recomendacao agricola (regras de manejo).

Este modulo NAO depende de Streamlit nem de scikit-learn: recebe as leituras
dos sensores e as PREVISOES geradas pelos modelos de regressao e devolve
recomendacoes de acao (irrigacao, fertilizacao e correcao de pH) para a
cultura do milho. Por ser independente, pode ser testado e reutilizado tanto
pelo dashboard quanto por um script de linha de comando.

Os limiares ("thresholds") seguem a logica da Fase 2/3 (solo seco < 40%) e
faixas agronomicas usuais para o milho.
"""

from dataclasses import dataclass, field
from typing import List, Dict

# ----------------------------------------------------------------------------
# Parametros agronomicos da cultura do milho
# ----------------------------------------------------------------------------
UMIDADE_CRITICA = 25.0      # abaixo disso o estresse hidrico e severo
UMIDADE_SECA = 40.0         # regra herdada da Fase 3 (solo seco)
UMIDADE_IDEAL_MIN = 60.0
UMIDADE_IDEAL_MAX = 85.0

PH_IDEAL_MIN = 5.5
PH_IDEAL_MAX = 6.8

NUTRIENTE_BAIXO = 40.0      # indice 0-100
NUTRIENTE_ALTO = 70.0


@dataclass
class Recomendacao:
    categoria: str          # "Irrigacao", "Fertilizacao", "Correcao de pH"
    acao: str               # texto curto da acao
    urgencia: str           # "Alta", "Media", "Baixa"
    justificativa: str


@dataclass
class ResultadoRecomendacao:
    produtividade_prevista: float
    volume_irrigacao_previsto: float
    recomendacoes: List[Recomendacao] = field(default_factory=list)

    def como_lista_dicts(self) -> List[Dict]:
        return [r.__dict__ for r in self.recomendacoes]


# ----------------------------------------------------------------------------
# Classificacoes auxiliares
# ----------------------------------------------------------------------------
def classificar_umidade(umidade: float) -> str:
    if umidade < UMIDADE_CRITICA:
        return "Solo muito seco (critico)"
    if umidade < UMIDADE_SECA:
        return "Solo seco"
    if umidade <= UMIDADE_IDEAL_MAX:
        return "Solo adequado"
    return "Solo encharcado"


def classificar_ph(ph: float) -> str:
    if ph < PH_IDEAL_MIN:
        return "Acido"
    if ph > PH_IDEAL_MAX:
        return "Alcalino"
    return "Adequado"


def classificar_nutriente(valor: float) -> str:
    if valor < NUTRIENTE_BAIXO:
        return "Baixo"
    if valor < NUTRIENTE_ALTO:
        return "Adequado"
    return "Alto"


# ----------------------------------------------------------------------------
# Regras de recomendacao
# ----------------------------------------------------------------------------
def recomendar_irrigacao(umidade: float, temperatura: float,
                         volume_previsto: float) -> Recomendacao:
    """Decide a acao de irrigacao a partir da umidade atual e do volume (mm)
    previsto pelo modelo de regressao."""
    if umidade < UMIDADE_CRITICA:
        return Recomendacao(
            "Irrigação",
            f"Irrigar com urgência ~{volume_previsto:.0f} mm",
            "Alta",
            "Umidade abaixo do nível crítico: risco de estresse hídrico severo "
            "e perda de produtividade.",
        )
    if umidade < UMIDADE_SECA:
        return Recomendacao(
            "Irrigação",
            f"Acionar irrigação ~{volume_previsto:.0f} mm",
            "Media",
            "Solo seco (umidade < 40%). Irrigação recomendada para repor a "
            "umidade até a faixa ideal do milho.",
        )
    if umidade <= UMIDADE_IDEAL_MAX:
        return Recomendacao(
            "Irrigação",
            "Manter monitoramento (sem irrigar agora)",
            "Baixa",
            "Umidade dentro da faixa adequada. Irrigação desnecessária no momento.",
        )
    return Recomendacao(
        "Irrigação",
        "Suspender irrigação",
        "Media",
        "Solo encharcado: irrigar agravaria o excesso de água e a lixiviação de "
        "nutrientes.",
    )


def recomendar_fertilizacao(n: float, p: float, k: float) -> List[Recomendacao]:
    recs: List[Recomendacao] = []
    mapa = {"Nitrogênio (N)": n, "Fósforo (P)": p, "Potássio (K)": k}
    baixos = [nome for nome, val in mapa.items() if val < NUTRIENTE_BAIXO]
    if baixos:
        recs.append(Recomendacao(
            "Fertilização",
            "Aplicar adubação de: " + ", ".join(baixos),
            "Alta" if len(baixos) >= 2 else "Media",
            "Nutriente(s) abaixo do nível adequado (índice < 40). A reposição "
            "tende a elevar a produtividade estimada.",
        ))
    else:
        recs.append(Recomendacao(
            "Fertilização",
            "Nenhuma adubação adicional necessária",
            "Baixa",
            "Níveis de N, P e K dentro ou acima da faixa adequada.",
        ))
    return recs


def recomendar_ph(ph: float) -> Recomendacao:
    classe = classificar_ph(ph)
    if classe == "Acido":
        return Recomendacao(
            "Correção de pH",
            "Realizar calagem (elevar o pH)",
            "Media",
            f"pH {ph:.1f} abaixo da faixa ideal ({PH_IDEAL_MIN}-{PH_IDEAL_MAX}). "
            "A calagem melhora a disponibilidade de nutrientes.",
        )
    if classe == "Alcalino":
        return Recomendacao(
            "Correção de pH",
            "Aplicar enxofre / matéria orgânica (reduzir o pH)",
            "Media",
            f"pH {ph:.1f} acima da faixa ideal ({PH_IDEAL_MIN}-{PH_IDEAL_MAX}).",
        )
    return Recomendacao(
        "Correção de pH",
        "pH adequado, sem correção",
        "Baixa",
        f"pH {ph:.1f} dentro da faixa ideal do milho.",
    )


def gerar_recomendacoes(umidade: float, temperatura: float, ph: float,
                        n: float, p: float, k: float,
                        produtividade_prevista: float,
                        volume_previsto: float) -> ResultadoRecomendacao:
    """Consolida todas as recomendacoes para um conjunto de leituras."""
    resultado = ResultadoRecomendacao(
        produtividade_prevista=round(float(produtividade_prevista), 2),
        volume_irrigacao_previsto=round(float(volume_previsto), 1),
    )
    resultado.recomendacoes.append(
        recomendar_irrigacao(umidade, temperatura, volume_previsto)
    )
    resultado.recomendacoes.extend(recomendar_fertilizacao(n, p, k))
    resultado.recomendacoes.append(recomendar_ph(ph))
    return resultado


# ----------------------------------------------------------------------------
# Execucao direta (demonstracao rapida)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    exemplo = gerar_recomendacoes(
        umidade=30, temperatura=31, ph=5.2,
        n=20, p=15, k=60,
        produtividade_prevista=7.4, volume_previsto=26,
    )
    print(f"Produtividade prevista: {exemplo.produtividade_prevista} t/ha")
    print(f"Volume de irrigacao previsto: {exemplo.volume_irrigacao_previsto} mm\n")
    for r in exemplo.recomendacoes:
        print(f"[{r.urgencia:5}] {r.categoria:16} -> {r.acao}")
        print(f"        {r.justificativa}")
