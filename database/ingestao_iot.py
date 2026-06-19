"""
FarmTech Solutions - Fase 4
Pipeline de ingestao de dados IoT no banco de dados - IR ALEM 1.

Simula a coleta dos sensores (ESP32/Wokwi) sendo gravada em um banco
relacional. Suporta dois bancos:

  * SQLite  (padrao) -> roda em qualquer maquina, sem credenciais. Ideal
                        para a demonstracao do video.
  * Oracle  (opcional) -> mesma modelagem da Fase 3/4 no Oracle da FIAP.
                          Requer a biblioteca 'oracledb' e as variaveis de
                          ambiente ORACLE_USER, ORACLE_PASSWORD e ORACLE_DSN.

Modos de ingestao:
  * completo -> carrega todas as leituras de uma vez (bulk).
  * stream   -> insere leitura a leitura com um intervalo, simulando a
                ingestao/atualizacao automatica em tempo real.

Exemplos:
  python database/ingestao_iot.py                       # SQLite, carga completa
  python database/ingestao_iot.py --reset               # recria as tabelas
  python database/ingestao_iot.py --modo stream --limite 20 --intervalo 0.5
  python database/ingestao_iot.py --banco oracle --reset
"""

import os
import sys
import csv
import time
import argparse

PASTA_DB = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(PASTA_DB)
CSV_DADOS = os.path.join(RAIZ, "data", "dataset_agricola.csv")
# Caminho do banco SQLite. Pode ser sobrescrito pela variavel de ambiente
# FARMTECH_SQLITE (util quando a pasta do projeto fica em um disco de rede
# que nao suporta o travamento de arquivos exigido pelo SQLite).
SQLITE_PATH = os.environ.get("FARMTECH_SQLITE", os.path.join(PASTA_DB, "farmtech.db"))

COLUNAS_LEITURA = [
    "id_cultura", "datahora", "umidade_solo", "temperatura_ar", "ph_solo",
    "nitrogenio_n", "fosforo_p", "potassio_k", "valor_ldr",
    "produtividade_t_ha", "volume_irrigacao_mm", "origem",
]


# ============================================================================
# Camada de banco de dados
# ============================================================================
class BancoFarmTech:
    """Abstracao simples para SQLite e Oracle com a mesma modelagem."""

    def __init__(self, banco="sqlite"):
        self.banco = banco
        self.conn = None
        self.ph = "?" if banco == "sqlite" else ":x"  # placeholder de bind

    # ---- conexao -----------------------------------------------------------
    def conectar(self):
        if self.banco == "sqlite":
            import sqlite3
            self.conn = sqlite3.connect(SQLITE_PATH)
            self.conn.execute("PRAGMA foreign_keys = ON")
        else:
            try:
                import oracledb
            except ImportError:
                sys.exit("ERRO: instale a biblioteca 'oracledb' (pip install oracledb).")
            usuario = os.environ.get("ORACLE_USER")
            senha = os.environ.get("ORACLE_PASSWORD")
            dsn = os.environ.get("ORACLE_DSN", "oracle.fiap.com.br:1521/ORCL")
            if not usuario or not senha:
                sys.exit("ERRO: defina ORACLE_USER e ORACLE_PASSWORD no ambiente.")
            self.conn = oracledb.connect(user=usuario, password=senha, dsn=dsn)
        return self.conn

    # ---- DDL ---------------------------------------------------------------
    def criar_tabelas(self, reset=False):
        cur = self.conn.cursor()
        if self.banco == "sqlite":
            if reset:
                cur.execute("DROP TABLE IF EXISTS LEITURA_SENSOR")
                cur.execute("DROP TABLE IF EXISTS CULTURA")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS CULTURA (
                    id_cultura        INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome              TEXT NOT NULL,
                    ph_ideal_min      REAL,
                    ph_ideal_max      REAL,
                    umidade_ideal_min REAL,
                    umidade_ideal_max REAL
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS LEITURA_SENSOR (
                    id_leitura          INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_cultura          INTEGER NOT NULL,
                    datahora            TEXT,
                    umidade_solo        REAL,
                    temperatura_ar      REAL,
                    ph_solo             REAL,
                    nitrogenio_n        REAL,
                    fosforo_p           REAL,
                    potassio_k          REAL,
                    valor_ldr           INTEGER,
                    produtividade_t_ha  REAL,
                    volume_irrigacao_mm REAL,
                    origem              TEXT,
                    FOREIGN KEY (id_cultura) REFERENCES CULTURA (id_cultura)
                )""")
        else:
            # Oracle: usa o DDL do arquivo 01 (aqui de forma resumida)
            if reset:
                for tab in ("LEITURA_SENSOR", "CULTURA"):
                    try:
                        cur.execute(f"DROP TABLE {tab} CASCADE CONSTRAINTS")
                    except Exception:
                        pass
            cur.execute("""
                CREATE TABLE CULTURA (
                    id_cultura        NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    nome              VARCHAR2(50) NOT NULL,
                    ph_ideal_min      NUMBER(4,2),
                    ph_ideal_max      NUMBER(4,2),
                    umidade_ideal_min NUMBER(5,2),
                    umidade_ideal_max NUMBER(5,2)
                )""")
            cur.execute("""
                CREATE TABLE LEITURA_SENSOR (
                    id_leitura          NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    id_cultura          NUMBER NOT NULL,
                    datahora            TIMESTAMP DEFAULT SYSTIMESTAMP,
                    umidade_solo        NUMBER(5,2),
                    temperatura_ar      NUMBER(4,1),
                    ph_solo             NUMBER(4,2),
                    nitrogenio_n        NUMBER(5,2),
                    fosforo_p           NUMBER(5,2),
                    potassio_k          NUMBER(5,2),
                    valor_ldr           NUMBER(6),
                    produtividade_t_ha  NUMBER(6,2),
                    volume_irrigacao_mm NUMBER(6,2),
                    origem              VARCHAR2(10),
                    CONSTRAINT fk_leitura_cultura
                        FOREIGN KEY (id_cultura) REFERENCES CULTURA (id_cultura)
                )""")
        self.conn.commit()

    # ---- cultura -----------------------------------------------------------
    def obter_ou_criar_cultura(self, nome="Milho"):
        cur = self.conn.cursor()
        p = self.ph
        cur.execute(f"SELECT id_cultura FROM CULTURA WHERE nome = {p}", (nome,))
        linha = cur.fetchone()
        if linha:
            return linha[0]
        cur.execute(
            f"INSERT INTO CULTURA (nome, ph_ideal_min, ph_ideal_max, "
            f"umidade_ideal_min, umidade_ideal_max) VALUES ({p},{p},{p},{p},{p})",
            (nome, 5.5, 6.8, 60, 85),
        )
        self.conn.commit()
        cur.execute(f"SELECT id_cultura FROM CULTURA WHERE nome = {p}", (nome,))
        return cur.fetchone()[0]

    # ---- insercao ----------------------------------------------------------
    def inserir_leitura(self, valores):
        cur = self.conn.cursor()
        binds = ", ".join([self.ph] * len(COLUNAS_LEITURA))
        sql = (f"INSERT INTO LEITURA_SENSOR ({', '.join(COLUNAS_LEITURA)}) "
               f"VALUES ({binds})")
        cur.execute(sql, valores)

    def contar(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM LEITURA_SENSOR")
        return cur.fetchone()[0]

    def commit(self):
        self.conn.commit()

    def fechar(self):
        if self.conn:
            self.conn.close()


# ============================================================================
# Leitura do CSV (fonte que simula o stream dos sensores IoT)
# ============================================================================
def ler_dataset(id_cultura, limite=None):
    linhas = []
    with open(CSV_DADOS, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        for i, r in enumerate(leitor):
            if limite is not None and i >= limite:
                break
            linhas.append((
                id_cultura,
                r["datahora"],
                float(r["umidade_solo"]),
                float(r["temperatura_ar"]),
                float(r["ph_solo"]),
                float(r["nitrogenio_n"]),
                float(r["fosforo_p"]),
                float(r["potassio_k"]),
                int(r["valor_ldr"]),
                float(r["produtividade_t_ha"]),
                float(r["volume_irrigacao_mm"]),
                r["origem"],
            ))
    return linhas


# ============================================================================
# Principal
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Ingestao de dados IoT - FarmTech Fase 4")
    parser.add_argument("--banco", choices=["sqlite", "oracle"], default="sqlite")
    parser.add_argument("--modo", choices=["completo", "stream"], default="completo")
    parser.add_argument("--intervalo", type=float, default=1.0,
                        help="segundos entre insercoes no modo stream")
    parser.add_argument("--limite", type=int, default=None,
                        help="limita o numero de leituras inseridas")
    parser.add_argument("--reset", action="store_true",
                        help="recria as tabelas antes de inserir")
    args = parser.parse_args()

    print("=" * 64)
    print(f"FarmTech Fase 4 | Ingestao IoT  ->  banco: {args.banco} | modo: {args.modo}")
    print("=" * 64)

    db = BancoFarmTech(args.banco)
    db.conectar()
    db.criar_tabelas(reset=args.reset)
    id_cultura = db.obter_ou_criar_cultura("Milho")
    print(f"Cultura 'Milho' -> id_cultura = {id_cultura}")

    leituras = ler_dataset(id_cultura, limite=args.limite)
    print(f"Leituras a inserir: {len(leituras)}")

    inicio = time.time()
    if args.modo == "completo":
        for v in leituras:
            db.inserir_leitura(v)
        db.commit()
        print("Carga completa finalizada.")
    else:
        # Modo stream: simula a chegada de dados em tempo real
        for i, v in enumerate(leituras, start=1):
            db.inserir_leitura(v)
            db.commit()
            print(f"  [{i:>4}/{len(leituras)}] leitura gravada | "
                  f"umidade={v[2]:.1f}% temp={v[3]:.1f}C ph={v[4]:.1f} "
                  f"-> prod={v[9]:.1f} t/ha", flush=True)
            time.sleep(args.intervalo)
        print("Stream finalizado.")

    total = db.contar()
    print("-" * 64)
    print(f"Total de leituras no banco: {total}")
    print(f"Tempo: {time.time() - inicio:.1f}s")
    if args.banco == "sqlite":
        print(f"Arquivo do banco: {os.path.relpath(SQLITE_PATH, RAIZ)}")
    db.fechar()


if __name__ == "__main__":
    main()
