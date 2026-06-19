# Relatório Técnico · FarmTech Solutions · Fase 4

**Inteligência Artificial aplicada ao Agronegócio**
Grupo FarmTech Solutions (FIAP)

---

## 1. Contexto e objetivo

A Fase 4 fecha o ciclo que começamos nas fases anteriores. Na Fase 2 montamos no Wokwi um sistema de irrigação inteligente com ESP32 para a cultura do milho, com sensores de umidade, pH (simulado por LDR) e nutrientes NPK. Na Fase 3 levamos essas leituras para um banco de dados Oracle (tabela `DADOS_SENSORES`).

Agora o objetivo é outro. Em vez de só coletar e armazenar, queremos aprender com os dados. Aplicamos aprendizado de máquina supervisionado (regressão) para prever variáveis do campo e, a partir dessas previsões, gerar recomendações de manejo. Tudo isso é apresentado em um dashboard interativo feito em Streamlit. Na prática, o protótipo percorre o caminho completo: dos sensores IoT ao banco de dados, dos modelos de Machine Learning até um painel de decisão para o gestor.

---

## 2. Dados utilizados

### 2.1. Origem

O dataset final (`data/dataset_agricola.csv`) tem 1.255 leituras e junta duas fontes:

- 55 leituras reais coletadas no Wokwi/ESP32 na Fase 3 (`origem = real`), convertidas para o schema novo;
- 1.200 leituras simuladas (`origem = simulado`), geradas a partir de relações agronômicas do milho.

### 2.2. Por que precisamos simular

Os dados da Fase 3 tinham dois problemas para um modelo de regressão. Primeiro, não existia uma variável-alvo de produtividade, já que o sistema da Fase 3 só decidia ligar ou não a bomba. Segundo, havia pouca variação: a umidade ficava concentrada em poucos valores e o NPK aparecia quase sempre como "Baixo". Com tão pouca variação, qualquer modelo aprenderia muito pouco.

Para conseguir treinar e validar de forma honesta, ampliamos a base com leituras simuladas. Mantivemos as leituras reais e marcamos a origem de cada registro para não perder a rastreabilidade. O gerador (`data/gerar_dataset.py`) usa semente fixa (42), então qualquer pessoa reproduz exatamente o mesmo dataset.

### 2.3. Variáveis

| Tipo | Variáveis |
|---|---|
| Features (entradas) | `umidade_solo`, `temperatura_ar`, `ph_solo`, `nitrogenio_n`, `fosforo_p`, `potassio_k` |
| Alvos (saídas) | `produtividade_t_ha`, `volume_irrigacao_mm` |

Decidimos prever a produtividade e o volume de irrigação porque são as duas variáveis que de fato interessam para a decisão do gestor. A umidade, o pH e o NPK já são medidos diretamente pelos sensores, então não faria sentido prevê-los: eles entram como entradas do modelo. Essa escolha está de acordo com a Parte 2 do enunciado, que pede previsão de "volume de irrigação (...) e estimativa de rendimento".

As relações que usamos na simulação seguem o que se conhece sobre o milho: a produtividade é maior perto de 65% de umidade, com pH em torno de 6,2, temperatura próxima de 27 °C e níveis altos de nutrientes (sendo o nitrogênio o de maior peso). Já o volume de irrigação é proporcional ao déficit de umidade em relação à meta de 70%, ajustado pela temperatura.

---

## 3. Pipeline de Machine Learning (Parte 2)

O pipeline está em `ml/treinar_modelos.py` e usa o Scikit-Learn.

### 3.1. Etapas

1. Tratamento dos dados: leitura do CSV, conversão de tipos e remoção de nulos.
2. Seleção das 6 features numéricas.
3. Divisão treino/teste em 80/20 (`random_state = 42`).
4. Treinamento de dois modelos por alvo: Regressão Linear (com `StandardScaler` dentro de um pipeline) e Random Forest (`n_estimators=150`, `max_depth=14`).
5. Validação cruzada de 5 folds, além do conjunto de teste.
6. Avaliação com MAE, MSE, RMSE e R².
7. Persistência: o melhor modelo de cada alvo vai para `ml/modelos/*.pkl`, as métricas para `metricas.json` e os gráficos para `assets/`.

### 3.2. Métricas obtidas

| Alvo | Modelo | R² | RMSE | MAE | MSE |
|---|---|---|---|---|---|
| Produtividade (t/ha) | Regressão Linear | 0,211 | 2,512 | 2,049 | 6,313 |
| Produtividade (t/ha) | Random Forest (melhor) | 0,916 | 0,820 | 0,640 | 0,672 |
| Volume de irrigação (mm) | Regressão Linear | 0,884 | 3,710 | 2,818 | 13,764 |
| Volume de irrigação (mm) | Random Forest (melhor) | 0,983 | 1,417 | 1,081 | 2,007 |

### 3.3. Interpretação

A produtividade depende de relações não lineares. Cada fator (umidade, pH, temperatura) tem uma faixa ótima em forma de sino, e ainda existe a interação entre umidade e nutrientes. Por isso a Regressão Linear captura pouco (R² = 0,21), enquanto o Random Forest chega a R² = 0,92. Esse contraste foi um dos pontos que mais nos chamou a atenção, porque mostra na prática em que situação vale a pena trocar um modelo linear por um não linear.

O volume de irrigação é diferente. Ele se comporta de forma quase linear, já que depende basicamente do déficit de umidade. Aqui a própria Regressão Linear já vai bem (R² = 0,88) e o Random Forest só refina o resultado (0,98).

Os gráficos de importância das variáveis (em `assets/`) confirmam que a umidade do solo e os nutrientes são os fatores que mais pesam na produtividade, o que é coerente com a cultura do milho.

Uma ressalva importante: como boa parte dos dados é simulada a partir de uma fórmula conhecida, parte do R² alto vem do fato de o modelo estar reaprendendo essa relação. Os números mostram que o pipeline está correto, mas em um cenário real, com mais ruído e fatores não medidos, seria natural esperar métricas um pouco menores.

---

## 4. Camada de recomendação de manejo

O módulo `ml/recomendacoes.py` transforma as previsões em ações práticas. Ele não depende do Streamlit nem do Scikit-Learn, o que facilita testar e reaproveitar o código. As regras seguem limiares agronômicos do milho:

- **Irrigação**: a partir da umidade atual e do volume previsto, o sistema indica irrigar com urgência (umidade < 25%), acionar a irrigação (< 40%, regra que herdamos da Fase 3), apenas monitorar (faixa adequada) ou suspender (solo encharcado);
- **Fertilização**: aponta quais nutrientes (N, P, K) estão abaixo do nível adequado (índice < 40);
- **Correção de pH**: sugere calagem para solo ácido (pH < 5,5) ou redução do pH para solo alcalino (pH > 6,8).

Alguns cenários que testamos:

| Cenário | Produtividade prevista | Recomendações |
|---|---|---|
| Solo seco, NPK baixo, pH ácido | 6,4 t/ha | Irrigar ~29 mm, adubar N e P, calagem |
| Condições ideais | 15,8 t/ha | Apenas monitorar |
| Solo encharcado | 8,8 t/ha | Suspender irrigação |

---

## 5. Banco de dados (Ir Além 1)

### 5.1. Modelagem

Usamos um modelo relacional simples, com duas tabelas:

- **`CULTURA`** (dimensão): guarda os parâmetros agronômicos da cultura;
- **`LEITURA_SENSOR`** (fato): guarda cada leitura dos sensores e os alvos, com chave estrangeira para `CULTURA`.

### 5.2. Ingestão IoT

O script `database/ingestao_iot.py` simula os sensores gravando no banco. Ele funciona de duas formas: em Oracle SQL Developer (continuidade da Fase 3), usando a biblioteca `oracledb`, e em SQLite (padrão, sem precisar de credenciais), que é o que usamos na demonstração.

Há também dois modos de carga. No modo completo, o script insere todas as leituras de uma vez. No modo stream, ele insere leitura a leitura com um intervalo, simulando a chegada contínua dos dados em tempo real. As consultas analíticas estão em `database/02_consultas_oracle.sql` e incluem contagem por origem, produtividade média por faixa de umidade, leituras com solo seco e volume médio de irrigação por faixa de temperatura.

---

## 6. Dashboard (Parte 1 e Ir Além 2)

O dashboard (`dashboard/app.py`) é onde tudo se junta para o gestor. Ele carrega os modelos treinados e os usa direto na interface, organizado em quatro abas:

1. **Visão Geral**: tabela de métricas (R², RMSE, MAE) com destaque para o melhor modelo, mais uma amostra dos dados;
2. **Correlações**: matriz de correlação interativa e gráficos de dispersão configuráveis;
3. **Previsão & Recomendações**: o gestor mexe nos sliders dos sensores e vê a previsão e as recomendações mudarem em tempo real;
4. **Tendências**: produtividade por faixa de umidade, distribuição, relação com o pH e série temporal.

A previsão em tempo real é justamente o que a Parte 1 pede. O dashboard carrega os arquivos `.pkl` e aplica o modelo às entradas do gestor, mostrando na prática a integração entre o Machine Learning e a interface.

---

## 7. Conclusão

A Fase 4 entrega um protótipo funcional de Assistente Agrícola Inteligente que vai da coleta à predição. Os modelos preveem produtividade (R² = 0,92) e volume de irrigação (R² = 0,98) com boa precisão, e o sistema traduz esses números em recomendações que o gestor consegue entender e usar. No fim, é uma demonstração concreta do que o enunciado chama de Agricultura Cognitiva: sensores, banco de dados e algoritmos trabalhando juntos.

### Limitações e próximos passos

- A maior parte dos dados ainda é simulada. O ideal seria coletar mais leituras reais ao longo de um ciclo da cultura.
- Dá para evoluir para séries temporais, prevendo a umidade dos próximos dias, e incorporar dados de clima.
- A camada de recomendação poderia virar um problema de otimização: quanto irrigar e adubar para maximizar o retorno, e não apenas corrigir o que está fora da faixa.
