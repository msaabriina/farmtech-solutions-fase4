-- ============================================================================
-- FarmTech Solutions - Fase 4
-- Consultas analiticas (Oracle SQL Developer)
-- Executar APOS criar as tabelas (01) e popular os dados (ingestao_iot.py).
-- ============================================================================

-- 1) Visao geral das leituras (join com a cultura)
SELECT l.id_leitura, c.nome AS cultura, l.datahora,
       l.umidade_solo, l.ph_solo, l.produtividade_t_ha, l.origem
FROM   LEITURA_SENSOR l
JOIN   CULTURA c ON c.id_cultura = l.id_cultura
ORDER  BY l.id_leitura
FETCH FIRST 20 ROWS ONLY;

-- 2) Total de leituras por origem (reais x simuladas)
SELECT origem, COUNT(*) AS quantidade
FROM   LEITURA_SENSOR
GROUP  BY origem;

-- 3) Produtividade media e umidade media por cultura
SELECT c.nome AS cultura,
       ROUND(AVG(l.produtividade_t_ha), 2) AS produtividade_media,
       ROUND(AVG(l.umidade_solo), 2)       AS umidade_media
FROM   LEITURA_SENSOR l
JOIN   CULTURA c ON c.id_cultura = l.id_cultura
GROUP  BY c.nome;

-- 4) Produtividade media por FAIXA de umidade do solo
--    (mostra a relacao entre umidade e rendimento)
SELECT CASE
         WHEN umidade_solo < 25 THEN '1. Critico (<25%)'
         WHEN umidade_solo < 40 THEN '2. Seco (25-40%)'
         WHEN umidade_solo < 60 THEN '3. Moderado (40-60%)'
         WHEN umidade_solo <= 85 THEN '4. Ideal (60-85%)'
         ELSE '5. Encharcado (>85%)'
       END AS faixa_umidade,
       COUNT(*) AS leituras,
       ROUND(AVG(produtividade_t_ha), 2) AS produtividade_media
FROM   LEITURA_SENSOR
GROUP  BY CASE
         WHEN umidade_solo < 25 THEN '1. Critico (<25%)'
         WHEN umidade_solo < 40 THEN '2. Seco (25-40%)'
         WHEN umidade_solo < 60 THEN '3. Moderado (40-60%)'
         WHEN umidade_solo <= 85 THEN '4. Ideal (60-85%)'
         ELSE '5. Encharcado (>85%)'
       END
ORDER  BY faixa_umidade;

-- 5) Leituras com solo seco que exigem irrigacao (volume previsto > 0)
SELECT id_leitura, umidade_solo, temperatura_ar,
       volume_irrigacao_mm, produtividade_t_ha
FROM   LEITURA_SENSOR
WHERE  umidade_solo < 40
ORDER  BY volume_irrigacao_mm DESC
FETCH FIRST 15 ROWS ONLY;

-- 6) Top 10 leituras de maior produtividade
SELECT id_leitura, umidade_solo, ph_solo,
       nitrogenio_n, fosforo_p, potassio_k, produtividade_t_ha
FROM   LEITURA_SENSOR
ORDER  BY produtividade_t_ha DESC
FETCH FIRST 10 ROWS ONLY;

-- 7) Volume medio de irrigacao recomendado por faixa de temperatura
SELECT CASE
         WHEN temperatura_ar < 20 THEN 'Ameno (<20C)'
         WHEN temperatura_ar < 30 THEN 'Moderado (20-30C)'
         ELSE 'Quente (>=30C)'
       END AS faixa_temperatura,
       ROUND(AVG(volume_irrigacao_mm), 2) AS volume_medio_mm
FROM   LEITURA_SENSOR
GROUP  BY CASE
         WHEN temperatura_ar < 20 THEN 'Ameno (<20C)'
         WHEN temperatura_ar < 30 THEN 'Moderado (20-30C)'
         ELSE 'Quente (>=30C)'
       END
ORDER  BY volume_medio_mm;
