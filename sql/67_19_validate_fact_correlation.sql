
-- STEP 67.3.22 - Validate imported ESS-Eurostat correlations
-- Read-only validation. No database modifications.

USE european_values_v1;

-- 1. Verify total correlations, coverage and sample sizes
SELECT
    COUNT(*) AS total_correlations,
    COUNT(DISTINCT ess_variable) AS ess_variables_n,
    COUNT(DISTINCT economic_variable) AS indicators_n,
    MIN(countries_n) AS min_countries_n,
    MAX(countries_n) AS max_countries_n
FROM fact_correlation;

-- 2. Verify six correlations per economic indicator
SELECT
    economic_variable,
    COUNT(*) AS correlations_n,
    MIN(countries_n) AS min_countries_n,
    MAX(countries_n) AS max_countries_n
FROM fact_correlation
GROUP BY economic_variable
ORDER BY economic_variable;

-- 3. Detect missing ESS-Eurostat combinations
SELECT
    e.ess_variable,
    i.economic_variable
FROM dim_ess_variable AS e
CROSS JOIN dim_economic_indicator AS i
LEFT JOIN fact_correlation AS f
    ON e.ess_variable = f.ess_variable
    AND i.economic_variable = f.economic_variable
WHERE f.ess_variable IS NULL;

-- 4. Detect invalid correlation coefficients
SELECT
    ess_variable,
    economic_variable,
    pearson_baseline,
    spearman_baseline
FROM fact_correlation
WHERE pearson_baseline NOT BETWEEN -1 AND 1
   OR spearman_baseline NOT BETWEEN -1 AND 1;
