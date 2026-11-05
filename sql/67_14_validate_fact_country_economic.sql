
-- STEP 67.3.15 - Validate imported Eurostat observations
-- Read-only validation. No database modifications.

USE european_values_v1;

-- 1. Validate total coverage and missing values
SELECT
    COUNT(*) AS total_observations,
    COUNT(DISTINCT cntry) AS countries_n,
    COUNT(DISTINCT economic_variable) AS indicators_n,
    COUNT(indicator_value) AS numeric_values_n,
    SUM(indicator_value IS NULL) AS missing_values_n
FROM fact_country_economic;

-- 2. Check availability by economic indicator
SELECT
    economic_variable,
    COUNT(*) AS observations_n,
    COUNT(indicator_value) AS available_n,
    SUM(indicator_value IS NULL) AS missing_n
FROM fact_country_economic
GROUP BY economic_variable
ORDER BY economic_variable;

-- 3. Detect missing country-indicator combinations
SELECT
    c.cntry,
    e.economic_variable
FROM dim_country AS c
CROSS JOIN dim_economic_indicator AS e
LEFT JOIN fact_country_economic AS f
    ON c.cntry = f.cntry
    AND e.economic_variable = f.economic_variable
WHERE f.cntry IS NULL;

-- 4. Detect inconsistencies in numeric availability flags
SELECT
    cntry,
    economic_variable,
    indicator_value,
    has_numeric_value
FROM fact_country_economic
WHERE has_numeric_value IS NULL
   OR has_numeric_value <> (indicator_value IS NOT NULL);
