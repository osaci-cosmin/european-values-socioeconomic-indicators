
-- STEP 67.3.23 - Export baseline correlations
-- Source: fact_correlation
-- Read-only: no database modifications.

USE european_values_v1;

SELECT
    ess_variable,
    economic_variable,
    countries_n,
    pearson_baseline,
    spearman_baseline
FROM fact_correlation
ORDER BY ess_variable, economic_variable;
