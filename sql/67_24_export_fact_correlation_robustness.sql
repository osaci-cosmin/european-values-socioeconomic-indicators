
-- STEP 67.3.29 - Export correlation robustness data
-- Source: fact_correlation_robustness
-- Read-only: no database modifications.

USE european_values_v1;

SELECT *
FROM fact_correlation_robustness
ORDER BY ess_variable, economic_variable;
