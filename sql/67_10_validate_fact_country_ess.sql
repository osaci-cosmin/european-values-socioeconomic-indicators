
-- STEP 67.3.10 - Validate imported ESS country observations
-- Read-only validation: no data modifications.

USE european_values_v1;

-- 1. Verify total observations and coverage
SELECT
    COUNT(*) AS total_observations,
    COUNT(DISTINCT cntry) AS total_countries,
    COUNT(DISTINCT ess_variable) AS total_variables,
    SUM(estimate_value IS NULL) AS missing_estimates,
    SUM(standard_error IS NULL) AS missing_standard_errors
FROM fact_country_ess;

-- 2. Verify the coverage of each ESS variable
SELECT
    ess_variable,
    COUNT(*) AS observations_n,
    COUNT(estimate_value) AS numeric_values_n,
    COUNT(standard_error) AS standard_errors_n
FROM fact_country_ess
GROUP BY ess_variable
ORDER BY ess_variable;

-- 3. Check missing country-variable combinations
SELECT
    c.cntry,
    v.ess_variable
FROM dim_country AS c
CROSS JOIN dim_ess_variable AS v
LEFT JOIN fact_country_ess AS f
    ON c.cntry = f.cntry
    AND v.ess_variable = f.ess_variable
WHERE f.cntry IS NULL;

-- 4. Check invalid confidence interval ordering
SELECT
    cntry,
    ess_variable,
    ci_lower,
    ci_upper
FROM fact_country_ess
WHERE ci_lower > ci_upper;
