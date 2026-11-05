
-- STEP 67.3.35 - Export Schwartz exclusion sensitivity
-- Read-only: no database modifications.
-- Preserve actual exclusions and all scenario coefficients.

USE european_values_v1;

SELECT
    ess_variable,
    economic_variable,
    scenario_code,
    actual_exclusions,
    scenario_n,
    removed_n,
    pearson_scenario,
    pearson_change,
    pearson_abs_change,
    pearson_sign_flip,
    spearman_scenario,
    spearman_change,
    spearman_abs_change,
    spearman_sign_flip
FROM fact_schwartz_exclusion
ORDER BY
    ess_variable,
    economic_variable,
    scenario_code;
