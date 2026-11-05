
-- STEP 67.3.16 - Export Eurostat observations for validation
-- Read-only: no database modifications.

USE european_values_v1;

SELECT
    cntry,
    economic_variable,
    indicator_value,
    context_year,
    requested_eurostat_year,
    eurostat_year,
    reference_year_offset,
    observation_match_status,
    source_cell_found,
    has_numeric_value,
    source_is_missing,
    source_has_status_flag,
    eurostat_status
FROM fact_country_economic
ORDER BY cntry, economic_variable;
