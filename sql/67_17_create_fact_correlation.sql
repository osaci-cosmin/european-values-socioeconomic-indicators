
-- STEP 67.3.19 - Create correlation fact table
-- Source: ESS11_scientific_evidence_matrix_v1.csv
-- Baseline Pearson and Spearman correlations
-- No statistical values are recalculated.

USE european_values_v1;

CREATE TABLE IF NOT EXISTS fact_correlation (

    ess_variable VARCHAR(64) NOT NULL,

    economic_variable VARCHAR(64) NOT NULL,

    countries_n SMALLINT UNSIGNED NOT NULL,

    pearson_baseline DOUBLE NOT NULL,

    spearman_baseline DOUBLE NOT NULL,

    -- Each ESS-Eurostat pair must be unique
    CONSTRAINT pk_fact_correlation
        PRIMARY KEY (ess_variable, economic_variable),

    -- ESS variable must exist in the dimension
    CONSTRAINT fk_fact_correlation_ess
        FOREIGN KEY (ess_variable)
        REFERENCES dim_ess_variable(ess_variable),

    -- Economic indicator must exist in the dimension
    CONSTRAINT fk_fact_correlation_economic
        FOREIGN KEY (economic_variable)
        REFERENCES dim_economic_indicator(economic_variable),

    -- Correlations require at least 3 observations
    CONSTRAINT chk_correlation_sample
        CHECK (countries_n >= 3),

    -- Pearson must be between -1 and 1
    CONSTRAINT chk_correlation_pearson
        CHECK (pearson_baseline BETWEEN -1 AND 1),

    -- Spearman must be between -1 and 1
    CONSTRAINT chk_correlation_spearman
        CHECK (spearman_baseline BETWEEN -1 AND 1)

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- Verify table structure
DESCRIBE fact_correlation;

-- Verify primary and foreign keys
SHOW CREATE TABLE fact_correlation;

-- Verify initial row count
SELECT COUNT(*) AS total_correlations
FROM fact_correlation;
