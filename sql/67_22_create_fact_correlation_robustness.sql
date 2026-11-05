
-- STEP 67.3.26 - Create correlation robustness table
-- Source: ESS11_scientific_evidence_matrix_v1.csv
-- Stores the 32 robustness fields without recalculation.
-- Empty country lists are preserved as empty strings, not NULL.

USE european_values_v1;

CREATE TABLE IF NOT EXISTS fact_correlation_robustness (

    -- Composite key: one robustness record per correlation
    ess_variable VARCHAR(64) NOT NULL,
    economic_variable VARCHAR(64) NOT NULL,

    -- Country coverage and temporal comparison metadata
    included_countries VARCHAR(255) NOT NULL,
    common_countries_n SMALLINT UNSIGNED NOT NULL,
    common_countries VARCHAR(255) NOT NULL,
    removed_from_primary_sample VARCHAR(255) NOT NULL,
    switched_context_countries_n SMALLINT UNSIGNED NOT NULL,
    changed_numeric_values_n SMALLINT UNSIGNED NOT NULL,

    -- Pearson-Spearman baseline agreement
    pearson_spearman_same_strict_sign BOOLEAN NOT NULL,
    pearson_minus_spearman DOUBLE NOT NULL,

    -- Pearson Leave-One-Country-Out results
    pearson_loo_min DOUBLE NOT NULL,
    pearson_loo_max DOUBLE NOT NULL,
    pearson_loo_max_abs_change DOUBLE NOT NULL,
    pearson_loo_largest_change_countries VARCHAR(255) NOT NULL,
    pearson_loo_sign_flip_n SMALLINT UNSIGNED NOT NULL,
    pearson_loo_sign_flip_countries VARCHAR(255) NOT NULL,
    pearson_loo_any_sign_flip BOOLEAN NOT NULL,

    -- Pearson temporal and sample sensitivity
    pearson_primary_common DOUBLE NOT NULL,
    pearson_sensitivity_common DOUBLE NOT NULL,
    pearson_temporal_change DOUBLE NOT NULL,
    pearson_sample_change DOUBLE NOT NULL,
    pearson_temporal_sign_flip BOOLEAN NOT NULL,

    -- Spearman Leave-One-Country-Out results
    spearman_loo_min DOUBLE NOT NULL,
    spearman_loo_max DOUBLE NOT NULL,
    spearman_loo_max_abs_change DOUBLE NOT NULL,
    spearman_loo_largest_change_countries VARCHAR(255) NOT NULL,
    spearman_loo_sign_flip_n SMALLINT UNSIGNED NOT NULL,
    spearman_loo_sign_flip_countries VARCHAR(255) NOT NULL,
    spearman_loo_any_sign_flip BOOLEAN NOT NULL,

    -- Spearman temporal and sample sensitivity
    spearman_primary_common DOUBLE NOT NULL,
    spearman_sensitivity_common DOUBLE NOT NULL,
    spearman_temporal_change DOUBLE NOT NULL,
    spearman_sample_change DOUBLE NOT NULL,
    spearman_temporal_sign_flip BOOLEAN NOT NULL,

    -- One record per ESS-Eurostat association
    CONSTRAINT pk_fact_correlation_robustness
        PRIMARY KEY (ess_variable, economic_variable),

    -- Link robustness results to their baseline correlation
    CONSTRAINT fk_robustness_correlation
        FOREIGN KEY (ess_variable, economic_variable)
        REFERENCES fact_correlation (
            ess_variable,
            economic_variable
        ),

    -- Valid sample size
    CONSTRAINT chk_robustness_common_sample
        CHECK (common_countries_n BETWEEN 3 AND 29),

    -- Valid LOO correlation ranges
    CONSTRAINT chk_robustness_loo_ranges
        CHECK (
            pearson_loo_min BETWEEN -1 AND 1
            AND pearson_loo_max BETWEEN -1 AND 1
            AND spearman_loo_min BETWEEN -1 AND 1
            AND spearman_loo_max BETWEEN -1 AND 1
            AND pearson_loo_min <= pearson_loo_max
            AND spearman_loo_min <= spearman_loo_max
        ),

    -- Absolute changes cannot be negative
    CONSTRAINT chk_robustness_abs_changes
        CHECK (
            pearson_loo_max_abs_change >= 0
            AND spearman_loo_max_abs_change >= 0
        ),

    -- Boolean indicators must be binary
    CONSTRAINT chk_robustness_boolean_flags
        CHECK (
            pearson_spearman_same_strict_sign IN (0, 1)
            AND pearson_loo_any_sign_flip IN (0, 1)
            AND pearson_temporal_sign_flip IN (0, 1)
            AND spearman_loo_any_sign_flip IN (0, 1)
            AND spearman_temporal_sign_flip IN (0, 1)
        )

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


-- Verify the table structure
DESCRIBE fact_correlation_robustness;


-- Verify keys and constraints
SHOW CREATE TABLE fact_correlation_robustness;


-- Verify that the new table is empty
SELECT COUNT(*) AS total_robustness_records
FROM fact_correlation_robustness;
