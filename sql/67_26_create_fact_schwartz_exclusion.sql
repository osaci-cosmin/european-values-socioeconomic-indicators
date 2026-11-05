
-- STEP 67.3.32 - Create Schwartz exclusion sensitivity table
-- Source: ESS11_scientific_evidence_matrix_v1.csv
-- One row per Schwartz-Eurostat pair and exclusion scenario.
-- No scientific results are recalculated.

USE european_values_v1;

CREATE TABLE IF NOT EXISTS fact_schwartz_exclusion (

    -- Composite key
    ess_variable VARCHAR(64) NOT NULL,
    economic_variable VARCHAR(64) NOT NULL,
    scenario_code VARCHAR(16) NOT NULL,

    -- Effective exclusion information
    actual_exclusions VARCHAR(8) NOT NULL,
    scenario_n SMALLINT UNSIGNED NOT NULL,
    removed_n TINYINT UNSIGNED NOT NULL,

    -- Pearson sensitivity results
    pearson_scenario DOUBLE NOT NULL,
    pearson_change DOUBLE NOT NULL,
    pearson_abs_change DOUBLE NOT NULL,
    pearson_sign_flip BOOLEAN NOT NULL,

    -- Spearman sensitivity results
    spearman_scenario DOUBLE NOT NULL,
    spearman_change DOUBLE NOT NULL,
    spearman_abs_change DOUBLE NOT NULL,
    spearman_sign_flip BOOLEAN NOT NULL,

    -- One record per pair and scenario
    CONSTRAINT pk_schwartz_exclusion
        PRIMARY KEY (
            ess_variable,
            economic_variable,
            scenario_code
        ),

    -- Reference the validated baseline correlation
    CONSTRAINT fk_schwartz_exclusion_correlation
        FOREIGN KEY (
            ess_variable,
            economic_variable
        )
        REFERENCES fact_correlation (
            ess_variable,
            economic_variable
        ),

    -- Only the four Schwartz dimensions are applicable
    CONSTRAINT chk_schwartz_dimension
        CHECK (
            ess_variable IN (
                'mean_openness',
                'mean_conservation',
                'mean_self_transcendence',
                'mean_self_enhancement'
            )
        ),

    -- Only the three defined exclusion scenarios
    CONSTRAINT chk_schwartz_scenario
        CHECK (
            scenario_code IN (
                'exclude_EE',
                'exclude_GB',
                'exclude_EE_GB'
            )
        ),

    -- Valid sample size
    CONSTRAINT chk_schwartz_sample
        CHECK (scenario_n BETWEEN 3 AND 29),

    -- Preserve actual, not merely requested exclusions
    CONSTRAINT chk_schwartz_removed
        CHECK (
            (removed_n = 0 AND actual_exclusions = '')
            OR
            (
                removed_n = 1
                AND actual_exclusions IN ('EE', 'GB')
            )
            OR
            (
                removed_n = 2
                AND actual_exclusions = 'EE|GB'
            )
        ),

    -- Valid correlation coefficients
    CONSTRAINT chk_schwartz_correlations
        CHECK (
            pearson_scenario BETWEEN -1 AND 1
            AND spearman_scenario BETWEEN -1 AND 1
        ),

    -- Absolute changes must be non-negative
    CONSTRAINT chk_schwartz_changes
        CHECK (
            pearson_abs_change >= 0
            AND spearman_abs_change >= 0
        ),

    -- Boolean indicators
    CONSTRAINT chk_schwartz_flags
        CHECK (
            pearson_sign_flip IN (0, 1)
            AND spearman_sign_flip IN (0, 1)
        )

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;


-- Verify table structure
DESCRIBE fact_schwartz_exclusion;


-- Verify keys and constraints
SHOW CREATE TABLE fact_schwartz_exclusion;


-- Verify that the table is initially empty
SELECT COUNT(*) AS total_exclusion_records
FROM fact_schwartz_exclusion;
