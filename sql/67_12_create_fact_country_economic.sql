
-- STEP 67.3.12 - Create Eurostat country fact table
-- Source: ESS11_country_integrated_main_v1.csv
-- Preserve missing values and temporal metadata.

USE european_values_v1;

CREATE TABLE IF NOT EXISTS fact_country_economic (

    cntry CHAR(2) NOT NULL,

    economic_variable VARCHAR(64) NOT NULL,

    indicator_value DOUBLE NULL,

    context_year SMALLINT NULL,

    requested_eurostat_year SMALLINT NULL,

    eurostat_year SMALLINT NULL,

    reference_year_offset SMALLINT NULL,

    observation_match_status VARCHAR(100) NULL,

    source_cell_found BOOLEAN NULL,

    has_numeric_value BOOLEAN NULL,

    source_is_missing BOOLEAN NULL,

    source_has_status_flag BOOLEAN NULL,

    eurostat_status VARCHAR(50) NULL,

    CONSTRAINT pk_fact_country_economic
        PRIMARY KEY (cntry, economic_variable),

    CONSTRAINT fk_fact_country_economic_country
        FOREIGN KEY (cntry)
        REFERENCES dim_country(cntry),

    CONSTRAINT fk_fact_country_economic_indicator
        FOREIGN KEY (economic_variable)
        REFERENCES dim_economic_indicator(economic_variable)

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- Verify table structure
DESCRIBE fact_country_economic;

-- Verify initial row count
SELECT COUNT(*) AS total_economic_observations
FROM fact_country_economic;
