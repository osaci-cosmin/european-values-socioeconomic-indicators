
-- STEP 67.3.7 - Create ESS country fact table
-- Source: ESS11_country_integrated_main_v1.csv

USE european_values_v1;

CREATE TABLE IF NOT EXISTS fact_country_ess (

    cntry CHAR(2) NOT NULL,

    ess_variable VARCHAR(64) NOT NULL,

    estimate_value DOUBLE NULL,

    valid_respondents_n INT NULL,

    standard_error DOUBLE NULL,

    ci_lower DOUBLE NULL,

    ci_upper DOUBLE NULL,

    CONSTRAINT pk_fact_country_ess
        PRIMARY KEY (cntry, ess_variable),

    CONSTRAINT fk_fact_country_ess_country
        FOREIGN KEY (cntry)
        REFERENCES dim_country(cntry),

    CONSTRAINT fk_fact_country_ess_variable
        FOREIGN KEY (ess_variable)
        REFERENCES dim_ess_variable(ess_variable)

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- Verify table structure
DESCRIBE fact_country_ess;

-- Verify initial row count
SELECT COUNT(*) AS total_ess_observations
FROM fact_country_ess;
