
-- STEP 67.3.3 - Create ESS variable dimension
-- Source: ESS11_country_integrated_main_v1.csv

USE european_values_v1;

CREATE TABLE IF NOT EXISTS dim_ess_variable (
    ess_variable VARCHAR(64) NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    construct_group VARCHAR(40) NOT NULL,

    CONSTRAINT pk_dim_ess_variable
        PRIMARY KEY (ess_variable)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- Inspect the table structure
DESCRIBE dim_ess_variable;

-- Check initial number of records
SELECT COUNT(*) AS total_ess_variables
FROM dim_ess_variable;
