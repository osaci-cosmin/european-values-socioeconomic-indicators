
-- STEP 67.3.5 - Create Eurostat indicator dimension
-- Source: ESS11_country_integrated_main_v1.csv

USE european_values_v1;

CREATE TABLE IF NOT EXISTS dim_economic_indicator (

    economic_variable VARCHAR(64) NOT NULL,

    display_name VARCHAR(150) NOT NULL,

    eurostat_dataset_code VARCHAR(40) NOT NULL,

    unit_label VARCHAR(60) NOT NULL,

    CONSTRAINT pk_dim_economic_indicator
        PRIMARY KEY (economic_variable)

) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- Verify table structure
DESCRIBE dim_economic_indicator;

-- Verify initial row count
SELECT COUNT(*) AS total_economic_indicators
FROM dim_economic_indicator;
