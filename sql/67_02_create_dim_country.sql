
-- STEP 67.3.1 — Create the country dimension

USE european_values_v1;

CREATE TABLE IF NOT EXISTS dim_country (
    cntry CHAR(2) NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    region_4 VARCHAR(50) NULL,

    CONSTRAINT pk_dim_country PRIMARY KEY (cntry)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;

-- Verify the table structure
DESCRIBE dim_country;

-- Verify the initial number of rows
SELECT COUNT(*) AS total_countries
FROM dim_country;
