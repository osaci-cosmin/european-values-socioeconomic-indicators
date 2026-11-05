
-- Step 67.1 - Create the project database

CREATE DATABASE IF NOT EXISTS european_values_v1
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE european_values_v1;

SELECT DATABASE() AS active_database;

SHOW TABLES;