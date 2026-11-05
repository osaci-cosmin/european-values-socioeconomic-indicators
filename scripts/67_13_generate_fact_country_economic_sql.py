
# STEP 67.3.13 - Generate Eurostat country SQL inserts
# Read-only source processing; generates a separate SQL file.
# Missing numeric values are preserved as SQL NULL.

import csv
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path


# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "processed"
    / "ESS11_country_integrated_main_v1.csv"
)

OUTPUT = (
    ROOT / "sql"
    / "67_13_insert_fact_country_economic.sql"
)


# 2. Economic indicators
INDICATORS = (
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
)

EXPECTED_AVAILABLE = {
    "gdp_pps_index": 28,
    "employment_rate": 26,
    "unemployment_rate": 26,
    "gini": 25,
    "material_deprivation_rate": 26,
    "inflation_rate": 27,
}


# 3. SQL value conversion functions
def sql_number(value):
    """Convert numeric text to SQL, preserving NULL."""
    value = value.strip()

    if not value:
        return "NULL"

    number = Decimal(value)

    if not number.is_finite():
        raise ValueError(f"Non-finite number: {value}")

    return format(number, "f")


def sql_integer(value):
    """Convert integer text to SQL."""
    value = value.strip()

    if not value:
        return "NULL"

    number = Decimal(value)

    if not number.is_finite():
        raise ValueError(f"Non-finite integer: {value}")

    if number != number.to_integral_value():
        raise ValueError(f"Invalid integer: {value}")

    return str(int(number))


def sql_boolean(value):
    """Convert CSV boolean text to SQL 1, 0 or NULL."""
    value = value.strip().lower()

    if not value:
        return "NULL"

    if value in ("true", "1"):
        return "1"

    if value in ("false", "0"):
        return "0"

    raise ValueError(f"Invalid boolean: {value}")


def sql_text(value):
    """Escape SQL text and preserve missing values."""
    value = value.strip()

    if not value:
        return "NULL"

    return "'" + value.replace("'", "''") + "'"


# 4. Read source data
with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    source_columns = set(reader.fieldnames or [])


# 5. Validate required source columns
required = {"cntry", "context_year"}

for indicator in INDICATORS:
    required.add(indicator)

    for suffix in (
        "requested_eurostat_year",
        "eurostat_year",
        "reference_year_offset",
        "observation_match_status",
        "source_cell_found",
        "has_numeric_value",
        "source_is_missing",
        "source_has_status_flag",
        "eurostat_status",
    ):
        required.add(f"{indicator}__{suffix}")

missing_columns = required - source_columns

if missing_columns:
    raise ValueError(
        f"Missing source columns: {sorted(missing_columns)}"
    )

if len(rows) != 29:
    raise ValueError(
        f"Expected 29 countries, found {len(rows)}"
    )

countries = [row["cntry"].strip() for row in rows]

if len(set(countries)) != 29:
    raise ValueError("Duplicate country codes detected")

if not all(
    re.fullmatch(r"[A-Z]{2}", country)
    for country in countries
):
    raise ValueError("Invalid country code detected")


# 6. Transform wide data into long SQL records
sql_rows = []
available_counts = Counter()
missing_counts = Counter()

for row in rows:
    country = row["cntry"].strip()

    for indicator in INDICATORS:
        value = row[indicator].strip()

        has_value = row[
            f"{indicator}__has_numeric_value"
        ].strip().lower()

        # Check source availability flag consistency.
        if has_value not in ("true", "false", "1", "0"):
            raise ValueError(
                f"Invalid availability flag: "
                f"{country}, {indicator}"
            )

        flag_available = has_value in ("true", "1")
        actual_available = bool(value)

        if flag_available != actual_available:
            raise ValueError(
                f"Numeric availability mismatch: "
                f"{country}, {indicator}"
            )

        if actual_available:
            available_counts[indicator] += 1
        else:
            missing_counts[indicator] += 1

        values = [
            sql_text(country),
            sql_text(indicator),
            sql_number(value),
            sql_integer(row["context_year"]),
            sql_integer(
                row[f"{indicator}__requested_eurostat_year"]
            ),
            sql_integer(
                row[f"{indicator}__eurostat_year"]
            ),
            sql_integer(
                row[f"{indicator}__reference_year_offset"]
            ),
            sql_text(
                row[f"{indicator}__observation_match_status"]
            ),
            sql_boolean(
                row[f"{indicator}__source_cell_found"]
            ),
            sql_boolean(
                row[f"{indicator}__has_numeric_value"]
            ),
            sql_boolean(
                row[f"{indicator}__source_is_missing"]
            ),
            sql_boolean(
                row[f"{indicator}__source_has_status_flag"]
            ),
            sql_text(
                row[f"{indicator}__eurostat_status"]
            ),
        ]

        if len(values) != 13:
            raise ValueError("Invalid SQL record width")

        sql_rows.append(
            "    (" + ", ".join(values) + ")"
        )


# 7. Validate expected coverage
if len(sql_rows) != 174:
    raise ValueError(
        f"Expected 174 rows, generated {len(sql_rows)}"
    )

for indicator in INDICATORS:
    if available_counts[indicator] != EXPECTED_AVAILABLE[indicator]:
        raise ValueError(
            f"Unexpected available values for {indicator}: "
            f"{available_counts[indicator]}"
        )

available_total = sum(available_counts.values())
missing_total = sum(missing_counts.values())

if available_total != 158 or missing_total != 16:
    raise ValueError(
        f"Unexpected availability: "
        f"{available_total} available, "
        f"{missing_total} missing"
    )


# 8. Generate SQL
sql = (
    "-- STEP 67.3.13 - Generated Eurostat country inserts\n"
    "-- Source: ESS11_country_integrated_main_v1.csv\n"
    "-- Preserve source NULL values and temporal metadata.\n"
    "-- Execute only once after reviewing the SQL.\n\n"

    "USE european_values_v1;\n\n"

    "INSERT INTO fact_country_economic (\n"
    "    cntry,\n"
    "    economic_variable,\n"
    "    indicator_value,\n"
    "    context_year,\n"
    "    requested_eurostat_year,\n"
    "    eurostat_year,\n"
    "    reference_year_offset,\n"
    "    observation_match_status,\n"
    "    source_cell_found,\n"
    "    has_numeric_value,\n"
    "    source_is_missing,\n"
    "    source_has_status_flag,\n"
    "    eurostat_status\n"
    ")\nVALUES\n"

    + ",\n".join(sql_rows)

    + ";\n\n"

    "-- Verify imported row count\n"
    "SELECT COUNT(*) AS total_economic_observations\n"
    "FROM fact_country_economic;\n"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(sql, encoding="utf-8")


# 9. Report generation results
print("=" * 60)
print("STEP 67.3.13 - EUROSTAT SQL GENERATION")
print("=" * 60)

print("Countries:", len(countries))
print("Economic indicators:", len(INDICATORS))
print("SQL rows generated:", len(sql_rows))
print("Available values:", available_total)
print("Missing values:", missing_total)

for indicator in INDICATORS:
    print(
        f"  {indicator}: "
        f"{available_counts[indicator]} available, "
        f"{missing_counts[indicator]} missing"
    )

print("OUTPUT:", OUTPUT)
print("=" * 60)
print("STATUS: SQL generation completed")
print("=" * 60)
