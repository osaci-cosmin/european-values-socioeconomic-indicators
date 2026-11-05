
# STEP 67.3.8 — Generate SQL inserts from validated ESS country data
# Uses Python standard library only.
# Original project data is never modified.

import csv
import re
from decimal import Decimal
from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.csv"
)

OUTPUT = (
    ROOT
    / "sql"
    / "67_09_insert_fact_country_ess.sql"
)


# --------------------------------------------------
# 2. Define ESS variables and respondent counts
# --------------------------------------------------

# Each ESS variable is linked to the appropriate
# valid respondent count from the validated dataset.

VARIABLES = {
    "mean_social_trust": "social_trust_valid_n",
    "mean_left_right": "ideology_valid_n",
    "mean_openness": "human_values_valid_n",
    "mean_conservation": "human_values_valid_n",
    "mean_self_transcendence": "human_values_valid_n",
    "mean_self_enhancement": "human_values_valid_n",
}


# --------------------------------------------------
# 3. Convert CSV numeric values to SQL literals
# --------------------------------------------------

def sql_number(value, is_integer=False):
    """
    Convert CSV numeric text into a SQL numeric literal.
    Missing values become SQL NULL, never zero.
    """

    value = value.strip()

    if value == "":
        return "NULL"

    number = Decimal(value)

    if not number.is_finite():
        raise ValueError(
            f"Invalid numeric value: {value}"
        )

    if is_integer:
        if (
            number != number.to_integral_value()
            or number < 0
        ):
            raise ValueError(
                f"Invalid respondent count: {value}"
            )

        return str(int(number))

    return format(number, "f")


# --------------------------------------------------
# 4. Read the validated country dataset
# --------------------------------------------------

with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    source_rows = list(reader)

    # Check that every required column exists.

    required_columns = {"cntry"}

    for variable, count_column in VARIABLES.items():

        required_columns.update({
            variable,
            count_column,
            f"{variable}_se",
            f"{variable}_ci_lower",
            f"{variable}_ci_upper",
        })

    missing_columns = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing_columns:
        raise ValueError(
            f"Missing source columns: {missing_columns}"
        )


# --------------------------------------------------
# 5. Validate country coverage
# --------------------------------------------------

if len(source_rows) != 29:
    raise ValueError(
        f"Expected 29 countries, found {len(source_rows)}"
    )

country_codes = {
    row["cntry"].strip()
    for row in source_rows
}

if len(country_codes) != 29:
    raise ValueError(
        "Duplicate country codes detected"
    )


# --------------------------------------------------
# 6. Convert wide data into long SQL records
# --------------------------------------------------

sql_rows = []

for row in source_rows:

    country = row["cntry"].strip()

    # Validate two-letter ESS country codes.

    if not re.fullmatch(r"[A-Z]{2}", country):
        raise ValueError(
            f"Invalid country code: {country}"
        )

    for variable, count_column in VARIABLES.items():

        values = [
            f"'{country}'",
            f"'{variable}'",
            sql_number(row[variable]),
            sql_number(
                row[count_column],
                is_integer=True
            ),
            sql_number(
                row[f"{variable}_se"]
            ),
            sql_number(
                row[f"{variable}_ci_lower"]
            ),
            sql_number(
                row[f"{variable}_ci_upper"]
            ),
        ]

        sql_rows.append(
            "    (" + ", ".join(values) + ")"
        )


# --------------------------------------------------
# 7. Validate generated record count
# --------------------------------------------------

expected_rows = 29 * 6

if len(sql_rows) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, "
        f"generated {len(sql_rows)}"
    )


# --------------------------------------------------
# 8. Generate the SQL INSERT statement
# --------------------------------------------------

sql = (
    "-- STEP 67.3.8 — Generated ESS country inserts\n"
    "-- Source: ESS11_country_integrated_main_v1.csv\n"
    "-- Run once after reviewing the generated SQL.\n\n"

    "USE european_values_v1;\n\n"

    "INSERT INTO fact_country_ess (\n"
    "    cntry,\n"
    "    ess_variable,\n"
    "    estimate_value,\n"
    "    valid_respondents_n,\n"
    "    standard_error,\n"
    "    ci_lower,\n"
    "    ci_upper\n"
    ")\n"

    "VALUES\n"

    + ",\n".join(sql_rows)

    + ";\n\n"

    "-- Verify imported records\n"
    "SELECT COUNT(*) AS total_ess_observations\n"
    "FROM fact_country_ess;\n"
)


# --------------------------------------------------
# 9. Save SQL output (without modifying source data)
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT.write_text(
    sql,
    encoding="utf-8"
)


# --------------------------------------------------
# 10. Report generation results
# --------------------------------------------------

print("=" * 60)
print("STEP 67.3.8 — ESS SQL GENERATION")
print("=" * 60)

print("SOURCE:", SOURCE.name)
print("Countries:", len(source_rows))
print("ESS variables:", len(VARIABLES))
print("SQL rows generated:", len(sql_rows))
print("OUTPUT:", OUTPUT)

print("=" * 60)
print("STATUS: SQL generation completed")
print("=" * 60)
