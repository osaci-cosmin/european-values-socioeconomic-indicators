
# STEP 67.3.20 - Generate validated correlation SQL inserts
# Read the scientific evidence matrix without modifying it.
# No statistical coefficients are recalculated.

import csv
from decimal import Decimal
from pathlib import Path


# 1. Define project paths
ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs" / "tables"
    / "ESS11_scientific_evidence_matrix_v1.csv"
)

OUTPUT = (
    ROOT / "sql"
    / "67_18_insert_fact_correlation.sql"
)


# 2. Define the expected analytical scope
ESS_VARIABLES = {
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
}

ECONOMIC_VARIABLES = {
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
}

EXPECTED_COUNTRIES_N = {
    "gdp_pps_index": 28,
    "employment_rate": 26,
    "unemployment_rate": 26,
    "gini": 25,
    "material_deprivation_rate": 26,
    "inflation_rate": 27,
}

REQUIRED_COLUMNS = {
    "ess_variable",
    "economic_variable",
    "countries_n",
    "pearson_baseline",
    "spearman_baseline",
}


# 3. Validate numeric values
def parse_number(value, field_name):
    """Parse a required finite numeric value."""
    text = (value or "").strip()

    if not text:
        raise ValueError(
            f"Missing numeric value: {field_name}"
        )

    number = Decimal(text)

    if not number.is_finite():
        raise ValueError(
            f"Non-finite value: {field_name}={text}"
        )

    return number


# 4. Read the validated matrix
with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)
    columns = reader.fieldnames or []
    rows = list(reader)

if len(columns) != 72:
    raise ValueError(
        f"Expected 72 source columns, found {len(columns)}"
    )

missing_columns = REQUIRED_COLUMNS - set(columns)

if missing_columns:
    raise ValueError(
        f"Missing source columns: {sorted(missing_columns)}"
    )

if len(rows) != 36:
    raise ValueError(
        f"Expected 36 correlations, found {len(rows)}"
    )


# 5. Validate and prepare SQL records
sql_rows = []
observed_pairs = set()

for row in rows:

    ess_variable = row["ess_variable"].strip()
    economic_variable = row["economic_variable"].strip()

    if ess_variable not in ESS_VARIABLES:
        raise ValueError(
            f"Unknown ESS variable: {ess_variable}"
        )

    if economic_variable not in ECONOMIC_VARIABLES:
        raise ValueError(
            f"Unknown economic indicator: {economic_variable}"
        )

    key = (ess_variable, economic_variable)

    if key in observed_pairs:
        raise ValueError(
            f"Duplicate correlation pair: {key}"
        )

    observed_pairs.add(key)

    # Validate the pairwise number of countries.
    countries_number = parse_number(
        row["countries_n"],
        "countries_n"
    )

    if countries_number != countries_number.to_integral_value():
        raise ValueError(
            f"Non-integer country count: {key}"
        )

    countries_n = int(countries_number)

    if countries_n != EXPECTED_COUNTRIES_N[economic_variable]:
        raise ValueError(
            f"Unexpected country count: {key}, N={countries_n}"
        )

    # Read existing validated coefficients.
    pearson = parse_number(
        row["pearson_baseline"],
        "pearson_baseline"
    )

    spearman = parse_number(
        row["spearman_baseline"],
        "spearman_baseline"
    )

    # Correlation coefficients must be in [-1, 1].
    if not (-1 <= pearson <= 1):
        raise ValueError(
            f"Invalid Pearson coefficient: {key}"
        )

    if not (-1 <= spearman <= 1):
        raise ValueError(
            f"Invalid Spearman coefficient: {key}"
        )

    sql_rows.append(
        "    ("
        f"'{ess_variable}', "
        f"'{economic_variable}', "
        f"{countries_n}, "
        f"{format(pearson, 'f')}, "
        f"{format(spearman, 'f')}"
        ")"
    )


# 6. Verify complete pair coverage
expected_pairs = {
    (ess, economic)
    for ess in ESS_VARIABLES
    for economic in ECONOMIC_VARIABLES
}

if observed_pairs != expected_pairs:
    raise ValueError(
        "Incomplete ESS-Eurostat correlation coverage"
    )

if len(sql_rows) != 36:
    raise ValueError("Expected exactly 36 SQL records")


# 7. Generate the SQL file
sql = (
    "-- STEP 67.3.20 - Validated correlation inserts\n"
    "-- Source: ESS11_scientific_evidence_matrix_v1.csv\n"
    "-- Pearson and Spearman values are not recalculated.\n"
    "-- Execute once after reviewing the generated file.\n\n"

    "USE european_values_v1;\n\n"

    "INSERT INTO fact_correlation (\n"
    "    ess_variable,\n"
    "    economic_variable,\n"
    "    countries_n,\n"
    "    pearson_baseline,\n"
    "    spearman_baseline\n"
    ")\nVALUES\n"

    + ",\n".join(sql_rows)

    + ";\n\n"

    "SELECT COUNT(*) AS total_correlations\n"
    "FROM fact_correlation;\n"
)

OUTPUT.write_text(sql, encoding="utf-8")


# 8. Report results
print("=" * 60)
print("STEP 67.3.20 - CORRELATION SQL GENERATION")
print("=" * 60)

print("Source:", SOURCE.name)
print("ESS variables:", len(ESS_VARIABLES))
print("Economic indicators:", len(ECONOMIC_VARIABLES))
print("Correlation pairs:", len(observed_pairs))
print("SQL rows generated:", len(sql_rows))
print("OUTPUT:", OUTPUT)

print("=" * 60)
print("STATUS: SQL generation completed")
print("Validated scientific source was not modified.")
print("=" * 60)
