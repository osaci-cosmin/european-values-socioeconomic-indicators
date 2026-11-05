
# STEP 67.3.11 - Independently validate MySQL ESS export
# Compare all imported ESS records against the validated CSV.
# Read-only validation: no project files are modified.

import csv
import math
from pathlib import Path


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "processed"
    / "ESS11_country_integrated_main_v1.csv"
)

MYSQL_EXPORT = (
    ROOT / "outputs" / "sql_validation"
    / "67_11_fact_country_ess_mysql_export.csv"
)


# --------------------------------------------------
# 2. Define the six ESS variables
# --------------------------------------------------

VARIABLES = {
    "mean_social_trust": "social_trust_valid_n",
    "mean_left_right": "ideology_valid_n",
    "mean_openness": "human_values_valid_n",
    "mean_conservation": "human_values_valid_n",
    "mean_self_transcendence": "human_values_valid_n",
    "mean_self_enhancement": "human_values_valid_n",
}

NUMERIC_FIELDS = (
    "estimate_value",
    "standard_error",
    "ci_lower",
    "ci_upper",
)


# --------------------------------------------------
# 3. Read CSV files
# --------------------------------------------------

def read_csv(path):
    """Read a CSV and return its header and records."""

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    return columns, rows


source_columns, source_rows = read_csv(SOURCE)
export_columns, export_rows = read_csv(MYSQL_EXPORT)


# --------------------------------------------------
# 4. Validate file structures
# --------------------------------------------------

required_source = {"cntry"}

for variable, count_column in VARIABLES.items():
    required_source.update({
        variable,
        count_column,
        f"{variable}_se",
        f"{variable}_ci_lower",
        f"{variable}_ci_upper",
    })

required_export = {
    "cntry",
    "ess_variable",
    "estimate_value",
    "valid_respondents_n",
    "standard_error",
    "ci_lower",
    "ci_upper",
}

assert required_source.issubset(source_columns), (
    "Missing columns in original source"
)

assert len(export_columns) == 7, (
    "Unexpected number of export columns"
)

assert set(export_columns) == required_export, (
    "Unexpected MySQL export schema"
)

assert len(source_rows) == 29, (
    "Expected 29 countries in source"
)

assert len(export_rows) == 174, (
    "Expected 174 rows in MySQL export"
)


# --------------------------------------------------
# 5. Reconstruct expected ESS records independently
# --------------------------------------------------

expected = {}

for row in source_rows:
    country = row["cntry"].strip()

    for variable, count_column in VARIABLES.items():
        key = (country, variable)

        assert key not in expected, (
            f"Duplicate source key: {key}"
        )

        expected[key] = {
            "estimate_value": row[variable],
            "valid_respondents_n": row[count_column],
            "standard_error": row[f"{variable}_se"],
            "ci_lower": row[f"{variable}_ci_lower"],
            "ci_upper": row[f"{variable}_ci_upper"],
        }


# --------------------------------------------------
# 6. Index the MySQL export
# --------------------------------------------------

actual = {}

for row in export_rows:
    key = (
        row["cntry"].strip(),
        row["ess_variable"].strip(),
    )

    assert key not in actual, (
        f"Duplicate MySQL key: {key}"
    )

    actual[key] = row

assert set(expected) == set(actual), (
    "Source and MySQL keys do not match"
)


# --------------------------------------------------
# 7. Compare all values
# --------------------------------------------------

def parse_number(value):
    """Preserve SQL NULL semantics during comparison."""

    if value is None:
        return None

    value = value.strip()

    if value.upper() in ("", "NULL", "\\N"):
        return None

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Invalid numeric value: {value}")

    return result


max_abs_difference = 0.0
numeric_comparisons = 0
count_comparisons = 0

for key in sorted(expected):
    source = expected[key]
    mysql = actual[key]

    # Respondent counts must match exactly.
    assert parse_number(
        source["valid_respondents_n"]
    ) == parse_number(
        mysql["valid_respondents_n"]
    ), f"Respondent count mismatch: {key}"

    count_comparisons += 1

    # Compare estimates, SE and confidence intervals.
    for field in NUMERIC_FIELDS:
        source_value = parse_number(source[field])
        mysql_value = parse_number(mysql[field])

        if source_value is None or mysql_value is None:
            assert source_value is mysql_value, (
                f"NULL mismatch: {key}, {field}"
            )
            continue

        difference = abs(source_value - mysql_value)

        max_abs_difference = max(
            max_abs_difference, difference
        )

        assert math.isclose(
            source_value,
            mysql_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ), (
            f"Numeric mismatch: {key}, {field}, "
            f"source={source_value}, mysql={mysql_value}"
        )

        numeric_comparisons += 1


# --------------------------------------------------
# 8. Report validation results
# --------------------------------------------------

print("=" * 60)
print("STEP 67.3.11 - INDEPENDENT MYSQL ESS VALIDATION")
print("=" * 60)

print("Source countries:", len(source_rows))
print("Expected ESS records:", len(expected))
print("MySQL export records:", len(actual))
print("Primary key comparison: PASSED")
print("Respondent counts checked:", count_comparisons)
print("Floating-point values checked:", numeric_comparisons)
print(
    "Maximum absolute difference:",
    f"{max_abs_difference:.3e}"
)

print("=" * 60)
print("STATUS: PASSED - ALL ESS RECORDS MATCH")
print("No source or database files were modified.")
print("=" * 60)
