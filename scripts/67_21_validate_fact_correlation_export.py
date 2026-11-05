
# STEP 67.3.24 - Independent correlation validation
# Compare MySQL baseline correlations against the validated matrix.
# Read-only: no source, export or database modifications.

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


# 1. Define project paths
ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs" / "tables"
    / "ESS11_scientific_evidence_matrix_v1.csv"
)

MYSQL_EXPORT = (
    ROOT / "outputs" / "sql_validation"
    / "67_20_fact_correlation_mysql_export.csv"
)

KEY_COLUMNS = ("ess_variable", "economic_variable")

NUMERIC_COLUMNS = (
    "countries_n",
    "pearson_baseline",
    "spearman_baseline",
)

EXPECTED_EXPORT_COLUMNS = set(KEY_COLUMNS + NUMERIC_COLUMNS)

TOLERANCE = Decimal("1e-12")


# 2. Read CSV files
def read_csv(path):
    """Read CSV records while preserving original text values."""
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Invalid CSV header: {path.name}")

    return columns, rows


# 3. Convert numeric text to Decimal
def parse_number(value, field):
    """Reject missing, invalid or non-finite numeric values."""
    value = (value or "").strip()

    if not value:
        raise ValueError(f"Missing value in {field}")

    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid numeric value in {field}: {value}"
        ) from error

    if not result.is_finite():
        raise ValueError(f"Non-finite value in {field}")

    return result


# 4. Load the validated source and MySQL export
source_columns, source_rows = read_csv(SOURCE)
export_columns, export_rows = read_csv(MYSQL_EXPORT)

if len(source_columns) != 72:
    raise AssertionError("Source must contain 72 columns")

if not EXPECTED_EXPORT_COLUMNS.issubset(source_columns):
    raise AssertionError("Missing source correlation columns")

if set(export_columns) != EXPECTED_EXPORT_COLUMNS:
    raise AssertionError("Unexpected MySQL export schema")

if len(export_columns) != 5:
    raise AssertionError("Expected exactly 5 export columns")

if len(source_rows) != 36:
    raise AssertionError("Expected 36 source correlations")

if len(export_rows) != 36:
    raise AssertionError("Expected 36 MySQL correlations")


# 5. Build keyed records independently
def build_records(rows):
    """Index rows by ESS variable and economic indicator."""
    records = {}

    for row in rows:
        key = tuple(row[column].strip() for column in KEY_COLUMNS)

        if not all(key):
            raise AssertionError(f"Empty correlation key: {key}")

        if key in records:
            raise AssertionError(f"Duplicate key: {key}")

        records[key] = {
            column: parse_number(row[column], column)
            for column in NUMERIC_COLUMNS
        }

    return records


expected = build_records(source_rows)
actual = build_records(export_rows)


# 6. Validate exact key coverage
if set(expected) != set(actual):
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))

    raise AssertionError(
        f"Key mismatch. Missing={missing}, Extra={unexpected}"
    )

ess_variables = {key[0] for key in expected}
economic_variables = {key[1] for key in expected}

if len(ess_variables) != 6 or len(economic_variables) != 6:
    raise AssertionError("Expected 6 ESS and 6 economic variables")

expected_pairs = {
    (ess, economic)
    for ess in ess_variables
    for economic in economic_variables
}

if set(expected) != expected_pairs:
    raise AssertionError("Incomplete 6 x 6 correlation coverage")


# 7. Compare sample sizes and correlation coefficients
sample_comparisons = 0
pearson_comparisons = 0
spearman_comparisons = 0
max_abs_difference = Decimal("0")

for key in sorted(expected):
    source = expected[key]
    mysql = actual[key]

    # Country counts must be exact integers.
    source_n = source["countries_n"]
    mysql_n = mysql["countries_n"]

    if source_n != source_n.to_integral_value():
        raise AssertionError(f"Invalid source N: {key}")

    if mysql_n != mysql_n.to_integral_value():
        raise AssertionError(f"Invalid MySQL N: {key}")

    if source_n != mysql_n:
        raise AssertionError(
            f"Sample size mismatch: {key}"
        )

    sample_comparisons += 1

    if not 3 <= source_n <= 29:
        raise AssertionError(f"Unexpected sample size: {key}")

    # Compare Pearson and Spearman independently.
    for method in ("pearson_baseline", "spearman_baseline"):
        source_value = source[method]
        mysql_value = mysql[method]

        if not (-1 <= source_value <= 1):
            raise AssertionError(f"Invalid source coefficient: {key}")

        if not (-1 <= mysql_value <= 1):
            raise AssertionError(f"Invalid MySQL coefficient: {key}")

        difference = abs(source_value - mysql_value)

        max_abs_difference = max(
            max_abs_difference, difference
        )

        if difference > TOLERANCE:
            raise AssertionError(
                f"Coefficient mismatch: {key}, {method}, "
                f"source={source_value}, mysql={mysql_value}"
            )

        if method == "pearson_baseline":
            pearson_comparisons += 1
        else:
            spearman_comparisons += 1


# 8. Report independent validation
print("=" * 60)
print("STEP 67.3.24 - INDEPENDENT CORRELATION VALIDATION")
print("=" * 60)

print("Source correlations:", len(expected))
print("MySQL correlations:", len(actual))
print("Primary key comparison: PASSED")
print("Country sample sizes checked:", sample_comparisons)
print("Pearson coefficients checked:", pearson_comparisons)
print("Spearman coefficients checked:", spearman_comparisons)

print(
    "Total numeric comparisons:",
    sample_comparisons
    + pearson_comparisons
    + spearman_comparisons
)

print(
    "Maximum absolute coefficient difference:",
    f"{max_abs_difference:.3e}"
)

print("=" * 60)
print("STATUS: PASSED - ALL CORRELATIONS MATCH")
print("No project files were modified.")
print("=" * 60)
