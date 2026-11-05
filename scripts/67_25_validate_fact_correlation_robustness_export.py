
# STEP 67.3.30 - Independent correlation robustness validation
# Compare the MySQL export with the validated scientific matrix.
# Read-only: this script does not modify any project files.

import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


# 1. Project paths

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs" / "tables"
    / "ESS11_scientific_evidence_matrix_v1.csv"
)

MYSQL_EXPORT = (
    ROOT / "outputs" / "sql_validation"
    / "67_24_fact_correlation_robustness_mysql_export.csv"
)

KEY_COLUMNS = ("ess_variable", "economic_variable")

BASELINE_COLUMNS = {
    "ess_variable",
    "economic_variable",
    "countries_n",
    "pearson_baseline",
    "spearman_baseline",
}

TOLERANCE = Decimal("1e-12")


# 2. Define field types independently of the SQL generator

TEXT_FIELDS = {
    "included_countries",
    "common_countries",
    "removed_from_primary_sample",
    "pearson_loo_largest_change_countries",
    "pearson_loo_sign_flip_countries",
    "spearman_loo_largest_change_countries",
    "spearman_loo_sign_flip_countries",
}

INTEGER_FIELDS = {
    "common_countries_n",
    "switched_context_countries_n",
    "changed_numeric_values_n",
    "pearson_loo_sign_flip_n",
    "spearman_loo_sign_flip_n",
}

BOOLEAN_FIELDS = {
    "pearson_spearman_same_strict_sign",
    "pearson_loo_any_sign_flip",
    "pearson_temporal_sign_flip",
    "spearman_loo_any_sign_flip",
    "spearman_temporal_sign_flip",
}

NUMBER_FIELDS = {
    "pearson_minus_spearman",
    "pearson_loo_min",
    "pearson_loo_max",
    "pearson_loo_max_abs_change",
    "pearson_primary_common",
    "pearson_sensitivity_common",
    "pearson_temporal_change",
    "pearson_sample_change",
    "spearman_loo_min",
    "spearman_loo_max",
    "spearman_loo_max_abs_change",
    "spearman_primary_common",
    "spearman_sensitivity_common",
    "spearman_temporal_change",
    "spearman_sample_change",
}

FIELD_GROUPS = {
    "text": TEXT_FIELDS,
    "integer": INTEGER_FIELDS,
    "boolean": BOOLEAN_FIELDS,
    "number": NUMBER_FIELDS,
}

all_typed_fields = [
    field
    for fields in FIELD_GROUPS.values()
    for field in fields
]

if len(all_typed_fields) != len(set(all_typed_fields)):
    raise AssertionError("Overlapping field categories")

if len(all_typed_fields) != 32:
    raise AssertionError("Expected exactly 32 robustness fields")


# 3. Read CSV files without modifying them

def read_csv(path):
    """Load CSV data and validate its basic structure."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        columns = reader.fieldnames or []
        rows = list(reader)

    if not columns:
        raise AssertionError(f"Empty CSV header: {path.name}")

    if len(columns) != len(set(columns)):
        raise AssertionError(f"Duplicate columns: {path.name}")

    for row in rows:
        if None in row or any(
            value is None for value in row.values()
        ):
            raise AssertionError(
                f"Malformed CSV row: {path.name}"
            )

    return columns, rows


source_columns, source_rows = read_csv(SOURCE)
export_columns, export_rows = read_csv(MYSQL_EXPORT)


# 4. Validate source and export schemas

if len(source_columns) != 72:
    raise AssertionError(
        f"Expected 72 source columns, found {len(source_columns)}"
    )

robustness_columns = [
    field
    for field in source_columns[:37]
    if field not in BASELINE_COLUMNS
]

if len(robustness_columns) != 32:
    raise AssertionError("Expected 32 source robustness fields")

if set(robustness_columns) != set(all_typed_fields):
    raise AssertionError("Unexpected robustness fields")

expected_export_columns = (
    list(KEY_COLUMNS) + robustness_columns
)

if export_columns != expected_export_columns:
    raise AssertionError(
        "MySQL export schema or column order does not match "
        "the validated source"
    )

if len(source_rows) != 36:
    raise AssertionError("Expected 36 source records")

if len(export_rows) != 36:
    raise AssertionError("Expected 36 MySQL export records")


# 5. Normalize values according to their semantic types

def normalize(value, field):
    """Normalize equivalent representations without losing meaning."""

    if field in TEXT_FIELDS:
        # Preserve text exactly, including empty strings.
        # An empty country list is not the same as NULL.
        return value

    text = value.strip()

    if field in BOOLEAN_FIELDS:

        if text.lower() in ("true", "1"):
            return True

        if text.lower() in ("false", "0"):
            return False

        raise ValueError(
            f"Invalid boolean in {field}: {value!r}"
        )

    if not text or text.upper() == "NULL":
        raise ValueError(
            f"Missing numeric value in {field}"
        )

    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid number in {field}: {value!r}"
        ) from error

    if not number.is_finite():
        raise ValueError(
            f"Non-finite number in {field}"
        )

    if field in INTEGER_FIELDS:

        if number != number.to_integral_value():
            raise ValueError(
                f"Non-integer value in {field}"
            )

        return int(number)

    if field in NUMBER_FIELDS:
        return number

    raise ValueError(f"Unknown field type: {field}")


# 6. Build independently keyed records

def build_records(rows):
    """Index records using their ESS-Eurostat composite keys."""

    records = {}

    for row in rows:

        key = tuple(
            row[column].strip()
            for column in KEY_COLUMNS
        )

        if not all(key):
            raise AssertionError("Empty correlation key")

        if key in records:
            raise AssertionError(
                f"Duplicate correlation key: {key}"
            )

        records[key] = {
            field: normalize(row[field], field)
            for field in robustness_columns
        }

    return records


expected = build_records(source_rows)
actual = build_records(export_rows)


# 7. Validate exact primary-key coverage

if set(expected) != set(actual):

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    raise AssertionError(
        f"Primary-key mismatch: missing={missing}, extra={extra}"
    )

ess_variables = {
    key[0] for key in expected
}

economic_variables = {
    key[1] for key in expected
}

expected_pairs = {
    (ess, economic)
    for ess in ess_variables
    for economic in economic_variables
}

if (
    len(ess_variables) != 6
    or len(economic_variables) != 6
    or set(expected) != expected_pairs
):
    raise AssertionError(
        "Incomplete 6 x 6 correlation coverage"
    )


# 8. Compare all 1,152 robustness values

comparison_counts = Counter()
maximum_difference = Decimal("0")

for key in sorted(expected):

    source_record = expected[key]
    mysql_record = actual[key]

    for field in robustness_columns:

        source_value = source_record[field]
        mysql_value = mysql_record[field]

        if field in NUMBER_FIELDS:

            difference = abs(source_value - mysql_value)

            maximum_difference = max(
                maximum_difference,
                difference
            )

            if difference > TOLERANCE:
                raise AssertionError(
                    f"Numeric mismatch: {key}, {field}: "
                    f"{source_value} != {mysql_value}"
                )

            comparison_counts["number"] += 1

        elif field in INTEGER_FIELDS:

            if source_value != mysql_value:
                raise AssertionError(
                    f"Integer mismatch: {key}, {field}"
                )

            comparison_counts["integer"] += 1

        elif field in BOOLEAN_FIELDS:

            if source_value != mysql_value:
                raise AssertionError(
                    f"Boolean mismatch: {key}, {field}"
                )

            comparison_counts["boolean"] += 1

        elif field in TEXT_FIELDS:

            if source_value != mysql_value:
                raise AssertionError(
                    f"Text mismatch: {key}, {field}: "
                    f"{source_value!r} != {mysql_value!r}"
                )

            comparison_counts["text"] += 1


# 9. Verify the number of comparisons

expected_counts = {
    "number": 540,
    "integer": 180,
    "boolean": 180,
    "text": 252,
}

if dict(comparison_counts) != expected_counts:
    raise AssertionError(
        f"Unexpected comparison counts: {comparison_counts}"
    )

total_comparisons = sum(comparison_counts.values())

if total_comparisons != 1152:
    raise AssertionError("Expected 1,152 comparisons")


# 10. Report validation results

if maximum_difference == 0:
    difference_display = "0.000e+00"
else:
    difference_display = f"{maximum_difference:.3e}"

print("=" * 60)
print("STEP 67.3.30 - INDEPENDENT ROBUSTNESS VALIDATION")
print("=" * 60)

print("Source records:", len(expected))
print("MySQL export records:", len(actual))
print("Primary key comparison: PASSED")
print("Robustness fields per record:", len(robustness_columns))
print("Numeric comparisons:", comparison_counts["number"])
print("Integer comparisons:", comparison_counts["integer"])
print("Boolean comparisons:", comparison_counts["boolean"])
print("Text comparisons:", comparison_counts["text"])
print("Total field comparisons:", total_comparisons)

print(
    "Maximum absolute numeric difference:",
    difference_display
)

print("=" * 60)
print("STATUS: PASSED - ALL ROBUSTNESS RECORDS MATCH")
print("No project files were modified.")
print("=" * 60)
