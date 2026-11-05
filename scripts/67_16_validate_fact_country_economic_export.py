
# STEP 67.3.17 - Independent MySQL-Eurostat validation
# Compare the MySQL export against the validated source CSV.
# Read-only: no existing project files are modified.

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "processed"
    / "ESS11_country_integrated_main_v1.csv"
)

MYSQL_EXPORT = (
    ROOT / "outputs" / "sql_validation"
    / "67_15_fact_country_economic_mysql_export.csv"
)

INDICATORS = (
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
)

# The fields to compare for every country-indicator pair.
FIELDS = {
    "indicator_value": "numeric",
    "context_year": "integer",
    "requested_eurostat_year": "integer",
    "eurostat_year": "integer",
    "reference_year_offset": "integer",
    "observation_match_status": "text",
    "source_cell_found": "boolean",
    "has_numeric_value": "boolean",
    "source_is_missing": "boolean",
    "source_has_status_flag": "boolean",
    "eurostat_status": "text",
}

TOLERANCE = Decimal("1e-12")


# 2. Read CSV files
def read_csv(path):
    """Read CSV records without modifying the file."""
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Invalid CSV header: {path.name}")

    return set(columns), rows


# 3. Normalize values for a fair comparison
def normalize(raw_value, field_type):
    """Preserve NULL and compare values by their real type."""
    value = (raw_value or "").strip()

    if value.upper() in ("", "NULL", "\\N"):
        return None

    if field_type == "text":
        return value

    if field_type == "boolean":
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
        raise ValueError(f"Invalid boolean: {value}")

    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid numeric value: {value}"
        ) from error

    if not number.is_finite():
        raise ValueError(f"Non-finite number: {value}")

    if field_type == "integer":
        if number != number.to_integral_value():
            raise ValueError(f"Invalid integer: {value}")
        return int(number)

    return number


# 4. Load and validate source structure
source_columns, source_rows = read_csv(SOURCE)
export_columns, export_rows = read_csv(MYSQL_EXPORT)

required_source = {"cntry", "context_year"}

for indicator in INDICATORS:
    required_source.add(indicator)

    for field in FIELDS:
        if field not in ("indicator_value", "context_year"):
            required_source.add(f"{indicator}__{field}")

required_export = {
    "cntry",
    "economic_variable",
    *FIELDS.keys(),
}

if not required_source.issubset(source_columns):
    raise ValueError(
        "Missing source columns: "
        f"{sorted(required_source - source_columns)}"
    )

if export_columns != required_export:
    raise ValueError(
        "MySQL export columns do not match the expected schema"
    )

if len(source_rows) != 29:
    raise ValueError("Expected exactly 29 source countries")

if len(export_rows) != 174:
    raise ValueError("Expected exactly 174 MySQL records")


# 5. Reconstruct the expected records from the original CSV
expected = {}

for row in source_rows:
    country = row["cntry"].strip()

    for indicator in INDICATORS:
        key = (country, indicator)

        if key in expected:
            raise ValueError(f"Duplicate source key: {key}")

        record = {}

        for field, field_type in FIELDS.items():
            if field == "indicator_value":
                source_column = indicator
            elif field == "context_year":
                source_column = "context_year"
            else:
                source_column = f"{indicator}__{field}"

            record[field] = normalize(
                row[source_column], field_type
            )

        expected[key] = record


# 6. Index exported MySQL records
actual = {}

for row in export_rows:
    key = (
        row["cntry"].strip(),
        row["economic_variable"].strip(),
    )

    if key in actual:
        raise ValueError(f"Duplicate MySQL key: {key}")

    actual[key] = {
        field: normalize(row[field], field_type)
        for field, field_type in FIELDS.items()
    }


# 7. Compare primary keys
if set(expected) != set(actual):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    raise AssertionError(
        f"Key mismatch. Missing={missing}, Extra={extra}"
    )


# 8. Compare every value and metadata field
max_abs_difference = Decimal("0")
field_comparisons = 0
numeric_comparisons = 0
available_values = 0
missing_values = 0

for key in sorted(expected):
    source = expected[key]
    mysql = actual[key]

    for field, field_type in FIELDS.items():
        source_value = source[field]
        mysql_value = mysql[field]

        if field_type == "numeric":
            if source_value is None or mysql_value is None:
                if source_value is not mysql_value:
                    raise AssertionError(
                        f"NULL mismatch: {key}, {field}"
                    )
            else:
                difference = abs(source_value - mysql_value)

                max_abs_difference = max(
                    max_abs_difference, difference
                )

                if difference > TOLERANCE:
                    raise AssertionError(
                        f"Numeric mismatch: {key}, {field}: "
                        f"{source_value} != {mysql_value}"
                    )

                numeric_comparisons += 1

        elif source_value != mysql_value:
            raise AssertionError(
                f"Metadata mismatch: {key}, {field}: "
                f"{source_value!r} != {mysql_value!r}"
            )

        field_comparisons += 1

    value = mysql["indicator_value"]

    if value is None:
        missing_values += 1
    else:
        available_values += 1

    if mysql["has_numeric_value"] != (value is not None):
        raise AssertionError(
            f"Inconsistent availability flag: {key}"
        )


# 9. Verify economic data coverage
if len(expected) != 174 or len(actual) != 174:
    raise AssertionError("Unexpected record count")

if available_values != 158 or missing_values != 16:
    raise AssertionError(
        "Unexpected numeric availability: "
        f"{available_values} available, "
        f"{missing_values} missing"
    )


# 10. Report results
print("=" * 60)
print("STEP 67.3.17 - INDEPENDENT MYSQL EUROSTAT VALIDATION")
print("=" * 60)

print("Source countries:", len(source_rows))
print("Economic indicators:", len(INDICATORS))
print("Expected records:", len(expected))
print("MySQL export records:", len(actual))
print("Primary key comparison: PASSED")
print("Fields compared per record:", len(FIELDS))
print("Total field comparisons:", field_comparisons)
print("Numeric values compared:", numeric_comparisons)
print("Available economic values:", available_values)
print("Missing economic values:", missing_values)
print(
    "Maximum absolute numeric difference:",
    f"{max_abs_difference:.3e}"
)

print("=" * 60)
print("STATUS: PASSED - ALL EUROSTAT RECORDS MATCH")
print("No project files were modified.")
print("=" * 60)
