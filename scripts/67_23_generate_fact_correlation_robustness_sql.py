
# STEP 67.3.27 - Generate correlation robustness SQL
# Read validated results without recalculating coefficients.
# Preserve empty country lists as empty strings.

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs" / "tables"
    / "ESS11_scientific_evidence_matrix_v1.csv"
)

OUTPUT = (
    ROOT / "sql"
    / "67_23_insert_fact_correlation_robustness.sql"
)


# 2. Define the 32 robustness fields and SQL value types.
# The dictionary order matches the validated matrix.

FIELD_TYPES = {
    "included_countries": "text",
    "common_countries_n": "integer",
    "common_countries": "text",
    "removed_from_primary_sample": "text",
    "switched_context_countries_n": "integer",
    "changed_numeric_values_n": "integer",
    "pearson_spearman_same_strict_sign": "boolean",
    "pearson_minus_spearman": "number",

    "pearson_loo_min": "number",
    "pearson_loo_max": "number",
    "pearson_loo_max_abs_change": "number",
    "pearson_loo_largest_change_countries": "text",
    "pearson_loo_sign_flip_n": "integer",
    "pearson_loo_sign_flip_countries": "text",
    "pearson_loo_any_sign_flip": "boolean",
    "pearson_primary_common": "number",
    "pearson_sensitivity_common": "number",
    "pearson_temporal_change": "number",
    "pearson_sample_change": "number",
    "pearson_temporal_sign_flip": "boolean",

    "spearman_loo_min": "number",
    "spearman_loo_max": "number",
    "spearman_loo_max_abs_change": "number",
    "spearman_loo_largest_change_countries": "text",
    "spearman_loo_sign_flip_n": "integer",
    "spearman_loo_sign_flip_countries": "text",
    "spearman_loo_any_sign_flip": "boolean",
    "spearman_primary_common": "number",
    "spearman_sensitivity_common": "number",
    "spearman_temporal_change": "number",
    "spearman_sample_change": "number",
    "spearman_temporal_sign_flip": "boolean",
}

KEY_COLUMNS = ("ess_variable", "economic_variable")

BASELINE_COLUMNS = {
    "ess_variable",
    "economic_variable",
    "countries_n",
    "pearson_baseline",
    "spearman_baseline",
}


# 3. Convert CSV values to valid SQL literals.
def sql_value(raw_value, value_type):
    """Preserve numeric precision, booleans and empty text."""
    value = (raw_value or "").strip()

    if value_type == "text":
        # Empty text is meaningful for country lists.
        return "'" + value.replace("'", "''") + "'"

    if value_type == "boolean":
        if value.lower() in ("true", "1"):
            return "1"
        if value.lower() in ("false", "0"):
            return "0"
        raise ValueError(f"Invalid boolean: {value!r}")

    if not value:
        raise ValueError(
            "Missing value in a required numeric field"
        )

    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid numeric value: {value!r}"
        ) from error

    if not number.is_finite():
        raise ValueError(
            f"Non-finite numeric value: {value!r}"
        )

    if value_type == "integer":
        if number != number.to_integral_value():
            raise ValueError(
                f"Expected integer, received: {value}"
            )

        integer = int(number)

        if integer < 0 or integer > 65535:
            raise ValueError(
                f"Integer outside SMALLINT UNSIGNED: {integer}"
            )

        return str(integer)

    if value_type == "number":
        return format(number, "f")

    raise ValueError(f"Unknown field type: {value_type}")


# 4. Read the original validated matrix.
with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:
    reader = csv.DictReader(file)
    source_columns = reader.fieldnames or []
    rows = list(reader)


# 5. Validate schema and record count.
if len(source_columns) != 72:
    raise AssertionError(
        f"Expected 72 columns, found {len(source_columns)}"
    )

if len(set(source_columns)) != 72:
    raise AssertionError("Duplicate source column names")

if len(rows) != 36:
    raise AssertionError(
        f"Expected 36 rows, found {len(rows)}"
    )

robustness_columns = [
    name
    for name in source_columns[:37]
    if name not in BASELINE_COLUMNS
]

if robustness_columns != list(FIELD_TYPES):
    raise AssertionError(
        "Robustness field names or order do not match "
        "the validated source schema"
    )

if len(robustness_columns) != 32:
    raise AssertionError("Expected 32 robustness fields")


# 6. Validate keys and generate SQL records.
observed_pairs = set()
sql_rows = []

for row in rows:
    ess = row["ess_variable"].strip()
    economic = row["economic_variable"].strip()

    if not ess or not economic:
        raise AssertionError("Empty correlation key")

    key = (ess, economic)

    if key in observed_pairs:
        raise AssertionError(f"Duplicate correlation: {key}")

    observed_pairs.add(key)

    # Validate country coverage without changing it.
    country_count = int(
        sql_value(row["countries_n"], "integer")
    )

    included_countries = [
        code for code in row["included_countries"].split("|")
        if code
    ]

    if len(included_countries) != country_count:
        raise AssertionError(
            f"Country coverage mismatch: {key}"
        )

    common_count = int(
        sql_value(row["common_countries_n"], "integer")
    )

    common_countries = [
        code for code in row["common_countries"].split("|")
        if code
    ]

    if len(common_countries) != common_count:
        raise AssertionError(
            f"Common country coverage mismatch: {key}"
        )

    # Build the two key values.
    values = [
        sql_value(ess, "text"),
        sql_value(economic, "text"),
    ]

    # Append the 32 robustness values in source order.
    for field, field_type in FIELD_TYPES.items():
        values.append(
            sql_value(row[field], field_type)
        )

    if len(values) != 34:
        raise AssertionError(
            f"Expected 34 SQL values for {key}"
        )

    sql_rows.append(
        "    (" + ", ".join(values) + ")"
    )


# 7. Validate complete 6 x 6 coverage.
ess_variables = {key[0] for key in observed_pairs}
economic_variables = {key[1] for key in observed_pairs}

expected_pairs = {
    (ess, economic)
    for ess in ess_variables
    for economic in economic_variables
}

if (
    len(ess_variables) != 6
    or len(economic_variables) != 6
    or observed_pairs != expected_pairs
):
    raise AssertionError(
        "Incomplete ESS-Eurostat correlation coverage"
    )


# 8. Build the SQL INSERT statement.
sql_columns = list(KEY_COLUMNS) + list(FIELD_TYPES)

column_section = ",\n".join(
    f"    {column}" for column in sql_columns
)

sql = (
    "-- STEP 67.3.27 - Generated correlation robustness data\n"
    "-- Source: ESS11_scientific_evidence_matrix_v1.csv\n"
    "-- Results are preserved, not recalculated.\n"
    "-- Run the INSERT only once.\n\n"

    "USE european_values_v1;\n\n"

    "INSERT INTO fact_correlation_robustness (\n"
    + column_section
    + "\n)\nVALUES\n"
    + ",\n".join(sql_rows)
    + ";\n\n"

    "SELECT COUNT(*) AS total_robustness_records\n"
    "FROM fact_correlation_robustness;\n"
)


# 9. Create only the new generated SQL file.
if OUTPUT.exists():
    raise FileExistsError(
        f"Generated SQL already exists: {OUTPUT}"
    )

OUTPUT.write_text(sql, encoding="utf-8")


# 10. Report results.
print("=" * 60)
print("STEP 67.3.27 - CORRELATION ROBUSTNESS SQL GENERATION")
print("=" * 60)

print("Source:", SOURCE.name)
print("Source rows:", len(rows))
print("Robustness columns:", len(FIELD_TYPES))
print("SQL columns:", len(sql_columns))
print("Unique correlation pairs:", len(observed_pairs))
print("SQL rows generated:", len(sql_rows))
print("OUTPUT:", OUTPUT)

print("=" * 60)
print("STATUS: SQL generation completed")
print("Validated scientific source was not modified.")
print("=" * 60)
