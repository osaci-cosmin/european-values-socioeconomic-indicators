
# STEP 67.3.33 - Generate Schwartz exclusion sensitivity SQL
# Read-only access to the validated Scientific Evidence Matrix V1.
# Generate 72 SQL records without recalculating correlations.

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

OUTPUT = (
    ROOT / "sql"
    / "67_27_insert_fact_schwartz_exclusion.sql"
)

TOLERANCE = Decimal("1e-12")


# 2. Define the three scenarios
SCENARIOS = {
    "exclude_EE": ("EE",),
    "exclude_GB": ("GB",),
    "exclude_EE_GB": ("EE", "GB"),
}

SCHWARTZ_VARIABLES = {
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
}


# 3. Define the 11 scenario fields
FIELD_TYPES = {
    "actual_exclusions": "text",
    "scenario_n": "integer",
    "removed_n": "integer",
    "pearson_scenario": "number",
    "pearson_change": "number",
    "pearson_abs_change": "number",
    "pearson_sign_flip": "boolean",
    "spearman_scenario": "number",
    "spearman_change": "number",
    "spearman_abs_change": "number",
    "spearman_sign_flip": "boolean",
}

SQL_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "scenario_code",
    *FIELD_TYPES,
]


# 4. Conversion helpers
def parse_decimal(value, field):
    """Validate finite numeric values without using float."""

    text = value.strip()

    if not text:
        raise ValueError(f"Missing numeric value: {field}")

    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid numeric value in {field}: {text!r}"
        ) from error

    if not number.is_finite():
        raise ValueError(f"Non-finite value: {field}")

    return number


def parse_integer(value, field):
    """Reject fractional and negative integer values."""

    number = parse_decimal(value, field)

    if number != number.to_integral_value():
        raise ValueError(f"Non-integer value: {field}")

    result = int(number)

    if result < 0:
        raise ValueError(f"Negative integer: {field}")

    return result


def parse_boolean(value, field):
    """Accept only explicitly valid boolean representations."""

    text = value.strip().lower()

    if text in ("true", "1"):
        return True

    if text in ("false", "0"):
        return False

    raise ValueError(
        f"Invalid boolean in {field}: {value!r}"
    )


def sql_text(value):
    """Quote text safely for MySQL."""

    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace("'", "''")

    return f"'{escaped}'"


def sql_literal(value, field_type, field):
    """Convert a validated CSV field into an SQL literal."""

    if field_type == "text":
        return sql_text(value)

    if field_type == "boolean":
        return "1" if parse_boolean(value, field) else "0"

    if field_type == "integer":
        number = parse_integer(value, field)

        if number > 65535:
            raise ValueError(f"Integer too large: {field}")

        return str(number)

    if field_type == "number":
        return format(parse_decimal(value, field), "f")

    raise ValueError(f"Unknown SQL type: {field_type}")


# 5. Read the validated source
with SOURCE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:

    reader = csv.DictReader(file)
    columns = reader.fieldnames or []
    rows = list(reader)


if len(columns) != 72 or len(set(columns)) != 72:
    raise AssertionError("Expected 72 unique source columns")

if len(rows) != 36:
    raise AssertionError("Expected 36 source correlations")

if any(
    None in row or any(value is None for value in row.values())
    for row in rows
):
    raise AssertionError("Malformed source CSV records")


# 6. Verify all scenario columns exist
scenario_columns = [
    column for column in columns
    if column.startswith("exclude_")
]

if len(scenario_columns) != 33:
    raise AssertionError("Expected 33 scenario columns")

expected_scenario_columns = {
    f"{scenario}__{field}"
    for scenario in SCENARIOS
    for field in FIELD_TYPES
}

if set(scenario_columns) != expected_scenario_columns:
    raise AssertionError("Unexpected scenario column names")


# 7. Validate applicability and source keys
source_pairs = set()
applicable_rows = []
non_applicable_rows = []

for row in rows:

    key = (
        row["ess_variable"].strip(),
        row["economic_variable"].strip(),
    )

    if not all(key) or key in source_pairs:
        raise AssertionError(f"Invalid or duplicate key: {key}")

    source_pairs.add(key)

    applicable = parse_boolean(
        row["schwartz_sensitivity_applicable"],
        "schwartz_sensitivity_applicable",
    )

    expected_applicable = key[0] in SCHWARTZ_VARIABLES

    if applicable != expected_applicable:
        raise AssertionError(
            f"Incorrect Schwartz applicability: {key}"
        )

    if applicable:
        applicable_rows.append(row)
    else:
        non_applicable_rows.append(row)

        # Non-Schwartz scenarios must be inapplicable,
        # not represented as numeric zero.
        if any(row[field].strip() for field in scenario_columns):
            raise AssertionError(
                f"Unexpected non-Schwartz scenario data: {key}"
            )


if len(source_pairs) != 36:
    raise AssertionError("Expected 36 unique source pairs")

if len(applicable_rows) != 24:
    raise AssertionError("Expected 24 Schwartz pairs")

if len(non_applicable_rows) != 12:
    raise AssertionError("Expected 12 non-Schwartz pairs")


# 8. Generate and validate each Schwartz scenario
sql_rows = []
exclusion_counts = Counter()
scenario_counts = Counter()

for row in sorted(
    applicable_rows,
    key=lambda item: (
        item["ess_variable"],
        item["economic_variable"],
    ),
):

    ess = row["ess_variable"].strip()
    economic = row["economic_variable"].strip()
    key = (ess, economic)

    included_countries = row["included_countries"].split("|")

    if len(included_countries) != len(set(included_countries)):
        raise AssertionError(f"Duplicate country codes: {key}")

    baseline_n = parse_integer(
        row["countries_n"], "countries_n"
    )

    if len(included_countries) != baseline_n:
        raise AssertionError(f"Baseline country mismatch: {key}")

    baseline_pearson = parse_decimal(
        row["pearson_baseline"], "pearson_baseline"
    )

    baseline_spearman = parse_decimal(
        row["spearman_baseline"], "spearman_baseline"
    )

    for scenario, requested_exclusions in SCENARIOS.items():

        values = [
            sql_text(ess),
            sql_text(economic),
            sql_text(scenario),
        ]

        # Preserve each of the 11 validated scenario fields.
        data = {
            field: row[f"{scenario}__{field}"]
            for field in FIELD_TYPES
        }

        actual_exclusions = data["actual_exclusions"]

        # Independently determine which requested countries
        # actually belonged to the baseline sample.
        expected_removed = [
            country
            for country in requested_exclusions
            if country in included_countries
        ]

        expected_exclusions = "|".join(expected_removed)

        if actual_exclusions != expected_exclusions:
            raise AssertionError(
                f"Effective exclusions mismatch: {key}, {scenario}"
            )

        removed_n = parse_integer(
            data["removed_n"], "removed_n"
        )

        scenario_n = parse_integer(
            data["scenario_n"], "scenario_n"
        )

        if removed_n != len(expected_removed):
            raise AssertionError(
                f"Removed country count mismatch: {key}, {scenario}"
            )

        if scenario_n != baseline_n - removed_n:
            raise AssertionError(
                f"Scenario sample size mismatch: {key}, {scenario}"
            )

        if not 3 <= scenario_n <= 29:
            raise AssertionError(
                f"Invalid scenario sample size: {key}, {scenario}"
            )

        # Check that the published changes are consistent
        # with the original validated baseline coefficients.
        for method, baseline in (
            ("pearson", baseline_pearson),
            ("spearman", baseline_spearman),
        ):

            coefficient = parse_decimal(
                data[f"{method}_scenario"],
                f"{method}_scenario",
            )

            change = parse_decimal(
                data[f"{method}_change"],
                f"{method}_change",
            )

            absolute_change = parse_decimal(
                data[f"{method}_abs_change"],
                f"{method}_abs_change",
            )

            sign_flip = parse_boolean(
                data[f"{method}_sign_flip"],
                f"{method}_sign_flip",
            )

            if not -1 <= coefficient <= 1:
                raise AssertionError(
                    f"Invalid correlation: {key}, {scenario}"
                )

            if abs((coefficient - baseline) - change) > TOLERANCE:
                raise AssertionError(
                    f"Coefficient change mismatch: "
                    f"{key}, {scenario}, {method}"
                )

            if abs(abs(change) - absolute_change) > TOLERANCE:
                raise AssertionError(
                    f"Absolute change mismatch: "
                    f"{key}, {scenario}, {method}"
                )

            # Strict sign reversal requires opposite,
            # nonzero signs.
            expected_flip = baseline * coefficient < 0

            if sign_flip != expected_flip:
                raise AssertionError(
                    f"Sign flip mismatch: "
                    f"{key}, {scenario}, {method}"
                )

        for field, field_type in FIELD_TYPES.items():
            values.append(
                sql_literal(
                    data[field],
                    field_type,
                    f"{scenario}__{field}",
                )
            )

        if len(values) != 14:
            raise AssertionError("Expected 14 SQL values")

        sql_rows.append("    (" + ", ".join(values) + ")")

        scenario_counts[scenario] += 1
        exclusion_counts[
            (scenario, actual_exclusions)
        ] += 1


# 9. Check expected scenario coverage
if len(sql_rows) != 72:
    raise AssertionError("Expected exactly 72 SQL records")

if any(scenario_counts[scenario] != 24 for scenario in SCENARIOS):
    raise AssertionError("Expected 24 rows per scenario")

expected_exclusion_counts = {
    ("exclude_EE", "EE"): 24,
    ("exclude_GB", "GB"): 4,
    ("exclude_GB", ""): 20,
    ("exclude_EE_GB", "EE|GB"): 4,
    ("exclude_EE_GB", "EE"): 20,
}

if dict(exclusion_counts) != expected_exclusion_counts:
    raise AssertionError(
        f"Unexpected effective exclusion coverage: {exclusion_counts}"
    )


# 10. Generate the INSERT file
column_sql = ",\n".join(
    f"    {column}" for column in SQL_COLUMNS
)

sql = (
    "-- STEP 67.3.33 - Generated Schwartz exclusion scenarios\n"
    "-- Source: ESS11_scientific_evidence_matrix_v1.csv\n"
    "-- Results preserved, not recalculated.\n"
    "-- Execute INSERT only once.\n\n"
    "USE european_values_v1;\n\n"
    "INSERT INTO fact_schwartz_exclusion (\n"
    + column_sql
    + "\n)\nVALUES\n"
    + ",\n".join(sql_rows)
    + ";\n\n"
    "SELECT COUNT(*) AS total_exclusion_records\n"
    "FROM fact_schwartz_exclusion;\n\n"
    "SELECT scenario_code, actual_exclusions,\n"
    "       COUNT(*) AS records_n\n"
    "FROM fact_schwartz_exclusion\n"
    "GROUP BY scenario_code, actual_exclusions\n"
    "ORDER BY scenario_code, actual_exclusions;\n"
)

# Never overwrite a previously generated SQL file.
if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}"
    )

OUTPUT.write_text(sql, encoding="utf-8")


# 11. Report results
print("=" * 60)
print("STEP 67.3.33 - SCHWARTZ EXCLUSION SQL GENERATION")
print("=" * 60)
print("Source:", SOURCE.name)
print("Source correlations:", len(rows))
print("Applicable Schwartz pairs:", len(applicable_rows))
print("Non-Schwartz pairs excluded:", len(non_applicable_rows))
print("Scenario types:", len(SCENARIOS))
print("SQL columns:", len(SQL_COLUMNS))
print("SQL rows generated:", len(sql_rows))

for scenario in SCENARIOS:
    print(f"{scenario}: {scenario_counts[scenario]} rows")

print("Effective exclusion distributions:")

for key, count in sorted(exclusion_counts.items()):
    print(f"  {key}: {count}")

print("OUTPUT:", OUTPUT)
print("=" * 60)
print("STATUS: SQL generation completed")
print("Validated scientific source was not modified.")
print("=" * 60)
