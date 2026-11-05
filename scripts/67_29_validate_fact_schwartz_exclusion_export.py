
# STEP 67.3.36 - Independent Schwartz exclusion validation
# Compare the validated Scientific Evidence Matrix V1 against
# the exported MySQL Schwartz exclusion table.
# Read-only: no files or database records are modified.

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
    / "67_28_fact_schwartz_exclusion_mysql_export.csv"
)

TOLERANCE = Decimal("1e-12")


# 2. Independently define the expected schema

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

KEY_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "scenario_code",
]

EXPECTED_EXPORT_COLUMNS = KEY_COLUMNS + list(FIELD_TYPES)


# 3. CSV reader with structural checks

def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    if not columns or len(columns) != len(set(columns)):
        raise AssertionError(
            f"Invalid CSV header: {path.name}"
        )

    for row in rows:
        if None in row or any(
            value is None for value in row.values()
        ):
            raise AssertionError(
                f"Malformed CSV record: {path.name}"
            )

    return columns, rows


# 4. Normalize values without changing their meaning

def normalize(value, field_type, field_name):

    if field_type == "text":
        # Preserve empty text exactly.
        return value

    text = value.strip()

    if field_type == "boolean":
        if text.lower() in ("true", "1"):
            return True
        if text.lower() in ("false", "0"):
            return False

        raise ValueError(
            f"Invalid boolean: {field_name}={value!r}"
        )

    if not text or text.upper() == "NULL":
        raise ValueError(
            f"Missing numeric value: {field_name}"
        )

    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid number: {field_name}={value!r}"
        ) from error

    if not number.is_finite():
        raise ValueError(
            f"Non-finite number: {field_name}"
        )

    if field_type == "integer":
        if number != number.to_integral_value():
            raise ValueError(
                f"Non-integer value: {field_name}"
            )
        return int(number)

    if field_type == "number":
        return number

    raise ValueError(
        f"Unexpected field type: {field_type}"
    )


def boolean(value):
    return normalize(value, "boolean", "boolean")


def integer(value):
    return normalize(value, "integer", "integer")


def number(value):
    return normalize(value, "number", "number")


# 5. Load and check the two datasets

source_columns, source_rows = read_csv(SOURCE)
export_columns, export_rows = read_csv(MYSQL_EXPORT)

if len(source_columns) != 72:
    raise AssertionError("Expected 72 source columns")

if export_columns != EXPECTED_EXPORT_COLUMNS:
    raise AssertionError(
        "MySQL export columns or their order are incorrect"
    )

if len(source_rows) != 36:
    raise AssertionError("Expected 36 source correlations")

if len(export_rows) != 72:
    raise AssertionError("Expected 72 MySQL scenarios")

expected_scenario_columns = {
    f"{scenario}__{field}"
    for scenario in SCENARIOS
    for field in FIELD_TYPES
}

actual_scenario_columns = {
    column for column in source_columns
    if column.startswith("exclude_")
}

if actual_scenario_columns != expected_scenario_columns:
    raise AssertionError(
        "Unexpected source scenario fields"
    )


# 6. Independently reconstruct the expected records

expected = {}
source_keys = set()
applicable_count = 0
non_applicable_count = 0

for row in source_rows:

    ess = row["ess_variable"].strip()
    economic = row["economic_variable"].strip()
    source_key = (ess, economic)

    if not all(source_key) or source_key in source_keys:
        raise AssertionError(
            f"Invalid source key: {source_key}"
        )

    source_keys.add(source_key)

    applicable = boolean(
        row["schwartz_sensitivity_applicable"]
    )

    if applicable != (ess in SCHWARTZ_VARIABLES):
        raise AssertionError(
            f"Incorrect applicability: {source_key}"
        )

    if not applicable:
        non_applicable_count += 1

        if any(
            row[column] != ""
            for column in expected_scenario_columns
        ):
            raise AssertionError(
                f"Non-Schwartz scenario contains values: "
                f"{source_key}"
            )

        continue

    applicable_count += 1

    included = row["included_countries"].split("|")
    baseline_n = integer(row["countries_n"])

    if (
        len(included) != baseline_n
        or len(set(included)) != baseline_n
    ):
        raise AssertionError(
            f"Invalid baseline country coverage: {source_key}"
        )

    baseline = {
        "pearson": number(row["pearson_baseline"]),
        "spearman": number(row["spearman_baseline"]),
    }

    for scenario, requested in SCENARIOS.items():

        key = (ess, economic, scenario)

        if key in expected:
            raise AssertionError(f"Duplicate key: {key}")

        data = {
            field: normalize(
                row[f"{scenario}__{field}"],
                field_type,
                f"{scenario}__{field}"
            )
            for field, field_type in FIELD_TYPES.items()
        }

        # Determine the exclusions from baseline coverage,
        # independently of the scenario columns.
        removed_countries = [
            country for country in requested
            if country in included
        ]

        actual_exclusions = "|".join(removed_countries)

        if data["actual_exclusions"] != actual_exclusions:
            raise AssertionError(
                f"Effective exclusions mismatch: {key}"
            )

        if data["removed_n"] != len(removed_countries):
            raise AssertionError(
                f"Removed count mismatch: {key}"
            )

        if data["scenario_n"] != baseline_n - len(
            removed_countries
        ):
            raise AssertionError(
                f"Scenario sample mismatch: {key}"
            )

        # Independently verify change, absolute change,
        # and strict sign reversal against the baseline.
        for method in ("pearson", "spearman"):

            coefficient = data[f"{method}_scenario"]
            change = data[f"{method}_change"]
            absolute_change = data[f"{method}_abs_change"]
            sign_flip = data[f"{method}_sign_flip"]

            if not Decimal("-1") <= coefficient <= Decimal("1"):
                raise AssertionError(
                    f"Invalid correlation: {key}, {method}"
                )

            calculated_change = coefficient - baseline[method]

            if abs(calculated_change - change) > TOLERANCE:
                raise AssertionError(
                    f"Incorrect change: {key}, {method}"
                )

            if abs(abs(change) - absolute_change) > TOLERANCE:
                raise AssertionError(
                    f"Incorrect absolute change: {key}, {method}"
                )

            calculated_flip = (
                baseline[method] * coefficient < 0
            )

            if sign_flip != calculated_flip:
                raise AssertionError(
                    f"Incorrect sign flip: {key}, {method}"
                )

        expected[key] = data


# 7. Check expected scenario coverage

if applicable_count != 24:
    raise AssertionError("Expected 24 Schwartz pairs")

if non_applicable_count != 12:
    raise AssertionError("Expected 12 non-Schwartz pairs")

if len(expected) != 72:
    raise AssertionError("Expected 72 reconstructed scenarios")

expected_distribution = Counter(
    key[2] for key in expected
)

if any(
    expected_distribution[scenario] != 24
    for scenario in SCENARIOS
):
    raise AssertionError(
        "Expected 24 records for each exclusion scenario"
)


# 8. Index MySQL records using three-part keys

actual = {}

for row in export_rows:

    key = tuple(
        row[column].strip()
        for column in KEY_COLUMNS
    )

    if not all(key):
        raise AssertionError("Empty MySQL primary key")

    if key in actual:
        raise AssertionError(
            f"Duplicate MySQL primary key: {key}"
        )

    actual[key] = {
        field: normalize(
            row[field],
            field_type,
            field
        )
        for field, field_type in FIELD_TYPES.items()
    }


# 9. Check exact key equality

if set(expected) != set(actual):

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    raise AssertionError(
        f"Primary-key mismatch: missing={missing}, extra={extra}"
    )


# 10. Compare every exported field independently

comparison_counts = Counter()
maximum_difference = Decimal("0")

for key in sorted(expected):

    for field, field_type in FIELD_TYPES.items():

        source_value = expected[key][field]
        mysql_value = actual[key][field]

        if field_type == "number":

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

        elif source_value != mysql_value:

            raise AssertionError(
                f"Mismatch: {key}, {field}: "
                f"{source_value!r} != {mysql_value!r}"
            )

        comparison_counts[field_type] += 1


# 11. Verify all expected comparison totals

expected_counts = {
    "number": 432,
    "integer": 144,
    "boolean": 144,
    "text": 72,
}

if dict(comparison_counts) != expected_counts:
    raise AssertionError(
        f"Unexpected comparison counts: {comparison_counts}"
    )

total_comparisons = sum(comparison_counts.values())

if total_comparisons != 792:
    raise AssertionError("Expected exactly 792 comparisons")


# 12. Verify exclusion distribution in the MySQL export

actual_exclusions = Counter(
    (key[2], data["actual_exclusions"])
    for key, data in actual.items()
)

expected_exclusions = {
    ("exclude_EE", "EE"): 24,
    ("exclude_GB", ""): 20,
    ("exclude_GB", "GB"): 4,
    ("exclude_EE_GB", "EE"): 20,
    ("exclude_EE_GB", "EE|GB"): 4,
}

if dict(actual_exclusions) != expected_exclusions:
    raise AssertionError(
        "Unexpected MySQL exclusion distribution"
    )


# 13. Print validation report

difference_display = (
    "0.000e+00"
    if maximum_difference == 0
    else f"{maximum_difference:.3e}"
)

print("=" * 60)
print("STEP 67.3.36 - INDEPENDENT SCHWARTZ SQL VALIDATION")
print("=" * 60)

print("Source correlations:", len(source_rows))
print("Applicable Schwartz pairs:", applicable_count)
print("Non-Schwartz pairs excluded:", non_applicable_count)
print("Expected scenarios:", len(expected))
print("MySQL export scenarios:", len(actual))
print("Primary key comparison: PASSED")
print("Scenario distributions: PASSED")

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
print("STATUS: PASSED - ALL SCHWARTZ SCENARIOS MATCH")
print("No project files were modified.")
print("=" * 60)
