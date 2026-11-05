
# ==============================================================
# STEP 63 - INDEPENDENT SCHWARTZ COUNTRY EXCLUSION VALIDATION
# ==============================================================
#
# Independently validate all Step 62 outputs.
#
# Recalculate Pearson using centered sums and math.fsum.
# Recalculate Spearman using independently constructed average ranks.
#
# Do not import or reuse the correlation function from Step 62.
# Do not create, delete, overwrite or modify project files.
#
# The validation is descriptive and does not establish
# causality or psychometric measurement invariance.
# ==============================================================

import csv
import hashlib
import json
import math
import re

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------
# 1. PATHS AND ANALYTICAL SCOPE
# --------------------------------------------------------------

# Resolve the project root from the location of this script.
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

# Previously validated analytical inputs.
INPUT = ROOT / "data/processed/ESS11_country_integrated_main_v1.parquet"

BASELINE = TABLES / "ESS11_country_correlations_main_v1.parquet"

SYNTHESIS = (
    TABLES
    / "ESS11_country_correlation_robustness_synthesis_v1.parquet"
)

# Step 62 outputs to validate.
STEM = "ESS11_schwartz_country_exclusion_sensitivity_v1"

RESULT_PARQUET = TABLES / f"{STEM}.parquet"
RESULT_CSV = TABLES / f"{STEM}.csv"
RESULT_METADATA = TABLES / f"{STEM}.metadata.json"

# The exact integrated dataset already validated in the project.
EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

# Permit only negligible floating-point differences.
TOLERANCE = 1e-12

# Four Schwartz higher-order dimensions.
SCHWARTZ = [
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

# Six socioeconomic indicators.
ECONOMIC = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

# Validated country universe of ESS11 V1.
COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# Define scenarios independently of Step 62.
SCENARIOS = {
    "primary_full": set(),
    "exclude_EE": {"EE"},
    "exclude_GB": {"GB"},
    "exclude_EE_GB": {"EE", "GB"},
}

# Define the 24 expected relationships.
EXPECTED_PAIRS = [
    (schwartz, economic)
    for schwartz in SCHWARTZ
    for economic in ECONOMIC
]

# Define the exact 96 expected output keys.
EXPECTED_KEYS = [
    (schwartz, economic, scenario)
    for schwartz, economic in EXPECTED_PAIRS
    for scenario in SCENARIOS
]


# --------------------------------------------------------------
# 2. EXPECTED STEP 62 OUTPUT SCHEMA
# --------------------------------------------------------------

# General information about each scenario and country sample.
GENERAL_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "scenario",
    "requested_exclusions",
    "actual_exclusions",
    "primary_n",
    "scenario_n",
    "removed_n",
    "scenario_countries",
]

# Pearson and Spearman share the same column structure.
METHOD_SUFFIXES = [
    "primary",
    "scenario",
    "change",
    "abs_change",
    "sign_flip",
]

METHODS = ("pearson", "spearman")

# Exactly 19 columns: 9 general + 5 Pearson + 5 Spearman.
EXPECTED_COLUMNS = GENERAL_COLUMNS + [
    f"{method}_{suffix}"
    for method in METHODS
    for suffix in METHOD_SUFFIXES
]

# Explicit schema categories.
# This avoids deprecated select_dtypes(include="object") behavior.
TEXT_COLUMNS = {
    "ess_variable",
    "economic_variable",
    "scenario",
    "requested_exclusions",
    "actual_exclusions",
    "scenario_countries",
}

INTEGER_COLUMNS = {
    "primary_n",
    "scenario_n",
    "removed_n",
}

BOOLEAN_COLUMNS = {
    "pearson_sign_flip",
    "spearman_sign_flip",
}

FLOAT_COLUMNS = (
    set(EXPECTED_COLUMNS)
    - TEXT_COLUMNS
    - INTEGER_COLUMNS
    - BOOLEAN_COLUMNS
)


# --------------------------------------------------------------
# 3. BASIC VALIDATION HELPERS
# --------------------------------------------------------------

def require(condition, message):

    # Stop immediately when a validation requirement fails.
    # Never silently correct unexpected source data.
    if not bool(condition):
        raise AssertionError(message)


def file_hash(path):

    # Calculate SHA-256 from exact file bytes.
    # Use streaming to avoid loading large files into memory.
    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def compare_float(actual, expected, label):

    # Missing numerical values cannot be compared as zero.
    require(
        not pd.isna(actual) and not pd.isna(expected),
        f"{label}: missing numerical value",
    )

    a = float(actual)
    b = float(expected)

    # Reject NaN and positive/negative infinity.
    require(
        math.isfinite(a) and math.isfinite(b),
        f"{label}: non-finite value",
    )

    difference = abs(a - b)

    # Use absolute tolerance only.
    require(
        difference <= TOLERANCE,
        f"{label}: actual={a!r}, expected={b!r}",
    )

    return difference


def compare_exact(actual, expected, label):

    # Text, integers, and Boolean flags must match exactly.
    require(
        not pd.isna(actual),
        f"{label}: unexpected missing value",
    )

    require(
        actual == expected,
        f"{label}: actual={actual!r}, expected={expected!r}",
    )


def check_columns(table, required, label):

    # Ensure unambiguous column names.
    require(
        table.columns.is_unique,
        f"{label}: duplicate columns",
    )

    # Check all fields required by the independent reconstruction.
    missing = sorted(
        set(required) - set(table.columns)
    )

    require(
        not missing,
        f"{label}: missing columns: {missing}",
    )


def parse_countries(value, label):

    # Project country lists use a pipe separator.
    require(
        isinstance(value, str),
        f"{label}: expected text country list",
    )

    countries = value.split("|") if value else []

    require(
        len(countries) == len(set(countries)),
        f"{label}: duplicate country",
    )

    require(
        set(countries).issubset(COUNTRIES),
        f"{label}: unknown country",
    )

    return countries


# --------------------------------------------------------------
# 4. INDEPENDENT PEARSON IMPLEMENTATION
# --------------------------------------------------------------

def independent_pearson(x, y):

    # Require paired observations from the same countries.
    require(
        len(x) == len(y) and len(x) >= 3,
        "Pearson requires at least three paired observations",
    )

    # Convert to Python floats for math.fsum operations.
    x = [float(value) for value in x]
    y = [float(value) for value in y]

    require(
        all(math.isfinite(value) for value in x + y),
        "Non-finite Pearson input",
    )

    n = len(x)

    # Compute both means using accurate floating-point summation.
    mean_x = math.fsum(x) / n
    mean_y = math.fsum(y) / n

    # Center every observation around its variable mean.
    centered_x = [
        value - mean_x
        for value in x
    ]

    centered_y = [
        value - mean_y
        for value in y
    ]

    # Pearson numerator: sum of centered cross-products.
    numerator = math.fsum(
        a * b
        for a, b in zip(centered_x, centered_y)
    )

    # Compute centered sums of squares separately.
    ss_x = math.fsum(
        value * value
        for value in centered_x
    )

    ss_y = math.fsum(
        value * value
        for value in centered_y
    )

    # A constant variable has undefined Pearson correlation.
    require(
        ss_x > 0 and ss_y > 0,
        "Undefined correlation: constant variable",
    )

    # Calculate Pearson without numpy.corrcoef.
    result = numerator / math.sqrt(ss_x * ss_y)

    require(
        math.isfinite(result)
        and -1 - TOLERANCE <= result <= 1 + TOLERANCE,
        "Invalid independently calculated Pearson coefficient",
    )

    return result


# --------------------------------------------------------------
# 5. INDEPENDENT SPEARMAN RANKING
# --------------------------------------------------------------

def average_ranks(values):

    # Spearman requires average ranks when ties are present.
    # Construct these ranks manually, without scipy.rankdata.
    numbers = [
        float(value)
        for value in values
    ]

    require(
        all(math.isfinite(value) for value in numbers),
        "Non-finite Spearman ranking input",
    )

    n = len(numbers)

    # Sort indices while preserving the original observations.
    order = sorted(
        range(n),
        key=lambda i: numbers[i],
    )

    ranks = [0.0] * n

    start = 0

    # Process each group of identical values.
    while start < n:

        end = start + 1

        while (
            end < n
            and numbers[order[end]] == numbers[order[start]]
        ):
            end += 1

        # Ranks begin at 1, not 0.
        # For a tied group, assign the average occupied rank.
        rank = ((start + 1) + end) / 2.0

        for position in range(start, end):

            original_index = order[position]

            ranks[original_index] = rank

        start = end

    return ranks


def independent_spearman(x, y):

    # Spearman is Pearson applied to ranked observations.
    ranked_x = average_ranks(x)
    ranked_y = average_ranks(y)

    return independent_pearson(
        ranked_x,
        ranked_y,
    )


# --------------------------------------------------------------
# 6. VERIFY THE OUTPUT SCHEMA
# --------------------------------------------------------------

def validate_schema(result):

    # Require the exact output column names and order.
    require(
        list(result.columns) == EXPECTED_COLUMNS,
        "Unexpected Step 62 output schema or column order",
    )

    require(
        len(EXPECTED_COLUMNS) == 19,
        "Incorrect independent schema definition",
    )

    require(
        len(result) == 96,
        "Expected exactly 96 scenario comparisons",
    )

    # Require the same pair and scenario ordering used in Step 62.
    actual_keys = list(
        result[
            ["ess_variable", "economic_variable", "scenario"]
        ].itertuples(index=False, name=None)
    )

    require(
        actual_keys == EXPECTED_KEYS,
        "Scenario pair coverage/order differs",
    )

    require(
        not result.isna().any().any(),
        "Unexpected null in Step 62 output",
    )

    # Verify stored pandas data types.
    for column in EXPECTED_COLUMNS:

        dtype = result[column].dtype

        if column in TEXT_COLUMNS:

            require(
                pd.api.types.is_string_dtype(dtype)
                and not pd.api.types.is_object_dtype(dtype),
                f"{column}: expected string dtype, got {dtype}",
            )

        elif column in INTEGER_COLUMNS:

            require(
                str(dtype) == "int64",
                f"{column}: expected int64, got {dtype}",
            )

        elif column in BOOLEAN_COLUMNS:

            require(
                str(dtype) == "bool",
                f"{column}: expected bool, got {dtype}",
            )

        else:

            require(
                str(dtype) == "float64",
                f"{column}: expected float64, got {dtype}",
            )

    print(
        "Output schema, 19 columns, 96 rows and dtypes: PASSED."
    )


# --------------------------------------------------------------
# 7. READ AND VALIDATE ORIGINAL ANALYTICAL SOURCES
# --------------------------------------------------------------

def load_sources():

    # Read previously validated sources without modifying them.
    integrated = pd.read_parquet(
        INPUT,
        engine="pyarrow",
    )

    baseline = pd.read_parquet(
        BASELINE,
        engine="pyarrow",
    )

    synthesis = pd.read_parquet(
        SYNTHESIS,
        engine="pyarrow",
    )

    check_columns(
        integrated,
        ["cntry"] + SCHWARTZ + ECONOMIC,
        "Integrated input",
    )

    check_columns(
        baseline,
        [
            "ess_variable",
            "economic_variable",
            "countries_n",
            "included_countries",
            "pearson_r",
            "spearman_rho",
            "input_sha256",
        ],
        "Baseline",
    )

    check_columns(
        synthesis,
        [
            "ess_variable",
            "economic_variable",
            "countries_n",
            "included_countries",
            "pearson_baseline",
            "spearman_baseline",
        ],
        "Robustness synthesis",
    )

    # The integrated dataset must have one row per V1 country.
    require(
        len(integrated) == 29,
        "Unexpected number of integrated countries",
    )

    require(
        integrated["cntry"].notna().all()
        and integrated["cntry"].is_unique,
        "Invalid integrated country identifiers",
    )

    require(
        set(integrated["cntry"]) == COUNTRIES,
        "Integrated country universe differs",
    )

    # Baseline and synthesis should each cover all 36 ESS pairs.
    require(
        len(baseline) == 36
        and len(synthesis) == 36,
        "Unexpected source correlation row count",
    )

    expected_source_pairs = {
        (ess, economic)
        for ess in (
            "mean_social_trust",
            "mean_left_right",
            *SCHWARTZ,
        )
        for economic in ECONOMIC
    }

    for name, table in (
        ("Baseline", baseline),
        ("Synthesis", synthesis),
    ):

        keys = list(
            table[
                ["ess_variable", "economic_variable"]
            ].itertuples(index=False, name=None)
        )

        require(
            len(keys) == len(set(keys))
            and set(keys) == expected_source_pairs,
            f"{name}: invalid pair coverage",
        )

    # Verify the original analysis provenance.
    require(
        baseline["input_sha256"].eq(
            EXPECTED_INPUT_SHA256
        ).all(),
        "Baseline input SHA-256 differs",
    )

    # Create country-indexed values for independent calculations.
    integrated = integrated.set_index(
        "cntry"
    ).sort_index()

    baseline = baseline.set_index(
        ["ess_variable", "economic_variable"]
    )

    synthesis = synthesis.set_index(
        ["ess_variable", "economic_variable"]
    )

    return integrated, baseline, synthesis


# --------------------------------------------------------------
# 8. RECONSTRUCT ALL 96 SCENARIOS INDEPENDENTLY
# --------------------------------------------------------------

def reconstruct(integrated, baseline, synthesis):

    expected_rows = []

    maximum_baseline_difference = 0.0

    checked_coefficients = 0

    # Inspect every Schwartz-Eurostat relationship.
    for ess, economic in EXPECTED_PAIRS:

        key = (ess, economic)

        primary = baseline.loc[key]
        consolidated = synthesis.loc[key]

        # Filter only observations where both variables exist.
        # Never impute economic or Schwartz values.
        original = integrated[
            [ess, economic]
        ].dropna()

        original_countries = set(
            original.index
        )

        # Confirm exact original country membership.
        baseline_countries = set(
            parse_countries(
                primary["included_countries"],
                f"{key}: baseline membership",
            )
        )

        synthesis_countries = set(
            parse_countries(
                consolidated["included_countries"],
                f"{key}: synthesis membership",
            )
        )

        require(
            original_countries
            == baseline_countries
            == synthesis_countries,
            f"{key}: baseline country samples differ",
        )

        require(
            len(original)
            == int(primary["countries_n"])
            == int(consolidated["countries_n"]),
            f"{key}: baseline N differs",
        )

        # Verify the sample structure observed in Steps 53-62.
        expected_n = {
            "gdp_pps_index": 28,
            "employment_rate": 26,
            "unemployment_rate": 26,
            "gini": 25,
            "material_deprivation_rate": 26,
            "inflation_rate": 27,
        }

        require(
            len(original) == expected_n[economic],
            f"{key}: unexpected primary country count",
        )

        # Estonia is included in every original relationship.
        require(
            "EE" in original_countries,
            f"{key}: Estonia unexpectedly absent",
        )

        # GB is available for GDP but not the other indicators.
        require(
            ("GB" in original_countries)
            == (economic == "gdp_pps_index"),
            f"{key}: unexpected GB availability",
        )

        # Independently reproduce the original correlations.
        x = original[ess].tolist()
        y = original[economic].tolist()

        primary_pearson = independent_pearson(x, y)
        primary_spearman = independent_spearman(x, y)

        # Verify Step 53 and the Step 59 synthesis separately.
        for actual, expected, label in [
            (
                primary_pearson,
                primary["pearson_r"],
                "baseline Pearson",
            ),
            (
                primary_spearman,
                primary["spearman_rho"],
                "baseline Spearman",
            ),
            (
                primary_pearson,
                consolidated["pearson_baseline"],
                "synthesis Pearson",
            ),
            (
                primary_spearman,
                consolidated["spearman_baseline"],
                "synthesis Spearman",
            ),
        ]:

            maximum_baseline_difference = max(
                maximum_baseline_difference,
                compare_float(
                    actual,
                    expected,
                    f"{key}: {label}",
                ),
            )

        # Apply each predefined exclusion scenario.
        for scenario_name, requested in SCENARIOS.items():

            # Only countries present in the original sample
            # can actually be removed from a correlation.
            removed = original_countries & requested

            retained = sorted(
                original_countries - requested
            )

            scenario_data = original.loc[
                retained
            ]

            require(
                len(scenario_data)
                == len(original) - len(removed),
                f"{key}: incorrect scenario N",
            )

            # Recalculate Pearson directly from the reduced sample.
            scenario_pearson = independent_pearson(
                scenario_data[ess].tolist(),
                scenario_data[economic].tolist(),
            )

            # Recalculate Spearman using newly constructed ranks.
            # Existing full-sample ranks must not be reused.
            scenario_spearman = independent_spearman(
                scenario_data[ess].tolist(),
                scenario_data[economic].tolist(),
            )

            checked_coefficients += 2

            # Reconstruct signed and absolute changes.
            pearson_change = (
                scenario_pearson - primary_pearson
            )

            spearman_change = (
                scenario_spearman - primary_spearman
            )

            # A strict sign reversal requires opposite signs.
            # Crossing to zero is not counted as a strict reversal.
            pearson_flip = (
                primary_pearson * scenario_pearson < 0
            )

            spearman_flip = (
                primary_spearman * scenario_spearman < 0
            )

            expected_rows.append({
                "ess_variable": ess,
                "economic_variable": economic,
                "scenario": scenario_name,
                "requested_exclusions":
                    "|".join(sorted(requested)),
                "actual_exclusions":
                    "|".join(sorted(removed)),
                "primary_n": len(original),
                "scenario_n": len(scenario_data),
                "removed_n": len(removed),
                "scenario_countries":
                    "|".join(retained),

                "pearson_primary": primary_pearson,
                "pearson_scenario": scenario_pearson,
                "pearson_change": pearson_change,
                "pearson_abs_change": abs(pearson_change),
                "pearson_sign_flip": pearson_flip,

                "spearman_primary": primary_spearman,
                "spearman_scenario": scenario_spearman,
                "spearman_change": spearman_change,
                "spearman_abs_change": abs(spearman_change),
                "spearman_sign_flip": spearman_flip,
            })

    require(
        len(expected_rows) == 96,
        "Independent reconstruction did not produce 96 rows",
    )

    require(
        checked_coefficients == 192,
        "Expected 192 independently calculated scenario coefficients",
    )

    print(
        "24 baseline pairs and 192 independent "
        "scenario coefficients: PASSED."
    )

    print(
        "Country exclusions, actual membership "
        "and sample sizes: PASSED."
    )

    return (
        expected_rows,
        maximum_baseline_difference,
        checked_coefficients,
    )


# --------------------------------------------------------------
# 9. COMPARE EVERY SAVED PARQUET CELL
# --------------------------------------------------------------

def validate_parquet(result, expected_rows):

    max_difference = 0.0

    # Compare all 96 observations in the original output order.
    for position, expected in enumerate(expected_rows):

        saved = result.iloc[position]

        for column in EXPECTED_COLUMNS:

            label = (
                f"row {position + 1}, "
                f"{expected['ess_variable']} / "
                f"{expected['economic_variable']} / "
                f"{expected['scenario']}, {column}"
            )

            if column in FLOAT_COLUMNS:

                max_difference = max(
                    max_difference,
                    compare_float(
                        saved[column],
                        expected[column],
                        label,
                    ),
                )

            else:

                compare_exact(
                    saved[column],
                    expected[column],
                    label,
                )

    print(
        "All 96 x 19 Parquet cells independently checked: PASSED."
    )

    return max_difference


# --------------------------------------------------------------
# 10. INDEPENDENT CSV COMPARISON
# --------------------------------------------------------------

def validate_csv(result, expected_rows):

    # csv.DictReader preserves text without pandas type inference.
    with RESULT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        require(
            reader.fieldnames == EXPECTED_COLUMNS,
            "CSV column names/order differ",
        )

        rows = list(reader)

    require(
        len(rows) == 96,
        "CSV does not contain exactly 96 rows",
    )

    max_difference = 0.0

    # Check CSV against both the reconstructed values and Parquet.
    for position, (csv_row, expected) in enumerate(
        zip(rows, expected_rows)
    ):

        saved = result.iloc[position]

        require(
            set(csv_row) == set(EXPECTED_COLUMNS),
            f"CSV row {position + 1}: unexpected fields",
        )

        for column in EXPECTED_COLUMNS:

            value = csv_row[column]

            label = f"CSV row {position + 1}, {column}"

            require(
                value is not None,
                f"{label}: missing CSV field",
            )

            if column in FLOAT_COLUMNS:

                max_difference = max(
                    max_difference,
                    compare_float(
                        value,
                        expected[column],
                        label,
                    ),
                    compare_float(
                        value,
                        saved[column],
                        label + " versus Parquet",
                    ),
                )

            elif column in INTEGER_COLUMNS:

                require(
                    re.fullmatch(r"-?\d+", value) is not None,
                    f"{label}: invalid integer text",
                )

                integer = int(value)

                compare_exact(
                    integer,
                    expected[column],
                    label,
                )

                compare_exact(
                    integer,
                    saved[column],
                    label + " versus Parquet",
                )

            elif column in BOOLEAN_COLUMNS:

                require(
                    value in ("True", "False"),
                    f"{label}: invalid Boolean text",
                )

                boolean = value == "True"

                compare_exact(
                    boolean,
                    expected[column],
                    label,
                )

                compare_exact(
                    boolean,
                    saved[column],
                    label + " versus Parquet",
                )

            else:

                compare_exact(
                    value,
                    expected[column],
                    label,
                )

                compare_exact(
                    value,
                    saved[column],
                    label + " versus Parquet",
                )

    print(
        "96 CSV rows, headers and CSV/Parquet "
        "consistency: PASSED."
    )

    return max_difference


# --------------------------------------------------------------
# 11. VERIFY METADATA AND SOURCE HASHES
# --------------------------------------------------------------

def validate_metadata(metadata):

    # Verify the basic methodological metadata.
    expected_fields = {
        "step": 62,
        "study_design": "cross-sectional exploratory",
        "analysis_unit": "country",
        "analysis_scope": "Schwartz-Eurostat correlations",
        "schwartz_dimensions_n": 4,
        "economic_indicators_n": 6,
        "analytical_pairs_n": 24,
        "scenarios_n": 4,
        "scenario_comparisons_n": 96,
        "baseline_unchanged": True,
        "independent_validation_status": "pending",
        "p_values_calculated": False,
        "causal_analysis": False,
        "measurement_invariance_test": False,
    }

    for key, expected in expected_fields.items():

        compare_exact(
            metadata.get(key),
            expected,
            f"metadata.{key}",
        )

    # Verify all predeclared exclusion scenarios.
    expected_scenarios = {
        name: sorted(countries)
        for name, countries in SCENARIOS.items()
    }

    require(
        metadata.get("requested_exclusion_scenarios")
        == expected_scenarios,
        "Metadata exclusion scenarios differ",
    )

    # Check that methodological limitation notes are present.
    notes = metadata.get("notes")

    require(
        isinstance(notes, list)
        and len(notes) >= 7
        and all(isinstance(note, str) for note in notes),
        "Invalid methodological notes",
    )

    # Check the timestamp format and UTC timezone.
    timestamp = metadata.get("created_at_utc")

    require(
        isinstance(timestamp, str),
        "Missing metadata timestamp",
    )

    try:

        parsed_timestamp = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

    except ValueError as exc:

        raise AssertionError(
            "Invalid metadata timestamp"
        ) from exc

    require(
        parsed_timestamp.tzinfo is not None
        and parsed_timestamp.utcoffset() == timedelta(0),
        "Metadata timestamp must be UTC",
    )

    # ----------------------------------------------------------
    # 11.1 VERIFY THE THREE SOURCE FILE HASHES
    # ----------------------------------------------------------

    source_files = [
        INPUT,
        BASELINE,
        SYNTHESIS,
    ]

    source_entries = metadata.get("sources")

    require(
        isinstance(source_entries, list)
        and len(source_entries) == 3,
        "Expected three source hash records",
    )

    for entry, path in zip(source_entries, source_files):

        require(
            isinstance(entry, dict),
            "Invalid source metadata record",
        )

        compare_exact(
            entry.get("filename"),
            path.name,
            "Source filename",
        )

        compare_exact(
            entry.get("sha256"),
            file_hash(path),
            f"SHA-256 {path.name}",
        )

    # ----------------------------------------------------------
    # 11.2 VERIFY THE TWO OUTPUT FILE HASHES
    # ----------------------------------------------------------

    output_files = [
        RESULT_PARQUET,
        RESULT_CSV,
    ]

    output_entries = metadata.get("outputs")

    require(
        isinstance(output_entries, list)
        and len(output_entries) == 2,
        "Expected two output hash records",
    )

    for entry, path in zip(output_entries, output_files):

        require(
            isinstance(entry, dict),
            "Invalid output metadata record",
        )

        compare_exact(
            entry.get("filename"),
            path.name,
            "Output filename",
        )

        compare_exact(
            entry.get("sha256"),
            file_hash(path),
            f"SHA-256 {path.name}",
        )

    print(
        "Metadata, provenance and five SHA-256 records: PASSED."
    )


# --------------------------------------------------------------
# 12. VERIFY AGGREGATE SUMMARY STATISTICS
# --------------------------------------------------------------

def validate_summary(result):

    # These expected totals come from the Step 62 terminal report.
    expected_ranges = {
        "primary_full": (25, 28),
        "exclude_EE": (24, 27),
        "exclude_GB": (25, 27),
        "exclude_EE_GB": (24, 26),
    }

    expected_changes = {
        "primary_full": (0.0, 0.0),
        "exclude_EE": (0.031088, 0.081344),
        "exclude_GB": (0.006686, 0.022082),
        "exclude_EE_GB": (0.033706, 0.081344),
    }

    for scenario in SCENARIOS:

        rows = result.loc[
            result["scenario"] == scenario
        ]

        require(
            len(rows) == 24,
            f"{scenario}: expected 24 pairs",
        )

        min_n, max_n = expected_ranges[scenario]

        require(
            int(rows["scenario_n"].min()) == min_n,
            f"{scenario}: minimum N differs",
        )

        require(
            int(rows["scenario_n"].max()) == max_n,
            f"{scenario}: maximum N differs",
        )

        # Compare rounded report values, not raw unrounded values.
        observed_pearson = round(
            float(rows["pearson_abs_change"].max()),
            6,
        )

        observed_spearman = round(
            float(rows["spearman_abs_change"].max()),
            6,
        )

        expected_pearson, expected_spearman = (
            expected_changes[scenario]
        )

        require(
            observed_pearson == expected_pearson,
            f"{scenario}: maximum Pearson change differs",
        )

        require(
            observed_spearman == expected_spearman,
            f"{scenario}: maximum Spearman change differs",
        )

        # No strict sign reversals were found in Step 62.
        require(
            not rows["pearson_sign_flip"].any()
            and not rows["spearman_sign_flip"].any(),
            f"{scenario}: unexpected sign reversal",
        )

    # Confirm the important limitation of the GB exclusion.
    gb = result.loc[
        result["scenario"] == "exclude_GB"
    ]

    # GB should be removed from exactly four GDP relationships.
    require(
        int(gb["removed_n"].sum()) == 4,
        "GB should affect exactly four GDP correlations",
    )

    require(
        gb.loc[
            gb["economic_variable"] == "gdp_pps_index",
            "removed_n",
        ].eq(1).all(),
        "GB was not removed from all GDP relationships",
    )

    require(
        gb.loc[
            gb["economic_variable"] != "gdp_pps_index",
            "removed_n",
        ].eq(0).all(),
        "GB unexpectedly affects non-GDP relationships",
    )

    # Estonia should affect all 24 Schwartz relationships.
    ee = result.loc[
        result["scenario"] == "exclude_EE"
    ]

    require(
        ee["removed_n"].eq(1).all(),
        "Estonia was not included in every baseline pair",
    )

    print(
        "Scenario ranges, maximum changes, sign flags "
        "and country coverage: PASSED."
    )


# --------------------------------------------------------------
# 13. MAIN READ-ONLY VALIDATION WORKFLOW
# --------------------------------------------------------------

def main():

    print("=" * 85)

    print(
        "STEP 63 - INDEPENDENT SCHWARTZ "
        "COUNTRY EXCLUSION VALIDATION"
    )

    print("=" * 85)

    # All six files are read-only.
    paths = [
        INPUT,
        BASELINE,
        SYNTHESIS,
        RESULT_PARQUET,
        RESULT_CSV,
        RESULT_METADATA,
    ]

    for path in paths:

        require(
            path.is_file(),
            f"Missing required file: {path}",
        )

    # Record file integrity before reading analytical data.
    before = {
        path: file_hash(path)
        for path in paths
    }

    # Verify the validated integrated dataset hash.
    require(
        before[INPUT] == EXPECTED_INPUT_SHA256,
        "Integrated input hash does not match",
    )

    # Read sources and Step 62 results.
    integrated, baseline, synthesis = load_sources()

    result = pd.read_parquet(
        RESULT_PARQUET,
        engine="pyarrow",
    )

    with RESULT_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:

        metadata = json.load(handle)

    require(
        isinstance(metadata, dict),
        "Step 62 metadata must be a JSON object",
    )

    # Validate the exact output structure.
    validate_schema(result)

    # Independently recalculate all country-level correlations.
    (
        reconstructed,
        baseline_difference,
        coefficients_checked,
    ) = reconstruct(
        integrated,
        baseline,
        synthesis,
    )

    # Validate every saved numerical and descriptive field.
    parquet_difference = validate_parquet(
        result,
        reconstructed,
    )

    csv_difference = validate_csv(
        result,
        reconstructed,
    )

    # Validate provenance and scenario-level summaries.
    validate_metadata(metadata)
    validate_summary(result)

    # Check file integrity again after all operations.
    after = {
        path: file_hash(path)
        for path in paths
    }

    require(
        before == after,
        "A project file changed during validation",
    )

    # ----------------------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------------------

    print("\nSUMMARY")
    print("-" * 85)

    print("Validated analytical relationships:", len(EXPECTED_PAIRS))
    print("Validated exclusion scenarios:", len(SCENARIOS))
    print("Validated output rows:", len(result))
    print("Validated output columns:", len(result.columns))
    print("Validated Parquet cells:", result.size)
    print("Independent scenario coefficients:", coefficients_checked)

    print(
        "Maximum baseline reproduction difference:",
        f"{baseline_difference:.3e}",
    )

    print(
        "Maximum independent Parquet difference:",
        f"{parquet_difference:.3e}",
    )

    print(
        "Maximum CSV/Parquet difference:",
        f"{csv_difference:.3e}",
    )

    print(
        "Pearson strict sign reversals:",
        int(result["pearson_sign_flip"].sum()),
    )

    print(
        "Spearman strict sign reversals:",
        int(result["spearman_sign_flip"].sum()),
    )

    print("\nInput and output file bytes unchanged: PASSED.")

    # Print PASSED only if all previous assertions succeeded.
    print(
        "\nINDEPENDENT SCHWARTZ COUNTRY EXCLUSION "
        "VALIDATION PASSED."
    )

    print(
        "No project files were created or modified."
    )


if __name__ == "__main__":
    main()
