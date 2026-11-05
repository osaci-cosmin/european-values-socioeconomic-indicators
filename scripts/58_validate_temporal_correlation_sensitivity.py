
# ============================================================
# STEP 58 - INDEPENDENT TEMPORAL SENSITIVITY VALIDATION
# ============================================================
#
# Purpose:
# Independently validate the results produced by Step 57.
#
# This script is READ-ONLY:
# - It never creates or overwrites project files.
# - It does not modify the validated RAW data.
# - It does not impute missing economic observations.
# - It recalculates Pearson and Spearman independently.
# - It verifies sample composition, metadata and provenance.


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# Calculate SHA-256 hashes for file integrity checks.
import hashlib

# Read binary Parquet contents from memory.
import io

# Read Step 57 JSON metadata.
import json

# Use independent mathematical formulas for correlations.
import math

# Validate the metadata timestamp.
from datetime import datetime

# Build file paths relative to the script location.
from pathlib import Path

# Inspect numerical data types.
import numpy as np

# Read Parquet and CSV tables.
import pandas as pd


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

# The script is stored inside the project's scripts/ folder.
# parents[1] therefore points to the project root.
ROOT = Path(__file__).resolve().parents[1]

# Statistical outputs are stored in outputs/tables/.
TABLES = ROOT / "outputs" / "tables"

# Main validated integrated ESS-Eurostat dataset.
INTEGRATED_PATH = (
    ROOT / "data" / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

# Validated primary and alternative Eurostat observations.
MATCHES_PATH = (
    ROOT / "data" / "interim"
    / "ess_eurostat_temporal_matches_v1.parquet"
)

# Previously validated baseline correlations.
BASELINE_PATH = (
    TABLES / "ESS11_country_correlations_main_v1.parquet"
)

# Step 57 economic values for both temporal scenarios.
PAIRED_PATH = (
    TABLES / "ESS11_temporal_sensitivity_values_v1.parquet"
)

# Step 57 temporal correlation results.
CORRELATIONS_PATH = (
    TABLES / "ESS11_temporal_sensitivity_correlations_v1.parquet"
)

# Historical Step 57 metadata.
METADATA_PATH = (
    TABLES / "ESS11_temporal_correlation_sensitivity_v1.metadata.json"
)


# ============================================================
# 3. DEFINE EXPECTED INPUT HASH
# ============================================================

# This is the exact SHA-256 of the validated integrated dataset.
# If the bytes change, the validator must stop.
EXPECTED_INTEGRATED_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)


# ============================================================
# 4. DEFINE EXPECTED COUNTRIES
# ============================================================

# These are the 29 European countries included in ESS V1.
# A set makes country membership independent of row order.
COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# These 13 countries have fieldwork in both 2023 and 2024.
# Only these countries use the alternative context year.
SWITCH_COUNTRIES = set(
    "BE BG CH CY FI FR HR IE IT LV PL PT RS".split()
)


# ============================================================
# 5. DEFINE ANALYTICAL VARIABLES
# ============================================================

# Six ESS outcomes.
# Their country estimates remain fixed across both scenarios.
ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

# Six Eurostat socioeconomic indicators.
ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

# Both correlation methods must be validated independently.
METHODS = ("pearson", "spearman")

# Absolute tolerance for floating-point comparisons.
# Exact equality is still required for codes, years and statuses.
ATOL = 1e-12


# ============================================================
# 6. DEFINE THE PAIRED-VALUES TABLE SCHEMA
# ============================================================

# Step 57 must have produced exactly these 18 columns.
# Column order is part of the expected output structure.
PAIRED_COLUMNS = [
    "cntry",
    "indicator_id",
    "sensitivity_source_scenario",
    "temporal_switch_flag",
    "primary_context_year",
    "sensitivity_context_year",
    "primary_requested_eurostat_year",
    "sensitivity_requested_eurostat_year",
    "primary_value",
    "sensitivity_value",
    "primary_available_flag",
    "sensitivity_available_flag",
    "both_available_flag",
    "value_changed_flag",
    "primary_match_status",
    "sensitivity_match_status",
    "primary_eurostat_status",
    "sensitivity_eurostat_status",
]


# ============================================================
# 7. DEFINE THE CORRELATION TABLE SCHEMA
# ============================================================

# Eight fields describe the country sample and variable pair.
# Six additional fields are generated for each method.
# Total: 8 + 6 + 6 = 20 columns.
CORRELATION_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "primary_full_n",
    "common_countries_n",
    "switched_context_countries_n",
    "changed_numeric_values_n",
    "common_countries",
    "removed_from_primary_sample",
] + [
    f"{method}_{suffix}"
    for method in METHODS
    for suffix in (
        "primary_full",
        "primary_common",
        "sensitivity_common",
        "temporal_change",
        "sample_change",
        "temporal_sign_flip",
    )
]


# ============================================================
# 8. DECLARE EXPECTED DATA TYPES
# ============================================================

# Text fields in the paired economic table.
PAIRED_STRING = {
    "cntry",
    "indicator_id",
    "sensitivity_source_scenario",
    "primary_match_status",
    "sensitivity_match_status",
    "primary_eurostat_status",
    "sensitivity_eurostat_status",
}

# Year fields must be 64-bit integers.
PAIRED_INTS = {
    "primary_context_year",
    "sensitivity_context_year",
    "primary_requested_eurostat_year",
    "sensitivity_requested_eurostat_year",
}

# Economic observations are floating-point values.
PAIRED_FLOATS = {
    "primary_value",
    "sensitivity_value",
}

# Ordinary boolean flags cannot contain missing values.
PAIRED_BOOLS = {
    "temporal_switch_flag",
    "primary_available_flag",
    "sensitivity_available_flag",
    "both_available_flag",
}

# Text fields in the correlation table.
CORRELATION_STRING = {
    "ess_variable",
    "economic_variable",
    "common_countries",
    "removed_from_primary_sample",
}

# Country counts must be exact integers.
CORRELATION_INTS = {
    "primary_full_n",
    "common_countries_n",
    "switched_context_countries_n",
    "changed_numeric_values_n",
}

# Coefficients and coefficient differences use float64.
CORRELATION_FLOATS = {
    f"{method}_{suffix}"
    for method in METHODS
    for suffix in (
        "primary_full",
        "primary_common",
        "sensitivity_common",
        "temporal_change",
        "sample_change",
    )
}

# Sign-reversal flags are ordinary boolean values.
CORRELATION_BOOLS = {
    f"{method}_temporal_sign_flip"
    for method in METHODS
}


# ============================================================
# 9. GENERAL VALIDATION HELPER
# ============================================================

def require(condition, message):

    # Stop immediately when a required condition fails.
    # Invalid data must never be silently accepted or repaired.
    if not bool(condition):
        raise AssertionError(message)


# ============================================================
# 10. MISSING-VALUE HELPER
# ============================================================

def is_missing(value):

    # Detect NumPy NaN and pandas pd.NA.
    # Missing values are not equivalent to the number zero.
    return bool(pd.isna(value))


# ============================================================
# 11. FLOATING-POINT COMPARISON HELPER
# ============================================================

def verify_number(actual, expected, label, tolerance=ATOL):

    # If one value is missing, both must be missing.
    # This prevents accidental replacement of missing values.
    if is_missing(actual) or is_missing(expected):

        require(
            is_missing(actual) and is_missing(expected),
            f"{label}: missing mismatch"
        )

        return 0.0

    # Convert numerical scalar values to Python floats.
    a = float(actual)
    b = float(expected)

    # Reject infinite observations.
    require(
        math.isfinite(a) and math.isfinite(b),
        f"{label}: non-finite value"
    )

    # Calculate the absolute numerical difference.
    difference = abs(a - b)

    # Compare against the requested tolerance.
    require(
        difference <= tolerance,
        (
            f"{label}: {a!r} != {b!r}; "
            f"difference={difference:.3e}"
        )
    )

    return difference


# ============================================================
# 12. EXACT COMPARISON HELPER
# ============================================================

def verify_exact(actual, expected, label):

    # Missing values and empty strings are not equivalent.
    # Eurostat status metadata must preserve this distinction.
    if is_missing(actual) or is_missing(expected):

        require(
            is_missing(actual) and is_missing(expected),
            f"{label}: missing mismatch"
        )

    else:

        # Country codes, statuses, years and flags must match exactly.
        require(
            actual == expected,
            f"{label}: {actual!r} != {expected!r}"
        )


# ============================================================
# 13. CALCULATE SHA-256 OF AN EXISTING FILE
# ============================================================

def sha256(path):

    # A required file must already exist.
    require(
        path.is_file(),
        f"Missing required file: {path}"
    )

    # Calculate the hash from the original bytes.
    # Reading a file does not modify it.
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ============================================================
# 14. READ PARQUET WITHOUT MODIFYING IT
# ============================================================

def read_parquet(path):

    # Read exactly the bytes later used for hashing.
    data = path.read_bytes()

    # Decode the Parquet data from an in-memory buffer.
    table = pd.read_parquet(
        io.BytesIO(data),
        engine="pyarrow"
    )

    # Calculate the source file's binary fingerprint.
    digest = hashlib.sha256(data).hexdigest()

    return table, digest


# ============================================================
# 15. CHECK PARQUET SCHEMA AND DATA TYPES
# ============================================================

def check_columns_and_types(
    df,
    columns,
    string_cols,
    int_cols,
    float_cols,
    bool_cols,
    label,
    nullable_bool_cols=frozenset(),
):

    # Duplicate column names are not acceptable.
    require(
        df.columns.is_unique,
        f"{label}: duplicate column names"
    )

    # Require identical column names and order.
    require(
        list(df.columns) == columns,
        f"{label}: column schema or order differs"
    )

    # Validate the declared type of each column.
    for name in columns:

        dtype = df[name].dtype

        if name in string_cols:

            # Nullable pandas string columns preserve missing text.
            require(
                isinstance(dtype, pd.StringDtype),
                f"{label}.{name}: expected string, got {dtype}"
            )

        elif name in int_cols:

            # Years and count fields must be int64.
            require(
                dtype == np.dtype("int64"),
                f"{label}.{name}: expected int64, got {dtype}"
            )

        elif name in float_cols:

            # Economic observations and correlations must be float64.
            require(
                dtype == np.dtype("float64"),
                f"{label}.{name}: expected float64, got {dtype}"
            )

        elif name in bool_cols:

            # Ordinary boolean fields cannot contain pd.NA.
            require(
                dtype == np.dtype("bool"),
                f"{label}.{name}: expected bool, got {dtype}"
            )

        elif name in nullable_bool_cols:

            # value_changed_flag can contain True, False or pd.NA.
            require(
                str(dtype) == "boolean",
                f"{label}.{name}: expected nullable boolean, got {dtype}"
            )

        else:

            # Reject columns without an expected dtype rule.
            raise AssertionError(
                f"{label}: no dtype rule for {name}"
            )


# ============================================================
# 16. VERIFY CSV AGAINST PARQUET
# ============================================================

def check_csv_against_parquet(
    parquet,
    path,
    string_cols,
    int_cols,
    float_cols,
    bool_cols,
    nullable_bool_cols,
    label,
):

    # Confirm that the corresponding CSV export exists.
    require(
        path.is_file(),
        f"{label}: missing CSV: {path}"
    )

    # Read all fields as text first.
    #
    # keep_default_na=False preserves empty CSV cells.
    # Parquet and the independent source distinguish empty
    # strings from genuinely missing status values.
    csv = pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8"
    )

    # Require the same columns in the same order.
    require(
        list(csv.columns) == list(parquet.columns),
        f"{label}: CSV column mismatch"
    )

    # Require identical numbers of rows.
    require(
        len(csv) == len(parquet),
        f"{label}: CSV row count differs"
    )

    max_difference = 0.0

    # Compare every exported cell against the Parquet cell.
    for col in parquet.columns:

        for i, (csv_item, item) in enumerate(
            zip(csv[col], parquet[col])
        ):

            where = f"{label} CSV row {i + 1}, {col}"

            if col in string_cols:

                # CSV represents missing text as an empty field.
                # Check source-level missingness separately.
                expected = (
                    ""
                    if is_missing(item)
                    else str(item)
                )

                verify_exact(
                    str(csv_item),
                    expected,
                    where
                )

            elif col in int_cols:

                # Integer fields must not be missing.
                require(
                    str(csv_item) != "",
                    f"{where}: missing integer"
                )

                try:
                    parsed = int(str(csv_item))

                except ValueError as exc:
                    raise AssertionError(
                        f"{where}: invalid integer"
                    ) from exc

                # Integer comparisons require exact equality.
                verify_exact(
                    parsed,
                    int(item),
                    where
                )

            elif col in float_cols:

                if is_missing(item):

                    # Missing numeric values must remain empty in CSV.
                    verify_exact(
                        str(csv_item),
                        "",
                        where
                    )

                else:

                    # Valid values must have a numeric representation.
                    require(
                        str(csv_item) != "",
                        f"{where}: missing float"
                    )

                    try:
                        parsed = float(str(csv_item))

                    except ValueError as exc:
                        raise AssertionError(
                            f"{where}: invalid float"
                        ) from exc

                    # Check floating-point serialization accuracy.
                    difference = verify_number(
                        parsed,
                        item,
                        where
                    )

                    max_difference = max(
                        max_difference,
                        difference
                    )

            elif col in bool_cols or col in nullable_bool_cols:

                # Boolean CSV entries use True, False or empty text.
                expected = (
                    ""
                    if is_missing(item)
                    else str(bool(item))
                )

                verify_exact(
                    str(csv_item),
                    expected,
                    where
                )

            else:

                raise AssertionError(
                    f"{where}: unrecognized column type"
                )

    return max_difference

# ============================================================
# 17. VALIDATE ORIGINAL SOURCE DATA
# ============================================================

def check_sources(integrated, matches, baseline, hashes):

    # Confirm that the integrated dataset has the expected hash.
    # This prevents accidental use of a modified input snapshot.
    require(
        hashes["integrated"] == EXPECTED_INTEGRATED_SHA256,
        "Integrated dataset hash differs from validated input"
    )

    # Define the minimum required columns in the integrated table.
    required_columns = [
        "cntry",
        "context_year",
        "multi_year_fieldwork_flag",
        "matched_input_sha256",
        *ESS_VARIABLES,
        *ECONOMIC_VARIABLES,
        *[
            f"{indicator}__requested_eurostat_year"
            for indicator in ECONOMIC_VARIABLES
        ],
    ]

    # Confirm that every required field exists.
    for column in required_columns:
        require(
            column in integrated.columns,
            f"Integrated source missing: {column}"
        )

    # There must be exactly one record per European country.
    require(
        len(integrated) == 29
        and integrated["cntry"].is_unique,
        "Integrated dataset must contain 29 unique countries"
    )

    # Verify the exact country universe.
    require(
        set(integrated["cntry"]) == COUNTRIES,
        "Integrated country scope differs"
    )

    # The integrated dataset records the hash of its matches input.
    # Verify that it refers to the actual file currently being read.
    require(
        integrated["matched_input_sha256"].notna().all()
        and integrated["matched_input_sha256"].eq(
            hashes["matches"]
        ).all(),
        "Integrated provenance differs from temporal matches"
    )

    # Independently identify countries with multi-year fieldwork.
    selected = set(
        integrated.loc[
            integrated["multi_year_fieldwork_flag"].eq(True),
            "cntry"
        ]
    )

    # Exactly the expected 13 countries must be marked.
    require(
        selected == SWITCH_COUNTRIES,
        "Unexpected multi-year country scope"
    )

    # No country may have an unknown fieldwork flag.
    require(
        integrated["multi_year_fieldwork_flag"].notna().all(),
        "Missing multi-year fieldwork flags"
    )

    # Each match is identified by country, indicator and scenario.
    match_keys = [
        "cntry",
        "indicator_id",
        "alignment_scenario",
    ]

    # Verify the required source matching fields.
    for column in (
        *match_keys,
        "context_year",
        "requested_eurostat_year",
        "value",
        "observation_match_status",
        "eurostat_status",
    ):

        require(
            column in matches.columns,
            f"Temporal matches missing: {column}"
        )

    # Matching keys must be unique and non-missing.
    require(
        not matches.duplicated(match_keys).any()
        and not matches[match_keys].isna().any().any(),
        "Missing or duplicate temporal match keys"
    )

    # Construct 174 primary keys independently.
    expected_keys = {
        (country, indicator, "primary")
        for country in COUNTRIES
        for indicator in ECONOMIC_VARIABLES
    }

    # Add 78 alternative keys for the 13 multi-year countries.
    expected_keys |= {
        (country, indicator, "alternative")
        for country in SWITCH_COUNTRIES
        for indicator in ECONOMIC_VARIABLES
    }

    # Extract all actual source matching keys.
    actual_keys = set(
        matches[match_keys].itertuples(
            index=False,
            name=None
        )
    )

    # Require exactly 252 records with the expected coverage.
    require(
        len(matches) == 252
        and actual_keys == expected_keys,
        "Temporal matches coverage differs"
    )

    # Define permitted observation availability categories.
    allowed_statuses = {
        "available",
        "source_value_missing",
        "source_cell_absent",
    }

    # Reject unrecognized source availability categories.
    require(
        matches["observation_match_status"]
        .isin(allowed_statuses).all(),
        "Unexpected source observation status"
    )

    # Numeric availability must agree with the availability label.
    # A missing source value must never be classified as available.
    require(
        matches["observation_match_status"]
        .eq("available")
        .eq(matches["value"].notna())
        .all(),
        "Observation status and numeric availability disagree"
    )

    # The baseline correlation table has a two-column composite key.
    pair_columns = [
        "ess_variable",
        "economic_variable",
    ]

    # Verify the columns necessary for independent comparison.
    for column in (
        *pair_columns,
        "countries_n",
        "included_countries",
        "pearson_r",
        "spearman_rho",
        "input_sha256",
    ):

        require(
            column in baseline.columns,
            f"Baseline correlations missing: {column}"
        )

    # Construct the complete set of 36 expected analytic pairs.
    expected_pairs = {
        (ess, indicator)
        for ess in ESS_VARIABLES
        for indicator in ECONOMIC_VARIABLES
    }

    # Reject missing or duplicate baseline correlation pairs.
    require(
        len(baseline) == 36
        and not baseline.duplicated(pair_columns).any(),
        "Invalid baseline correlation structure"
    )

    # Confirm that all 36 expected pairs occur exactly once.
    require(
        set(
            baseline[pair_columns].itertuples(
                index=False,
                name=None
            )
        ) == expected_pairs,
        "Baseline pair coverage differs"
    )

    # The baseline correlations must use the same integrated input.
    require(
        baseline["input_sha256"].notna().all()
        and baseline["input_sha256"].eq(
            hashes["integrated"]
        ).all(),
        "Baseline input provenance differs"
    )

    print(
        "Input hashes, coverage and provenance: PASSED."
    )


# ============================================================
# 18. VALIDATE STEP 57 METADATA
# ============================================================

def check_metadata(metadata, hashes):

    # The historical output metadata must identify Step 57.
    require(
        metadata.get("step") == 57,
        "Unexpected creator step"
    )

    # Independent validation was pending when Step 57 ran.
    # Do not rewrite historical metadata during validation.
    require(
        metadata.get("validation_status")
        == "independent_validation_pending",
        "Unexpected historical validation status"
    )

    # Define the methodological properties of the analysis.
    static_fields = {
        "analysis_unit": "country",
        "across_country_weighting": "equal",
        "scenario_rule":
            "alternative year for multi-year countries; primary otherwise",
        "comparison_sample":
            "same countries with ESS and both economic values",
        "ess_estimates_changed": False,
        "missing_alternative_fallback": False,
        "imputation_performed": False,
        "gini_offset": 1,
        "paired_economic_rows_n": 174,
        "correlation_pairs_n": 36,
        "temporal_change_definition":
            "sensitivity common minus primary common",
        "sample_change_definition":
            "primary common minus primary full",
        "survey_uncertainty_propagated": False,
        "p_values_calculated": False,
        "country_list_separator": "|",
    }

    # Verify both the value and the Python type of each field.
    # This prevents values such as 0 from passing as False.
    for key, expected in static_fields.items():

        require(
            key in metadata
            and type(metadata[key]) is type(expected)
            and metadata[key] == expected,
            f"Metadata: invalid {key}"
        )

    # The metadata must list the exact 13 switching countries.
    require(
        metadata.get("switch_countries")
        == sorted(SWITCH_COUNTRIES),
        "Metadata country list differs"
    )

    # Parse the recorded creation date and time.
    timestamp = datetime.fromisoformat(
        metadata["created_at_utc"]
    )

    # The recorded timestamp must have UTC timezone information.
    require(
        timestamp.tzinfo is not None
        and timestamp.utcoffset().total_seconds() == 0,
        "Metadata timestamp must be UTC"
    )

    # Record library versions support computational reproducibility.
    for key in (
        "numpy_version",
        "pandas_version",
        "scipy_version",
    ):

        require(
            isinstance(metadata.get(key), str)
            and metadata[key].strip(),
            f"Missing library version: {key}"
        )

    # Exactly three source files must be recorded.
    require(
        isinstance(metadata.get("sources"), list)
        and len(metadata["sources"]) == 3,
        "Metadata must list three input files"
    )

    # Recalculate input file hashes instead of trusting metadata.
    expected_sources = [
        (INTEGRATED_PATH.name, hashes["integrated"]),
        (MATCHES_PATH.name, hashes["matches"]),
        (BASELINE_PATH.name, hashes["baseline"]),
    ]

    # Verify the exact original input manifest.
    require(
        metadata["sources"] == [
            {
                "filename": name,
                "sha256": digest,
            }
            for name, digest in expected_sources
        ],
        "Metadata input manifest differs"
    )

    # Four tabular output files were generated in Step 57.
    expected_outputs = [
        PAIRED_PATH,
        PAIRED_PATH.with_suffix(".csv"),
        CORRELATIONS_PATH,
        CORRELATIONS_PATH.with_suffix(".csv"),
    ]

    require(
        isinstance(metadata.get("outputs"), list)
        and len(metadata["outputs"]) == 4,
        "Metadata must list four output files"
    )

    # Independently hash each existing CSV and Parquet output.
    for record, path in zip(
        metadata["outputs"],
        expected_outputs
    ):

        require(
            record == {
                "filename": path.name,
                "sha256": sha256(path),
            },
            f"Output hash mismatch: {path.name}"
        )

    print(
        "Metadata and output SHA-256 checks: PASSED."
    )


# ============================================================
# 19. RECONSTRUCT ALL 174 ECONOMIC PAIRS
# ============================================================

def validate_paired_rows(integrated, matches, paired):

    # The saved paired table must contain 29 x 6 observations.
    require(
        len(paired) == 174,
        "Paired table must contain 174 rows"
    )

    # A country-indicator combination must occur only once.
    require(
        not paired.duplicated(
            ["cntry", "indicator_id"]
        ).any(),
        "Duplicate country-indicator pairs"
    )

    # Both parts of the composite key must be non-missing.
    require(
        paired["cntry"].notna().all()
        and paired["indicator_id"].notna().all(),
        "Missing paired-table keys"
    )

    # Create lookups from the original validated source tables.
    source = integrated.set_index("cntry")

    matches_lookup = matches.set_index(
        [
            "cntry",
            "indicator_id",
            "alignment_scenario",
        ]
    )

    # Track numerical differences and the expected row position.
    max_difference = 0.0
    row_index = 0

    # Reconstruct country-indicator combinations in output order.
    for country in sorted(COUNTRIES):

        # A multi-year country switches to its alternative year.
        switch = country in SWITCH_COUNTRIES

        # Single-year countries retain their primary observation.
        scenario = (
            "alternative"
            if switch
            else "primary"
        )

        for indicator in ECONOMIC_VARIABLES:

            # Retrieve the saved Step 57 row.
            row = paired.iloc[row_index]

            row_index += 1

            label = (
                f"Paired row {row_index}: "
                f"{country}/{indicator}"
            )

            # Obtain both observations from validated temporal matches.
            primary = matches_lookup.loc[
                (country, indicator, "primary")
            ]

            alternative = matches_lookup.loc[
                (country, indicator, scenario)
            ]

            # Read the original country context year.
            primary_year = int(
                source.loc[country, "context_year"]
            )

            # Only 2023 and 2024 are accepted in V1.
            require(
                primary_year in (2023, 2024),
                f"{label}: invalid primary context year"
            )

            # Independently derive the sensitivity context year.
            # Countries with two-year fieldwork switch years.
            # Single-year countries retain their original year.
            sensitivity_year = (
                2024 if primary_year == 2023 else 2023
            ) if switch else primary_year

            # Preserve the project's Gini year-offset rule.
            # Gini uses context_year + 1.
            # All other indicators use context_year.
            offset = (
                1
                if indicator == "gini"
                else 0
            )

            # Check the original context year against the matches source.
            verify_exact(
                int(primary["context_year"]),
                primary_year,
                f"{label}: primary context year"
            )

            # Check the sensitivity year against the source.
            verify_exact(
                int(alternative["context_year"]),
                sensitivity_year,
                f"{label}: sensitivity context year"
            )

            # Validate the primary requested Eurostat year.
            verify_exact(
                int(primary["requested_eurostat_year"]),
                primary_year + offset,
                f"{label}: primary Eurostat year"
            )

            # Validate the sensitivity requested Eurostat year.
            verify_exact(
                int(alternative["requested_eurostat_year"]),
                sensitivity_year + offset,
                f"{label}: sensitivity Eurostat year"
            )

            # Confirm year alignment in the integrated dataset.
            verify_exact(
                int(
                    source.loc[
                        country,
                        f"{indicator}__requested_eurostat_year"
                    ]
                ),
                primary_year + offset,
                f"{label}: integrated reference year"
            )

            # The primary economic value must match the integrated table.
            # Both refer to the exact same upstream observation.
            difference = verify_number(
                source.loc[country, indicator],
                primary["value"],
                f"{label}: integrated primary value",
                tolerance=0
            )

            max_difference = max(
                max_difference,
                difference
            )

            # Extract the original primary and selected economic values.
            primary_value = primary["value"]
            sensitivity_value = alternative["value"]

            # A country enters the common sample only if both exist.
            both_available = (
                not is_missing(primary_value)
                and not is_missing(sensitivity_value)
            )

            # Build the expected record independently of Step 57 output.
            expected = {
                "cntry": country,
                "indicator_id": indicator,
                "sensitivity_source_scenario": scenario,
                "temporal_switch_flag": switch,
                "primary_context_year": primary_year,
                "sensitivity_context_year": sensitivity_year,
                "primary_requested_eurostat_year":
                    primary_year + offset,
                "sensitivity_requested_eurostat_year":
                    sensitivity_year + offset,
                "primary_value": primary_value,
                "sensitivity_value": sensitivity_value,
                "primary_available_flag":
                    not is_missing(primary_value),
                "sensitivity_available_flag":
                    not is_missing(sensitivity_value),
                "both_available_flag": both_available,

                # Do not report a value change when a value is missing.
                # Use pd.NA rather than False for unknown comparisons.
                "value_changed_flag": (
                    bool(
                        float(primary_value)
                        != float(sensitivity_value)
                    )
                    if both_available
                    else pd.NA
                ),

                # Preserve source availability classifications.
                "primary_match_status":
                    primary["observation_match_status"],
                "sensitivity_match_status":
                    alternative["observation_match_status"],

                # Preserve Eurostat flags, including empty versus missing.
                "primary_eurostat_status":
                    primary["eurostat_status"],
                "sensitivity_eurostat_status":
                    alternative["eurostat_status"],
            }

            # Compare all 18 saved fields with reconstructed values.
            for column in PAIRED_COLUMNS:

                where = f"{label}.{column}"

                if column in PAIRED_FLOATS:

                    # Source economic observations must match exactly.
                    difference = verify_number(
                        row[column],
                        expected[column],
                        where,
                        tolerance=0
                    )

                    max_difference = max(
                        max_difference,
                        difference
                    )

                else:

                    # Codes, years, flags and statuses require exact equality.
                    verify_exact(
                        row[column],
                        expected[column],
                        where
                    )

    # ========================================================
    # 20. SPECIAL CASE: SWITZERLAND AND GINI
    # ========================================================

    # Switzerland has a primary Gini value but no alternative
    # Gini value for the required reference year.
    #
    # The alternative value must remain missing.
    # It must never be replaced with the primary observation.
    ch = paired.loc[
        paired["cntry"].eq("CH")
        & paired["indicator_id"].eq("gini")
    ]

    require(
        len(ch) == 1
        and not is_missing(
            ch.iloc[0]["primary_value"]
        )
        and is_missing(
            ch.iloc[0]["sensitivity_value"]
        )
        and not bool(
            ch.iloc[0]["both_available_flag"]
        ),
        "Switzerland alternative Gini must remain missing"
    )

    print(
        "174 country-indicator pairs, years, values, "
        "flags and source statuses: PASSED."
    )

    return max_difference

# ============================================================
# 21. INDEPENDENT PEARSON CORRELATION
# ============================================================

def independent_pearson(x, y):

    # Both variables must have the same number of observations.
    # A minimum of three paired countries is required here.
    require(
        len(x) == len(y) and len(x) >= 3,
        "Correlation requires at least three paired countries"
    )

    # Reject missing or infinite correlation inputs.
    require(
        all(
            math.isfinite(value)
            for value in x + y
        ),
        "Non-finite correlation input"
    )

    # Calculate arithmetic means using high-precision summation.
    # This implementation does not use numpy.corrcoef.
    mean_x = math.fsum(x) / len(x)
    mean_y = math.fsum(y) / len(y)

    # Center each observation by subtracting its variable mean.
    centered_x = [
        value - mean_x
        for value in x
    ]

    centered_y = [
        value - mean_y
        for value in y
    ]

    # Calculate the centered cross-product sum.
    numerator = math.fsum(
        a * b
        for a, b in zip(centered_x, centered_y)
    )

    # Calculate the centered sum of squares for X.
    squared_x = math.fsum(
        value * value
        for value in centered_x
    )

    # Calculate the centered sum of squares for Y.
    squared_y = math.fsum(
        value * value
        for value in centered_y
    )

    # Pearson denominator is the geometric mean
    # of the two centered sums of squares.
    denominator = math.sqrt(
        squared_x * squared_y
    )

    # Pearson correlation is undefined for constant variables.
    require(
        denominator > 0,
        "Constant variable in correlation sample"
    )

    # Calculate Pearson's correlation coefficient.
    result = numerator / denominator

    # Correlations must be finite and within [-1, 1].
    require(
        math.isfinite(result)
        and abs(result) <= 1 + ATOL,
        "Invalid Pearson correlation result"
    )

    # Restrict insignificant numerical boundary deviations.
    return max(
        -1.0,
        min(1.0, result)
    )


# ============================================================
# 22. INDEPENDENT AVERAGE RANKS FOR SPEARMAN
# ============================================================

def independent_average_ranks(values):

    # Spearman correlation uses ranked observations.
    #
    # Step 57 used scipy.stats.rankdata.
    # This validator constructs ranks without SciPy.
    #
    # Tied observations receive their average rank.
    #
    # Example:
    # Values: 10, 20, 20, 30
    # Ranks:   1, 2.5, 2.5, 4

    ranks = []

    for value in values:

        # Count how many observations are smaller.
        less = sum(
            other < value
            for other in values
        )

        # Count observations equal to the current value.
        ties = sum(
            other == value
            for other in values
        )

        # Compute the average occupied rank.
        rank = less + (ties + 1) / 2

        ranks.append(rank)

    return ranks


# ============================================================
# 23. CALCULATE BOTH CORRELATION METHODS
# ============================================================

def independent_correlations(x, y):

    # Pearson operates on the original numerical observations.
    pearson = independent_pearson(x, y)

    # Spearman is Pearson correlation applied to ranked values.
    #
    # Reconstruct ranks separately for every scenario and sample.
    # Country exclusions may change the ranking structure.
    ranks_x = independent_average_ranks(x)
    ranks_y = independent_average_ranks(y)

    spearman = independent_pearson(
        ranks_x,
        ranks_y
    )

    return {
        "pearson": pearson,
        "spearman": spearman,
    }


# ============================================================
# 24. INDEPENDENTLY VALIDATE ALL 36 CORRELATION PAIRS
# ============================================================

def validate_correlation_rows(
    integrated,
    paired,
    baseline,
    comparisons,
):

    # Require exactly 6 ESS variables x 6 economic indicators.
    require(
        len(comparisons) == 36,
        "Correlation table must contain 36 rows"
    )

    # Each analytic pair must be unique.
    require(
        not comparisons.duplicated(
            ["ess_variable", "economic_variable"]
        ).any(),
        "Duplicate correlation pairs"
    )

    # Index original ESS estimates by country.
    country_ess = integrated.set_index("cntry")

    # Index previously validated baseline correlations.
    baseline_lookup = baseline.set_index(
        ["ess_variable", "economic_variable"]
    )

    # Track the largest observed numerical discrepancies.
    max_baseline = 0.0
    max_recalculated = 0.0

    row_number = 0

    # Iterate all 36 combinations in the expected output order.
    for ess in ESS_VARIABLES:

        for indicator in ECONOMIC_VARIABLES:

            # Retrieve the saved Step 57 correlation record.
            row = comparisons.iloc[row_number]

            row_number += 1

            label = (
                f"Correlation row {row_number}: "
                f"{ess}/{indicator}"
            )

            # Verify the exact ESS outcome name.
            verify_exact(
                row["ess_variable"],
                ess,
                f"{label}: ESS variable"
            )

            # Verify the exact economic indicator name.
            verify_exact(
                row["economic_variable"],
                indicator,
                f"{label}: economic variable"
            )

            # Select all candidate countries for this economic indicator.
            indicator_rows = paired.loc[
                paired["indicator_id"].eq(indicator)
            ]

            # Build two separate country samples.
            #
            # full:
            # Countries with valid primary economic and ESS values.
            #
            # common:
            # Countries with valid primary, sensitivity and ESS values.
            full = []
            common = []

            # Reconstruct sample membership country by country.
            for record in indicator_rows.itertuples(index=False):

                country = record.cntry

                # Retrieve the same fixed ESS estimate for both scenarios.
                outcome = country_ess.loc[
                    country,
                    ess
                ]

                # Countries with missing ESS outcomes are excluded
                # only from the corresponding correlation pair.
                if is_missing(outcome):
                    continue

                # Valid ESS estimates must be finite.
                require(
                    math.isfinite(float(outcome)),
                    f"{label}: non-finite ESS estimate"
                )

                # Construct the full primary sample.
                if bool(record.primary_available_flag):

                    full.append(
                        (
                            country,
                            float(record.primary_value),
                            float(outcome),
                        )
                    )

                # Construct the common paired sample.
                if bool(record.both_available_flag):

                    common.append(
                        (
                            country,
                            float(record.primary_value),
                            float(record.sensitivity_value),
                            float(outcome),
                            bool(record.temporal_switch_flag),
                            bool(record.value_changed_flag),
                        )
                    )

            # ====================================================
            # PRIMARY FULL
            # ====================================================

            # Recalculate the original correlations on the full
            # country sample used for the main analysis.
            primary_full = independent_correlations(
                [value[1] for value in full],
                [value[2] for value in full]
            )

            # ====================================================
            # PRIMARY COMMON
            # ====================================================

            # Recalculate primary correlations only for countries
            # available in both temporal scenarios.
            primary_common = independent_correlations(
                [value[1] for value in common],
                [value[3] for value in common]
            )

            # ====================================================
            # SENSITIVITY COMMON
            # ====================================================

            # Replace only the economic observations.
            #
            # Countries and ESS values remain identical to the
            # primary common scenario.
            sensitivity_common = independent_correlations(
                [value[2] for value in common],
                [value[3] for value in common]
            )

            # Retrieve the original validated baseline results.
            base = baseline_lookup.loc[
                (ess, indicator)
            ]

            # Reconstruct the complete primary country list.
            full_countries = "|".join(
                value[0]
                for value in full
            )

            # Reconstruct the identical common country list.
            common_countries = "|".join(
                value[0]
                for value in common
            )

            # Independently identify lost countries.
            removed = "|".join(
                sorted(
                    set(value[0] for value in full)
                    - set(value[0] for value in common)
                )
            )

            # Independently calculate all sample counts.
            expected_counts = {
                "primary_full_n": len(full),
                "common_countries_n": len(common),
                "switched_context_countries_n": sum(
                    value[4]
                    for value in common
                ),
                "changed_numeric_values_n": sum(
                    value[5]
                    for value in common
                ),
            }

            # Every saved count must match exactly.
            for column, count in expected_counts.items():

                verify_exact(
                    row[column],
                    count,
                    f"{label}.{column}"
                )

            # Verify the exact common-sample country list.
            verify_exact(
                row["common_countries"],
                common_countries,
                f"{label}: common countries"
            )

            # Verify countries excluded from the original sample.
            verify_exact(
                row["removed_from_primary_sample"],
                removed,
                f"{label}: removed countries"
            )

            # Baseline sample size must match Step 53.
            verify_exact(
                base["countries_n"],
                len(full),
                f"{label}: baseline N"
            )

            # Baseline country composition must match Step 53.
            verify_exact(
                base["included_countries"],
                full_countries,
                f"{label}: baseline countries"
            )

            # Switzerland must be the only country lost
            # when constructing common samples for Gini.
            #
            # No primary-to-common losses are expected
            # for the other five indicators.
            verify_exact(
                removed,
                "CH" if indicator == "gini" else "",
                f"{label}: unexpected sample loss"
            )

            # Common sample sizes must remain between 24 and 28.
            require(
                24 <= len(common) <= 28,
                f"{label}: unexpected common sample size"
            )

            # ====================================================
            # VERIFY PEARSON AND SPEARMAN
            # ====================================================

            for method, base_column in (
                ("pearson", "pearson_r"),
                ("spearman", "spearman_rho"),
            ):

                # Compare the independently recalculated
                # full-primary result with the validated baseline.
                baseline_difference = verify_number(
                    primary_full[method],
                    base[base_column],
                    f"{label}: baseline {method}"
                )

                max_baseline = max(
                    max_baseline,
                    baseline_difference
                )

                # Reconstruct all saved numerical results.
                #
                # temporal_change:
                # sensitivity_common - primary_common
                #
                # sample_change:
                # primary_common - primary_full
                #
                # This separates temporal sensitivity from
                # changes caused by country sample composition.
                expected_results = {
                    "primary_full":
                        primary_full[method],
                    "primary_common":
                        primary_common[method],
                    "sensitivity_common":
                        sensitivity_common[method],
                    "temporal_change":
                        sensitivity_common[method]
                        - primary_common[method],
                    "sample_change":
                        primary_common[method]
                        - primary_full[method],
                }

                # Verify all coefficients and calculated differences.
                for suffix, coefficient in expected_results.items():

                    difference = verify_number(
                        row[f"{method}_{suffix}"],
                        coefficient,
                        f"{label}.{method}_{suffix}"
                    )

                    max_recalculated = max(
                        max_recalculated,
                        difference
                    )

                # ====================================================
                # VERIFY SIGN REVERSALS
                # ====================================================

                # A sign reversal exists only if the product
                # of the two coefficients is strictly negative.
                #
                # A zero coefficient is not a strict sign reversal.
                sign_flip = (
                    primary_common[method]
                    * sensitivity_common[method]
                    < 0
                )

                verify_exact(
                    row[f"{method}_temporal_sign_flip"],
                    sign_flip,
                    f"{label}: {method} sign flip"
                )

    # ========================================================
    # VERIFY OVERALL SIGN-REVERSAL COUNTS
    # ========================================================

    # Step 57 reported exactly one Pearson sign reversal.
    require(
        int(
            comparisons["pearson_temporal_sign_flip"].sum()
        ) == 1,
        "Expected exactly one Pearson sign reversal"
    )

    # Step 57 reported exactly two Spearman sign reversals.
    require(
        int(
            comparisons["spearman_temporal_sign_flip"].sum()
        ) == 2,
        "Expected exactly two Spearman sign reversals"
    )

    print(
        "36 pairs, identical country samples "
        "and independent correlations: PASSED."
    )

    print(
        "Maximum baseline coefficient difference: "
        f"{max_baseline:.3e}"
    )

    print(
        "Maximum Step 57 recalculation difference: "
        f"{max_recalculated:.3e}"
    )

    return max_baseline, max_recalculated


# ============================================================
# 25. MAIN VALIDATION FUNCTION
# ============================================================

def main():

    # Display the validation title.
    print("=" * 88)

    print(
        "STEP 58 - INDEPENDENT TEMPORAL "
        "CORRELATION SENSITIVITY VALIDATION"
    )

    print("=" * 88)

    # ========================================================
    # LOAD PREVIOUSLY VALIDATED INPUTS
    # ========================================================

    # Read the integrated ESS-Eurostat dataset.
    integrated, integrated_hash = read_parquet(
        INTEGRATED_PATH
    )

    # Read the temporal matching table.
    matches, matches_hash = read_parquet(
        MATCHES_PATH
    )

    # Read the validated baseline correlations.
    baseline, baseline_hash = read_parquet(
        BASELINE_PATH
    )

    # ========================================================
    # LOAD STEP 57 OUTPUTS
    # ========================================================

    # Read saved primary and alternative economic values.
    paired, _ = read_parquet(
        PAIRED_PATH
    )

    # Read saved temporal correlation comparisons.
    comparisons, _ = read_parquet(
        CORRELATIONS_PATH
    )

    # ========================================================
    # LOAD ORIGINAL METADATA
    # ========================================================

    # Metadata must already exist.
    require(
        METADATA_PATH.is_file(),
        f"Metadata not found: {METADATA_PATH}"
    )

    # Open metadata in read-only mode.
    with METADATA_PATH.open(
        "r",
        encoding="utf-8"
    ) as handle:

        metadata = json.load(handle)

    # Group input file hashes for provenance checks.
    hashes = {
        "integrated": integrated_hash,
        "matches": matches_hash,
        "baseline": baseline_hash,
    }

    # ========================================================
    # VALIDATE UPSTREAM DATA AND METADATA
    # ========================================================

    # Confirm source integrity, country coverage and provenance.
    check_sources(
        integrated,
        matches,
        baseline,
        hashes
    )

    # Verify methodological metadata and output hashes.
    check_metadata(
        metadata,
        hashes
    )

    # ========================================================
    # VALIDATE PARQUET SCHEMAS
    # ========================================================

    # Verify 18 paired-value columns and their expected dtypes.
    check_columns_and_types(
        paired,
        PAIRED_COLUMNS,
        PAIRED_STRING,
        PAIRED_INTS,
        PAIRED_FLOATS,
        PAIRED_BOOLS,
        "Paired Parquet",
        {"value_changed_flag"},
    )

    # Verify 20 correlation columns and their expected dtypes.
    check_columns_and_types(
        comparisons,
        CORRELATION_COLUMNS,
        CORRELATION_STRING,
        CORRELATION_INTS,
        CORRELATION_FLOATS,
        CORRELATION_BOOLS,
        "Correlations Parquet",
    )

    print(
        "Output schemas and pandas dtypes: PASSED."
    )

    # ========================================================
    # INDEPENDENTLY RECONSTRUCT ECONOMIC VALUES
    # ========================================================

    # Verify all 174 country-indicator pairs against original matches.
    difference_values = validate_paired_rows(
        integrated,
        matches,
        paired
    )

    # ========================================================
    # INDEPENDENTLY RECALCULATE CORRELATIONS
    # ========================================================

    # Recalculate primary-full, primary-common and
    # sensitivity-common coefficients for all 36 analytic pairs.
    baseline_diff, corr_diff = validate_correlation_rows(
        integrated,
        paired,
        baseline,
        comparisons
    )

    # ========================================================
    # VALIDATE CSV / PARQUET CONSISTENCY
    # ========================================================

    # Compare every cell in the paired-values CSV and Parquet.
    csv_difference_values = check_csv_against_parquet(
        paired,
        PAIRED_PATH.with_suffix(".csv"),
        PAIRED_STRING,
        PAIRED_INTS,
        PAIRED_FLOATS,
        PAIRED_BOOLS,
        {"value_changed_flag"},
        "Paired values",
    )

    # Compare every cell in the correlations CSV and Parquet.
    csv_difference_correlations = check_csv_against_parquet(
        comparisons,
        CORRELATIONS_PATH.with_suffix(".csv"),
        CORRELATION_STRING,
        CORRELATION_INTS,
        CORRELATION_FLOATS,
        CORRELATION_BOOLS,
        set(),
        "Correlations",
    )

    print(
        "CSV versus Parquet and independently reconstructed "
        "source missingness: PASSED."
    )

    # ========================================================
    # PRINT FINAL VALIDATION SUMMARY
    # ========================================================

    print("\nSUMMARY")

    # Number of independently verified country-indicator rows.
    print(
        "Validated paired economic rows:",
        len(paired)
    )

    # Number of independently verified analytic pairs.
    print(
        "Validated correlation comparisons:",
        len(comparisons)
    )

    # 36 pairs x 2 correlation methods x 3 scenarios.
    print(
        "Independent Pearson and Spearman coefficients:",
        len(comparisons) * 2 * 3
    )

    # Number of countries using an alternative context year.
    print(
        "Switched countries:",
        len(SWITCH_COUNTRIES)
    )

    # Minimum and maximum country sample sizes.
    print(
        "Common sample N range:",
        comparisons["common_countries_n"].min(),
        "to",
        comparisons["common_countries_n"].max()
    )

    # Count strict Pearson sign reversals.
    print(
        "Pearson sign changes:",
        int(
            comparisons["pearson_temporal_sign_flip"].sum()
        )
    )

    # Count strict Spearman sign reversals.
    print(
        "Spearman sign changes:",
        int(
            comparisons["spearman_temporal_sign_flip"].sum()
        )
    )

    # Report the maximum numerical discrepancies.
    print(
        "Maximum paired numerical difference:",
        f"{difference_values:.3e}"
    )

    print(
        "Maximum baseline coefficient difference:",
        f"{baseline_diff:.3e}"
    )

    print(
        "Maximum recalculated coefficient difference:",
        f"{corr_diff:.3e}"
    )

    print(
        "Maximum paired CSV numerical difference:",
        f"{csv_difference_values:.3e}"
    )

    print(
        "Maximum correlations CSV numerical difference:",
        f"{csv_difference_correlations:.3e}"
    )

    # ========================================================
    # FINAL SUCCESS MESSAGE
    # ========================================================

    # This message appears only if every validation passed.
    print(
        "\nINDEPENDENT TEMPORAL SENSITIVITY VALIDATION PASSED."
    )

    # The script contains no operations that modify project files.
    print(
        "No project files were created or modified."
    )


# ============================================================
# 26. EXECUTE THE VALIDATOR
# ============================================================

# Execute main() only when this file is run directly.
# Importing the module does not trigger project validation.
if __name__ == "__main__":
    main()
