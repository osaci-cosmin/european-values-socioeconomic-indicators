
"""Step 60: read-only independent validation of the Step 59 robustness synthesis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 1. PROJECT PATHS AND FIXED ANALYTICAL DEFINITIONS
# ------------------------------------------------------------

# Find the project root relative to this script.
# This works regardless of the current terminal directory.
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

# The four analytical source tables and two historical metadata files.
# All are read-only inputs for this validation.
SOURCE_PATHS = {
    "baseline": TABLES / "ESS11_country_correlations_main_v1.parquet",
    "loo_detail": TABLES / "ESS11_country_correlations_loo_detail_v1.parquet",
    "loo_summary": TABLES / "ESS11_country_correlations_loo_summary_v1.parquet",
    "temporal": TABLES / "ESS11_temporal_sensitivity_correlations_v1.parquet",
    "loo_metadata": TABLES / "ESS11_country_correlations_loo_v1.metadata.json",
    "temporal_metadata": TABLES / "ESS11_temporal_correlation_sensitivity_v1.metadata.json",
}

# The three outputs previously created by Step 59.
OUTPUT_STEM = "ESS11_country_correlation_robustness_synthesis_v1"
RESULT_PARQUET = TABLES / f"{OUTPUT_STEM}.parquet"
RESULT_CSV = TABLES / f"{OUTPUT_STEM}.csv"
RESULT_METADATA = TABLES / f"{OUTPUT_STEM}.metadata.json"

# Define the exact analytical variables independently.
ESS = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

ECON = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

METHODS = ("pearson", "spearman")
PAIR = ("ess_variable", "economic_variable")

# Six ESS outcomes times six economic variables give 36 pairs.
EXPECTED_PAIRS = [(e, x) for e in ESS for x in ECON]

# Preserve the country universe used by the validated V1 project.
COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# This is the already validated SHA-256 of the integrated dataset.
INTEGRATED_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

# The tolerance permits negligible floating-point differences.
TOL = 1e-12


# ------------------------------------------------------------
# 2. DEFINE THE EXACT EXPECTED OUTPUT SCHEMA
# ------------------------------------------------------------

# General fields shared by both correlation methods.
BASE_COLS = [
    "ess_variable",
    "economic_variable",
    "countries_n",
    "included_countries",
    "common_countries_n",
    "common_countries",
    "removed_from_primary_sample",
    "switched_context_countries_n",
    "changed_numeric_values_n",
    "pearson_spearman_same_strict_sign",
    "pearson_minus_spearman",
]

# Fields expected separately for Pearson and Spearman.
METHOD_SUFFIXES = [
    "baseline",
    "loo_min",
    "loo_max",
    "loo_max_abs_change",
    "loo_largest_change_countries",
    "loo_sign_flip_n",
    "loo_sign_flip_countries",
    "loo_any_sign_flip",
    "primary_common",
    "sensitivity_common",
    "temporal_change",
    "sample_change",
    "temporal_sign_flip",
]

# Construct the complete 37-column schema.
OUTPUT_COLS = BASE_COLS + [
    f"{method}_{suffix}"
    for method in METHODS
    for suffix in METHOD_SUFFIXES
]

# Identify text fields explicitly.
TEXT_COLS = {
    "ess_variable",
    "economic_variable",
    "included_countries",
    "common_countries",
    "removed_from_primary_sample",
} | {
    f"{method}_{suffix}"
    for method in METHODS
    for suffix in (
        "loo_largest_change_countries",
        "loo_sign_flip_countries",
    )
}

# Identify integer fields explicitly.
INT_COLS = {
    "countries_n",
    "common_countries_n",
    "switched_context_countries_n",
    "changed_numeric_values_n",
} | {
    f"{method}_loo_sign_flip_n"
    for method in METHODS
}

# Identify Boolean fields explicitly.
BOOL_COLS = {
    "pearson_spearman_same_strict_sign"
} | {
    f"{method}_{suffix}"
    for method in METHODS
    for suffix in (
        "loo_any_sign_flip",
        "temporal_sign_flip",
    )
}

# All remaining fields contain floating-point coefficients.
FLOAT_COLS = (
    set(OUTPUT_COLS)
    - TEXT_COLS
    - INT_COLS
    - BOOL_COLS
)


# ------------------------------------------------------------
# 3. INDEPENDENT VALIDATION HELPERS
# ------------------------------------------------------------

def ensure(condition, message):

    # Stop immediately when a required condition is false.
    # A validator must never silently repair its input data.
    if not condition:
        raise AssertionError(message)


def sha256(path):

    # Calculate the SHA-256 fingerprint from the original bytes.
    # Reading in chunks avoids loading the entire file into memory.
    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def numeric_match(actual, expected, where):

    # Reject missing numerical observations.
    # Missing must not be converted to zero.
    ensure(
        not pd.isna(actual) and not pd.isna(expected),
        f"{where}: missing number",
    )

    a = float(actual)
    b = float(expected)

    # Reject NaN, infinity, and negative infinity.
    ensure(
        math.isfinite(a) and math.isfinite(b),
        f"{where}: non-finite number",
    )

    difference = abs(a - b)

    # Use absolute tolerance with no relative tolerance.
    ensure(
        difference <= TOL,
        f"{where}: {a!r} != {b!r}",
    )

    return difference


def strict_value(actual, expected, where):

    # Compare non-floating-point fields exactly.
    # This includes country codes, counts and Boolean flags.
    ensure(
        not pd.isna(actual),
        f"{where}: unexpected null",
    )

    ensure(
        actual == expected,
        f"{where}: {actual!r} != {expected!r}",
    )


def codes(value, where):

    # Historical project country lists use the "|" separator.
    # An empty string means that there are no countries.
    ensure(
        isinstance(value, str),
        f"{where}: country list is not text",
    )

    items = value.split("|") if value else []

    # A country may not appear twice in the same list.
    ensure(
        len(items) == len(set(items)),
        f"{where}: duplicate country",
    )

    # No country outside the validated 29-country universe is allowed.
    ensure(
        set(items) <= COUNTRIES,
        f"{where}: unexpected country code",
    )

    return items


def require_columns(table, columns, name):

    # Check that all required source fields are present.
    missing = sorted(
        set(columns) - set(table.columns)
    )

    ensure(
        not missing,
        f"{name}: missing columns: {missing}",
    )

    # Duplicate column names make selection ambiguous.
    ensure(
        table.columns.is_unique,
        f"{name}: duplicated column names",
    )


def require_pairs(table, name):

    # Every summary must contain each of the 36 pairs exactly once.
    require_columns(table, PAIR, name)

    ensure(
        table[list(PAIR)].notna().all().all(),
        f"{name}: null pair identifiers",
    )

    seen = [
        tuple(row)
        for row in table[
            list(PAIR)
        ].itertuples(index=False, name=None)
    ]

    ensure(
        len(seen) == 36
        and set(seen) == set(EXPECTED_PAIRS),
        f"{name}: wrong 36-pair coverage",
    )

    ensure(
        len(seen) == len(set(seen)),
        f"{name}: duplicated pairs",
    )


# ------------------------------------------------------------
# 4. VERIFY THE STEP 59 OUTPUT SCHEMA AND DTYPES
# ------------------------------------------------------------

def inspect_output_schema(result):

    # Column ORDER matters for reproducible exports and BI integration.
    ensure(
        list(result.columns) == OUTPUT_COLS,
        "Step 59 Parquet schema/order differs",
    )

    ensure(
        len(OUTPUT_COLS) == 37,
        "Independent schema definition is not 37 columns",
    )

    ensure(
        len(result) == 36,
        "Step 59 output does not have 36 rows",
    )

    require_pairs(result, "Step 59 Parquet")

    # Validate pandas data types, not just the displayed values.
    for col in OUTPUT_COLS:

        dtype = result[col].dtype

        if col in TEXT_COLS:

            ensure(
                isinstance(dtype, pd.StringDtype),
                f"{col}: expected pandas StringDtype, got {dtype}",
            )

        elif col in INT_COLS:

            ensure(
                str(dtype) == "int64",
                f"{col}: expected int64, got {dtype}",
            )

        elif col in BOOL_COLS:

            ensure(
                str(dtype) == "bool",
                f"{col}: expected bool, got {dtype}",
            )

        else:

            ensure(
                str(dtype) == "float64",
                f"{col}: expected float64, got {dtype}",
            )

    # Every synthesis cell must contain a valid, non-null value.
    # Empty country lists are represented by "", not missing data.
    ensure(
        not result.isna().any().any(),
        "Step 59 output contains unexpected nulls",
    )

    print(
        "Output columns, order, pandas dtypes "
        "and non-null values: PASSED."
    )


# ------------------------------------------------------------
# 5. RECONSTRUCT THE SYNTHESIS FROM ORIGINAL RESULT TABLES
# ------------------------------------------------------------

def reconstruct(baseline, details, loo_summary, temporal):

    # Rebuild each expected result independently.
    # Do not call or import any function from Step 59.
    require_pairs(baseline, "Baseline")
    require_pairs(loo_summary, "LOO summary")
    require_pairs(temporal, "Temporal sensitivity")

    # Check the required columns of the baseline correlation table.
    require_columns(
        baseline,
        [
            *PAIR,
            "countries_n",
            "included_countries",
            "pearson_r",
            "spearman_rho",
            "input_sha256",
        ],
        "Baseline",
    )

    # Check the original country-omission records.
    require_columns(
        details,
        [
            *PAIR,
            "omitted_country",
            "pearson_full",
            "pearson_loo",
            "pearson_sign_flip",
            "spearman_full",
            "spearman_loo",
            "spearman_sign_flip",
        ],
        "LOO detail",
    )

    # Check the saved maximum-variation LOO summary.
    require_columns(
        loo_summary,
        [
            *PAIR,
            *[
                f"{m}_{suffix}"
                for m in METHODS
                for suffix in (
                    "full",
                    "max_abs_change",
                    "largest_change_countries",
                )
            ],
        ],
        "LOO summary",
    )

    # Check the saved temporal-sensitivity comparison table.
    require_columns(
        temporal,
        [
            *PAIR,
            "primary_full_n",
            "common_countries_n",
            "common_countries",
            "removed_from_primary_sample",
            "switched_context_countries_n",
            "changed_numeric_values_n",
            *[
                f"{m}_{suffix}"
                for m in METHODS
                for suffix in (
                    "primary_full",
                    "primary_common",
                    "sensitivity_common",
                    "temporal_change",
                    "sample_change",
                    "temporal_sign_flip",
                )
            ],
        ],
        "Temporal sensitivity",
    )

    # Make sure the baseline comes from the validated integrated data.
    ensure(
        baseline["input_sha256"].notna().all(),
        "Missing baseline provenance",
    )

    ensure(
        baseline["input_sha256"].eq(
            INTEGRATED_SHA256
        ).all(),
        "Baseline does not match the validated integrated dataset",
    )

    # Confirm complete LOO coverage before reconstructing anything.
    ensure(
        len(details) == 948,
        "Expected 948 LOO omission rows",
    )

    ensure(
        details[
            list(PAIR) + ["omitted_country"]
        ].notna().all().all(),
        "Missing LOO identifiers",
    )

    ensure(
        not details.duplicated(
            list(PAIR) + ["omitted_country"]
        ).any(),
        "Repeated LOO omitted-country key",
    )

    detail_pairs = set(
        map(
            tuple,
            details[
                list(PAIR)
            ].itertuples(index=False, name=None),
        )
    )

    ensure(
        detail_pairs == set(EXPECTED_PAIRS),
        "Unexpected LOO pair coverage",
    )

    # Index the three summary tables by their analytical pair.
    base_ix = baseline.set_index(list(PAIR))
    summary_ix = loo_summary.set_index(list(PAIR))
    time_ix = temporal.set_index(list(PAIR))

    # Group the 948 detailed records by ESS-economic relationship.
    grouped = dict(
        tuple(
            details.groupby(
                list(PAIR),
                sort=False,
            )
        )
    )

    reconstructed = []
    checked_omissions = 0

    # Process all 36 pairs in the independently defined order.
    for pair in EXPECTED_PAIRS:

        b = base_ix.loc[pair]
        s = summary_ix.loc[pair]
        t = time_ix.loc[pair]
        omit = grouped[pair]

        tag = " / ".join(pair)

        # ----------------------------------------------------
        # 5.1 VERIFY THE ORIGINAL COUNTRY SAMPLE
        # ----------------------------------------------------

        original_countries = codes(
            b["included_countries"],
            f"{tag} baseline countries",
        )

        omitted_countries = omit[
            "omitted_country"
        ].astype(str).tolist()

        # Confirm the baseline sample size.
        ensure(
            len(original_countries)
            == int(b["countries_n"]),
            f"{tag}: incorrect baseline N",
        )

        # Each eligible country must be omitted exactly once.
        ensure(
            len(omitted_countries) == len(original_countries)
            and set(omitted_countries) == set(original_countries),
            f"{tag}: LOO does not omit every baseline country exactly once",
        )

        # Step 57's full primary sample must equal the baseline sample.
        ensure(
            int(t["primary_full_n"])
            == len(original_countries),
            f"{tag}: temporal N differs from baseline",
        )

        checked_omissions += len(omitted_countries)

        # ----------------------------------------------------
        # 5.2 VERIFY THE TEMPORAL COMMON SAMPLE
        # ----------------------------------------------------

        common = codes(
            t["common_countries"],
            f"{tag} common countries",
        )

        removed = codes(
            t["removed_from_primary_sample"],
            f"{tag} removed countries",
        )

        ensure(
            len(common)
            == int(t["common_countries_n"]),
            f"{tag}: common sample N differs",
        )

        # Common countries must belong to the original sample.
        ensure(
            set(common) <= set(original_countries),
            f"{tag}: common sample is not a baseline subset",
        )

        # Verify that removed countries are exactly the set difference.
        ensure(
            set(removed)
            == set(original_countries) - set(common),
            f"{tag}: removed countries do not reconcile",
        )

        # Switzerland is removed only from Gini common comparisons.
        # It remains in the primary full Gini analysis.
        ensure(
            removed == (
                ["CH"] if pair[1] == "gini" else []
            ),
            f"{tag}: unexpected common-sample loss",
        )

        switched_n = int(
            t["switched_context_countries_n"]
        )

        changed_n = int(
            t["changed_numeric_values_n"]
        )

        # Reject impossible count values.
        ensure(
            0 <= switched_n <= len(common),
            f"{tag}: invalid switched count",
        )

        ensure(
            0 <= changed_n <= len(common),
            f"{tag}: invalid changed-value count",
        )

        # ----------------------------------------------------
        # 5.3 RECONSTRUCT GENERAL SUMMARY FIELDS
        # ----------------------------------------------------

        pearson = float(b["pearson_r"])
        spearman = float(b["spearman_rho"])

        # Correlations must lie within [-1, +1].
        ensure(
            all(
                math.isfinite(x) and -1 <= x <= 1
                for x in (pearson, spearman)
            ),
            f"{tag}: invalid baseline coefficient",
        )

        record = {
            "ess_variable": pair[0],
            "economic_variable": pair[1],
            "countries_n": len(original_countries),
            "included_countries":
                "|".join(original_countries),
            "common_countries_n": len(common),
            "common_countries": "|".join(common),
            "removed_from_primary_sample":
                "|".join(removed),
            "switched_context_countries_n": switched_n,
            "changed_numeric_values_n": changed_n,

            # Strict sign agreement requires both coefficients
            # to be positive or both to be negative.
            "pearson_spearman_same_strict_sign":
                pearson * spearman > 0,

            # Preserve the numerical difference between methods.
            "pearson_minus_spearman":
                pearson - spearman,
        }

        # ----------------------------------------------------
        # 5.4 VALIDATE PEARSON AND SPEARMAN INDEPENDENTLY
        # ----------------------------------------------------

        for method, full in (
            ("pearson", pearson),
            ("spearman", spearman),
        ):

            # The LOO summary must reproduce the main coefficient.
            numeric_match(
                s[f"{method}_full"],
                full,
                f"{tag}: {method} LOO full",
            )

            # The temporal full coefficient must also agree.
            numeric_match(
                t[f"{method}_primary_full"],
                full,
                f"{tag}: {method} temporal primary full",
            )

            reduced_values = []
            abs_changes = {}
            reversed_sign_countries = []

            # ------------------------------------------------
            # 5.5 RECONSTRUCT EACH COUNTRY-OMISSION RESULT
            # ------------------------------------------------

            for _, observation in omit.iterrows():

                country = str(
                    observation["omitted_country"]
                )

                coefficient = float(
                    observation[f"{method}_loo"]
                )

                # Check the baseline repeated in each detail row.
                numeric_match(
                    observation[f"{method}_full"],
                    full,
                    f"{tag}: {method} repeated full for {country}",
                )

                ensure(
                    math.isfinite(coefficient)
                    and -1 <= coefficient <= 1,
                    f"{tag}: invalid LOO coefficient for {country}",
                )

                # Independently reconstruct the strict sign flip.
                actual_flip = full * coefficient < 0

                strict_value(
                    observation[f"{method}_sign_flip"],
                    actual_flip,
                    f"{tag}: {method} sign flag for {country}",
                )

                # Preserve every reduced-sample coefficient.
                reduced_values.append(coefficient)

                # Measure the absolute coefficient change.
                abs_changes[country] = abs(
                    coefficient - full
                )

                if actual_flip:
                    reversed_sign_countries.append(country)

            # ------------------------------------------------
            # 5.6 RECONSTRUCT MAXIMUM LOO CHANGES
            # ------------------------------------------------

            maximum = max(
                abs_changes.values()
            )

            # Keep all exact ties for the largest change.
            tied_countries = sorted(
                country
                for country, amount in abs_changes.items()
                if amount == maximum
            )

            # Compare the independent maximum with Step 55.
            numeric_match(
                s[f"{method}_max_abs_change"],
                maximum,
                f"{tag}: {method} LOO maximum difference",
            )

            summary_countries = codes(
                s[f"{method}_largest_change_countries"],
                f"{tag} LOO summary countries",
            )

            ensure(
                summary_countries == tied_countries,
                f"{tag}: {method} summary maximum country set differs",
            )

            # ------------------------------------------------
            # 5.7 RECONSTRUCT TEMPORAL SENSITIVITY
            # ------------------------------------------------

            # Primary and sensitivity use the same common sample.
            pc = float(
                t[f"{method}_primary_common"]
            )

            sc = float(
                t[f"{method}_sensitivity_common"]
            )

            # Isolate the economic-context-year effect.
            delta_year = sc - pc

            # Isolate the sample-composition effect.
            delta_sample = pc - full

            # Verify both stored differences independently.
            numeric_match(
                t[f"{method}_temporal_change"],
                delta_year,
                f"{tag}: {method} temporal delta",
            )

            numeric_match(
                t[f"{method}_sample_change"],
                delta_sample,
                f"{tag}: {method} sample delta",
            )

            # Verify the strict sign reversal on common samples.
            temporal_flip = pc * sc < 0

            strict_value(
                t[f"{method}_temporal_sign_flip"],
                temporal_flip,
                f"{tag}: {method} temporal sign flag",
            )

            # ------------------------------------------------
            # 5.8 REBUILD ALL METHOD-SPECIFIC OUTPUT FIELDS
            # ------------------------------------------------

            # Each field is independently reconstructed.
            # No robustness score or significance test is created.
            record.update({
                f"{method}_baseline":
                    full,

                f"{method}_loo_min":
                    min(reduced_values),

                f"{method}_loo_max":
                    max(reduced_values),

                f"{method}_loo_max_abs_change":
                    maximum,

                f"{method}_loo_largest_change_countries":
                    "|".join(tied_countries),

                f"{method}_loo_sign_flip_n":
                    len(reversed_sign_countries),

                f"{method}_loo_sign_flip_countries":
                    "|".join(sorted(reversed_sign_countries)),

                f"{method}_loo_any_sign_flip":
                    bool(reversed_sign_countries),

                f"{method}_primary_common":
                    pc,

                f"{method}_sensitivity_common":
                    sc,

                f"{method}_temporal_change":
                    delta_year,

                f"{method}_sample_change":
                    delta_sample,

                f"{method}_temporal_sign_flip":
                    temporal_flip,
            })

        reconstructed.append(record)

    # Ensure that every detailed omission was covered.
    ensure(
        checked_omissions == 948,
        "LOO omissions were not fully checked",
    )

    # Verify the independently reconstructed field set.
    ensure(
        all(
            set(row) == set(OUTPUT_COLS)
            for row in reconstructed
        ),
        "Internal independent reconstruction schema differs",
    )

    print(
        "36 independently reconstructed pairs "
        "and 948 omission records: PASSED."
    )

    return reconstructed


# ------------------------------------------------------------
# 6. COMPARE ALL SAVED PARQUET CELLS
# ------------------------------------------------------------

def validate_parquet_rows(result, reconstructed):

    # Compare analytical pair order, not just set membership.
    observed_keys = list(
        result[
            list(PAIR)
        ].itertuples(index=False, name=None)
    )

    ensure(
        observed_keys == EXPECTED_PAIRS,
        "Unexpected Step 59 row order",
    )

    maximum_difference = 0.0

    # Compare each of the 36 rows.
    for i, expected in enumerate(reconstructed):

        saved = result.iloc[i]

        # Compare all 37 columns within this row.
        for col in OUTPUT_COLS:

            where = (
                f"row {i + 1} "
                f"({expected['ess_variable']} / "
                f"{expected['economic_variable']}): {col}"
            )

            if col in FLOAT_COLS:

                maximum_difference = max(
                    maximum_difference,
                    numeric_match(
                        saved[col],
                        expected[col],
                        where,
                    ),
                )

            else:

                strict_value(
                    saved[col],
                    expected[col],
                    where,
                )

    print(
        "All 36 x 37 saved Parquet cells "
        "independently checked: PASSED."
    )

    return maximum_difference


# ------------------------------------------------------------
# 7. INDEPENDENT CSV VERSUS PARQUET VALIDATION
# ------------------------------------------------------------

def validate_csv(result, reconstructed):

    # Use Python's CSV reader to preserve original text.
    # Do not let pandas infer missing values or numeric types.
    with RESULT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        ensure(
            reader.fieldnames == OUTPUT_COLS,
            "CSV column names/order differs",
        )

        csv_rows = list(reader)

    ensure(
        len(csv_rows) == 36,
        "CSV does not have 36 data rows",
    )

    max_difference = 0.0

    # Compare CSV against both Parquet and independent calculations.
    for i, (saved_row, expected_row) in enumerate(
        zip(csv_rows, reconstructed)
    ):

        for col in OUTPUT_COLS:

            value = saved_row[col]
            where = f"CSV row {i + 1}: {col}"

            ensure(
                value is not None,
                f"{where}: missing CSV cell",
            )

            if col in FLOAT_COLS:

                max_difference = max(
                    max_difference,

                    numeric_match(
                        value,
                        expected_row[col],
                        where,
                    ),

                    numeric_match(
                        value,
                        result.iloc[i][col],
                        where + " vs Parquet",
                    ),
                )

            elif col in INT_COLS:

                # A count must be an integer, not a decimal.
                ensure(
                    re.fullmatch(r"-?\d+", value) is not None,
                    f"{where}: invalid integer representation",
                )

                strict_value(
                    int(value),
                    expected_row[col],
                    where,
                )

                strict_value(
                    int(value),
                    int(result.iloc[i][col]),
                    where + " vs Parquet",
                )

            elif col in BOOL_COLS:

                # pandas.to_csv writes ordinary Boolean values
                # as the exact strings True and False.
                ensure(
                    value in ("True", "False"),
                    f"{where}: invalid boolean representation",
                )

                actual = value == "True"

                strict_value(
                    actual,
                    expected_row[col],
                    where,
                )

                strict_value(
                    actual,
                    bool(result.iloc[i][col]),
                    where + " vs Parquet",
                )

            else:

                # Country lists and identifiers must match exactly.
                strict_value(
                    value,
                    expected_row[col],
                    where,
                )

                strict_value(
                    value,
                    result.iloc[i][col],
                    where + " vs Parquet",
                )

    print(
        "CSV headers, 36 rows, all cells "
        "and Parquet consistency: PASSED."
    )

    return max_difference


# ------------------------------------------------------------
# 8. VERIFY METADATA, PROVENANCE AND SHA-256 HASHES
# ------------------------------------------------------------

def validate_metadata(metadata, result):

    # Step 59 metadata must still describe Step 59.
    # We do not modify historical metadata to mark it validated.
    ensure(
        metadata.get("step") == 59,
        "Metadata step number differs",
    )

    ensure(
        metadata.get("analytic_pairs_n") == len(result),
        "Metadata pair count differs",
    )

    # Check the declared analytical scope.
    ensure(
        metadata.get("analysis_unit") == "country",
        "Wrong analysis unit",
    )

    ensure(
        metadata.get("study_design")
        == "cross-sectional exploratory",
        "Wrong declared study design",
    )

    ensure(
        metadata.get("country_weighting")
        == "equal across countries",
        "Wrong country weighting statement",
    )

    # This status remains historically accurate in the Step 59 file.
    ensure(
        metadata.get("independent_validation_status")
        == "pending",
        "Step 59 metadata was unexpectedly edited",
    )

    # Step 59 was descriptive: no inferential tests or causal claims.
    inference = metadata.get("inference")

    ensure(
        isinstance(inference, dict),
        "Missing inference metadata",
    )

    for field in (
        "p_values_calculated",
        "confidence_intervals_calculated",
        "causal_claims",
        "robustness_score_created",
    ):

        ensure(
            inference.get(field) is False,
            f"Unexpected inference property: {field}",
        )

    # Preserve interpretation and methodological limitations.
    ensure(
        isinstance(
            metadata.get("interpretation_notes"),
            list,
        )
        and len(metadata["interpretation_notes"]) > 0,
        "Missing interpretation notes",
    )

    # Validate the metadata creation timestamp.
    timestamp = metadata.get("created_at_utc")

    ensure(
        isinstance(timestamp, str),
        "Missing creation timestamp",
    )

    try:

        datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

    except ValueError as exc:

        raise AssertionError(
            "Invalid metadata timestamp"
        ) from exc

    # --------------------------------------------------------
    # 8.1 VERIFY THE SIX RECORDED SOURCE HASHES
    # --------------------------------------------------------

    source_entries = metadata.get("sources")

    ensure(
        isinstance(source_entries, list)
        and len(source_entries) == 6,
        "Expected six metadata source records",
    )

    recorded_source_names = [
        item.get("filename")
        for item in source_entries
    ]

    actual_source_names = [
        path.name
        for path in SOURCE_PATHS.values()
    ]

    ensure(
        recorded_source_names == actual_source_names,
        "Source names/order in metadata differ",
    )

    # Recalculate each source hash from the actual file bytes.
    for entry, source_path in zip(
        source_entries,
        SOURCE_PATHS.values(),
    ):

        strict_value(
            entry.get("sha256"),
            sha256(source_path),
            f"metadata source hash: {source_path.name}",
        )

    # --------------------------------------------------------
    # 8.2 VERIFY THE TWO RECORDED OUTPUT HASHES
    # --------------------------------------------------------

    output_entries = metadata.get("outputs")

    ensure(
        isinstance(output_entries, list)
        and len(output_entries) == 2,
        "Expected two metadata output records",
    )

    output_files = (
        RESULT_PARQUET,
        RESULT_CSV,
    )

    ensure(
        [
            item.get("filename")
            for item in output_entries
        ]
        == [
            path.name
            for path in output_files
        ],
        "Output names/order in metadata differ",
    )

    for entry, path in zip(
        output_entries,
        output_files,
    ):

        strict_value(
            entry.get("sha256"),
            sha256(path),
            f"metadata output hash: {path.name}",
        )

    print(
        "Provenance metadata, six source hashes "
        "and two output hashes: PASSED."
    )


# ------------------------------------------------------------
# 9. VERIFY HISTORICAL SUMMARY COUNTS
# ------------------------------------------------------------

def validate_summary(result):

    # The preceding independent validators established these totals.
    # Step 60 must confirm that consolidation preserved them.
    expected_counts = {
        "pearson_loo_any_sign_flip": 9,
        "spearman_loo_any_sign_flip": 7,
        "pearson_temporal_sign_flip": 1,
        "spearman_temporal_sign_flip": 2,
    }

    for column, expected in expected_counts.items():

        actual = int(
            result[column].sum()
        )

        ensure(
            actual == expected,
            f"{column}: expected {expected}, got {actual}",
        )

    # Verify the common sample size range from Step 58.
    ensure(
        result["common_countries_n"].min() == 24,
        "Unexpected minimum common sample N",
    )

    ensure(
        result["common_countries_n"].max() == 28,
        "Unexpected maximum common sample N",
    )

    # The six Gini comparisons use 24 common countries.
    ensure(
        result.loc[
            result["economic_variable"] == "gini",
            "common_countries_n",
        ].eq(24).all(),
        "Gini should use a common sample of 24 countries",
    )

    print(
        "LOO/temporal sign-change totals "
        "and common sample sizes: PASSED."
    )


# ------------------------------------------------------------
# 10. MAIN READ-ONLY VALIDATION WORKFLOW
# ------------------------------------------------------------

def main():

    print("=" * 78)

    print(
        "STEP 60 - INDEPENDENT CORRELATION "
        "ROBUSTNESS SYNTHESIS VALIDATION"
    )

    print("=" * 78)

    # --------------------------------------------------------
    # 10.1 CONFIRM ALL REQUIRED INPUT FILES EXIST
    # --------------------------------------------------------

    all_paths = list(
        SOURCE_PATHS.values()
    ) + [
        RESULT_PARQUET,
        RESULT_CSV,
        RESULT_METADATA,
    ]

    for path in all_paths:

        ensure(
            path.is_file(),
            f"Required file missing: {path}",
        )

    # Record exact hashes before the validation starts.
    before = {
        path: sha256(path)
        for path in all_paths
    }

    # --------------------------------------------------------
    # 10.2 READ THE FOUR ANALYTICAL SOURCE TABLES
    # --------------------------------------------------------

    baseline = pd.read_parquet(
        SOURCE_PATHS["baseline"],
        engine="pyarrow",
    )

    details = pd.read_parquet(
        SOURCE_PATHS["loo_detail"],
        engine="pyarrow",
    )

    summary = pd.read_parquet(
        SOURCE_PATHS["loo_summary"],
        engine="pyarrow",
    )

    temporal = pd.read_parquet(
        SOURCE_PATHS["temporal"],
        engine="pyarrow",
    )

    # Read the Step 59 Parquet without modifying it.
    result = pd.read_parquet(
        RESULT_PARQUET,
        engine="pyarrow",
    )

    # --------------------------------------------------------
    # 10.3 CHECK HISTORICAL METADATA FILES
    # --------------------------------------------------------

    for key in (
        "loo_metadata",
        "temporal_metadata",
    ):

        with SOURCE_PATHS[key].open(
            "r",
            encoding="utf-8",
        ) as handle:

            ensure(
                isinstance(json.load(handle), dict),
                f"Invalid source metadata: {key}",
            )

    # Read the Step 59 metadata for detailed validation.
    with RESULT_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:

        metadata = json.load(handle)

    ensure(
        isinstance(metadata, dict),
        "Step 59 metadata is not a JSON object",
    )

    # --------------------------------------------------------
    # 10.4 VALIDATE SCHEMA AND RECONSTRUCT ALL RESULTS
    # --------------------------------------------------------

    inspect_output_schema(result)

    expected_rows = reconstruct(
        baseline,
        details,
        summary,
        temporal,
    )

    # --------------------------------------------------------
    # 10.5 COMPARE PARQUET AND CSV AGAINST THE RECONSTRUCTION
    # --------------------------------------------------------

    parquet_diff = validate_parquet_rows(
        result,
        expected_rows,
    )

    csv_diff = validate_csv(
        result,
        expected_rows,
    )

    # --------------------------------------------------------
    # 10.6 VALIDATE METADATA AND SUMMARY STATISTICS
    # --------------------------------------------------------

    validate_metadata(
        metadata,
        result,
    )

    validate_summary(result)

    # --------------------------------------------------------
    # 10.7 VERIFY THAT INPUTS AND OUTPUTS ARE UNCHANGED
    # --------------------------------------------------------

    after = {
        path: sha256(path)
        for path in all_paths
    }

    ensure(
        before == after,
        "At least one inspected file changed during validation",
    )

    # --------------------------------------------------------
    # 10.8 PRINT THE FINAL VALIDATION REPORT
    # --------------------------------------------------------

    print("\nSUMMARY")

    print(
        "Validated synthesis rows:",
        len(result),
    )

    print(
        "Validated synthesis columns:",
        len(result.columns),
    )

    print(
        "Validated synthesis cells:",
        result.size,
    )

    print(
        "Independently checked omissions:",
        len(details),
    )

    print(
        "Maximum independent Parquet difference:",
        f"{parquet_diff:.3e}",
    )

    print(
        "Maximum CSV/Parquet/reconstruction difference:",
        f"{csv_diff:.3e}",
    )

    print(
        "Input and output file bytes unchanged: PASSED."
    )

    # Print PASSED only after all validation requirements succeed.
    print(
        "\nINDEPENDENT CORRELATION ROBUSTNESS "
        "SYNTHESIS VALIDATION PASSED."
    )

    print(
        "No project files were created or modified."
    )


# ------------------------------------------------------------
# 11. EXECUTE ONLY WHEN THE SCRIPT IS RUN DIRECTLY
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
