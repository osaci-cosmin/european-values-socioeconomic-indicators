
# ============================================================
# STEP 64 - SCIENTIFIC EVIDENCE MATRIX V1
# ============================================================
#
# Purpose:
# Integrate the independently validated Step 59 synthesis
# with the Schwartz country-exclusion scenarios from Step 62.
#
# Preserve all 37 original synthesis columns.
# Add exclusion sensitivity information for Schwartz only.
#
# Do not recalculate the original correlations.
# Do not change or overwrite historical analytical outputs.
#
# Study: Cross-sectional, exploratory, country-level.
# No causal claims, p-value screening or robustness scoring.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import pandas as pd


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

INTEGRATED = (
    ROOT / "data" / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

BASELINE = (
    TABLES / "ESS11_country_correlations_main_v1.parquet"
)

SYNTHESIS = (
    TABLES
    / "ESS11_country_correlation_robustness_synthesis_v1.parquet"
)

SYNTHESIS_METADATA = (
    TABLES
    / "ESS11_country_correlation_robustness_synthesis_v1.metadata.json"
)

SCHWARTZ = (
    TABLES
    / "ESS11_schwartz_country_exclusion_sensitivity_v1.parquet"
)

SCHWARTZ_METADATA = (
    TABLES
    / "ESS11_schwartz_country_exclusion_sensitivity_v1.metadata.json"
)

STEM = "ESS11_scientific_evidence_matrix_v1"

OUTPUT_PARQUET = TABLES / f"{STEM}.parquet"
OUTPUT_CSV = TABLES / f"{STEM}.csv"
OUTPUT_METADATA = TABLES / f"{STEM}.metadata.json"

EXPECTED_INTEGRATED_HASH = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

ATOL = 1e-12


# ------------------------------------------------------------
# 2. DEFINE THE SCIENTIFIC SCOPE
# ------------------------------------------------------------

ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

SCHWARTZ_VARIABLES = {
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
}

ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# Primary results are already represented by the Step 59 columns.
# Only the three exclusion scenarios need new output columns.
EXCLUSION_SCENARIOS = {
    "exclude_EE": {"EE"},
    "exclude_GB": {"GB"},
    "exclude_EE_GB": {"EE", "GB"},
}

ALL_SCENARIOS = {
    "primary_full": set(),
    **EXCLUSION_SCENARIOS,
}

# The following 11 fields will be added for each exclusion.
SCENARIO_FIELDS = [
    "actual_exclusions",
    "scenario_n",
    "removed_n",
    "pearson_scenario",
    "pearson_change",
    "pearson_abs_change",
    "pearson_sign_flip",
    "spearman_scenario",
    "spearman_change",
    "spearman_abs_change",
    "spearman_sign_flip",
]

TEXT_FIELDS = {"actual_exclusions"}

INTEGER_FIELDS = {
    "scenario_n",
    "removed_n",
}

BOOLEAN_FIELDS = {
    "pearson_sign_flip",
    "spearman_sign_flip",
}

FLOAT_FIELDS = (
    set(SCENARIO_FIELDS)
    - TEXT_FIELDS
    - INTEGER_FIELDS
    - BOOLEAN_FIELDS
)

# Define the validated schema of Step 62 explicitly.
SCHWARTZ_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "scenario",
    "requested_exclusions",
    "actual_exclusions",
    "primary_n",
    "scenario_n",
    "removed_n",
    "scenario_countries",
] + [
    f"{method}_{suffix}"
    for method in ("pearson", "spearman")
    for suffix in (
        "primary",
        "scenario",
        "change",
        "abs_change",
        "sign_flip",
    )
]


# ------------------------------------------------------------
# 3. GENERAL VALIDATION FUNCTIONS
# ------------------------------------------------------------

def require(condition, message):

    # Fail explicitly instead of silently correcting input.
    if not bool(condition):
        raise AssertionError(message)


def sha256(path):

    # Read source bytes in chunks to calculate the exact hash.
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def compare_number(actual, expected, label):

    # Compare floating-point values with absolute tolerance.
    require(
        not pd.isna(actual) and not pd.isna(expected),
        f"{label}: unexpected missing numerical value",
    )

    a = float(actual)
    b = float(expected)

    require(
        math.isfinite(a) and math.isfinite(b),
        f"{label}: non-finite numerical value",
    )

    difference = abs(a - b)

    require(
        difference <= ATOL,
        f"{label}: difference={difference:.3e}",
    )

    return difference


def expected_pairs():

    # Independently define the complete 6 x 6 pair universe.
    return {
        (ess, economic)
        for ess in ESS_VARIABLES
        for economic in ECONOMIC_VARIABLES
    }


def pair_keys(table):

    return list(
        table[
            ["ess_variable", "economic_variable"]
        ].itertuples(index=False, name=None)
    )


def parse_country_list(value):

    # Historical project tables use pipe-separated countries.
    require(
        isinstance(value, str),
        "Country list must be a string",
    )

    values = value.split("|") if value else []

    require(
        len(values) == len(set(values)),
        "Duplicate country in a saved sample",
    )

    require(
        set(values).issubset(COUNTRIES),
        "Unknown country in a saved sample",
    )

    return set(values)


# ------------------------------------------------------------
# 4. VERIFY THE EXISTING INPUTS AND PROVENANCE
# ------------------------------------------------------------

def load_and_check_inputs():

    paths = [
        INTEGRATED,
        BASELINE,
        SYNTHESIS,
        SYNTHESIS_METADATA,
        SCHWARTZ,
        SCHWARTZ_METADATA,
    ]

    for path in paths:
        require(
            path.is_file(),
            f"Required source not found: {path}",
        )

    # Verify the original validated ESS-Eurostat snapshot.
    require(
        sha256(INTEGRATED) == EXPECTED_INTEGRATED_HASH,
        "Integrated dataset SHA-256 differs",
    )

    synthesis = pd.read_parquet(
        SYNTHESIS,
        engine="pyarrow",
    )

    sensitivity = pd.read_parquet(
        SCHWARTZ,
        engine="pyarrow",
    )

    baseline = pd.read_parquet(
        BASELINE,
        engine="pyarrow",
    )

    with SYNTHESIS_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        synthesis_metadata = json.load(handle)

    with SCHWARTZ_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        sensitivity_metadata = json.load(handle)

    require(
        isinstance(synthesis_metadata, dict),
        "Invalid Step 59 metadata",
    )

    require(
        isinstance(sensitivity_metadata, dict),
        "Invalid Step 62 metadata",
    )

    # Check the independently validated Step 62 output hash.
    output_hashes = sensitivity_metadata.get("outputs")

    require(
        isinstance(output_hashes, list)
        and len(output_hashes) == 2,
        "Unexpected Step 62 output provenance",
    )

    recorded_hashes = {
        entry["filename"]: entry["sha256"]
        for entry in output_hashes
    }

    require(
        recorded_hashes.get(SCHWARTZ.name) == sha256(SCHWARTZ),
        "Step 62 Parquet hash differs from metadata",
    )

    # Verify source hashes recorded by Step 62.
    source_hashes = sensitivity_metadata.get("sources")

    require(
        isinstance(source_hashes, list)
        and len(source_hashes) == 3,
        "Unexpected Step 62 source provenance",
    )

    recorded_sources = {
        entry["filename"]: entry["sha256"]
        for entry in source_hashes
    }

    for path in (INTEGRATED, BASELINE, SYNTHESIS):

        require(
            recorded_sources.get(path.name) == sha256(path),
            f"Step 62 source hash differs: {path.name}",
        )

    # The main synthesis must preserve all historical pairs.
    require(
        len(synthesis) == 36,
        "Step 59 synthesis must contain 36 relationships",
    )

    require(
        len(synthesis.columns) == 37,
        "Unexpected Step 59 synthesis column count",
    )

    require(
        synthesis.columns.is_unique,
        "Duplicate Step 59 column names",
    )

    required_synthesis = {
        "ess_variable",
        "economic_variable",
        "countries_n",
        "included_countries",
        "pearson_baseline",
        "spearman_baseline",
    }

    require(
        required_synthesis.issubset(synthesis.columns),
        "Required Step 59 analytical fields are missing",
    )

    keys = pair_keys(synthesis)

    require(
        len(keys) == len(set(keys)),
        "Duplicate Step 59 relationships",
    )

    require(
        set(keys) == expected_pairs(),
        "Step 59 relationship universe differs",
    )

    # Check the exact validated 19-column Step 62 structure.
    require(
        sensitivity.columns.tolist() == SCHWARTZ_COLUMNS,
        "Unexpected Step 62 output schema",
    )

    require(
        len(sensitivity) == 96,
        "Expected 96 Step 62 scenario records",
    )

    require(
        not sensitivity.isna().any().any(),
        "Unexpected null in Step 62 scenario data",
    )

    require(
        len(baseline) == 36,
        "Unexpected baseline correlation count",
    )

    require(
        set(pair_keys(baseline)) == expected_pairs(),
        "Baseline pair universe differs",
    )

    # Verify all four scenarios for the 24 Schwartz pairs.
    actual_sensitivity_keys = set(
        sensitivity[
            ["ess_variable", "economic_variable", "scenario"]
        ].itertuples(index=False, name=None)
    )

    expected_sensitivity_keys = {
        (ess, economic, scenario)
        for ess in SCHWARTZ_VARIABLES
        for economic in ECONOMIC_VARIABLES
        for scenario in ALL_SCENARIOS
    }

    require(
        len(actual_sensitivity_keys) == 96
        and actual_sensitivity_keys == expected_sensitivity_keys,
        "Step 62 scenario coverage differs",
    )

    print(
        "Source hashes, metadata and validated input schemas: PASSED."
    )

    print(
        "36 Step 59 pairs and 96 Step 62 scenarios: PASSED."
    )

    return synthesis, sensitivity


# ------------------------------------------------------------
# 5. CHECK EACH SCHWARTZ SCENARIO AGAINST STEP 59
# ------------------------------------------------------------

def validate_sensitivity_links(synthesis, sensitivity):

    # Index historical rows by analytical pair.
    original = synthesis.set_index(
        ["ess_variable", "economic_variable"]
    )

    maximum_difference = 0.0

    for _, row in sensitivity.iterrows():

        ess = row["ess_variable"]
        economic = row["economic_variable"]
        scenario = row["scenario"]

        key = (ess, economic)

        require(
            key in original.index,
            f"Missing synthesis pair: {key}",
        )

        base = original.loc[key]

        primary_n = int(base["countries_n"])

        countries = parse_country_list(
            base["included_countries"]
        )

        require(
            len(countries) == primary_n,
            f"{key}: inconsistent baseline sample",
        )

        require(
            int(row["primary_n"]) == primary_n,
            f"{key}: Step 62 primary N differs",
        )

        # Verify both primary coefficients before integration.
        maximum_difference = max(
            maximum_difference,
            compare_number(
                row["pearson_primary"],
                base["pearson_baseline"],
                f"{key}: Pearson baseline",
            ),
            compare_number(
                row["spearman_primary"],
                base["spearman_baseline"],
                f"{key}: Spearman baseline",
            ),
        )

        requested = ALL_SCENARIOS[scenario]

        actual_exclusions = countries & requested

        expected_remaining = countries - requested

        require(
            row["requested_exclusions"]
            == "|".join(sorted(requested)),
            f"{key}: requested exclusions differ",
        )

        require(
            row["actual_exclusions"]
            == "|".join(sorted(actual_exclusions)),
            f"{key}: actual exclusions differ",
        )

        require(
            parse_country_list(row["scenario_countries"])
            == expected_remaining,
            f"{key}: scenario country sample differs",
        )

        require(
            int(row["removed_n"]) == len(actual_exclusions),
            f"{key}: removed country count differs",
        )

        require(
            int(row["scenario_n"]) == len(expected_remaining),
            f"{key}: scenario N differs",
        )

        # Verify the documented ESS-Eurostat availability.
        # Estonia is present for all 24 Schwartz relationships.
        require(
            "EE" in countries,
            f"{key}: Estonia unexpectedly absent",
        )

        # GB is available for GDP, but not for the other
        # five economic indicators in the validated V1 data.
        require(
            ("GB" in countries)
            == (economic == "gdp_pps_index"),
            f"{key}: GB inclusion differs from V1",
        )

        # Recheck saved differences without recomputing correlations.
        for method in ("pearson", "spearman"):

            primary = float(row[f"{method}_primary"])
            scenario_value = float(row[f"{method}_scenario"])

            change = scenario_value - primary

            maximum_difference = max(
                maximum_difference,
                compare_number(
                    row[f"{method}_change"],
                    change,
                    f"{key}/{scenario}/{method}: change",
                ),
                compare_number(
                    row[f"{method}_abs_change"],
                    abs(change),
                    f"{key}/{scenario}/{method}: absolute change",
                ),
            )

            expected_flip = (
                primary * scenario_value < 0
            )

            require(
                bool(row[f"{method}_sign_flip"])
                == expected_flip,
                f"{key}/{scenario}/{method}: sign flag differs",
            )

    print(
        "All 96 scenarios linked to the correct baseline "
        "coefficients and country samples: PASSED."
    )

    return maximum_difference


# ------------------------------------------------------------
# 6. CONSTRUCT THE SCIENTIFIC EVIDENCE MATRIX
# ------------------------------------------------------------

def create_matrix(synthesis, sensitivity):

    # Preserve the original 37 columns, their values,
    # pandas dtypes and historical row ordering.
    matrix = synthesis.copy().reset_index(drop=True)

    original_columns = synthesis.columns.tolist()

    # Distinguish applicable sensitivity from non-applicability.
    applicable = matrix["ess_variable"].isin(
        SCHWARTZ_VARIABLES
    )

    matrix["schwartz_sensitivity_applicable"] = (
        applicable.astype("boolean")
    )

    # Explain the effective GB sample coverage explicitly.
    limitations = []

    for index in range(len(matrix)):

        ess = matrix.at[index, "ess_variable"]
        economic = matrix.at[index, "economic_variable"]

        if ess not in SCHWARTZ_VARIABLES:

            limitations.append(
                "not_applicable_non_schwartz"
            )

        elif economic == "gdp_pps_index":

            limitations.append(
                "EE_and_GB_both_in_primary_sample"
            )

        else:

            limitations.append(
                "only_EE_in_primary_sample_GB_unavailable"
            )

    matrix["schwartz_sensitivity_scope_note"] = (
        pd.Series(limitations, dtype="string")
    )

    # Create nullable fields for all three exclusion scenarios.
    # NA means not applicable; it must never become zero.
    for scenario in EXCLUSION_SCENARIOS:

        for field in SCENARIO_FIELDS:

            column = f"{scenario}__{field}"

            if field in TEXT_FIELDS:
                dtype = "string"

            elif field in INTEGER_FIELDS:
                dtype = "Int64"

            elif field in BOOLEAN_FIELDS:
                dtype = "boolean"

            else:
                dtype = "Float64"

            matrix[column] = pd.Series(
                pd.NA,
                index=matrix.index,
                dtype=dtype,
            )

    # Construct an exact lookup of the saved Step 62 records.
    lookup = {
        (
            record["ess_variable"],
            record["economic_variable"],
            record["scenario"],
        ): record
        for record in sensitivity.to_dict("records")
    }

    populated = 0

    # Add exclusion details only to Schwartz relationships.
    for index in range(len(matrix)):

        ess = matrix.at[index, "ess_variable"]
        economic = matrix.at[index, "economic_variable"]

        if ess not in SCHWARTZ_VARIABLES:

            # All 33 scenario fields remain missing.
            continue

        for scenario in EXCLUSION_SCENARIOS:

            key = (ess, economic, scenario)

            require(
                key in lookup,
                f"Missing Schwartz scenario: {key}",
            )

            record = lookup[key]

            for field in SCENARIO_FIELDS:

                output_column = f"{scenario}__{field}"

                matrix.at[index, output_column] = (
                    record[field]
                )

            populated += 1

    require(
        populated == 72,
        "Expected 24 x 3 populated Schwartz scenarios",
    )

    # --------------------------------------------------------
    # 6.1 VERIFY THE ORIGINAL STEP 59 DATA WAS PRESERVED
    # --------------------------------------------------------

    require(
        matrix[original_columns].equals(
            synthesis.reset_index(drop=True)
        ),
        "Original Step 59 columns changed during integration",
    )

    # --------------------------------------------------------
    # 6.2 VERIFY THE FINAL DIMENSIONS
    # --------------------------------------------------------

    require(
        len(matrix) == 36,
        "Evidence matrix must have 36 rows",
    )

    require(
        len(matrix.columns) == 72,
        "Evidence matrix must have 72 columns",
    )

    require(
        matrix.columns.is_unique,
        "Duplicate evidence matrix columns",
    )

    require(
        matrix[
            "schwartz_sensitivity_applicable"
        ].sum() == 24,
        "Expected 24 Schwartz relationships",
    )

    require(
        (~matrix[
            "schwartz_sensitivity_applicable"
        ]).sum() == 12,
        "Expected 12 non-Schwartz relationships",
    )

    # --------------------------------------------------------
    # 6.3 VERIFY TRUE NOT-APPLICABLE VALUES
    # --------------------------------------------------------

    scenario_columns = [
        f"{scenario}__{field}"
        for scenario in EXCLUSION_SCENARIOS
        for field in SCENARIO_FIELDS
    ]

    non_schwartz = matrix.loc[
        ~matrix["schwartz_sensitivity_applicable"]
    ]

    schwartz = matrix.loc[
        matrix["schwartz_sensitivity_applicable"]
    ]

    require(
        non_schwartz[scenario_columns].isna().all().all(),
        "Non-Schwartz sensitivity must remain NA",
    )

    require(
        schwartz[scenario_columns].notna().all().all(),
        "Schwartz scenarios contain unexpected nulls",
    )

    # --------------------------------------------------------
    # 6.4 VERIFY THE SPECIAL GB COVERAGE
    # --------------------------------------------------------

    gdp_rows = schwartz.loc[
        schwartz["economic_variable"] == "gdp_pps_index"
    ]

    other_rows = schwartz.loc[
        schwartz["economic_variable"] != "gdp_pps_index"
    ]

    require(
        len(gdp_rows) == 4 and len(other_rows) == 20,
        "Unexpected Schwartz indicator distribution",
    )

    require(
        gdp_rows["exclude_GB__removed_n"].eq(1).all(),
        "GB must be present in the four GDP correlations",
    )

    require(
        other_rows["exclude_GB__removed_n"].eq(0).all(),
        "GB must be unavailable for non-GDP relationships",
    )

    require(
        schwartz["exclude_EE__removed_n"].eq(1).all(),
        "EE must be present in all Schwartz correlations",
    )

    # --------------------------------------------------------
    # 6.5 VERIFY SIGN-CHANGE RESULTS
    # --------------------------------------------------------

    for scenario in EXCLUSION_SCENARIOS:

        for method in ("pearson", "spearman"):

            flag = f"{scenario}__{method}_sign_flip"

            require(
                not schwartz[flag].any(),
                f"Unexpected sign reversal: {flag}",
            )

    print(
        "Original 37 synthesis columns preserved: PASSED."
    )

    print(
        "72-column Scientific Evidence Matrix constructed: PASSED."
    )

    print(
        "24 Schwartz and 12 non-Schwartz pairs "
        "correctly distinguished: PASSED."
    )

    print(
        "GB/EE coverage, missingness semantics "
        "and sign flags: PASSED."
    )

    return matrix


# ------------------------------------------------------------
# 7. PRINT SCIENTIFIC SUMMARY
# ------------------------------------------------------------

def report(matrix, maximum_difference):

    print()
    print("SCIENTIFIC EVIDENCE MATRIX SUMMARY")
    print("-" * 90)

    print("Analytical relationships:", len(matrix))
    print("Columns:", len(matrix.columns))
    print("Total cells:", matrix.size)

    print(
        "Schwartz relationships:",
        int(
            matrix["schwartz_sensitivity_applicable"].sum()
        ),
    )

    print(
        "Non-Schwartz relationships:",
        int(
            (~matrix["schwartz_sensitivity_applicable"]).sum()
        ),
    )

    print(
        "Additional applicable exclusion scenarios:",
        24 * len(EXCLUSION_SCENARIOS),
    )

    print()
    print("MAXIMUM ABSOLUTE SCHWARTZ CHANGES")
    print("-" * 90)

    schwartz = matrix.loc[
        matrix["schwartz_sensitivity_applicable"]
    ]

    for scenario in EXCLUSION_SCENARIOS:

        pearson = (
            schwartz[
                f"{scenario}__pearson_abs_change"
            ].max()
        )

        spearman = (
            schwartz[
                f"{scenario}__spearman_abs_change"
            ].max()
        )

        print()
        print("Scenario:", scenario)
        print(f"  Pearson maximum:  {pearson:.6f}")
        print(f"  Spearman maximum: {spearman:.6f}")

    print()
    print(
        "Maximum cross-source numerical difference:",
        f"{maximum_difference:.3e}",
    )

    print()
    print("INTERPRETATION POLICY")
    print("-" * 90)

    print(
        "Primary: coefficient direction/magnitude, "
        "Pearson-Spearman concordance and N."
    )

    print(
        "Sensitivity: LOO, alternative context year "
        "and Schwartz country exclusions."
    )

    print(
        "No arbitrary robustness scores or new p-values."
    )

    print(
        "Schwartz cross-country psychometric invariance "
        "is not established."
    )

    print(
        "Country-level associations must not be "
        "interpreted as causal effects."
    )


# ------------------------------------------------------------
# 8. SAVE OUTPUTS AND METADATA
# ------------------------------------------------------------

def save_outputs(matrix, maximum_difference):

    # Never overwrite an existing scientific evidence matrix.
    for path in (
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ):

        require(
            not path.exists(),
            f"Output already exists: {path}",
        )

    # Record all source files used for this integration.
    source_paths = [
        INTEGRATED,
        BASELINE,
        SYNTHESIS,
        SYNTHESIS_METADATA,
        SCHWARTZ,
        SCHWARTZ_METADATA,
    ]

    sources = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "filename": path.name,
            "sha256": sha256(path),
        }
        for path in source_paths
    ]

    # Save the new Parquet table without an index column.
    with OUTPUT_PARQUET.open("xb") as handle:

        matrix.to_parquet(
            handle,
            engine="pyarrow",
            index=False,
        )

    # Export a CSV for Power BI and general inspection.
    with OUTPUT_CSV.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:

        matrix.to_csv(
            handle,
            index=False,
            na_rep="",
        )

    # Store schema information for independent validation.
    schema = [
        {
            "name": column,
            "pandas_dtype": str(matrix[column].dtype),
        }
        for column in matrix.columns
    ]

    metadata = {
        "step": 64,
        "dataset_name": "Scientific Evidence Matrix V1",
        "created_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "country",
        "study_design": "cross-sectional exploratory",
        "country_universe_n": 29,
        "analytical_pairs_n": 36,
        "schwartz_pairs_n": 24,
        "non_schwartz_pairs_n": 12,
        "original_synthesis_columns_n": 37,
        "added_columns_n": 35,
        "output_columns_n": 72,
        "output_rows_n": 36,
        "source_synthesis_step": 59,
        "source_sensitivity_step": 62,
        "sensitivity_validation_step": 63,
        "independent_step_64_validation": "pending",
        "baseline_correlations_recalculated": False,
        "loo_correlations_recalculated": False,
        "temporal_correlations_recalculated": False,
        "causal_analysis": False,
        "measurement_invariance_test": False,
        "statistical_significance_testing": False,
        "arbitrary_robustness_score": False,
        "numeric_comparison_tolerance": ATOL,
        "maximum_cross_source_difference":
            maximum_difference,
        "schwartz_exclusion_scenarios":
            {
                name: sorted(countries)
                for name, countries in EXCLUSION_SCENARIOS.items()
            },
        "interpretation_notes": [
            "The 37 Step 59 columns are preserved unchanged.",
            "Country-level associations are exploratory.",
            "The Schwartz exclusion scenarios apply only to 24 pairs.",
            "Non-Schwartz sensitivity values are NA, not zero.",
            "GB affects only GDP Schwartz correlations because other economic indicators are unavailable for GB.",
            "Estonia contributes to all 24 baseline Schwartz correlations.",
            "LOO and temporal results remain those of the independently validated Step 59 synthesis.",
            "Country exclusion is not a test of psychometric measurement invariance.",
            "Missing economic data are not imputed.",
            "Correlations do not establish causality.",
            "No p-values, causal effects or arbitrary robustness categories are introduced.",
        ],
        "schema": schema,
        "sources": sources,
        "outputs": [
            {
                "filename": path.name,
                "sha256": sha256(path),
            }
            for path in (OUTPUT_PARQUET, OUTPUT_CSV)
        ],
    }

    # Save the new provenance and methodology manifest.
    with OUTPUT_METADATA.open(
        "x",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        handle.write("\n")

    print()
    print("CREATED OUTPUT FILES")
    print("-" * 90)

    for path in (
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ):
        print(path)


# ------------------------------------------------------------
# 9. MAIN EXECUTION
# ------------------------------------------------------------

def main():

    print("=" * 90)
    print("STEP 64 - SCIENTIFIC EVIDENCE MATRIX V1")
    print("=" * 90)

    # Protect previous results before performing any analysis.
    for path in (
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ):

        require(
            not path.exists(),
            f"Step 64 output already exists: {path}",
        )

    # Load existing validated results.
    synthesis, sensitivity = load_and_check_inputs()

    # Check each sensitivity record against the baseline.
    maximum_difference = validate_sensitivity_links(
        synthesis,
        sensitivity,
    )

    # Integrate existing evidence without recomputing correlations.
    matrix = create_matrix(
        synthesis,
        sensitivity,
    )

    # Show the key analytical diagnostics.
    report(
        matrix,
        maximum_difference,
    )

    # Save only new Step 64 outputs.
    save_outputs(
        matrix,
        maximum_difference,
    )

    print()
    print("=" * 90)
    print("STEP 64 SCIENTIFIC EVIDENCE MATRIX CREATED.")
    print("=" * 90)

    print(
        "All previous source and analytical files remain unchanged."
    )

    print(
        "Independent Step 64 validation is still pending."
    )


if __name__ == "__main__":
    main()
