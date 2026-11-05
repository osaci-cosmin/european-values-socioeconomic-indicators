
# ============================================================
# STEP 62 - SCHWARTZ CROSS-COUNTRY EXCLUSION SENSITIVITY
# ============================================================
#
# Purpose:
# Evaluate how the 24 Schwartz-Eurostat associations change
# when temporarily excluding Estonia and/or Great Britain.
#
# The original 29-country dataset remains unchanged.
# The original baseline correlations remain unchanged.
#
# This is a descriptive sensitivity analysis, not a test
# of measurement invariance or statistical significance.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ------------------------------------------------------------
# 1. DEFINE PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/"
    "ESS11_country_integrated_main_v1.parquet"
)

BASELINE = (
    ROOT / "outputs/tables/"
    "ESS11_country_correlations_main_v1.parquet"
)

SYNTHESIS = (
    ROOT / "outputs/tables/"
    "ESS11_country_correlation_robustness_synthesis_v1.parquet"
)

TABLES = ROOT / "outputs/tables"

STEM = "ESS11_schwartz_country_exclusion_sensitivity_v1"

OUTPUT_PARQUET = TABLES / f"{STEM}.parquet"
OUTPUT_CSV = TABLES / f"{STEM}.csv"
OUTPUT_METADATA = TABLES / f"{STEM}.metadata.json"

EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

TOLERANCE = 1e-12


# ------------------------------------------------------------
# 2. DEFINE THE ANALYTICAL SCOPE
# ------------------------------------------------------------

# The four centered Schwartz higher-order dimensions.
ESS_VARIABLES = [
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

# The six Eurostat economic indicators.
ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

# The original ESS11 European country universe.
COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# Each scenario is defined before examining the results.
# This avoids choosing exclusions based on desired correlations.
SCENARIOS = {
    "primary_full": set(),
    "exclude_EE": {"EE"},
    "exclude_GB": {"GB"},
    "exclude_EE_GB": {"EE", "GB"},
}


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def require(condition, message):

    # Stop execution when a methodological assumption fails.
    if not bool(condition):
        raise AssertionError(message)


def sha256(path):

    # Calculate the hash of an existing source or output file.
    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check_columns(df, required, name):

    # Prevent calculations with missing or ambiguous columns.
    require(
        df.columns.is_unique,
        f"{name}: duplicate columns",
    )

    missing = sorted(set(required) - set(df.columns))

    require(
        not missing,
        f"{name}: missing columns {missing}",
    )


def correlation_coefficients(x, y):

    # Convert observations to ordinary floating-point arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Both variables must refer to the same country sample.
    require(
        len(x) == len(y) and len(x) >= 3,
        "Insufficient paired country observations",
    )

    # Missing or infinite values are not allowed here.
    # Missing observations are filtered before this function.
    require(
        np.isfinite(x).all() and np.isfinite(y).all(),
        "Non-finite correlation input",
    )

    # Constant variables cannot have a defined correlation.
    require(
        np.unique(x).size > 1
        and np.unique(y).size > 1,
        "Constant variable in correlation sample",
    )

    # Pearson describes linear association.
    pearson = float(
        np.corrcoef(x, y)[0, 1]
    )

    # Spearman is Pearson correlation between average ranks.
    # Tied values receive average ranks.
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")

    spearman = float(
        np.corrcoef(x_rank, y_rank)[0, 1]
    )

    require(
        np.isfinite([pearson, spearman]).all(),
        "Invalid correlation coefficient",
    )

    return pearson, spearman


def verify_baseline(actual, expected, description):

    # Check that the original coefficients are reproduced.
    # This prevents silently changing the Step 53 analysis.
    difference = abs(float(actual) - float(expected))

    require(
        difference <= TOLERANCE,
        f"{description}: difference {difference:.3e}",
    )

    return difference


# ------------------------------------------------------------
# 4. LOAD AND VALIDATE THE EXISTING DATA
# ------------------------------------------------------------

def load_sources():

    for path in (INPUT, BASELINE, SYNTHESIS):

        require(
            path.is_file(),
            f"Required input missing: {path}",
        )

    # Lock the analysis to the validated integrated snapshot.
    require(
        sha256(INPUT) == EXPECTED_INPUT_SHA256,
        "Integrated dataset SHA-256 does not match",
    )

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

    # Confirm the country-level integrated data structure.
    check_columns(
        integrated,
        ["cntry"] + ESS_VARIABLES + ECONOMIC_VARIABLES,
        "Integrated data",
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
        "Baseline correlations",
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
        "Validated robustness synthesis",
    )

    require(
        len(integrated) == 29,
        "Integrated dataset must contain 29 countries",
    )

    require(
        integrated["cntry"].is_unique
        and integrated["cntry"].notna().all(),
        "Country identifiers are missing or duplicated",
    )

    require(
        set(integrated["cntry"]) == COUNTRIES,
        "Unexpected integrated country coverage",
    )

    require(
        len(baseline) == 36
        and len(synthesis) == 36,
        "Unexpected baseline or synthesis pair count",
    )

    require(
        baseline["input_sha256"].eq(
            EXPECTED_INPUT_SHA256
        ).all(),
        "Baseline provenance differs",
    )

    for name, table in (
        ("Baseline", baseline),
        ("Synthesis", synthesis),
    ):

        require(
            table[
                ["ess_variable", "economic_variable"]
            ].notna().all().all(),
            f"{name}: null pair identifiers",
        )

        require(
            not table.duplicated(
                ["ess_variable", "economic_variable"]
            ).any(),
            f"{name}: duplicate analytical pair",
        )

    # Index summary tables by their exact analytical keys.
    baseline = baseline.set_index(
        ["ess_variable", "economic_variable"]
    )

    synthesis = synthesis.set_index(
        ["ess_variable", "economic_variable"]
    )

    integrated = integrated.set_index(
        "cntry"
    ).sort_index()

    print(
        "Input SHA-256, 29-country scope "
        "and analytical source structure: PASSED."
    )

    return integrated, baseline, synthesis


# ------------------------------------------------------------
# 5. CALCULATE THE FOUR SCENARIOS
# ------------------------------------------------------------

def calculate_sensitivity(integrated, baseline, synthesis):

    records = []

    maximum_baseline_difference = 0.0

    # Four Schwartz dimensions x six economic indicators.
    for ess in ESS_VARIABLES:

        for economic in ECONOMIC_VARIABLES:

            key = (ess, economic)

            require(
                key in baseline.index
                and key in synthesis.index,
                f"Missing validated analytical pair: {key}",
            )

            original = baseline.loc[key]
            consolidated = synthesis.loc[key]

            # ------------------------------------------------
            # 5.1 CREATE THE ORIGINAL PAIRWISE COUNTRY SAMPLE
            # ------------------------------------------------

            # Only countries with both variables available
            # enter the original correlation.
            paired = integrated[
                [ess, economic]
            ].dropna().copy()

            # Convert both variables to numeric values.
            paired[ess] = pd.to_numeric(
                paired[ess],
                errors="raise",
            )

            paired[economic] = pd.to_numeric(
                paired[economic],
                errors="raise",
            )

            # Ensure that no infinite observations enter the analysis.
            require(
                np.isfinite(
                    paired[
                        [ess, economic]
                    ].to_numpy(dtype=float)
                ).all(),
                f"Non-finite values: {key}",
            )

            original_countries = set(
                paired.index
            )

            # Verify original country membership and sample size.
            stored_countries = set(
                str(original["included_countries"]).split("|")
            )

            require(
                len(paired) == int(original["countries_n"]),
                f"Primary sample size mismatch: {key}",
            )

            require(
                original_countries == stored_countries,
                f"Primary country membership mismatch: {key}",
            )

            require(
                original_countries
                == set(
                    str(
                        consolidated["included_countries"]
                    ).split("|")
                ),
                f"Synthesis membership mismatch: {key}",
            )

            require(
                len(paired) == int(
                    consolidated["countries_n"]
                ),
                f"Synthesis sample size mismatch: {key}",
            )

            # ------------------------------------------------
            # 5.2 REPRODUCE THE ORIGINAL PEARSON AND SPEARMAN
            # ------------------------------------------------

            primary_pearson, primary_spearman = (
                correlation_coefficients(
                    paired[ess].to_numpy(dtype=float),
                    paired[economic].to_numpy(dtype=float),
                )
            )

            maximum_baseline_difference = max(
                maximum_baseline_difference,

                verify_baseline(
                    primary_pearson,
                    original["pearson_r"],
                    f"{key}: Pearson original",
                ),

                verify_baseline(
                    primary_spearman,
                    original["spearman_rho"],
                    f"{key}: Spearman original",
                ),

                verify_baseline(
                    primary_pearson,
                    consolidated["pearson_baseline"],
                    f"{key}: Pearson synthesis",
                ),

                verify_baseline(
                    primary_spearman,
                    consolidated["spearman_baseline"],
                    f"{key}: Spearman synthesis",
                ),
            )

            # ------------------------------------------------
            # 5.3 APPLY EACH PREDEFINED COUNTRY EXCLUSION
            # ------------------------------------------------

            for scenario_name, exclusions in SCENARIOS.items():

                # Only exclude countries actually present
                # in the original pairwise sample.
                actual_exclusions = (
                    original_countries & exclusions
                )

                # Keep all other available countries unchanged.
                scenario = paired.loc[
                    ~paired.index.isin(exclusions)
                ].copy()

                require(
                    len(scenario)
                    == len(paired) - len(actual_exclusions),
                    f"Unexpected scenario sample size: {key}",
                )

                require(
                    set(scenario.index)
                    == original_countries - exclusions,
                    f"Unexpected scenario country set: {key}",
                )

                # ------------------------------------------------
                # 5.4 RECALCULATE BOTH CORRELATION METHODS
                # ------------------------------------------------

                pearson, spearman = correlation_coefficients(
                    scenario[ess].to_numpy(dtype=float),
                    scenario[economic].to_numpy(dtype=float),
                )

                # Signed changes show direction of movement.
                pearson_change = pearson - primary_pearson
                spearman_change = spearman - primary_spearman

                # Strict sign reversals use the historical
                # project definition: baseline * scenario < 0.
                pearson_flip = (
                    primary_pearson * pearson < 0
                )

                spearman_flip = (
                    primary_spearman * spearman < 0
                )

                # ------------------------------------------------
                # 5.5 PRESERVE FULL DESCRIPTIVE INFORMATION
                # ------------------------------------------------

                records.append({
                    "ess_variable": ess,
                    "economic_variable": economic,
                    "scenario": scenario_name,
                    "requested_exclusions":
                        "|".join(sorted(exclusions)),
                    "actual_exclusions":
                        "|".join(sorted(actual_exclusions)),
                    "primary_n": len(paired),
                    "scenario_n": len(scenario),
                    "removed_n": len(actual_exclusions),
                    "scenario_countries":
                        "|".join(sorted(scenario.index)),
                    "pearson_primary": primary_pearson,
                    "pearson_scenario": pearson,
                    "pearson_change": pearson_change,
                    "pearson_abs_change":
                        abs(pearson_change),
                    "pearson_sign_flip": bool(pearson_flip),
                    "spearman_primary": primary_spearman,
                    "spearman_scenario": spearman,
                    "spearman_change": spearman_change,
                    "spearman_abs_change":
                        abs(spearman_change),
                    "spearman_sign_flip": bool(spearman_flip),
                })

    # Construct the final long-format analytical table.
    result = pd.DataFrame(records)

    # Standardize text and Boolean representation.
    for column in result.select_dtypes(
        include="object"
    ).columns:

        result[column] = result[column].astype("string")

    # Ensure exactly 24 relationships x four scenarios.
    require(
        len(result) == 96,
        "Expected 96 country-exclusion comparisons",
    )

    require(
        not result.duplicated(
            ["ess_variable", "economic_variable", "scenario"]
        ).any(),
        "Duplicate country-exclusion scenario",
    )

    # The full-sample scenario must exactly reproduce baseline.
    full = result.loc[
        result["scenario"] == "primary_full"
    ]

    require(
        len(full) == 24,
        "Expected 24 full-sample Schwartz pairs",
    )

    require(
        full["removed_n"].eq(0).all(),
        "Full scenario unexpectedly excludes a country",
    )

    require(
        full["pearson_abs_change"].le(TOLERANCE).all()
        and full["spearman_abs_change"].le(TOLERANCE).all(),
        "Full scenario does not reproduce original coefficients",
    )

    print(
        "24 original Schwartz-Eurostat correlations "
        "reproduced: PASSED."
    )

    print(
        "96 country-exclusion scenario comparisons "
        "constructed: PASSED."
    )

    return result, maximum_baseline_difference


# ------------------------------------------------------------
# 6. REPORT RESULTS
# ------------------------------------------------------------

def report_results(result, baseline_difference):

    print("\nSUMMARY BY EXCLUSION SCENARIO")
    print("-" * 85)

    summary = result.groupby("scenario", sort=False).agg(
        comparisons=("ess_variable", "size"),
        min_n=("scenario_n", "min"),
        max_n=("scenario_n", "max"),
        pearson_max_abs_change=("pearson_abs_change", "max"),
        spearman_max_abs_change=("spearman_abs_change", "max"),
        pearson_sign_flips=("pearson_sign_flip", "sum"),
        spearman_sign_flips=("spearman_sign_flip", "sum"),
    )

    print(
        summary.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    # Focus on the scenario excluding both flagged countries.
    both = result.loc[
        result["scenario"] == "exclude_EE_GB"
    ].copy()

    print("\nEXCLUDING EE AND GB - ALL 24 RELATIONSHIPS")
    print("-" * 85)

    print(
        both[
            [
                "ess_variable",
                "economic_variable",
                "primary_n",
                "scenario_n",
                "pearson_primary",
                "pearson_scenario",
                "pearson_change",
                "spearman_primary",
                "spearman_scenario",
                "spearman_change",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nTOP 10 ABSOLUTE PEARSON CHANGES")
    print("-" * 85)

    print(
        result.loc[
            result["scenario"] != "primary_full"
        ].sort_values(
            "pearson_abs_change",
            ascending=False
        )[
            [
                "ess_variable",
                "economic_variable",
                "scenario",
                "actual_exclusions",
                "pearson_change",
                "pearson_abs_change",
            ]
        ].head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSTRICT SIGN REVERSALS")
    print("-" * 85)

    flips = result.loc[
        (
            result["pearson_sign_flip"]
            | result["spearman_sign_flip"]
        )
    ]

    if flips.empty:

        print("No strict sign reversals observed.")

    else:

        print(
            flips[
                [
                    "ess_variable",
                    "economic_variable",
                    "scenario",
                    "pearson_sign_flip",
                    "spearman_sign_flip",
                ]
            ].to_string(index=False)
        )

    print("\nMaximum baseline reproduction difference:")
    print(f"{baseline_difference:.3e}")


# ------------------------------------------------------------
# 7. SAVE NEW OUTPUTS WITH PROVENANCE
# ------------------------------------------------------------

def save_results(result):

    # Never replace existing files.
    for path in (
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ):

        require(
            not path.exists(),
            f"Output already exists: {path}",
        )

    # Source hashes preserve the exact versions examined.
    source_hashes = [
        {
            "filename": path.name,
            "sha256": sha256(path),
        }
        for path in (
            INPUT,
            BASELINE,
            SYNTHESIS,
        )
    ]

    # Save analytical results in machine-readable formats.
    with OUTPUT_PARQUET.open("xb") as handle:

        result.to_parquet(
            handle,
            engine="pyarrow",
            index=False,
        )

    with OUTPUT_CSV.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:

        result.to_csv(
            handle,
            index=False,
        )

    # Document the analytic design and limitations.
    metadata = {
        "step": 62,
        "created_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "study_design": "cross-sectional exploratory",
        "analysis_unit": "country",
        "analysis_scope": "Schwartz-Eurostat correlations",
        "schwartz_dimensions_n": 4,
        "economic_indicators_n": 6,
        "analytical_pairs_n": 24,
        "scenarios_n": 4,
        "scenario_comparisons_n": 96,
        "requested_exclusion_scenarios": {
            name: sorted(values)
            for name, values in SCENARIOS.items()
        },
        "baseline_unchanged": True,
        "independent_validation_status": "pending",
        "p_values_calculated": False,
        "causal_analysis": False,
        "measurement_invariance_test": False,
        "notes": [
            "Countries are excluded only in temporary scenarios.",
            "Each coefficient uses a pairwise-complete country sample.",
            "Pearson and Spearman are recalculated after exclusion.",
            "Strict sign reversals near zero require caution.",
            "Changes caused by excluding countries do not identify the cause.",
            "The analysis does not establish measurement invariance.",
            "Existing ESS-Eurostat source files remain unchanged.",
        ],
        "sources": source_hashes,
        "outputs": [
            {
                "filename": path.name,
                "sha256": sha256(path),
            }
            for path in (OUTPUT_PARQUET, OUTPUT_CSV)
        ],
    }

    with OUTPUT_METADATA.open(
        "x",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        handle.write("\n")

    print("\nCREATED FILES")

    for path in (
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ):

        print(path)


# ------------------------------------------------------------
# 8. MAIN EXECUTION
# ------------------------------------------------------------

def main():

    print("=" * 85)
    print("STEP 62 - SCHWARTZ COUNTRY EXCLUSION SENSITIVITY")
    print("=" * 85)

    # Protect existing outputs before performing analysis.
    for path in (
        OUTPUT_PARQUET,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ):

        require(
            not path.exists(),
            f"Existing Step 62 output: {path}",
        )

    # Load the previously validated country-level data.
    integrated, baseline, synthesis = load_sources()

    # Reproduce the baseline and calculate all scenarios.
    result, baseline_difference = calculate_sensitivity(
        integrated,
        baseline,
        synthesis,
    )

    # Display descriptive sensitivity results.
    report_results(
        result,
        baseline_difference,
    )

    # Save the three new outputs only after checks pass.
    save_results(result)

    print("\nSTEP 62 SENSITIVITY OUTPUTS CREATED.")

    print(
        "Existing project inputs and previous outputs "
        "were not modified."
    )

    print(
        "Independent Step 62 validation is still pending."
    )


if __name__ == "__main__":
    main()
