
# ============================================================
# STEP 59 - CONSOLIDATED CORRELATION ROBUSTNESS SYNTHESIS
# ============================================================
#
# Combine previously validated baseline, leave-one-country-out
# (LOO), and temporal sensitivity results into one country-level
# descriptive synthesis. This is NOT a new significance test.
# Original data and results are never modified or overwritten.

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 1. PATHS AND EXPECTED ANALYTICAL SCOPE
# ------------------------------------------------------------

# Resolve paths from the actual location of this script.
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

BASELINE_PATH = TABLES / "ESS11_country_correlations_main_v1.parquet"
LOO_DETAIL_PATH = TABLES / "ESS11_country_correlations_loo_detail_v1.parquet"
LOO_SUMMARY_PATH = TABLES / "ESS11_country_correlations_loo_summary_v1.parquet"
TEMPORAL_PATH = TABLES / "ESS11_temporal_sensitivity_correlations_v1.parquet"
LOO_METADATA_PATH = TABLES / "ESS11_country_correlations_loo_v1.metadata.json"
TEMPORAL_METADATA_PATH = TABLES / "ESS11_temporal_correlation_sensitivity_v1.metadata.json"

OUTPUT_PATH = TABLES / "ESS11_country_correlation_robustness_synthesis_v1.parquet"
OUTPUT_CSV = OUTPUT_PATH.with_suffix(".csv")
OUTPUT_METADATA = OUTPUT_PATH.with_suffix(".metadata.json")

# Pin the validated ESS-Eurostat integrated input used by Steps 53-58.
EXPECTED_INTEGRATED_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

# Define the six ESS outcomes included in the analysis.
ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

# Define the six socioeconomic indicators from Eurostat.
ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

# Both methods are kept separate throughout the synthesis.
METHODS = ("pearson", "spearman")

# Allow only negligible floating-point discrepancies.
ATOL = 1e-12

# These columns uniquely identify one analytical relationship.
PAIR_KEYS = ["ess_variable", "economic_variable"]


# ------------------------------------------------------------
# 2. SMALL VALIDATION HELPERS
# ------------------------------------------------------------

def require(condition, message):

    # Fail immediately instead of silently correcting data.
    if not bool(condition):
        raise ValueError(message)


def near(actual, expected, label):

    # Check numerical consistency between validated sources.
    # Floating-point differences below ATOL are acceptable.
    a = float(actual)
    b = float(expected)

    require(
        math.isfinite(a) and math.isfinite(b),
        f"{label}: non-finite value"
    )

    require(
        abs(a - b) <= ATOL,
        f"{label}: {a!r} != {b!r}"
    )


def read_table(path):

    # Existing analytical files must be available locally.
    require(
        path.is_file(),
        f"Missing source file: {path}"
    )

    # Read Parquet without modifying the original file.
    table = pd.read_parquet(path, engine="pyarrow")

    # Duplicate column names may cause ambiguous selections.
    require(
        table.columns.is_unique,
        f"Duplicate columns: {path.name}"
    )

    return table


def check_columns(table, required, label):

    # Stop if any required source column is missing.
    missing = sorted(
        set(required) - set(table.columns)
    )

    require(
        not missing,
        f"{label}: missing columns: {missing}"
    )


def unique_pairs(table, label):

    # Every summary must contain precisely 36 analytical pairs.
    check_columns(table, PAIR_KEYS, label)

    # Analytical identifiers cannot be missing.
    require(
        table[PAIR_KEYS].notna().all().all(),
        f"{label}: missing pair keys"
    )

    # No combination can appear twice.
    require(
        not table.duplicated(PAIR_KEYS).any(),
        f"{label}: duplicate pairs"
    )

    # Construct the expected 6 x 6 combinations independently.
    expected = {
        (ess, economic)
        for ess in ESS_VARIABLES
        for economic in ECONOMIC_VARIABLES
    }

    # Extract the combinations actually stored in the table.
    actual = set(
        table[PAIR_KEYS].itertuples(
            index=False,
            name=None
        )
    )

    require(
        len(table) == 36 and actual == expected,
        f"{label}: pair coverage differs"
    )


def hash_file(path):

    # Preserve provenance using the exact original file bytes.
    require(
        path.is_file(),
        f"Missing source file: {path}"
    )

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def country_list(text):

    # Country lists use the project's historical '|' separator.
    require(
        pd.notna(text),
        "Unexpected missing country list"
    )

    # An empty string represents an empty country list.
    if str(text) == "":
        return []

    return str(text).split("|")


# ------------------------------------------------------------
# 3. LOAD AND CHECK THE VALIDATED SOURCE TABLES
# ------------------------------------------------------------

def load_sources():

    # Load the four previously generated statistical tables.
    baseline = read_table(BASELINE_PATH)
    loo_detail = read_table(LOO_DETAIL_PATH)
    loo_summary = read_table(LOO_SUMMARY_PATH)
    temporal = read_table(TEMPORAL_PATH)

    # Validate essential baseline fields from Step 53.
    check_columns(
        baseline,
        PAIR_KEYS + [
            "countries_n",
            "included_countries",
            "pearson_r",
            "spearman_rho",
            "input_sha256",
        ],
        "Baseline",
    )

    # Validate Step 55 country-level omission records.
    check_columns(
        loo_detail,
        PAIR_KEYS + ["omitted_country"] + [
            f"{method}_{field}"
            for method in METHODS
            for field in ("full", "loo", "sign_flip")
        ],
        "LOO detail",
    )

    # Validate Step 55 summary information.
    check_columns(
        loo_summary,
        PAIR_KEYS + [
            f"{method}_{field}"
            for method in METHODS
            for field in (
                "full",
                "max_abs_change",
                "largest_change_countries",
            )
        ],
        "LOO summary",
    )

    # Validate the temporal sensitivity output from Step 57.
    check_columns(
        temporal,
        PAIR_KEYS + [
            "primary_full_n",
            "common_countries_n",
            "common_countries",
            "removed_from_primary_sample",
            "switched_context_countries_n",
            "changed_numeric_values_n",
        ] + [
            f"{method}_{field}"
            for method in METHODS
            for field in (
                "primary_full",
                "primary_common",
                "sensitivity_common",
                "temporal_change",
                "sample_change",
                "temporal_sign_flip",
            )
        ],
        "Temporal sensitivity",
    )

    # All three summary-level tables must have the same 36 pairs.
    for name, table in [
        ("Baseline", baseline),
        ("LOO summary", loo_summary),
        ("Temporal sensitivity", temporal),
    ]:
        unique_pairs(table, name)

    # LOO detail contains one record per omitted country.
    # Previous validation established exactly 948 omissions.
    require(
        len(loo_detail) == 948,
        "LOO detail: expected 948 omissions"
    )

    # A country must not be omitted twice for the same pair.
    require(
        not loo_detail.duplicated(
            PAIR_KEYS + ["omitted_country"]
        ).any(),
        "LOO detail: duplicate omitted country"
    )

    # Keys in the detailed results cannot be missing.
    require(
        loo_detail[
            PAIR_KEYS + ["omitted_country"]
        ].notna().all().all(),
        "LOO detail: missing keys"
    )

    # Detailed LOO observations must cover all 36 relationships.
    expected_pairs = set(
        baseline[PAIR_KEYS].itertuples(
            index=False,
            name=None
        )
    )

    detail_pairs = set(
        loo_detail[PAIR_KEYS].itertuples(
            index=False,
            name=None
        )
    )

    require(
        detail_pairs == expected_pairs,
        "LOO detail: unexpected analytic pairs"
    )

    # The baseline must originate from the validated input.
    require(
        baseline["input_sha256"].notna().all(),
        "Baseline: missing provenance"
    )

    require(
        baseline["input_sha256"].eq(
            EXPECTED_INTEGRATED_SHA256
        ).all(),
        "Baseline: unexpected integrated input hash"
    )

    # Historical metadata must exist and contain valid JSON.
    for path in (
        LOO_METADATA_PATH,
        TEMPORAL_METADATA_PATH,
    ):
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        require(
            isinstance(metadata, dict),
            f"Invalid JSON metadata: {path.name}"
        )

    print(
        "Four source tables, 36 analytic pairs "
        "and 948 omissions: PASSED."
    )

    return baseline, loo_detail, loo_summary, temporal


# ------------------------------------------------------------
# 4. CONSOLIDATE ONE ESS-EUROSTAT RELATIONSHIP AT A TIME
# ------------------------------------------------------------

def build_synthesis(
    baseline,
    loo_detail,
    loo_summary,
    temporal,
):

    # Create efficient lookups using the analytical pair keys.
    base_lookup = baseline.set_index(PAIR_KEYS)
    loo_lookup = loo_summary.set_index(PAIR_KEYS)
    temporal_lookup = temporal.set_index(PAIR_KEYS)

    # Keep all LOO omissions grouped by their analytical pair.
    detail_groups = {
        key: group
        for key, group in loo_detail.groupby(
            PAIR_KEYS,
            sort=False
        )
    }

    records = []

    # Build one consolidated record for each of the 36 pairs.
    for ess in ESS_VARIABLES:

        for economic in ECONOMIC_VARIABLES:

            key = (ess, economic)

            # Retrieve previously validated source results.
            base = base_lookup.loc[key]
            saved_loo = loo_lookup.loc[key]
            time = temporal_lookup.loc[key]
            detail = detail_groups[key]

            label = f"{ess} / {economic}"

            # ==================================================
            # 4.1 VERIFY THE BASELINE COUNTRY SAMPLE
            # ==================================================

            # Recover the exact countries in the main analysis.
            included = country_list(
                base["included_countries"]
            )

            # Recover every omitted country in Step 55.
            omitted = detail[
                "omitted_country"
            ].astype(str).tolist()

            # Baseline sample size must match its country list.
            require(
                len(included) == int(base["countries_n"]),
                f"{label}: baseline country count differs"
            )

            # Each country can appear only once.
            require(
                len(included) == len(set(included)),
                f"{label}: repeated baseline countries"
            )

            # LOO must omit each eligible country exactly once.
            require(
                len(omitted) == len(included)
                and set(omitted) == set(included),
                f"{label}: LOO country coverage differs"
            )

            # ==================================================
            # 4.2 VERIFY THE TEMPORAL COUNTRY SAMPLES
            # ==================================================

            # Primary-full must reproduce the original sample size.
            require(
                int(time["primary_full_n"]) == len(included),
                f"{label}: temporal full N differs"
            )

            # Recover the common sample used for temporal comparison.
            common = country_list(
                time["common_countries"]
            )

            # Recover countries excluded only from the common sample.
            removed = country_list(
                time["removed_from_primary_sample"]
            )

            # Common sample size must agree with the saved count.
            require(
                len(common) == int(time["common_countries_n"]),
                f"{label}: common sample count differs"
            )

            # The common sample must contain unique countries.
            require(
                len(common) == len(set(common)),
                f"{label}: repeated common country"
            )

            # Common countries must originate from the baseline sample.
            require(
                set(common).issubset(set(included)),
                f"{label}: unknown common-sample country"
            )

            # Verify the exact country sample difference.
            require(
                set(removed)
                == set(included) - set(common),
                f"{label}: removed country list differs"
            )

            # For Gini, Switzerland is absent only from the
            # paired temporal comparison.
            #
            # Other indicators should not lose any country.
            require(
                removed == (
                    ["CH"] if economic == "gini" else []
                ),
                f"{label}: unexpected temporal sample loss"
            )

            # ==================================================
            # 4.3 CREATE GENERAL DESCRIPTIVE FIELDS
            # ==================================================

            # These fields describe the association and its samples.
            # They are not a subjective robustness score.
            result = {
                "ess_variable": ess,
                "economic_variable": economic,
                "countries_n": len(included),
                "included_countries": "|".join(included),
                "common_countries_n": len(common),
                "common_countries": "|".join(common),
                "removed_from_primary_sample": "|".join(removed),
                "switched_context_countries_n": int(
                    time["switched_context_countries_n"]
                ),
                "changed_numeric_values_n": int(
                    time["changed_numeric_values_n"]
                ),

                # Agreement means both coefficients have the same
                # strict positive or negative sign.
                #
                # A coefficient equal to zero does not satisfy this rule.
                "pearson_spearman_same_strict_sign": (
                    float(base["pearson_r"])
                    * float(base["spearman_rho"]) > 0
                ),

                # Preserve the numerical difference between methods.
                "pearson_minus_spearman": (
                    float(base["pearson_r"])
                    - float(base["spearman_rho"])
                ),
            }

            # ==================================================
            # 4.4 PROCESS PEARSON AND SPEARMAN SEPARATELY
            # ==================================================

            for method, baseline_col in (
                ("pearson", "pearson_r"),
                ("spearman", "spearman_rho"),
            ):

                # Read the original correlation coefficient.
                original = float(
                    base[baseline_col]
                )

                # Step 55 and Step 57 must reproduce the baseline.
                near(
                    saved_loo[f"{method}_full"],
                    original,
                    f"{label}: LOO baseline"
                )

                near(
                    time[f"{method}_primary_full"],
                    original,
                    f"{label}: temporal baseline"
                )

                # ==================================================
                # 4.5 RECONSTRUCT ALL LOO CHANGES
                # ==================================================

                values = []
                change_by_country = {}
                flip_countries = []

                # Inspect every country-omission result.
                for row in detail.itertuples(index=False):

                    country = row.omitted_country

                    # Read the coefficient after omitting this country.
                    loo_value = float(
                        getattr(row, f"{method}_loo")
                    )

                    # Read the baseline repeated in LOO detail.
                    full_value = float(
                        getattr(row, f"{method}_full")
                    )

                    # Repeated baselines must match Step 53.
                    near(
                        full_value,
                        original,
                        f"{label}: detail baseline for {country}"
                    )

                    # Reject non-finite correlation coefficients.
                    require(
                        math.isfinite(loo_value),
                        f"{label}: non-finite LOO coefficient "
                        f"for {country}"
                    )

                    # Read the saved strict sign-flip flag.
                    flag = bool(
                        getattr(row, f"{method}_sign_flip")
                    )

                    # Independently verify that sign-flip flag.
                    require(
                        flag == (original * loo_value < 0),
                        f"{label}: incorrect sign reversal "
                        f"for {country}"
                    )

                    # Retain the reduced-sample coefficient.
                    values.append(loo_value)

                    # Measure how much this country omission changes r.
                    change_by_country[country] = abs(
                        loo_value - original
                    )

                    # Preserve countries causing strict sign reversals.
                    if flag:
                        flip_countries.append(country)

                # ==================================================
                # 4.6 IDENTIFY MAXIMUM LOO CHANGES
                # ==================================================

                # Identify the largest absolute change across omissions.
                max_change = max(
                    change_by_country.values()
                )

                # Retain all countries tied at exactly the maximum.
                largest_countries = sorted(
                    country
                    for country, change in change_by_country.items()
                    if change == max_change
                )

                # Compare against the validated LOO summary.
                near(
                    saved_loo[f"{method}_max_abs_change"],
                    max_change,
                    f"{label}: largest LOO change"
                )

                # Confirm the countries responsible for that maximum.
                require(
                    str(
                        saved_loo[
                            f"{method}_largest_change_countries"
                        ]
                    ) == "|".join(largest_countries),
                    f"{label}: largest-change country list differs"
                )

                # ==================================================
                # 4.7 VERIFY TEMPORAL DECOMPOSITION
                # ==================================================

                # Original coefficient using the full sample.
                full = float(
                    time[f"{method}_primary_full"]
                )

                # Original economic values on the common sample.
                primary_common = float(
                    time[f"{method}_primary_common"]
                )

                # Alternative economic values on the same sample.
                sensitivity_common = float(
                    time[f"{method}_sensitivity_common"]
                )

                # Isolate the change in socioeconomic context year.
                temporal_change = (
                    sensitivity_common - primary_common
                )

                # Isolate the change caused by sample composition.
                sample_change = (
                    primary_common - full
                )

                # Verify both changes against Step 57.
                near(
                    time[f"{method}_temporal_change"],
                    temporal_change,
                    f"{label}: temporal change"
                )

                near(
                    time[f"{method}_sample_change"],
                    sample_change,
                    f"{label}: sample change"
                )

                # Strict sign reversal on identical common samples.
                sign_flip = (
                    primary_common * sensitivity_common < 0
                )

                require(
                    bool(
                        time[f"{method}_temporal_sign_flip"]
                    ) == sign_flip,
                    f"{label}: incorrect temporal sign flag"
                )

                # ==================================================
                # 4.8 PRESERVE DESCRIPTIVE ROBUSTNESS INDICATORS
                # ==================================================

                # Store transparent measures in separate columns.
                # No arbitrary overall robustness score is calculated.
                result.update({
                    f"{method}_baseline": original,
                    f"{method}_loo_min": min(values),
                    f"{method}_loo_max": max(values),
                    f"{method}_loo_max_abs_change": max_change,
                    f"{method}_loo_largest_change_countries":
                        "|".join(largest_countries),
                    f"{method}_loo_sign_flip_n":
                        len(flip_countries),
                    f"{method}_loo_sign_flip_countries":
                        "|".join(sorted(flip_countries)),
                    f"{method}_loo_any_sign_flip":
                        bool(flip_countries),
                    f"{method}_primary_common":
                        primary_common,
                    f"{method}_sensitivity_common":
                        sensitivity_common,
                    f"{method}_temporal_change":
                        temporal_change,
                    f"{method}_sample_change":
                        sample_change,
                    f"{method}_temporal_sign_flip":
                        sign_flip,
                })

            # Append the completed ESS-economic relationship.
            records.append(result)

    # --------------------------------------------------------
    # 4.9 BUILD THE CONSOLIDATED DATAFRAME
    # --------------------------------------------------------

    # Create one tabular output containing all 36 relationships.
    result = pd.DataFrame(records)

    # Require one unique row per ESS-economic combination.
    require(
        len(result) == 36
        and not result.duplicated(PAIR_KEYS).any(),
        "Invalid consolidated output pair structure"
    )

    # Standardize text columns to pandas nullable string dtype.
    for column in result.columns:

        if result[column].dtype == object:
            result[column] = result[column].astype("string")

    # --------------------------------------------------------
    # 4.10 VERIFY KNOWN AGGREGATE SENSITIVITY RESULTS
    # --------------------------------------------------------

    # Step 55 reported nine pairs with Pearson LOO sign changes.
    require(
        sum(result["pearson_loo_any_sign_flip"]) == 9,
        "Unexpected Pearson LOO sign-flip pair count"
    )

    # Step 55 reported seven pairs with Spearman LOO sign changes.
    require(
        sum(result["spearman_loo_any_sign_flip"]) == 7,
        "Unexpected Spearman LOO sign-flip pair count"
    )

    # Step 57 reported one Pearson temporal sign reversal.
    require(
        sum(result["pearson_temporal_sign_flip"]) == 1,
        "Unexpected Pearson temporal sign-flip pair count"
    )

    # Step 57 reported two Spearman temporal sign reversals.
    require(
        sum(result["spearman_temporal_sign_flip"]) == 2,
        "Unexpected Spearman temporal sign-flip pair count"
    )

    return result


# ------------------------------------------------------------
# 5. SAVE NEW OUTPUTS EXCLUSIVELY AND RECORD PROVENANCE
# ------------------------------------------------------------

def save_new_outputs(result):

    # Define the three new files generated by Step 59.
    outputs = [
        OUTPUT_PATH,
        OUTPUT_CSV,
        OUTPUT_METADATA,
    ]

    # Never overwrite previously generated analytical outputs.
    existing = [
        str(path)
        for path in outputs
        if path.exists()
    ]

    require(
        not existing,
        "Output already exists; inspect it before rerunning:\n"
        + "\n".join(existing)
    )

    # This step expects the analytical output directory to exist.
    require(
        TABLES.is_dir(),
        f"Expected output directory missing: {TABLES}"
    )

    # --------------------------------------------------------
    # 5.1 RECORD INPUT PROVENANCE
    # --------------------------------------------------------

    # Preserve the exact six upstream source files.
    source_paths = [
        BASELINE_PATH,
        LOO_DETAIL_PATH,
        LOO_SUMMARY_PATH,
        TEMPORAL_PATH,
        LOO_METADATA_PATH,
        TEMPORAL_METADATA_PATH,
    ]

    # Calculate SHA-256 from the original file contents.
    sources = [
        {
            "filename": path.name,
            "sha256": hash_file(path),
        }
        for path in source_paths
    ]

    # --------------------------------------------------------
    # 5.2 SAVE PARQUET AND CSV
    # --------------------------------------------------------

    # 'xb' creates a binary file only if it does not exist.
    # The original source files are never opened for writing.
    with OUTPUT_PATH.open("xb") as handle:

        result.to_parquet(
            handle,
            engine="pyarrow",
            index=False
        )

    # Save a CSV representation for inspection and reporting.
    with OUTPUT_CSV.open(
        "x",
        encoding="utf-8",
        newline=""
    ) as handle:

        result.to_csv(
            handle,
            index=False
        )

    # --------------------------------------------------------
    # 5.3 CREATE HISTORICAL OUTPUT METADATA
    # --------------------------------------------------------

    # Metadata describe what was created, from which sources,
    # using which methodological assumptions.
    metadata = {
        "step": 59,
        "created_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "analysis_unit": "country",
        "study_design": "cross-sectional exploratory",
        "country_weighting": "equal across countries",

        "purpose":
            "descriptive consolidation of validated correlation sensitivities",

        "analytic_pairs_n": len(result),

        # The new synthesis still requires independent validation.
        "independent_validation_status": "pending",

        "inference": {
            "p_values_calculated": False,
            "confidence_intervals_calculated": False,
            "causal_claims": False,
            "robustness_score_created": False,
        },

        "interpretation_notes": [
            "LOO ranges are not confidence intervals.",
            "Temporal changes compare identical common country samples.",
            "For Gini, CH is excluded only from the common comparison.",
            "Sign changes near zero need contextual interpretation.",
            "Numerical stability does not establish causal or measurement validity.",
        ],

        # Record all upstream source hashes.
        "sources": sources,

        # Record the hashes of both new tabular outputs.
        "outputs": [
            {
                "filename": path.name,
                "sha256": hash_file(path),
            }
            for path in (
                OUTPUT_PATH,
                OUTPUT_CSV,
            )
        ],
    }

    # Use exclusive creation for the new metadata file.
    with OUTPUT_METADATA.open(
        "x",
        encoding="utf-8"
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )

        handle.write("\n")

    # Display the created file paths.
    for path in outputs:
        print(f"Created: {path}")


# ------------------------------------------------------------
# 6. MAIN: VALIDATE, CONSOLIDATE, THEN WRITE NEW OUTPUTS
# ------------------------------------------------------------

def main():

    print(
        "STEP 59 - CONSOLIDATED CORRELATION ROBUSTNESS SYNTHESIS"
    )

    # --------------------------------------------------------
    # 6.1 PREVENT ACCIDENTAL OVERWRITING
    # --------------------------------------------------------

    # Check planned output names before processing the sources.
    require(
        not any(
            path.exists()
            for path in (
                OUTPUT_PATH,
                OUTPUT_CSV,
                OUTPUT_METADATA,
            )
        ),
        "Step 59 output exists; inspect it before rerunning"
    )

    # --------------------------------------------------------
    # 6.2 LOAD AND VERIFY VALIDATED SOURCES
    # --------------------------------------------------------

    baseline, detail, summary, temporal = load_sources()

    # --------------------------------------------------------
    # 6.3 CREATE THE DESCRIPTIVE SYNTHESIS
    # --------------------------------------------------------

    # Combine baseline, LOO and temporal results in memory.
    result = build_synthesis(
        baseline,
        detail,
        summary,
        temporal
    )

    # --------------------------------------------------------
    # 6.4 REPORT THE MAIN DIAGNOSTIC COUNTS
    # --------------------------------------------------------

    print(
        f"Analytic pairs consolidated: {len(result)}"
    )

    print(
        "Pairs with Pearson LOO sign flip:",
        int(result["pearson_loo_any_sign_flip"].sum())
    )

    print(
        "Pairs with Spearman LOO sign flip:",
        int(result["spearman_loo_any_sign_flip"].sum())
    )

    print(
        "Pairs with Pearson temporal sign flip:",
        int(result["pearson_temporal_sign_flip"].sum())
    )

    print(
        "Pairs with Spearman temporal sign flip:",
        int(result["spearman_temporal_sign_flip"].sum())
    )

    # --------------------------------------------------------
    # 6.5 DISPLAY THE LARGEST TEMPORAL CHANGES
    # --------------------------------------------------------

    # This is a descriptive inspection, not a robustness ranking.
    preview = result.assign(
        absolute_pearson_temporal_change=
            result["pearson_temporal_change"].abs()
    ).nlargest(
        5,
        "absolute_pearson_temporal_change"
    )

    print(
        "\nLargest absolute Pearson temporal changes (descriptive):"
    )

    print(
        preview[
            [
                "ess_variable",
                "economic_variable",
                "pearson_temporal_change",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 6.6 SAVE ONLY AFTER ALL CHECKS HAVE PASSED
    # --------------------------------------------------------

    save_new_outputs(result)

    print(
        "\nSTEP 59 SYNTHESIS OUTPUTS CREATED."
    )

    print(
        "Existing input and previously validated output "
        "files were not modified."
    )

    print(
        "Independent validation of Step 59 outputs "
        "is still pending."
    )


# ------------------------------------------------------------
# 7. RUN THE SCRIPT ONLY WHEN EXECUTED DIRECTLY
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
