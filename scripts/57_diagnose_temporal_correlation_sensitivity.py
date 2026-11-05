# Step 57: Diagnose sensitivity to the alternative context year.
# Switch years only for countries with fieldwork in both years.
# Compare primary and sensitivity coefficients on identical country samples.

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


# Resolve paths from the script location.
ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

MATCHES_PATH = (
    ROOT / "data" / "interim"
    / "ess_eurostat_temporal_matches_v1.parquet"
)

BASELINE_PATH = TABLE_DIR / "ESS11_country_correlations_main_v1.parquet"

PAIRS_PATH = TABLE_DIR / "ESS11_temporal_sensitivity_values_v1.parquet"
PAIRS_CSV_PATH = PAIRS_PATH.with_suffix(".csv")

RESULTS_PATH = (
    TABLE_DIR / "ESS11_temporal_sensitivity_correlations_v1.parquet"
)
RESULTS_CSV_PATH = RESULTS_PATH.with_suffix(".csv")

METADATA_PATH = (
    TABLE_DIR / "ESS11_temporal_correlation_sensitivity_v1.metadata.json"
)

EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# These countries have a validated alternative fieldwork year.
SWITCH_COUNTRIES = set(
    "BE BG CH CY FI FR HR IE IT LV PL PT RS".split()
)

ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

TOLERANCE = 1e-12


def require(condition, message):
    # Stop explicitly when an expected condition fails.
    if not condition:
        raise ValueError(message)


def read_snapshot(path):
    require(path.is_file(), f"Required file not found: {path}")

    # Parse and hash the same original bytes.
    content = path.read_bytes()
    table = pd.read_parquet(io.BytesIO(content), engine="pyarrow")
    digest = hashlib.sha256(content).hexdigest()

    return table, digest


def require_columns(table, columns, name):
    require(table.columns.is_unique, f"{name}: duplicate column names.")

    missing = sorted(set(columns) - set(table.columns))
    require(not missing, f"{name}: missing columns: {missing}")


def check_output_paths():
    # Check every planned output before creating any files.
    paths = [
        PAIRS_PATH,
        PAIRS_CSV_PATH,
        RESULTS_PATH,
        RESULTS_CSV_PATH,
        METADATA_PATH,
    ]

    existing = [str(path) for path in paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Inspect them before rerunning:\n"
            + "\n".join(existing)
        )


def number_or_nan(value):
    # Preserve missing values; never replace them with zero.
    if pd.isna(value):
        return float("nan")

    result = float(value)
    require(np.isfinite(result), "Infinite economic value found.")
    return result


def same_value(left, right):
    # Two missing values match; a missing and a numeric value do not.
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))

    return bool(left == right)


def validate_inputs(data, matches, baseline, hashes):
    require(
        hashes["integrated"] == EXPECTED_INPUT_SHA256,
        "Integrated input differs from the validated source.",
    )

    require_columns(
        data,
        [
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
        ],
        "Integrated input",
    )

    require(
        len(data) == 29
        and data["cntry"].notna().all()
        and data["cntry"].is_unique
        and set(data["cntry"]) == EXPECTED_COUNTRIES,
        "Unexpected integrated country scope.",
    )

    # Verify that these matches are the source used in the main table.
    require(
        data["matched_input_sha256"].notna().all()
        and data["matched_input_sha256"].eq(hashes["matches"]).all(),
        "Matched input hash differs from integrated provenance.",
    )

    require(
        data["multi_year_fieldwork_flag"].notna().all(),
        "Missing multi-year fieldwork flags.",
    )

    actual_switch_countries = set(
        data.loc[data["multi_year_fieldwork_flag"].eq(True), "cntry"]
    )

    require(
        actual_switch_countries == SWITCH_COUNTRIES,
        "Unexpected alternative-year country scope.",
    )

    match_keys = ["cntry", "indicator_id", "alignment_scenario"]

    require_columns(
        matches,
        [
            *match_keys,
            "context_year",
            "requested_eurostat_year",
            "value",
            "observation_match_status",
            "eurostat_status",
        ],
        "Matched observations",
    )

    require(
        not matches[match_keys].isna().any().any()
        and not matches.duplicated(match_keys).any(),
        "Missing or duplicate matched-observation keys.",
    )

    expected_keys = {
        (country, indicator, "primary")
        for country in EXPECTED_COUNTRIES
        for indicator in ECONOMIC_VARIABLES
    } | {
        (country, indicator, "alternative")
        for country in SWITCH_COUNTRIES
        for indicator in ECONOMIC_VARIABLES
    }

    actual_keys = set(
        matches[match_keys].itertuples(index=False, name=None)
    )

    require(
        len(matches) == 252 and actual_keys == expected_keys,
        "Unexpected temporal-match coverage.",
    )

    # Availability categories must agree with numeric-value availability.
    require(
        matches["observation_match_status"].isin(
            ["available", "source_value_missing", "source_cell_absent"]
        ).all(),
        "Unexpected observation-match status.",
    )
    require(
        matches["observation_match_status"].eq("available").eq(
            matches["value"].notna()
        ).all(),
        "Match status and numeric availability disagree.",
    )

    require_columns(
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

    expected_pairs = {
        (outcome, indicator)
        for outcome in ESS_VARIABLES
        for indicator in ECONOMIC_VARIABLES
    }

    require(
        len(baseline) == 36
        and not baseline.duplicated(
            ["ess_variable", "economic_variable"]
        ).any()
        and set(
            baseline[["ess_variable", "economic_variable"]]
            .itertuples(index=False, name=None)
        ) == expected_pairs,
        "Unexpected baseline correlation coverage.",
    )

    require(
        baseline["input_sha256"].notna().all()
        and baseline["input_sha256"].eq(hashes["integrated"]).all(),
        "Baseline input provenance differs.",
    )


def build_paired_values(data, matches):
    source = data.set_index("cntry")

    lookup = matches.set_index(
        ["cntry", "indicator_id", "alignment_scenario"]
    )

    records = []

    for country in sorted(EXPECTED_COUNTRIES):
        switch_year = country in SWITCH_COUNTRIES

        # Single-year countries retain their primary economic observation.
        selected_scenario = "alternative" if switch_year else "primary"

        for indicator in ECONOMIC_VARIABLES:
            primary = lookup.loc[(country, indicator, "primary")]
            sensitivity = lookup.loc[
                (country, indicator, selected_scenario)
            ]

            primary_value = number_or_nan(primary["value"])
            sensitivity_value = number_or_nan(sensitivity["value"])

            primary_year = int(primary["context_year"])
            sensitivity_year = int(sensitivity["context_year"])

            require(
                primary_year == int(source.loc[country, "context_year"]),
                f"{country}: primary context year differs.",
            )

            expected_year = (
                2024 if primary_year == 2023 else 2023
            ) if switch_year else primary_year

            require(
                primary_year in [2023, 2024]
                and sensitivity_year == expected_year,
                f"{country}: incorrect sensitivity context year.",
            )

            primary_requested = int(primary["requested_eurostat_year"])
            sensitivity_requested = int(
                sensitivity["requested_eurostat_year"]
            )

            require(
                primary_requested == int(
                    source.loc[
                        country,
                        f"{indicator}__requested_eurostat_year",
                    ]
                ),
                f"{country} / {indicator}: primary requested year differs.",
            )

            # Retain the project's existing indicator-specific alignment.
            offset = 1 if indicator == "gini" else 0

            require(
                primary_requested == primary_year + offset
                and sensitivity_requested == sensitivity_year + offset,
                f"{country} / {indicator}: inconsistent year offset.",
            )

            require(
                same_value(
                    primary_value,
                    source.loc[country, indicator],
                ),
                f"{country} / {indicator}: primary value differs.",
            )

            primary_available = bool(pd.notna(primary_value))
            sensitivity_available = bool(pd.notna(sensitivity_value))
            both_available = primary_available and sensitivity_available

            records.append(
                {
                    "cntry": country,
                    "indicator_id": indicator,
                    "sensitivity_source_scenario": selected_scenario,
                    "temporal_switch_flag": switch_year,
                    "primary_context_year": primary_year,
                    "sensitivity_context_year": sensitivity_year,
                    "primary_requested_eurostat_year": primary_requested,
                    "sensitivity_requested_eurostat_year": sensitivity_requested,
                    "primary_value": primary_value,
                    "sensitivity_value": sensitivity_value,
                    "primary_available_flag": primary_available,
                    "sensitivity_available_flag": sensitivity_available,
                    "both_available_flag": both_available,
                    # A year switch may still produce the same numeric value.
                    "value_changed_flag": (
                        primary_value != sensitivity_value
                        if both_available else pd.NA
                    ),
                    "primary_match_status": primary["observation_match_status"],
                    "sensitivity_match_status": sensitivity[
                        "observation_match_status"
                    ],
                    # Preserve source quality flags in both scenarios.
                    "primary_eurostat_status": primary["eurostat_status"],
                    "sensitivity_eurostat_status": sensitivity[
                        "eurostat_status"
                    ],
                }
            )

    paired = pd.DataFrame(records)

    text_columns = [
        "cntry",
        "indicator_id",
        "sensitivity_source_scenario",
        "primary_match_status",
        "sensitivity_match_status",
        "primary_eurostat_status",
        "sensitivity_eurostat_status",
    ]

    paired = paired.astype(
        {
            **{column: "string" for column in text_columns},
            "value_changed_flag": "boolean",
        }
    )

    require(
        len(paired) == 174
        and not paired.duplicated(["cntry", "indicator_id"]).any(),
        "Unexpected paired economic table structure.",
    )

    return paired


def coefficients(x, y, label):
    require(
        len(x) == len(y) and len(x) >= 3,
        f"{label}: insufficient or inconsistent sample size.",
    )
    require(
        np.isfinite(x).all() and np.isfinite(y).all(),
        f"{label}: non-finite correlation inputs.",
    )
    require(
        np.unique(x).size > 1 and np.unique(y).size > 1,
        f"{label}: constant correlation input.",
    )

    values = {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        # Rank each scenario separately within the identical country sample.
        "spearman": float(
            np.corrcoef(
                rankdata(x, method="average"),
                rankdata(y, method="average"),
            )[0, 1]
        ),
    }

    require(
        all(np.isfinite(value) and -1 <= value <= 1 for value in values.values()),
        f"{label}: invalid correlation results.",
    )

    return values


def compare_scenarios(data, paired, baseline):
    ess_lookup = data.set_index("cntry")
    baseline_lookup = baseline.set_index(
        ["ess_variable", "economic_variable"]
    )

    records = []

    for outcome in ESS_VARIABLES:
        for indicator in ECONOMIC_VARIABLES:
            part = paired.loc[
                paired["indicator_id"].eq(indicator)
            ].copy()

            # Attach the same ESS estimate to both economic scenarios.
            part["ess_value"] = part["cntry"].map(ess_lookup[outcome])

            full_mask = (
                part["primary_available_flag"]
                & part["ess_value"].notna()
            )

            common_mask = (
                part["both_available_flag"]
                & part["ess_value"].notna()
            )

            full_sample = part.loc[full_mask]
            common = part.loc[common_mask]

            label = f"{outcome} / {indicator}"

            primary_full = coefficients(
                full_sample["primary_value"].to_numpy(dtype=float),
                full_sample["ess_value"].to_numpy(dtype=float),
                label,
            )

            saved = baseline_lookup.loc[(outcome, indicator)]

            require(
                len(full_sample) == saved["countries_n"]
                and "|".join(full_sample["cntry"]) == saved["included_countries"],
                f"{label}: baseline country sample differs.",
            )

            for method, column in [
                ("pearson", "pearson_r"),
                ("spearman", "spearman_rho"),
            ]:
                require(
                    abs(primary_full[method] - float(saved[column]))
                    <= TOLERANCE,
                    f"{label}: baseline coefficient differs.",
                )
                primary_full[method] = float(saved[column])

            # Use identical countries and ESS values in both calculations.
            y_common = common["ess_value"].to_numpy(dtype=float)

            primary_common = coefficients(
                common["primary_value"].to_numpy(dtype=float),
                y_common,
                f"{label}: primary common sample",
            )

            sensitivity_common = coefficients(
                common["sensitivity_value"].to_numpy(dtype=float),
                y_common,
                f"{label}: sensitivity common sample",
            )

            removed = sorted(
                set(full_sample["cntry"]) - set(common["cntry"])
            )

            record = {
                "ess_variable": outcome,
                "economic_variable": indicator,
                "primary_full_n": len(full_sample),
                "common_countries_n": len(common),
                "switched_context_countries_n": int(
                    common["temporal_switch_flag"].sum()
                ),
                "changed_numeric_values_n": int(
                    common["value_changed_flag"].sum()
                ),
                "common_countries": "|".join(common["cntry"]),
                "removed_from_primary_sample": "|".join(removed),
            }

            for method in ["pearson", "spearman"]:
                record.update(
                    {
                        f"{method}_primary_full": primary_full[method],
                        f"{method}_primary_common": primary_common[method],
                        f"{method}_sensitivity_common": sensitivity_common[method],
                        # This isolates scenario change on the common sample.
                        f"{method}_temporal_change": (
                            sensitivity_common[method] - primary_common[method]
                        ),
                        # This separately describes the sample-composition change.
                        f"{method}_sample_change": (
                            primary_common[method] - primary_full[method]
                        ),
                        f"{method}_temporal_sign_flip": (
                            primary_common[method] * sensitivity_common[method] < 0
                        ),
                    }
                )

            records.append(record)

    results = pd.DataFrame(records).astype(
        {
            "ess_variable": "string",
            "economic_variable": "string",
            "common_countries": "string",
            "removed_from_primary_sample": "string",
        }
    )

    require(
        len(results) == 36
        and not results.duplicated(
            ["ess_variable", "economic_variable"]
        ).any(),
        "Unexpected scenario-comparison structure.",
    )

    return results


def save_outputs(paired, results, hashes):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    for table, parquet_path, csv_path in [
        (paired, PAIRS_PATH, PAIRS_CSV_PATH),
        (results, RESULTS_PATH, RESULTS_CSV_PATH),
    ]:
        # Exclusive creation protects existing results.
        with parquet_path.open("xb") as handle:
            table.to_parquet(handle, index=False, engine="pyarrow")

        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            table.to_csv(handle, index=False)

        for path in [parquet_path, csv_path]:
            outputs.append(
                {
                    "filename": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            print(f"Saved: {path}")

    metadata = {
        "step": 57,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "filename": path.name,
                "sha256": hashes[key],
            }
            for key, path in [
                ("integrated", INPUT_PATH),
                ("matches", MATCHES_PATH),
                ("baseline", BASELINE_PATH),
            ]
        ],
        "analysis_unit": "country",
        "across_country_weighting": "equal",
        "switch_countries": sorted(SWITCH_COUNTRIES),
        "scenario_rule": "alternative year for multi-year countries; primary otherwise",
        "comparison_sample": "same countries with ESS and both economic values",
        "ess_estimates_changed": False,
        "missing_alternative_fallback": False,
        "imputation_performed": False,
        "gini_offset": 1,
        "paired_economic_rows_n": len(paired),
        "correlation_pairs_n": len(results),
        "temporal_change_definition": "sensitivity common minus primary common",
        "sample_change_definition": "primary common minus primary full",
        "survey_uncertainty_propagated": False,
        "p_values_calculated": False,
        "country_list_separator": "|",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "validation_status": "independent_validation_pending",
        "outputs": outputs,
    }

    with METADATA_PATH.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"Saved: {METADATA_PATH}")


def main():
    print("STEP 57 — TEMPORAL CORRELATION SENSITIVITY")

    check_output_paths()

    data, input_hash = read_snapshot(INPUT_PATH)
    matches, matches_hash = read_snapshot(MATCHES_PATH)
    baseline, baseline_hash = read_snapshot(BASELINE_PATH)

    hashes = {
        "integrated": input_hash,
        "matches": matches_hash,
        "baseline": baseline_hash,
    }

    validate_inputs(data, matches, baseline, hashes)

    paired = build_paired_values(data, matches)
    results = compare_scenarios(data, paired, baseline)

    print("\nPrimary values, years and baseline correlations: PASSED.")

    coverage = paired.groupby("indicator_id").agg(
        primary_available_n=("primary_available_flag", "sum"),
        sensitivity_available_n=("sensitivity_available_flag", "sum"),
        both_available_n=("both_available_flag", "sum"),
        switched_context_n=("temporal_switch_flag", "sum"),
    ).reindex(ECONOMIC_VARIABLES)

    print("\nECONOMIC AVAILABILITY BY SCENARIO")
    print(coverage.to_string())

    for method in ["pearson", "spearman"]:
        columns = [
            "ess_variable",
            "economic_variable",
            "primary_full_n",
            "common_countries_n",
            f"{method}_primary_full",
            f"{method}_primary_common",
            f"{method}_sensitivity_common",
            f"{method}_temporal_change",
        ]

        print(f"\n{method.upper()} — TEMPORAL SENSITIVITY")
        print(
            results[columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    # Show where the common sample differs from the original main sample.
    losses = results.loc[
        results["removed_from_primary_sample"].ne(""),
        [
            "ess_variable",
            "economic_variable",
            "primary_full_n",
            "common_countries_n",
            "removed_from_primary_sample",
        ],
    ]

    print("\nCOUNTRIES UNAVAILABLE FOR THE PAIRED COMPARISON")
    print("None." if losses.empty else losses.to_string(index=False))

    print("\nSUMMARY")
    print(f"Countries retained in economic scenario table: {paired['cntry'].nunique()}")
    print(f"Countries assigned an alternative context year: {len(SWITCH_COUNTRIES)}")
    print(f"Countries retaining their primary context year: {29 - len(SWITCH_COUNTRIES)}")
    print(f"Paired economic rows: {len(paired)}")
    print(f"Correlation comparisons: {len(results)}")
    print(f"Minimum common countries: {results['common_countries_n'].min()}")
    print(f"Maximum common countries: {results['common_countries_n'].max()}")

    for method in ["pearson", "spearman"]:
        print(
            f"{method.title()} temporal sign reversals: "
            f"{int(results[f'{method}_temporal_sign_flip'].sum())}"
        )

    print()
    save_outputs(paired, results, hashes)

    print("\nTEMPORAL CORRELATION SENSITIVITY OUTPUTS CREATED.")
    print("Scenario comparisons use identical country samples.")
    print("ESS estimates were held fixed.")
    print("Missing alternative values were not replaced with primary values.")
    print("No input files were modified.")
    print("Independent temporal-sensitivity validation is still pending.")


if __name__ == "__main__":
    main()