# Step 55: Diagnose sensitivity to omitting one country at a time.
# The main analysis retains every available country.
# Leave-one-country-out ranges are not confidence intervals.

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import rankdata


# Resolve paths from the script location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

BASELINE_PATH = TABLE_DIR / "ESS11_country_correlations_main_v1.parquet"

DETAIL_PATH = TABLE_DIR / "ESS11_country_correlations_loo_detail_v1.parquet"
DETAIL_CSV_PATH = DETAIL_PATH.with_suffix(".csv")

SUMMARY_PATH = TABLE_DIR / "ESS11_country_correlations_loo_summary_v1.parquet"
SUMMARY_CSV_PATH = SUMMARY_PATH.with_suffix(".csv")

METADATA_PATH = (
    TABLE_DIR / "ESS11_country_correlations_loo_v1.metadata.json"
)

EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
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

PAIR_KEYS = ["ess_variable", "economic_variable"]
METHODS = ["pearson", "spearman"]

# Use the same arithmetic tolerance as the independent validation.
ABSOLUTE_TOLERANCE = 1e-12


def require(condition, message):
    # Stop when a required condition fails.
    if not condition:
        raise ValueError(message)


def read_snapshot(path):
    require(path.is_file(), f"Required file not found: {path}")

    # Parse and hash the same bytes.
    content = path.read_bytes()
    table = pd.read_parquet(io.BytesIO(content), engine="pyarrow")
    digest = hashlib.sha256(content).hexdigest()

    return table, digest


def check_outputs():
    # Protect all planned outputs before creating any files.
    paths = [
        DETAIL_PATH,
        DETAIL_CSV_PATH,
        SUMMARY_PATH,
        SUMMARY_CSV_PATH,
        METADATA_PATH,
    ]

    existing = [str(path) for path in paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Inspect them before rerunning:\n"
            + "\n".join(existing)
        )


def check_inputs(data, baseline, input_sha256):
    require(
        input_sha256 == EXPECTED_INPUT_SHA256,
        "Integrated input differs from the source used in Step 50.",
    )

    require(data.columns.is_unique, "Duplicate source column names.")

    required_source = [
        "cntry",
        "analysis_unit",
        "alignment_scenario",
        *ESS_VARIABLES,
        *ECONOMIC_VARIABLES,
    ]

    missing_source = sorted(set(required_source) - set(data.columns))
    require(not missing_source, f"Missing source columns: {missing_source}")

    require(len(data) == 29, "Expected 29 source countries.")
    require(data["cntry"].notna().all(), "Missing source country codes.")
    require(data["cntry"].is_unique, "Duplicate source country codes.")
    require(
        set(data["cntry"]) == EXPECTED_COUNTRIES,
        "Unexpected source country scope.",
    )

    for column, expected in [
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
    ]:
        require(
            data[column].notna().all()
            and data[column].eq(expected).all(),
            f"Unexpected or missing source values in {column}.",
        )

    for variable in ESS_VARIABLES + ECONOMIC_VARIABLES:
        series = data[variable]

        require(
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype),
            f"{variable}: expected a numeric measurement.",
        )

        require(
            np.isfinite(series.dropna().to_numpy(dtype=float)).all(),
            f"{variable}: infinite values found.",
        )

    required_baseline = [
        *PAIR_KEYS,
        "countries_n",
        "included_countries",
        "pearson_r",
        "spearman_rho",
        "input_sha256",
        "analysis_unit",
        "alignment_scenario",
        "across_country_weighting",
    ]

    require(baseline.columns.is_unique, "Duplicate baseline column names.")

    missing_baseline = sorted(
        set(required_baseline) - set(baseline.columns)
    )
    require(
        not missing_baseline,
        f"Missing baseline columns: {missing_baseline}",
    )

    require(len(baseline) == 36, "Expected 36 baseline pairs.")
    require(
        not baseline[required_baseline].isna().any().any(),
        "Missing required baseline values.",
    )
    require(
        not baseline.duplicated(PAIR_KEYS).any(),
        "Duplicate baseline pairs.",
    )

    expected_pairs = {
        (outcome, economic)
        for outcome in ESS_VARIABLES
        for economic in ECONOMIC_VARIABLES
    }

    actual_pairs = set(
        baseline[PAIR_KEYS].itertuples(index=False, name=None)
    )

    require(actual_pairs == expected_pairs, "Unexpected baseline pairs.")

    for column, expected in [
        ("input_sha256", input_sha256),
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
        ("across_country_weighting", "equal"),
    ]:
        require(
            baseline[column].eq(expected).all(),
            f"Baseline mismatch in {column}.",
        )


def correlations(x, y, label):
    # Both arrays must describe the same countries in the same order.
    require(
        len(x) == len(y) and len(x) >= 3,
        f"{label}: insufficient or inconsistent sample size.",
    )

    require(
        np.unique(x).size > 1 and np.unique(y).size > 1,
        f"{label}: correlation is undefined for a constant variable.",
    )

    pearson = float(np.corrcoef(x, y)[0, 1])

    # Rebuild ranks for the current subset.
    # Do not reuse ranks from the full sample after omitting a country.
    x_ranks = rankdata(x, method="average")
    y_ranks = rankdata(y, method="average")
    spearman = float(np.corrcoef(x_ranks, y_ranks)[0, 1])

    for coefficient in [pearson, spearman]:
        require(
            np.isfinite(coefficient) and -1 <= coefficient <= 1,
            f"{label}: invalid correlation coefficient.",
        )

    return {"pearson": pearson, "spearman": spearman}


def summarize_pair(pair_details, outcome, economic, full, full_n):
    # Summarize every omission, not only countries noticed in the plots.
    summary = {
        "ess_variable": outcome,
        "economic_variable": economic,
        "full_countries_n": full_n,
        "loo_countries_n": full_n - 1,
        "omissions_n": len(pair_details),
    }

    for method in METHODS:
        values = pair_details[f"{method}_loo"]
        absolute_changes = pair_details[f"{method}_change"].abs()
        maximum_change = float(absolute_changes.max())

        # Retain all exact ties for the largest observed change.
        largest_change_countries = pair_details.loc[
            absolute_changes.eq(maximum_change),
            "omitted_country",
        ].tolist()

        summary.update(
            {
                f"{method}_full": full[method],
                f"{method}_loo_min": float(values.min()),
                f"{method}_loo_max": float(values.max()),
                f"{method}_max_abs_change": maximum_change,
                f"{method}_largest_change_countries": "|".join(
                    largest_change_countries
                ),
                f"{method}_sign_flips_n": int(
                    pair_details[f"{method}_sign_flip"].sum()
                ),
            }
        )

    return summary


def build_sensitivity(data, baseline):
    # Sort once so country lists and omission order are reproducible.
    data = data.sort_values("cntry").reset_index(drop=True)
    baseline_lookup = baseline.set_index(PAIR_KEYS)

    all_details = []
    all_summaries = []

    for outcome in ESS_VARIABLES:
        for economic in ECONOMIC_VARIABLES:
            label = f"{outcome} versus {economic}"

            # Preserve the pair-specific sample used in Step 53.
            available = data[[outcome, economic]].notna().all(axis=1)
            pair = data.loc[
                available,
                ["cntry", outcome, economic],
            ].reset_index(drop=True)

            countries = pair["cntry"].tolist()
            x = pair[economic].to_numpy(dtype=float)
            y = pair[outcome].to_numpy(dtype=float)

            saved = baseline_lookup.loc[(outcome, economic)]
            full = correlations(x, y, label)

            require(
                len(pair) == saved["countries_n"],
                f"{label}: baseline sample size differs.",
            )
            require(
                "|".join(countries) == saved["included_countries"],
                f"{label}: baseline country membership differs.",
            )

            # Reproduce the validated baseline before starting omissions.
            for method, saved_column in [
                ("pearson", "pearson_r"),
                ("spearman", "spearman_rho"),
            ]:
                require(
                    abs(full[method] - float(saved[saved_column]))
                    <= ABSOLUTE_TOLERANCE,
                    f"{label}: baseline {method} differs.",
                )

                # Express changes relative to the saved validated coefficient.
                full[method] = float(saved[saved_column])

            pair_records = []

            for index, country in enumerate(countries):
                # Omit exactly one country from the current pair.
                keep = np.arange(len(pair)) != index

                reduced = correlations(
                    x[keep],
                    y[keep],
                    label=f"{label}, omitting {country}",
                )

                record = {
                    "ess_variable": outcome,
                    "economic_variable": economic,
                    "omitted_country": country,
                    "full_countries_n": len(pair),
                    "loo_countries_n": int(keep.sum()),
                }

                for method in METHODS:
                    record[f"{method}_full"] = full[method]
                    record[f"{method}_loo"] = reduced[method]

                    # Positive change means a numerically larger coefficient.
                    # It does not necessarily mean a stronger association.
                    record[f"{method}_change"] = (
                        reduced[method] - full[method]
                    )

                    # Count only a strict change between positive and negative.
                    # An exactly zero coefficient is not assigned a sign.
                    record[f"{method}_sign_flip"] = (
                        full[method] * reduced[method] < 0
                    )

                pair_records.append(record)

            pair_details = pd.DataFrame(pair_records)

            all_details.extend(pair_records)
            all_summaries.append(
                summarize_pair(
                    pair_details,
                    outcome,
                    economic,
                    full,
                    len(pair),
                )
            )

    detail = pd.DataFrame(all_details)
    summary = pd.DataFrame(all_summaries)

    # Give text columns explicit types for consistent Parquet storage.
    detail = detail.astype(
        {
            "ess_variable": "string",
            "economic_variable": "string",
            "omitted_country": "string",
        }
    )

    summary = summary.astype(
        {
            "ess_variable": "string",
            "economic_variable": "string",
            "pearson_largest_change_countries": "string",
            "spearman_largest_change_countries": "string",
        }
    )

    # Each available country must be omitted once for its analytical pair.
    expected_omissions = int(baseline["countries_n"].sum())

    require(
        len(detail) == expected_omissions,
        "Unexpected total number of country omissions.",
    )
    require(len(summary) == 36, "Expected 36 sensitivity summaries.")
    require(
        not detail.duplicated([*PAIR_KEYS, "omitted_country"]).any(),
        "Duplicate country-omission records.",
    )
    require(
        not summary.duplicated(PAIR_KEYS).any(),
        "Duplicate summary pairs.",
    )
    require(
        detail["loo_countries_n"].eq(
            detail["full_countries_n"] - 1
        ).all(),
        "An omission did not remove exactly one country.",
    )
    require(
        summary["omissions_n"].eq(summary["full_countries_n"]).all(),
        "A pair does not contain one omission per available country.",
    )

    return detail, summary


def save_outputs(detail, summary, input_sha256, baseline_sha256):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_records = []

    for table, parquet_path, csv_path in [
        (detail, DETAIL_PATH, DETAIL_CSV_PATH),
        (summary, SUMMARY_PATH, SUMMARY_CSV_PATH),
    ]:
        # Exclusive file creation prevents overwriting prior results.
        with parquet_path.open("xb") as handle:
            table.to_parquet(handle, index=False, engine="pyarrow")

        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            table.to_csv(handle, index=False)

        for path in [parquet_path, csv_path]:
            output_records.append(
                {
                    "filename": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            print(f"Saved: {path}")

    metadata = {
        "step": 55,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": INPUT_PATH.name,
        "input_sha256": input_sha256,
        "baseline_filename": BASELINE_PATH.name,
        "baseline_sha256": baseline_sha256,
        "analysis_unit": "country",
        "alignment_scenario": "primary",
        "across_country_weighting": "equal",
        "pairs_n": len(summary),
        "country_omissions_n": len(detail),
        "recalculated_coefficients_n": len(detail) * 2,
        "sample_rule": "omit one country from each baseline pair sample",
        "spearman_rule": "recalculate average ranks after each omission",
        "change_definition": "leave-one-out coefficient minus baseline",
        "sign_flip_definition": "baseline times leave-one-out coefficient < 0",
        "largest_change_ties": "retain all exact ties",
        "country_list_separator": "|",
        "baseline_tolerance": ABSOLUTE_TOLERANCE,
        "main_sample_modified": False,
        "imputation_performed": False,
        "survey_uncertainty_propagated": False,
        "p_values_calculated": False,
        "ranges_are_confidence_intervals": False,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "validation_status": "independent_validation_pending",
        "outputs": output_records,
    }

    with METADATA_PATH.open("x", encoding="utf-8") as handle:
        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    print(f"Saved: {METADATA_PATH}")


def main():
    print("STEP 55 — LEAVE-ONE-COUNTRY-OUT CORRELATION SENSITIVITY")

    check_outputs()

    data, input_sha256 = read_snapshot(INPUT_PATH)
    baseline, baseline_sha256 = read_snapshot(BASELINE_PATH)

    check_inputs(data, baseline, input_sha256)
    detail, summary = build_sensitivity(data, baseline)

    print("\nBaseline coefficients and country membership: PASSED.")
    print("One-country omission structure: PASSED.")

    display_columns = [
        *PAIR_KEYS,
        "pearson_full",
        "pearson_max_abs_change",
        "pearson_largest_change_countries",
        "spearman_full",
        "spearman_max_abs_change",
        "spearman_largest_change_countries",
    ]

    print("\nLARGEST ABSOLUTE CHANGES PER PAIR")
    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Show the exact omission events that reverse a coefficient's sign.
    sign_changes = detail.loc[
        detail["pearson_sign_flip"] | detail["spearman_sign_flip"],
        [
            *PAIR_KEYS,
            "omitted_country",
            "pearson_full",
            "pearson_loo",
            "pearson_sign_flip",
            "spearman_full",
            "spearman_loo",
            "spearman_sign_flip",
        ],
    ]

    print("\nSTRICT SIGN REVERSALS")

    if sign_changes.empty:
        print("No strict sign reversals detected.")
    else:
        print(
            sign_changes.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\nSUMMARY")
    print(f"Analytical pairs: {len(summary)}")
    print(f"Country-omission calculations: {len(detail)}")
    print(f"Recalculated coefficients: {len(detail) * 2}")
    print(f"Minimum countries after omission: {detail['loo_countries_n'].min()}")
    print(f"Maximum countries after omission: {detail['loo_countries_n'].max()}")

    print(
        "Pairs with at least one Pearson sign reversal: "
        f"{int(summary['pearson_sign_flips_n'].gt(0).sum())}"
    )
    print(
        "Pairs with at least one Spearman sign reversal: "
        f"{int(summary['spearman_sign_flips_n'].gt(0).sum())}"
    )

    print()
    save_outputs(detail, summary, input_sha256, baseline_sha256)

    print("\nCOUNTRY CORRELATION SENSITIVITY OUTPUTS CREATED.")
    print("All available countries remain in the main analysis.")
    print("Spearman ranks were recalculated after each omission.")
    print("Sensitivity ranges are not confidence intervals.")
    print("Sign reversals near zero require cautious interpretation.")
    print("No input files were modified.")
    print("Independent sensitivity validation is still pending.")


if __name__ == "__main__":
    main()