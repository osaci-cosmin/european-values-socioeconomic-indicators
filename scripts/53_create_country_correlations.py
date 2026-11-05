# Step 53: Calculate descriptive country-level correlations.
# Pearson uses the observed values; Spearman uses their ranks.
# Each available country receives equal weight in these calculations.

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import rankdata


# Resolve all paths relative to the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

PARQUET_PATH = OUTPUT_DIR / "ESS11_country_correlations_main_v1.parquet"
CSV_PATH = OUTPUT_DIR / "ESS11_country_correlations_main_v1.csv"
METADATA_PATH = (
    OUTPUT_DIR / "ESS11_country_correlations_main_v1.metadata.json"
)

# Use the same input inspected in Steps 50–52.
EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# Keep every planned main outcome, regardless of the visual pattern.
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


def require(condition, message):
    # Stop explicitly when an input or calculation requirement fails.
    if not condition:
        raise ValueError(message)


def check_outputs():
    # Protect existing results before starting any file creation.
    existing = [
        str(path)
        for path in [PARQUET_PATH, CSV_PATH, METADATA_PATH]
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Output files already exist. Inspect them before rerunning:\n"
            + "\n".join(existing)
        )


def load_input():
    require(INPUT_PATH.is_file(), f"Input file not found: {INPUT_PATH}")

    # Read once so the hash and the table refer to identical bytes.
    content = INPUT_PATH.read_bytes()
    input_sha256 = hashlib.sha256(content).hexdigest()

    require(
        input_sha256 == EXPECTED_INPUT_SHA256,
        "Input differs from Step 50. Investigate before continuing.",
    )

    data = pd.read_parquet(io.BytesIO(content), engine="pyarrow")

    required_columns = [
        "cntry",
        "context_year",
        "analysis_unit",
        "alignment_scenario",
        *ESS_VARIABLES,
        *ECONOMIC_VARIABLES,
    ]

    require(data.columns.is_unique, "Duplicate column names found.")

    missing_columns = sorted(set(required_columns) - set(data.columns))

    require(
        not missing_columns,
        f"Missing required columns: {missing_columns}",
    )

    # One row per country prevents accidental duplication of observations.
    require(len(data) == 29, "Expected 29 country rows.")
    require(data["cntry"].notna().all(), "Missing country codes found.")
    require(data["cntry"].is_unique, "Duplicate country codes found.")
    require(
        set(data["cntry"]) == EXPECTED_COUNTRIES,
        "Unexpected V1 country scope.",
    )

    for column, expected in [
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
    ]:
        require(
            data[column].notna().all()
            and data[column].eq(expected).all(),
            f"Unexpected or missing values in {column}.",
        )

    require(
        data["context_year"].isin([2023, 2024]).all(),
        "Unexpected or missing context year.",
    )

    for variable in ESS_VARIABLES + ECONOMIC_VARIABLES:
        series = data[variable]

        # Reject unexpected types rather than silently converting them.
        require(
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype),
            f"{variable}: expected a numeric measurement.",
        )

        observed = series.dropna().to_numpy(dtype=float)

        # Missing values are permitted; infinite values are not.
        require(
            np.isfinite(observed).all(),
            f"{variable}: infinite values found.",
        )

    return data.sort_values("cntry").reset_index(drop=True), input_sha256


def calculate_coefficients(x, y, pair_name):
    # x and y must contain the same countries in the same order.
    require(len(x) == len(y), f"{pair_name}: unequal array lengths.")
    require(len(x) >= 3, f"{pair_name}: fewer than three available pairs.")

    # Correlation is undefined if either variable is constant.
    require(
        np.unique(x).size > 1 and np.unique(y).size > 1,
        f"{pair_name}: a variable is constant within the available sample.",
    )

    # The off-diagonal entry is the Pearson correlation between x and y.
    pearson_r = float(np.corrcoef(x, y)[0, 1])

    # Rank within this pair's sample, after removing incomplete pairs.
    # Equal values receive the average of their occupied rank positions.
    x_ranks = rankdata(x, method="average")
    y_ranks = rankdata(y, method="average")

    # Spearman is Pearson correlation applied to these ranks.
    spearman_rho = float(np.corrcoef(x_ranks, y_ranks)[0, 1])

    # Check the numerical results before saving them.
    for name, coefficient in [
        ("Pearson", pearson_r),
        ("Spearman", spearman_rho),
    ]:
        require(
            np.isfinite(coefficient) and -1 <= coefficient <= 1,
            f"{pair_name}: invalid {name} coefficient.",
        )

    return pearson_r, spearman_rho


def build_results(data, input_sha256):
    rows = []

    for outcome in ESS_VARIABLES:
        for economic in ECONOMIC_VARIABLES:
            # Select complete pairs separately for every relationship.
            # Do not require completeness on the other economic indicators.
            available = data[[outcome, economic]].notna().all(axis=1)

            pair = data.loc[
                available,
                ["cntry", "context_year", outcome, economic],
            ]

            x = pair[economic].to_numpy(dtype=float)
            y = pair[outcome].to_numpy(dtype=float)

            pearson_r, spearman_rho = calculate_coefficients(
                x,
                y,
                pair_name=f"{outcome} versus {economic}",
            )

            included = pair["cntry"].tolist()
            unavailable = data.loc[~available, "cntry"].tolist()

            rows.append(
                {
                    "ess_variable": outcome,
                    "economic_variable": economic,
                    "countries_n": len(pair),
                    "unavailable_countries_n": len(unavailable),
                    "pearson_r": pearson_r,
                    "spearman_rho": spearman_rho,
                    # This difference is descriptive, not a statistical test.
                    "pearson_minus_spearman": pearson_r - spearman_rho,
                    "included_countries": "|".join(included),
                    "unavailable_countries": "|".join(unavailable),
                    "context_2023_n": int(
                        pair["context_year"].eq(2023).sum()
                    ),
                    "context_2024_n": int(
                        pair["context_year"].eq(2024).sum()
                    ),
                    # Counts reveal whether the input contains tied values.
                    "economic_unique_values_n": int(np.unique(x).size),
                    "ess_unique_values_n": int(np.unique(y).size),
                    "analysis_unit": "country",
                    "alignment_scenario": "primary",
                    "across_country_weighting": "equal",
                    "input_filename": INPUT_PATH.name,
                    "input_sha256": input_sha256,
                }
            )

    # Construct the table once to avoid repeated column insertion.
    results = pd.DataFrame(rows)

    text_columns = [
        "ess_variable",
        "economic_variable",
        "included_countries",
        "unavailable_countries",
        "analysis_unit",
        "alignment_scenario",
        "across_country_weighting",
        "input_filename",
        "input_sha256",
    ]

    # Store text columns explicitly for stable Parquet types.
    results = results.astype(
        {column: "string" for column in text_columns}
    )

    # Each of the six outcomes must appear with each economic indicator.
    expected_pairs = {
        (outcome, economic)
        for outcome in ESS_VARIABLES
        for economic in ECONOMIC_VARIABLES
    }

    actual_pairs = set(
        results[["ess_variable", "economic_variable"]]
        .itertuples(index=False, name=None)
    )

    require(
        len(results) == 36 and actual_pairs == expected_pairs,
        "The results do not contain exactly the 36 planned pairs.",
    )

    require(
        not results.duplicated(
            ["ess_variable", "economic_variable"]
        ).any(),
        "Duplicate correlation pairs found.",
    )

    # Every country must be either included or unavailable for each pair.
    require(
        (
            results["countries_n"]
            + results["unavailable_countries_n"]
        ).eq(len(data)).all(),
        "Country availability counts are inconsistent.",
    )

    require(
        (
            results["context_2023_n"]
            + results["context_2024_n"]
        ).eq(results["countries_n"]).all(),
        "Context-year counts are inconsistent.",
    )

    return results


def print_matrix(results, value_column, title):
    # Keep the same row and column order across all displayed matrices.
    matrix = results.pivot(
        index="ess_variable",
        columns="economic_variable",
        values=value_column,
    ).reindex(
        index=ESS_VARIABLES,
        columns=ECONOMIC_VARIABLES,
    )

    print(f"\n{title}")

    # Round only the display; saved coefficients retain calculation precision.
    print(
        matrix.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )


def save_results(results, input_sha256):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents accidental overwriting.
    with PARQUET_PATH.open("xb") as handle:
        results.to_parquet(handle, index=False, engine="pyarrow")

    with CSV_PATH.open("x", encoding="utf-8", newline="") as handle:
        results.to_csv(handle, index=False)

    # Record the methods and their limits alongside the numerical results.
    metadata = {
        "step": 53,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": INPUT_PATH.name,
        "input_sha256": input_sha256,
        "analysis_unit": "country",
        "alignment_scenario": "primary",
        "pairs_n": len(results),
        "coefficients_n": len(results) * 2,
        "ess_variables": ESS_VARIABLES,
        "economic_variables": ECONOMIC_VARIABLES,
        "across_country_weighting": "equal",
        "within_country_ess_weighting": "existing survey estimates retained",
        "missing_policy": "complete pairs separately per relationship",
        "country_list_separator": "|",
        "pearson_method": "numpy.corrcoef on observed values",
        "spearman_method": "numpy.corrcoef on scipy average ranks",
        "year_adjustment": "none",
        "survey_uncertainty_propagated": False,
        "p_values_calculated": False,
        "confidence_intervals_calculated": False,
        "extreme_values_removed": False,
        "validation_status": "independent_validation_pending",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "outputs": [
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in [PARQUET_PATH, CSV_PATH]
        ],
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

    for path in [PARQUET_PATH, CSV_PATH, METADATA_PATH]:
        print(f"Saved: {path}")


def main():
    print("STEP 53 — DESCRIPTIVE COUNTRY CORRELATIONS")

    check_outputs()
    data, input_sha256 = load_input()
    results = build_results(data, input_sha256)

    print_matrix(
        results,
        "pearson_r",
        "PEARSON CORRELATIONS",
    )

    print_matrix(
        results,
        "spearman_rho",
        "SPEARMAN CORRELATIONS",
    )

    print_matrix(
        results,
        "countries_n",
        "AVAILABLE COUNTRIES PER PAIR",
    )

    print("\nSUMMARY")
    print(f"ESS outcomes: {len(ESS_VARIABLES)}")
    print(f"Economic indicators: {len(ECONOMIC_VARIABLES)}")
    print(f"Analytical pairs: {len(results)}")
    print(f"Coefficients calculated: {len(results) * 2}")
    print(f"Minimum countries per pair: {results['countries_n'].min()}")
    print(f"Maximum countries per pair: {results['countries_n'].max()}")
    print("Analysis unit: country")
    print("Across-country weighting: equal")

    print()
    save_results(results, input_sha256)

    print("\nDESCRIPTIVE COUNTRY CORRELATIONS CREATED.")
    print("Country membership was recorded for every pair.")
    print("No imputation or automatic extreme-value removal was performed.")
    print("No p-values or confidence intervals were calculated.")
    print("Survey uncertainty was not propagated into the correlations.")
    print("No input files were modified.")
    print("Independent correlation validation is still pending.")


if __name__ == "__main__":
    main()