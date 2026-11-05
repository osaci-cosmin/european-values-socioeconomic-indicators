# Step 50: Describe the validated country-level dataset.
# Each row represents one country.
# Results describe the countries with available data for each variable.

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


# Find the project folder using this script's location.
# This avoids hard-coding a Windows username or desktop path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)


# Keep the established V1 country scope.
EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)


# These six outcomes represent the main analytical domains.
MAIN_ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]


# These variables provide additional descriptive detail.
# Ideology shares remain proportions between 0 and 1.
ADDITIONAL_ESS_VARIABLES = [
    "left_share",
    "centre_share",
    "right_share",
    "mean_ppltrst",
    "mean_pplfair",
    "mean_pplhlp",
]


# Preserve the original units of all six economic indicators.
ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]


ALL_VARIABLES = (
    MAIN_ESS_VARIABLES
    + ADDITIONAL_ESS_VARIABLES
    + ECONOMIC_VARIABLES
)


def require(condition, message):
    # Stop with a clear explanation if an input requirement fails.
    if not condition:
        raise ValueError(message)


def load_input():
    # Check that the validated input file exists.
    require(INPUT_PATH.is_file(), f"Input file not found: {INPUT_PATH}")

    # Read the exact bytes used for this diagnostic.
    content = INPUT_PATH.read_bytes()

    # The hash identifies the specific input version being analysed.
    input_sha256 = hashlib.sha256(content).hexdigest()

    # Read from the same bytes used to calculate the hash.
    data = pd.read_parquet(io.BytesIO(content), engine="pyarrow")

    return data, input_sha256


def check_input(data):
    # Duplicate column names would make variable selection ambiguous.
    require(data.columns.is_unique, "Duplicate column names found.")

    required_columns = [
        "cntry",
        "country_name",
        "context_year",
        "alignment_scenario",
        "analysis_unit",
        *ALL_VARIABLES,
    ]

    missing_columns = sorted(set(required_columns) - set(data.columns))

    require(
        not missing_columns,
        f"Required columns are missing: {missing_columns}",
    )

    # One unique row per country is essential for this analysis.
    require(len(data) == 29, "Expected 29 country rows.")
    require(data["cntry"].notna().all(), "Missing country codes found.")
    require(data["cntry"].is_unique, "Duplicate country codes found.")

    require(
        set(data["cntry"]) == EXPECTED_COUNTRIES,
        "The country scope differs from the established V1 scope.",
    )

    # Confirm the analysis level and the selected alignment scenario.
    for column, expected_value in [
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
    ]:
        require(
            data[column].notna().all()
            and data[column].eq(expected_value).all(),
            f"Unexpected or missing values in {column}.",
        )

    # Context years may differ between countries.
    require(
        data["context_year"].isin([2023, 2024]).all(),
        "Unexpected or missing context years found.",
    )

    for variable in ALL_VARIABLES:
        series = data[variable]

        # Do not silently convert text or Boolean fields into measurements.
        require(
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype),
            f"{variable}: expected a numeric measurement column.",
        )

        # Missing values are allowed; infinite values are not.
        observed = series.dropna().to_numpy(dtype=float)

        require(
            np.isfinite(observed).all(),
            f"{variable}: infinite values found.",
        )


def describe_variables(data, variables):
    # Describe only analytical variables, excluding numeric metadata.
    # Transpose so that each output row describes one variable.
    summary = data[variables].describe(
        percentiles=[0.25, 0.50, 0.75]
    ).T

    summary = summary.rename(
        columns={
            "count": "countries_n",
            "mean": "country_mean",
            "std": "country_sd",
            "25%": "q1",
            "50%": "median",
            "75%": "q3",
        }
    )

    # Each available country contributes equally to these summaries.
    # ESS estimates already contain the within-country survey weighting.
    summary["countries_n"] = summary["countries_n"].astype("int64")
    summary["missing_n"] = len(data) - summary["countries_n"]

    # The denominator is all 29 countries, not the number of respondents.
    summary["missing_pct"] = summary["missing_n"] / len(data) * 100

    # IQR measures the spread of the middle half of the country values.
    summary["iqr"] = summary["q3"] - summary["q1"]

    # Sample SD uses n - 1; it is not a survey standard error.
    summary.index.name = "variable"

    return summary[
        [
            "countries_n",
            "missing_n",
            "missing_pct",
            "country_mean",
            "country_sd",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "iqr",
        ]
    ]


def identify_extremes(data):
    # Identify countries at each observed minimum and maximum.
    # An extreme value is not automatically an error or an outlier.
    rows = []

    for variable in MAIN_ESS_VARIABLES + ECONOMIC_VARIABLES:
        observed = data.loc[
            data[variable].notna(),
            ["cntry", variable],
        ]

        # Avoid calculating extremes if a variable has no observations.
        if observed.empty:
            rows.append(
                {
                    "variable": variable,
                    "min_value": np.nan,
                    "min_countries": "",
                    "max_value": np.nan,
                    "max_countries": "",
                }
            )
            continue

        minimum = observed[variable].min()
        maximum = observed[variable].max()

        # Keep every tied country rather than selecting only the first.
        minimum_countries = sorted(
            observed.loc[
                observed[variable].eq(minimum), "cntry"
            ].tolist()
        )

        maximum_countries = sorted(
            observed.loc[
                observed[variable].eq(maximum), "cntry"
            ].tolist()
        )

        rows.append(
            {
                "variable": variable,
                "min_value": minimum,
                "min_countries": ", ".join(minimum_countries),
                "max_value": maximum,
                "max_countries": ", ".join(maximum_countries),
            }
        )

    # Build the table once from the collected records.
    return pd.DataFrame(rows).set_index("variable")


def count_available_pairs(data):
    # Count countries with both variables observed for each future analysis.
    # This table contains sample sizes, not correlation coefficients.
    counts = {
        economic: [
            int(
                data[[outcome, economic]]
                .notna()
                .all(axis=1)
                .sum()
            )
            for outcome in MAIN_ESS_VARIABLES
        ]
        for economic in ECONOMIC_VARIABLES
    }

    pairs = pd.DataFrame(counts, index=MAIN_ESS_VARIABLES)
    pairs.index.name = "ess_outcome"

    return pairs


def describe_country_coverage(data):
    # Build a missingness report without changing the input table.
    rows = []

    for _, row in data.iterrows():
        missing_indicators = [
            variable
            for variable in ECONOMIC_VARIABLES
            if pd.isna(row[variable])
        ]

        # Include only countries with at least one missing indicator.
        if missing_indicators:
            rows.append(
                {
                    "cntry": row["cntry"],
                    "country_name": row["country_name"],
                    "missing_n": len(missing_indicators),
                    "missing_indicators": ", ".join(missing_indicators),
                }
            )

    # Explicit columns keep the structure defined even if no rows exist.
    return pd.DataFrame(
        rows,
        columns=[
            "cntry",
            "country_name",
            "missing_n",
            "missing_indicators",
        ],
    )


def print_table(title, table):
    print(f"\n{title}")

    # Round only the terminal display, keeping calculation precision.
    print(
        table.to_string(
            float_format=lambda value: f"{value:.4f}",
            na_rep="NA",
        )
    )


def main():
    print("STEP 50 — INTEGRATED COUNTRY DESCRIPTIVE DIAGNOSTICS")

    data, input_sha256 = load_input()
    check_input(data)

    # Sort only in memory to make the displayed results reproducible.
    data = data.sort_values("cntry").reset_index(drop=True)

    print(f"\nInput: {INPUT_PATH.name}")
    print(f"Input SHA-256: {input_sha256}")
    print("Unit of analysis: country")
    print("Across-country weighting: equal weight per available country")
    print("ESS within-country estimates: existing survey weights retained")

    print_table(
        "MAIN ESS OUTCOMES — COUNTRY DISTRIBUTIONS",
        describe_variables(data, MAIN_ESS_VARIABLES),
    )

    print_table(
        "ADDITIONAL ESS VARIABLES — COUNTRY DISTRIBUTIONS",
        describe_variables(data, ADDITIONAL_ESS_VARIABLES),
    )

    print_table(
        "ECONOMIC INDICATORS — COUNTRY DISTRIBUTIONS",
        describe_variables(data, ECONOMIC_VARIABLES),
    )

    print_table(
        "COUNTRIES AT OBSERVED MINIMUM AND MAXIMUM",
        identify_extremes(data),
    )

    print_table(
        "AVAILABLE COUNTRIES PER ESS–ECONOMIC PAIR",
        count_available_pairs(data),
    )

    print("\nCOUNTRIES WITH MISSING ECONOMIC INDICATORS")
    coverage = describe_country_coverage(data)

    if coverage.empty:
        print("All countries have all six economic indicators.")
    else:
        print(coverage.to_string(index=False))

    # This is the ESS context year, not necessarily the Eurostat label year.
    # Gini follows the separate reference-year alignment already documented.
    print("\nCOUNTRIES BY CONTEXT YEAR")

    year_counts = (
        data.groupby("context_year")
        .size()
        .rename("countries_n")
    )

    print(year_counts.to_string())

    # Count complete countries without removing incomplete countries.
    all_economic_available = (
        data[ECONOMIC_VARIABLES]
        .notna()
        .all(axis=1)
    )

    all_main_available = (
        data[MAIN_ESS_VARIABLES + ECONOMIC_VARIABLES]
        .notna()
        .all(axis=1)
    )

    ess_variables_n = (
        len(MAIN_ESS_VARIABLES)
        + len(ADDITIONAL_ESS_VARIABLES)
    )

    print("\nSUMMARY")
    print(f"Countries retained: {len(data)}")
    print(f"ESS variables described: {ess_variables_n}")
    print(f"Economic indicators described: {len(ECONOMIC_VARIABLES)}")

    print(
        "Countries with all six economic indicators: "
        f"{int(all_economic_available.sum())}"
    )

    print(
        "Countries with all main ESS and economic variables: "
        f"{int(all_main_available.sum())}"
    )

    print("\nCOUNTRY DESCRIPTIVE DIAGNOSTICS COMPLETED.")
    print("Summaries describe available country values in the primary scenario.")
    print("Country SD is not a survey standard error.")
    print("No correlations, significance tests or models were calculated.")
    print("No countries or extreme values were removed.")
    print("No files were created or modified.")


# Run the diagnostic only when this script is executed directly.
if __name__ == "__main__":
    main()