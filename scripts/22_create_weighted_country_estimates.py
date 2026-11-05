from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input file
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)

# Define output directory and files
output_dir = (
    project_root
    / "data"
    / "processed"
)

parquet_file = (
    output_dir
    / "ESS11_country_weighted_estimates.parquet"
)

csv_file = (
    output_dir
    / "ESS11_country_weighted_estimates.csv"
)


# Check that the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Analytical dataset not found: {input_file}"
    )

# Protect existing output files
if parquet_file.exists() or csv_file.exists():
    raise FileExistsError(
        "Country weighted estimates already exist."
    )


# Load the analytical respondent-level dataset
df = pd.read_parquet(input_file)


# ---------------------------------------------------------------------
# Weighted helper functions
# ---------------------------------------------------------------------

def weighted_mean(data, value_column, weight_column="anweight"):

    valid = data[
        data[value_column].notna()
        & data[weight_column].notna()
    ]

    if valid.empty:
        return np.nan

    return np.average(
        valid[value_column].astype(float),
        weights=valid[weight_column].astype(float),
    )


def weighted_share(data, flag_column, weight_column="anweight"):

    valid = data[
        data[flag_column].notna()
        & data[weight_column].notna()
    ]

    if valid.empty:
        return np.nan

    values = (
        valid[flag_column]
        .astype(float)
    )

    weights = (
        valid[weight_column]
        .astype(float)
    )

    return np.average(
        values,
        weights=weights,
    )


# ---------------------------------------------------------------------
# Define higher-order Human Values
# ---------------------------------------------------------------------

human_value_columns = {
    "mean_openness": "hv_openness_to_change_centered",
    "mean_conservation": "hv_conservation_centered",
    "mean_self_transcendence": "hv_self_transcendence_centered",
    "mean_self_enhancement": "hv_self_enhancement_centered",
}


# ---------------------------------------------------------------------
# Calculate country-level weighted estimates
# ---------------------------------------------------------------------

country_results = []


for (
    cntry,
    country_name,
    region_4
), group in df.groupby(
    [
        "cntry",
        "country_name",
        "region_4",
    ],
    dropna=False,
):

    result = {
        "cntry": cntry,
        "country_name": country_name,
        "region_4": region_4,

        "respondents_n": len(group),

        "ideology_valid_n": (
            group["lrscale"]
            .notna()
            .sum()
        ),

        "human_values_valid_n": (
            group["hv_scoring_eligible"]
            .sum()
        ),

        "social_trust_valid_n": (
            group["trust_scoring_eligible"]
            .sum()
        ),

        "mean_left_right": weighted_mean(
            group,
            "lrscale",
        ),

        "left_share": weighted_share(
            group,
            "ideology_left_flag",
        ),

        "centre_share": weighted_share(
            group,
            "ideology_centre_flag",
        ),

        "right_share": weighted_share(
            group,
            "ideology_right_flag",
        ),

        "mean_social_trust": weighted_mean(
            group,
            "social_trust_score",
        ),

        "mean_ppltrst": weighted_mean(
            group,
            "ppltrst",
        ),

        "mean_pplfair": weighted_mean(
            group,
            "pplfair",
        ),

        "mean_pplhlp": weighted_mean(
            group,
            "pplhlp",
        ),
    }


    for output_name, variable in human_value_columns.items():

        result[output_name] = weighted_mean(
            group,
            variable,
        )


    country_results.append(result)


# Convert results to DataFrame
country_estimates = pd.DataFrame(
    country_results
)


# ---------------------------------------------------------------------
# Add missing-data indicators
# ---------------------------------------------------------------------

country_estimates["ideology_missing_n"] = (
    country_estimates["respondents_n"]
    - country_estimates["ideology_valid_n"]
)

country_estimates["ideology_missing_rate"] = (
    country_estimates["ideology_missing_n"]
    / country_estimates["respondents_n"]
)

country_estimates["human_values_missing_n"] = (
    country_estimates["respondents_n"]
    - country_estimates["human_values_valid_n"]
)

country_estimates["human_values_missing_rate"] = (
    country_estimates["human_values_missing_n"]
    / country_estimates["respondents_n"]
)

country_estimates["social_trust_missing_n"] = (
    country_estimates["respondents_n"]
    - country_estimates["social_trust_valid_n"]
)

country_estimates["social_trust_missing_rate"] = (
    country_estimates["social_trust_missing_n"]
    / country_estimates["respondents_n"]
)


# ---------------------------------------------------------------------
# Validate ideology shares
# ---------------------------------------------------------------------

country_estimates["ideology_share_sum"] = (
    country_estimates["left_share"]
    + country_estimates["centre_share"]
    + country_estimates["right_share"]
)

share_sum_valid = np.allclose(
    country_estimates["ideology_share_sum"],
    1.0,
    atol=1e-10,
)


# ---------------------------------------------------------------------
# Sort countries
# ---------------------------------------------------------------------

country_estimates = (
    country_estimates
    .sort_values("country_name")
    .reset_index(drop=True)
)


# Save processed outputs
country_estimates.to_parquet(
    parquet_file,
    index=False,
)

country_estimates.to_csv(
    csv_file,
    index=False,
)


print("=" * 110)
print("ESS ROUND 11 WEIGHTED COUNTRY ESTIMATES")
print("=" * 110)

print()
print(f"Countries: {len(country_estimates)}")

print()
print("IDEOLOGY SHARE VALIDATION")
print("-" * 110)

print(
    f"Left + Centre + Right = 1: "
    f"{'PASSED' if share_sum_valid else 'FAILED'}"
)


print()
print("WEIGHTED COUNTRY ESTIMATES")
print("-" * 110)

display_columns = [
    "cntry",
    "country_name",
    "respondents_n",
    "ideology_valid_n",
    "mean_left_right",
    "left_share",
    "centre_share",
    "right_share",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
    "mean_social_trust",
]

print(
    country_estimates[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "mean_left_right": "{:.4f}".format,
            "left_share": "{:.4f}".format,
            "centre_share": "{:.4f}".format,
            "right_share": "{:.4f}".format,
            "mean_openness": "{:.4f}".format,
            "mean_conservation": "{:.4f}".format,
            "mean_self_transcendence": "{:.4f}".format,
            "mean_self_enhancement": "{:.4f}".format,
            "mean_social_trust": "{:.4f}".format,
        }
    )
)


print()
print("=" * 110)
print("COUNTRY ESTIMATES CREATED")
print("=" * 110)

print(f"Parquet: {parquet_file}")
print(f"CSV:     {csv_file}")

print()
print(
    "All point estimates use the ESS analysis weight (anweight)."
)

print(
    "Confidence intervals have not yet been calculated."
)

print(
    "Country estimates must not yet be interpreted as causal effects."
)