from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
respondent_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)

country_file = (
    project_root
    / "data"
    / "processed"
    / "ESS11_country_weighted_estimates.parquet"
)


# Check that input files exist
if not respondent_file.exists():
    raise FileNotFoundError(
        f"Respondent dataset not found: {respondent_file}"
    )

if not country_file.exists():
    raise FileNotFoundError(
        f"Country estimates not found: {country_file}"
    )


# Load datasets
df = pd.read_parquet(respondent_file)
country = pd.read_parquet(country_file)


# ---------------------------------------------------------------------
# Weighted helper functions
# ---------------------------------------------------------------------

def weighted_mean(data, value_column, weight_column):

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


def weighted_share(data, flag_column, weight_column):

    valid = data[
        data[flag_column].notna()
        & data[weight_column].notna()
    ]

    if valid.empty:
        return np.nan

    return np.average(
        valid[flag_column].astype(float),
        weights=valid[weight_column].astype(float),
    )


print("=" * 110)
print("ESS ROUND 11 WEIGHTED COUNTRY ESTIMATE VALIDATION")
print("=" * 110)

validation_passed = True


# ---------------------------------------------------------------------
# 1. Validate country coverage
# ---------------------------------------------------------------------

print()
print("COUNTRY COVERAGE")
print("-" * 110)

country_count_valid = len(country) == 29

unique_country_valid = (
    country["cntry"].nunique() == 29
)

israel_absent = (
    "IL" not in set(country["cntry"])
)

cyprus_present = (
    "CY" in set(country["cntry"])
)

print(f"Country rows:       {len(country)}")
print(f"Unique countries:   {country['cntry'].nunique()}")

print(
    f"29 countries:       "
    f"{'PASSED' if country_count_valid else 'FAILED'}"
)

print(
    f"Unique country rows:"
    f" {'PASSED' if unique_country_valid else 'FAILED'}"
)

print(
    f"Israel excluded:    "
    f"{'PASSED' if israel_absent else 'FAILED'}"
)

print(
    f"Cyprus included:    "
    f"{'PASSED' if cyprus_present else 'FAILED'}"
)

if not all([
    country_count_valid,
    unique_country_valid,
    israel_absent,
    cyprus_present,
]):
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Validate respondent counts
# ---------------------------------------------------------------------

print()
print("RESPONDENT COUNTS")
print("-" * 110)

respondent_total_valid = (
    country["respondents_n"].sum()
    == len(df)
)

print(
    f"Microdata respondents: "
    f"{len(df):,}"
)

print(
    f"Aggregated respondents:"
    f" {country['respondents_n'].sum():,}"
)

print(
    f"Status:               "
    f"{'PASSED' if respondent_total_valid else 'FAILED'}"
)

if not respondent_total_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 3. Validate country-level sample counts
# ---------------------------------------------------------------------

print()
print("COUNTRY-LEVEL SAMPLE COUNTS")
print("-" * 110)

expected_counts = (
    df
    .groupby("cntry")
    .size()
)

stored_counts = (
    country
    .set_index("cntry")["respondents_n"]
)

count_alignment_valid = (
    expected_counts
    .sort_index()
    .equals(
        stored_counts
        .sort_index()
    )
)

print(
    f"Respondent counts by country: "
    f"{'PASSED' if count_alignment_valid else 'FAILED'}"
)

if not count_alignment_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 4. Validate valid-N indicators
# ---------------------------------------------------------------------

print()
print("VALID-N INDICATORS")
print("-" * 110)

valid_n_checks = {
    "ideology_valid_n": "lrscale",
    "human_values_valid_n": "hv_scoring_eligible",
    "social_trust_valid_n": "trust_scoring_eligible",
}


for output_column, source_column in valid_n_checks.items():

    if source_column in [
        "hv_scoring_eligible",
        "trust_scoring_eligible",
    ]:

        expected = (
            df
            .groupby("cntry")[source_column]
            .sum()
            .astype(int)
        )

    else:

        expected = (
            df
            .groupby("cntry")[source_column]
            .apply(
                lambda x: x.notna().sum()
            )
        )

    stored = (
        country
        .set_index("cntry")[output_column]
        .astype(int)
    )

    valid = (
        expected
        .sort_index()
        .equals(
            stored
            .sort_index()
        )
    )

    print(
        f"{output_column:<25} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate ideology shares
# ---------------------------------------------------------------------

print()
print("IDEOLOGY SHARES")
print("-" * 110)

share_sum = (
    country["left_share"]
    + country["centre_share"]
    + country["right_share"]
)

share_sum_valid = np.allclose(
    share_sum,
    1.0,
    atol=1e-10,
)

share_ranges_valid = (
    country[
        [
            "left_share",
            "centre_share",
            "right_share",
        ]
    ]
    .apply(
        lambda x: x.between(0, 1).all()
    )
    .all()
)

print(
    f"Shares sum to 1: "
    f"{'PASSED' if share_sum_valid else 'FAILED'}"
)

print(
    f"Shares within 0-1: "
    f"{'PASSED' if share_ranges_valid else 'FAILED'}"
)

if not share_sum_valid or not share_ranges_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 6. Validate analytical ranges
# ---------------------------------------------------------------------

print()
print("ANALYTICAL RANGES")
print("-" * 110)

range_checks = {
    "mean_left_right": (0, 10),
    "mean_social_trust": (0, 10),
    "mean_ppltrst": (0, 10),
    "mean_pplfair": (0, 10),
    "mean_pplhlp": (0, 10),

    "mean_openness": (-5, 5),
    "mean_conservation": (-5, 5),
    "mean_self_transcendence": (-5, 5),
    "mean_self_enhancement": (-5, 5),
}


for variable, (minimum, maximum) in range_checks.items():

    valid = (
        country[variable]
        .dropna()
        .between(
            minimum,
            maximum
        )
        .all()
    )

    print(
        f"{variable:<30} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 7. Independently recalculate weighted estimates
# ---------------------------------------------------------------------

print()
print("INDEPENDENT ANWEIGHT RECALCULATION")
print("-" * 110)

estimate_mapping = {
    "mean_left_right": (
        "lrscale",
        "mean",
    ),

    "left_share": (
        "ideology_left_flag",
        "share",
    ),

    "centre_share": (
        "ideology_centre_flag",
        "share",
    ),

    "right_share": (
        "ideology_right_flag",
        "share",
    ),

    "mean_openness": (
        "hv_openness_to_change_centered",
        "mean",
    ),

    "mean_conservation": (
        "hv_conservation_centered",
        "mean",
    ),

    "mean_self_transcendence": (
        "hv_self_transcendence_centered",
        "mean",
    ),

    "mean_self_enhancement": (
        "hv_self_enhancement_centered",
        "mean",
    ),

    "mean_social_trust": (
        "social_trust_score",
        "mean",
    ),
}


max_errors = []


for output_column, (
    source_column,
    estimate_type,
) in estimate_mapping.items():

    recalculated = {}

    for cntry, group in df.groupby("cntry"):

        if estimate_type == "mean":

            value = weighted_mean(
                group,
                source_column,
                "anweight",
            )

        else:

            value = weighted_share(
                group,
                source_column,
                "anweight",
            )

        recalculated[cntry] = value


    recalculated = pd.Series(
        recalculated
    )

    stored = (
        country
        .set_index("cntry")[output_column]
    )

    aligned = pd.concat(
        [
            recalculated.rename("recalculated"),
            stored.rename("stored"),
        ],
        axis=1,
    )

    max_error = (
        aligned["recalculated"]
        - aligned["stored"]
    ).abs().max()

    valid = np.allclose(
        aligned["recalculated"],
        aligned["stored"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )

    max_errors.append(max_error)

    print(
        f"{output_column:<30} "
        f"{'PASSED' if valid else 'FAILED'} "
        f"| max error = {max_error:.12f}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 8. Validate pspwght / anweight equivalence within countries
# ---------------------------------------------------------------------

print()
print("PSPWGHT VS ANWEIGHT COUNTRY ESTIMATES")
print("-" * 110)

equivalence_passed = True

comparison_variables = [
    "lrscale",
    "social_trust_score",
    "hv_openness_to_change_centered",
    "hv_conservation_centered",
    "hv_self_transcendence_centered",
    "hv_self_enhancement_centered",
]


for variable in comparison_variables:

    maximum_difference = 0.0

    for _, group in df.groupby("cntry"):

        psp_estimate = weighted_mean(
            group,
            variable,
            "pspwght",
        )

        an_estimate = weighted_mean(
            group,
            variable,
            "anweight",
        )

        difference = abs(
            psp_estimate
            - an_estimate
        )

        maximum_difference = max(
            maximum_difference,
            difference,
        )


    valid = np.isclose(
        maximum_difference,
        0.0,
        atol=1e-10,
    )

    print(
        f"{variable:<40} "
        f"{'PASSED' if valid else 'FAILED'} "
        f"| max difference = {maximum_difference:.12f}"
    )

    if not valid:
        equivalence_passed = False
        validation_passed = False


# ---------------------------------------------------------------------
# 9. Validate missing-data calculations
# ---------------------------------------------------------------------

print()
print("MISSING-DATA INDICATORS")
print("-" * 110)

missing_checks = [
    (
        "ideology_valid_n",
        "ideology_missing_n",
        "ideology_missing_rate",
    ),
    (
        "human_values_valid_n",
        "human_values_missing_n",
        "human_values_missing_rate",
    ),
    (
        "social_trust_valid_n",
        "social_trust_missing_n",
        "social_trust_missing_rate",
    ),
]


for (
    valid_column,
    missing_column,
    rate_column,
) in missing_checks:

    expected_missing = (
        country["respondents_n"]
        - country[valid_column]
    )

    missing_valid = (
        expected_missing
        .equals(
            country[missing_column]
        )
    )

    expected_rate = (
        expected_missing
        / country["respondents_n"]
    )

    rate_valid = np.allclose(
        expected_rate,
        country[rate_column],
        atol=1e-12,
    )

    valid = (
        missing_valid
        and rate_valid
    )

    print(
        f"{rate_column:<30} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("VALIDATION SUMMARY")
print("=" * 110)

if validation_passed:
    print("Weighted country estimate validation PASSED.")
else:
    print("Weighted country estimate validation requires review.")

print()

print(
    "Processed country estimates were not modified."
)