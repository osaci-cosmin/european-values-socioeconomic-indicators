from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
respondent_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)

point_file = (
    project_root
    / "data"
    / "processed"
    / "ESS11_country_weighted_estimates.parquet"
)

survey_file = (
    project_root
    / "data"
    / "processed"
    / "ESS11_country_survey_estimates.parquet"
)


# Check input files
for file_path in [
    respondent_file,
    point_file,
    survey_file,
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# Load datasets
df = pd.read_parquet(
    respondent_file
)

point_estimates = pd.read_parquet(
    point_file
)

survey_estimates = pd.read_parquet(
    survey_file
)


# ---------------------------------------------------------------------
# Define analytical metrics
# ---------------------------------------------------------------------

metrics = {
    "mean_left_right": "lrscale",

    "left_share": "ideology_left_flag",
    "centre_share": "ideology_centre_flag",
    "right_share": "ideology_right_flag",

    "mean_openness": (
        "hv_openness_to_change_centered"
    ),

    "mean_conservation": (
        "hv_conservation_centered"
    ),

    "mean_self_transcendence": (
        "hv_self_transcendence_centered"
    ),

    "mean_self_enhancement": (
        "hv_self_enhancement_centered"
    ),

    "mean_social_trust": (
        "social_trust_score"
    ),

    "mean_ppltrst": "ppltrst",
    "mean_pplfair": "pplfair",
    "mean_pplhlp": "pplhlp",
}


print("=" * 120)
print("ESS ROUND 11 COUNTRY SURVEY ESTIMATE VALIDATION")
print("=" * 120)

validation_passed = True


# ---------------------------------------------------------------------
# 1. Country coverage
# ---------------------------------------------------------------------

print()
print("COUNTRY COVERAGE")
print("-" * 120)

same_country_count = (
    len(point_estimates)
    == len(survey_estimates)
    == 29
)

same_country_codes = (
    set(point_estimates["cntry"])
    == set(survey_estimates["cntry"])
    == set(df["cntry"].unique())
)

print(
    f"29 countries preserved: "
    f"{'PASSED' if same_country_count else 'FAILED'}"
)

print(
    f"Country codes aligned:  "
    f"{'PASSED' if same_country_codes else 'FAILED'}"
)

if not same_country_count or not same_country_codes:
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Point-estimate preservation
# ---------------------------------------------------------------------

print()
print("POINT ESTIMATE PRESERVATION")
print("-" * 120)

for metric in metrics:

    point_values = (
        point_estimates
        .set_index("cntry")[metric]
        .sort_index()
    )

    survey_values = (
        survey_estimates
        .set_index("cntry")[metric]
        .sort_index()
    )

    valid = np.allclose(
        point_values,
        survey_values,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )

    maximum_difference = (
        point_values
        - survey_values
    ).abs().max()

    print(
        f"{metric:<30} "
        f"{'PASSED' if valid else 'FAILED'} "
        f"| max difference = "
        f"{maximum_difference:.12f}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 3. Independently calculate country design structure
# ---------------------------------------------------------------------

print()
print("SURVEY DESIGN STRUCTURE")
print("-" * 120)

design_valid = True


for cntry, group in df.groupby("cntry"):

    psu_structure = (
        group[
            [
                "stratum",
                "psu",
            ]
        ]
        .drop_duplicates()
    )

    expected_psu_n = len(
        psu_structure
    )

    expected_strata_n = (
        psu_structure[
            "stratum"
        ]
        .nunique()
    )

    expected_df = (
        expected_psu_n
        - expected_strata_n
    )


    row = (
        survey_estimates[
            survey_estimates["cntry"]
            == cntry
        ]
        .iloc[0]
    )


    valid = (
        row["survey_psu_n"]
        == expected_psu_n
        and
        row["survey_strata_n"]
        == expected_strata_n
        and
        row["survey_design_df"]
        == expected_df
    )


    print(
        f"{cntry:<5} "
        f"{'PASSED' if valid else 'FAILED'} "
        f"| PSU={expected_psu_n}, "
        f"strata={expected_strata_n}, "
        f"df={expected_df}"
    )


    if not valid:
        design_valid = False
        validation_passed = False


# ---------------------------------------------------------------------
# 4. Independently recalculate Taylor SE
# ---------------------------------------------------------------------

print()
print("INDEPENDENT TAYLOR SE RECALCULATION")
print("-" * 120)


def recalculate_taylor_se(
    data,
    value_column,
    weight_column="anweight",
):

    valid = (
        data[value_column].notna()
        & data[weight_column].notna()
        & (data[weight_column] > 0)
    )


    y = (
        data.loc[
            valid,
            value_column
        ]
        .astype(float)
    )

    w = (
        data.loc[
            valid,
            weight_column
        ]
        .astype(float)
    )


    weighted_mean = (
        (w * y).sum()
        / w.sum()
    )


    # Create linearized contribution
    work = data[
        [
            "stratum",
            "psu",
        ]
    ].copy()

    work["linearized"] = 0.0


    work.loc[
        valid,
        "linearized"
    ] = (
        w
        * (
            y
            - weighted_mean
        )
    )


    # Sum linearized contributions within PSU
    psu_totals = (
        work
        .groupby(
            [
                "stratum",
                "psu",
            ],
            sort=False,
        )[
            "linearized"
        ]
        .sum()
        .reset_index()
    )


    variance_total = 0.0


    for _, stratum_data in psu_totals.groupby(
        "stratum",
        sort=False,
    ):

        m_h = len(
            stratum_data
        )

        values = (
            stratum_data[
                "linearized"
            ]
            .astype(float)
        )

        stratum_mean = (
            values.mean()
        )

        variance_total += (
            m_h
            / (m_h - 1)
            * (
                (
                    values
                    - stratum_mean
                )
                ** 2
            )
            .sum()
        )


    variance = (
        variance_total
        / (
            w.sum()
            ** 2
        )
    )


    return np.sqrt(
        variance
    )


for metric, source_variable in metrics.items():

    max_difference = 0.0
    metric_valid = True


    for cntry, group in df.groupby("cntry"):

        recalculated_se = (
            recalculate_taylor_se(
                group,
                source_variable,
            )
        )

        stored_se = float(
            survey_estimates.loc[
                survey_estimates["cntry"]
                == cntry,
                f"{metric}_se",
            ]
            .iloc[0]
        )

        difference = abs(
            recalculated_se
            - stored_se
        )

        max_difference = max(
            max_difference,
            difference,
        )

        if not np.isclose(
            recalculated_se,
            stored_se,
            rtol=1e-10,
            atol=1e-12,
        ):
            metric_valid = False


    print(
        f"{metric:<30} "
        f"{'PASSED' if metric_valid else 'FAILED'} "
        f"| max SE difference = "
        f"{max_difference:.12f}"
    )


    if not metric_valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate confidence interval formula
# ---------------------------------------------------------------------

print()
print("CONFIDENCE INTERVAL FORMULA")
print("-" * 120)

ci_formula_valid = True


for _, row in survey_estimates.iterrows():

    design_df = int(
        row["survey_design_df"]
    )

    critical_value = t.ppf(
        0.975,
        df=design_df,
    )


    for metric in metrics:

        estimate = float(
            row[metric]
        )

        se = float(
            row[
                f"{metric}_se"
            ]
        )

        expected_lower = (
            estimate
            - critical_value * se
        )

        expected_upper = (
            estimate
            + critical_value * se
        )


        stored_lower = float(
            row[
                f"{metric}_ci_lower"
            ]
        )

        stored_upper = float(
            row[
                f"{metric}_ci_upper"
            ]
        )


        if not (
            np.isclose(
                expected_lower,
                stored_lower,
                rtol=1e-10,
                atol=1e-12,
            )
            and
            np.isclose(
                expected_upper,
                stored_upper,
                rtol=1e-10,
                atol=1e-12,
            )
        ):

            ci_formula_valid = False


print(
    f"95% t-based confidence intervals: "
    f"{'PASSED' if ci_formula_valid else 'FAILED'}"
)

if not ci_formula_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 6. Validate CI symmetry
# ---------------------------------------------------------------------

print()
print("CONFIDENCE INTERVAL SYMMETRY")
print("-" * 120)

symmetry_valid = True


for metric in metrics:

    lower_distance = (
        survey_estimates[metric]
        - survey_estimates[
            f"{metric}_ci_lower"
        ]
    )

    upper_distance = (
        survey_estimates[
            f"{metric}_ci_upper"
        ]
        - survey_estimates[metric]
    )


    valid = np.allclose(
        lower_distance,
        upper_distance,
        rtol=1e-10,
        atol=1e-12,
    )


    print(
        f"{metric:<30} "
        f"{'PASSED' if valid else 'FAILED'}"
    )


    if not valid:
        symmetry_valid = False
        validation_passed = False


# ---------------------------------------------------------------------
# 7. Validate standard errors
# ---------------------------------------------------------------------

print()
print("STANDARD ERROR VALIDATION")
print("-" * 120)

all_se_valid = True


for metric in metrics:

    se = survey_estimates[
        f"{metric}_se"
    ]

    valid = (
        se.notna().all()
        and
        np.isfinite(se).all()
        and
        (se >= 0).all()
    )


    print(
        f"{metric:<30} "
        f"{'PASSED' if valid else 'FAILED'}"
    )


    if not valid:
        all_se_valid = False
        validation_passed = False


# ---------------------------------------------------------------------
# 8. Check CI boundaries
# ---------------------------------------------------------------------

print()
print("THEORETICAL CI BOUNDARIES")
print("-" * 120)

bounded_metrics = {
    "mean_left_right": (0, 10),
    "left_share": (0, 1),
    "centre_share": (0, 1),
    "right_share": (0, 1),

    "mean_social_trust": (0, 10),
    "mean_ppltrst": (0, 10),
    "mean_pplfair": (0, 10),
    "mean_pplhlp": (0, 10),
}


boundary_violations = 0


for metric, (
    minimum,
    maximum,
) in bounded_metrics.items():

    violations = (
        (
            survey_estimates[
                f"{metric}_ci_lower"
            ]
            < minimum
        )
        |
        (
            survey_estimates[
                f"{metric}_ci_upper"
            ]
            > maximum
        )
    ).sum()


    boundary_violations += int(
        violations
    )


    print(
        f"{metric:<30} "
        f"violations = {violations}"
    )


# ---------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("VALIDATION SUMMARY")
print("=" * 120)


if validation_passed:

    print(
        "Country survey estimate validation PASSED."
    )

else:

    print(
        "Country survey estimate validation "
        "requires review."
    )


print()

print(
    f"Total theoretical CI boundary violations: "
    f"{boundary_violations}"
)

print()

print(
    "No respondent or processed data were modified."
)