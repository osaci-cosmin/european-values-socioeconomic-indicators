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

point_estimate_file = (
    project_root
    / "data"
    / "processed"
    / "ESS11_country_weighted_estimates.parquet"
)

# Define output files
output_parquet = (
    project_root
    / "data"
    / "processed"
    / "ESS11_country_survey_estimates.parquet"
)

output_csv = (
    project_root
    / "data"
    / "processed"
    / "ESS11_country_survey_estimates.csv"
)


# Check input files
if not respondent_file.exists():
    raise FileNotFoundError(
        f"Respondent dataset not found: {respondent_file}"
    )

if not point_estimate_file.exists():
    raise FileNotFoundError(
        f"Point-estimate dataset not found: {point_estimate_file}"
    )

# Protect existing outputs
if output_parquet.exists() or output_csv.exists():
    raise FileExistsError(
        "Survey-estimate output already exists."
    )


# Load data
df = pd.read_parquet(respondent_file)

country_estimates = pd.read_parquet(
    point_estimate_file
)


# ---------------------------------------------------------------------
# Taylor-linearized survey mean
# ---------------------------------------------------------------------

def survey_mean_taylor(
    data,
    value_column,
    weight_column="anweight",
    confidence_level=0.95,
):

    # Define valid observations for the analytical variable
    valid = (
        data[value_column].notna()
        & data[weight_column].notna()
        & (data[weight_column] > 0)
    )

    valid_n = int(
        valid.sum()
    )

    if valid_n == 0:
        return {
            "estimate": np.nan,
            "standard_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "valid_n": 0,
            "psu_n": np.nan,
            "strata_n": np.nan,
            "design_df": np.nan,
        }

    # Extract valid values and weights
    y = (
        data.loc[
            valid,
            value_column
        ]
        .astype(float)
        .to_numpy()
    )

    w = (
        data.loc[
            valid,
            weight_column
        ]
        .astype(float)
        .to_numpy()
    )

    weight_sum = w.sum()

    # Weighted point estimate
    estimate = (
        np.sum(
            w * y
        )
        / weight_sum
    )


    # -------------------------------------------------------------
    # Linearized respondent contributions
    #
    # Missing outcome observations remain in the survey design
    # with zero contribution to the linearized variable.
    # -------------------------------------------------------------

    linearized = np.zeros(
        len(data),
        dtype=float,
    )

    linearized[
        valid.to_numpy()
    ] = (
        w
        * (
            y
            - estimate
        )
    )


    design_data = data[
        [
            "stratum",
            "psu",
        ]
    ].copy()

    design_data[
        "linearized"
    ] = linearized


    # -------------------------------------------------------------
    # Aggregate linearized values to PSU level
    #
    # PSU identity is defined inside stratum.
    # Country is already fixed by the calling loop.
    # -------------------------------------------------------------

    psu_totals = (
        design_data
        .groupby(
            [
                "stratum",
                "psu",
            ],
            dropna=False,
            sort=False,
        )[
            "linearized"
        ]
        .sum()
        .reset_index()
    )


    strata_n = (
        psu_totals[
            "stratum"
        ]
        .nunique()
    )

    psu_n = len(
        psu_totals
    )


    # -------------------------------------------------------------
    # Taylor-linearized variance
    #
    # With-replacement approximation.
    # -------------------------------------------------------------

    variance_numerator = 0.0


    for _, stratum_group in psu_totals.groupby(
        "stratum",
        dropna=False,
        sort=False,
    ):

        m_h = len(
            stratum_group
        )

        if m_h < 2:
            raise ValueError(
                "Singleton stratum detected during "
                "variance estimation."
            )

        psu_values = (
            stratum_group[
                "linearized"
            ]
            .to_numpy(
                dtype=float
            )
        )

        stratum_mean = (
            psu_values.mean()
        )

        stratum_component = (
            m_h
            / (m_h - 1)
            * np.sum(
                (
                    psu_values
                    - stratum_mean
                )
                ** 2
            )
        )

        variance_numerator += (
            stratum_component
        )


    variance = (
        variance_numerator
        / (
            weight_sum ** 2
        )
    )

    standard_error = np.sqrt(
        max(
            variance,
            0.0,
        )
    )


    # -------------------------------------------------------------
    # Survey design degrees of freedom
    # -------------------------------------------------------------

    design_df = (
        psu_n
        - strata_n
    )

    if design_df <= 0:
        raise ValueError(
            "Survey design degrees of freedom "
            "must be positive."
        )


    # -------------------------------------------------------------
    # Confidence interval
    # -------------------------------------------------------------

    alpha = (
        1.0
        - confidence_level
    )

    critical_value = t.ppf(
        1.0
        - alpha / 2.0,
        df=design_df,
    )

    margin_error = (
        critical_value
        * standard_error
    )

    ci_lower = (
        estimate
        - margin_error
    )

    ci_upper = (
        estimate
        + margin_error
    )


    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "valid_n": valid_n,
        "psu_n": psu_n,
        "strata_n": strata_n,
        "design_df": design_df,
    }


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


# ---------------------------------------------------------------------
# Prepare output dataset
# ---------------------------------------------------------------------

survey_estimates = (
    country_estimates
    .copy()
)


# Store one survey-design df per country
survey_estimates[
    "survey_psu_n"
] = pd.NA

survey_estimates[
    "survey_strata_n"
] = pd.NA

survey_estimates[
    "survey_design_df"
] = pd.NA

survey_estimates[
    "survey_ci_level"
] = 0.95


# Track point-estimate agreement
maximum_point_difference = 0.0


# ---------------------------------------------------------------------
# Calculate country-level survey estimates
# ---------------------------------------------------------------------

for row_index, country_row in survey_estimates.iterrows():

    cntry = (
        country_row[
            "cntry"
        ]
    )

    country_data = (
        df[
            df["cntry"] == cntry
        ]
        .copy()
    )


    for (
        output_name,
        source_variable,
    ) in metrics.items():

        result = survey_mean_taylor(
            country_data,
            source_variable,
            weight_column="anweight",
            confidence_level=0.95,
        )


        # Compare with previously validated point estimate
        stored_estimate = float(
            country_row[
                output_name
            ]
        )

        point_difference = abs(
            result[
                "estimate"
            ]
            - stored_estimate
        )

        maximum_point_difference = max(
            maximum_point_difference,
            point_difference,
        )


        # Store uncertainty estimates
        survey_estimates.loc[
            row_index,
            f"{output_name}_se",
        ] = result[
            "standard_error"
        ]

        survey_estimates.loc[
            row_index,
            f"{output_name}_ci_lower",
        ] = result[
            "ci_lower"
        ]

        survey_estimates.loc[
            row_index,
            f"{output_name}_ci_upper",
        ] = result[
            "ci_upper"
        ]


        # Design structure is identical across metrics
        survey_estimates.loc[
            row_index,
            "survey_psu_n",
        ] = result[
            "psu_n"
        ]

        survey_estimates.loc[
            row_index,
            "survey_strata_n",
        ] = result[
            "strata_n"
        ]

        survey_estimates.loc[
            row_index,
            "survey_design_df",
        ] = result[
            "design_df"
        ]


# Convert design columns to integer
for variable in [
    "survey_psu_n",
    "survey_strata_n",
    "survey_design_df",
]:

    survey_estimates[
        variable
    ] = (
        survey_estimates[
            variable
        ]
        .astype(int)
    )


# ---------------------------------------------------------------------
# Basic CI validation
# ---------------------------------------------------------------------

ci_order_valid = True

se_nonnegative = True


for output_name in metrics:

    se_column = (
        f"{output_name}_se"
    )

    lower_column = (
        f"{output_name}_ci_lower"
    )

    upper_column = (
        f"{output_name}_ci_upper"
    )

    if not (
        survey_estimates[
            se_column
        ] >= 0
    ).all():

        se_nonnegative = False

    if not (
        (
            survey_estimates[
                lower_column
            ]
            <= survey_estimates[
                output_name
            ]
        )
        &
        (
            survey_estimates[
                output_name
            ]
            <= survey_estimates[
                upper_column
            ]
        )
    ).all():

        ci_order_valid = False


# ---------------------------------------------------------------------
# Check constrained-scale CI boundaries
# ---------------------------------------------------------------------

bounded_metrics = {
    "mean_left_right": (
        0,
        10,
    ),

    "left_share": (
        0,
        1,
    ),

    "centre_share": (
        0,
        1,
    ),

    "right_share": (
        0,
        1,
    ),

    "mean_social_trust": (
        0,
        10,
    ),

    "mean_ppltrst": (
        0,
        10,
    ),

    "mean_pplfair": (
        0,
        10,
    ),

    "mean_pplhlp": (
        0,
        10,
    ),
}


outside_boundary_count = 0


for (
    metric,
    bounds,
) in bounded_metrics.items():

    minimum, maximum = bounds

    outside = (
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
    )

    outside_boundary_count += int(
        outside.sum()
    )


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

survey_estimates.to_parquet(
    output_parquet,
    index=False,
)

survey_estimates.to_csv(
    output_csv,
    index=False,
)


# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

print("=" * 120)
print("ESS ROUND 11 COUNTRY SURVEY ESTIMATES")
print("=" * 120)

print()
print(f"Countries: {len(survey_estimates)}")

print()
print("POINT ESTIMATE CONSISTENCY")
print("-" * 120)

print(
    f"Maximum difference from Step 22: "
    f"{maximum_point_difference:.12f}"
)

point_estimates_valid = np.isclose(
    maximum_point_difference,
    0.0,
    atol=1e-10,
)

print(
    f"Point estimates reproduced: "
    f"{'PASSED' if point_estimates_valid else 'FAILED'}"
)


print()
print("UNCERTAINTY VALIDATION")
print("-" * 120)

print(
    f"All standard errors non-negative: "
    f"{'PASSED' if se_nonnegative else 'FAILED'}"
)

print(
    f"Estimate inside every confidence interval: "
    f"{'PASSED' if ci_order_valid else 'FAILED'}"
)

print(
    f"CIs outside theoretical scale boundaries: "
    f"{outside_boundary_count}"
)


print()
print("SURVEY DESIGN DEGREES OF FREEDOM")
print("-" * 120)

print(
    survey_estimates[
        [
            "cntry",
            "country_name",
            "survey_psu_n",
            "survey_strata_n",
            "survey_design_df",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("SELECTED 95% CONFIDENCE INTERVALS")
print("-" * 120)

display_columns = [
    "cntry",
    "country_name",

    "mean_left_right",
    "mean_left_right_se",
    "mean_left_right_ci_lower",
    "mean_left_right_ci_upper",

    "mean_social_trust",
    "mean_social_trust_se",
    "mean_social_trust_ci_lower",
    "mean_social_trust_ci_upper",
]


print(
    survey_estimates[
        display_columns
    ]
    .to_string(
        index=False,
        formatters={
            "mean_left_right": "{:.4f}".format,
            "mean_left_right_se": "{:.4f}".format,
            "mean_left_right_ci_lower": "{:.4f}".format,
            "mean_left_right_ci_upper": "{:.4f}".format,

            "mean_social_trust": "{:.4f}".format,
            "mean_social_trust_se": "{:.4f}".format,
            "mean_social_trust_ci_lower": "{:.4f}".format,
            "mean_social_trust_ci_upper": "{:.4f}".format,
        }
    )
)


print()
print("=" * 120)
print("COUNTRY SURVEY ESTIMATES CREATED")
print("=" * 120)

print(f"Parquet: {output_parquet}")
print(f"CSV:     {output_csv}")

print()

print(
    "Variance method: Taylor linearization."
)

print(
    "Survey design: anweight + stratum + PSU."
)

print(
    "PSU identity is defined within country and stratum."
)

print(
    "No finite population correction was applied."
)

print(
    "Missing analytical outcomes were retained in the "
    "survey design with zero linearized contribution."
)