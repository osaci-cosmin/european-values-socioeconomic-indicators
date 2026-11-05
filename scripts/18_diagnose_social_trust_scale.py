from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define the respondent-level dataset
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values_ideology.parquet"
)

# Check that the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"V1 dataset not found: {input_file}"
    )


# Load the dataset
df = pd.read_parquet(input_file)


# Define the three ESS Social Trust items
trust_variables = [
    "ppltrst",
    "pplfair",
    "pplhlp",
]


# Calculate Cronbach's alpha
def cronbach_alpha(data):

    complete = data.dropna()

    if len(complete) < 2:
        return np.nan

    item_variances = complete.var(
        axis=0,
        ddof=1
    )

    total_score = complete.sum(axis=1)

    total_variance = total_score.var(
        ddof=1
    )

    n_items = complete.shape[1]

    if total_variance == 0:
        return np.nan

    alpha = (
        n_items
        / (n_items - 1)
        * (
            1
            - item_variances.sum()
            / total_variance
        )
    )

    return alpha


print("=" * 100)
print("ESS ROUND 11 SOCIAL TRUST SCALE DIAGNOSTIC")
print("=" * 100)

print()
print(f"Respondents in V1: {len(df):,}")


# ---------------------------------------------------------------------
# 1. Item completeness
# ---------------------------------------------------------------------

print()
print("SOCIAL TRUST ITEM COMPLETENESS")
print("-" * 100)

for variable in trust_variables:

    valid_n = df[variable].notna().sum()
    missing_n = df[variable].isna().sum()
    missing_rate = missing_n / len(df) * 100

    print(
        f"{variable:<10} "
        f"Valid: {valid_n:>7,} | "
        f"Missing: {missing_n:>5,} "
        f"({missing_rate:>5.2f}%)"
    )


# ---------------------------------------------------------------------
# 2. Complete-case distribution
# ---------------------------------------------------------------------

trust_answered_n = (
    df[trust_variables]
    .notna()
    .sum(axis=1)
)

complete_cases = (
    trust_answered_n == 3
)

print()
print("ANSWERED SOCIAL TRUST ITEMS")
print("-" * 100)

print(
    trust_answered_n
    .value_counts()
    .sort_index(ascending=False)
    .to_string()
)

complete_n = complete_cases.sum()
complete_rate = complete_n / len(df) * 100

print()
print(
    f"Complete 3/3 cases: "
    f"{complete_n:,} ({complete_rate:.2f}%)"
)


# ---------------------------------------------------------------------
# 3. Descriptive statistics
# ---------------------------------------------------------------------

print()
print("ITEM DESCRIPTIVE STATISTICS")
print("-" * 100)

print(
    df[trust_variables]
    .describe()
    .T
    .to_string()
)


# ---------------------------------------------------------------------
# 4. Pairwise correlations
# ---------------------------------------------------------------------

print()
print("PAIRWISE PEARSON CORRELATIONS")
print("-" * 100)

correlations = (
    df.loc[
        complete_cases,
        trust_variables
    ]
    .corr(method="pearson")
)

print(correlations.to_string())


# ---------------------------------------------------------------------
# 5. Overall Cronbach's alpha
# ---------------------------------------------------------------------

print()
print("OVERALL SCALE RELIABILITY")
print("-" * 100)

complete_trust = df.loc[
    complete_cases,
    trust_variables
]

overall_alpha = cronbach_alpha(
    complete_trust
)

print(
    f"Cronbach's alpha: {overall_alpha:.4f}"
)

print(
    f"Complete cases:   {len(complete_trust):,}"
)


# ---------------------------------------------------------------------
# 6. Alpha if item deleted
# ---------------------------------------------------------------------

print()
print("ALPHA IF ITEM DELETED")
print("-" * 100)

for variable in trust_variables:

    remaining_variables = [
        item
        for item in trust_variables
        if item != variable
    ]

    alpha_deleted = cronbach_alpha(
        complete_trust[
            remaining_variables
        ]
    )

    print(
        f"Remove {variable:<8} "
        f"alpha = {alpha_deleted:.4f}"
    )


# ---------------------------------------------------------------------
# 7. Country-level reliability
# ---------------------------------------------------------------------

print()
print("COUNTRY-LEVEL RELIABILITY")
print("-" * 100)

country_results = []

for (
    cntry,
    country_name
), group in df.groupby(
    ["cntry", "country_name"]
):

    complete_group = (
        group[trust_variables]
        .dropna()
    )

    alpha = cronbach_alpha(
        complete_group
    )

    country_results.append(
        {
            "cntry": cntry,
            "country_name": country_name,
            "complete_n": len(complete_group),
            "cronbach_alpha": alpha,
        }
    )


country_reliability = pd.DataFrame(
    country_results
).sort_values(
    "cronbach_alpha"
)


print(
    country_reliability.to_string(
        index=False,
        formatters={
            "cronbach_alpha": "{:.4f}".format
        }
    )
)


# ---------------------------------------------------------------------
# 8. Reliability range
# ---------------------------------------------------------------------

print()
print("COUNTRY RELIABILITY RANGE")
print("-" * 100)

print(
    f"Minimum alpha: "
    f"{country_reliability['cronbach_alpha'].min():.4f}"
)

print(
    f"Median alpha:  "
    f"{country_reliability['cronbach_alpha'].median():.4f}"
)

print(
    f"Maximum alpha: "
    f"{country_reliability['cronbach_alpha'].max():.4f}"
)


print()
print("=" * 100)
print("DIAGNOSTIC COMPLETED")
print("=" * 100)

print(
    "No Social Trust composite score was created."
)

print(
    "No respondent data were modified."
)

print(
    "Reliability estimates are unweighted diagnostics."
)