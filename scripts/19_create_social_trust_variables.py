from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input file
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values_ideology.parquet"
)

# Define output file
output_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)


# Check that the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"V1 dataset not found: {input_file}"
    )

# Protect an existing output file
if output_file.exists():
    raise FileExistsError(
        f"Analytical output already exists: {output_file}"
    )


# Load the respondent-level dataset
df = pd.read_parquet(input_file)


# Define the three ESS Social Trust items
trust_variables = [
    "ppltrst",
    "pplfair",
    "pplhlp",
]


# Count valid Social Trust items per respondent
df["trust_answered_n"] = (
    df[trust_variables]
    .notna()
    .sum(axis=1)
)


# Define complete Social Trust cases
df["trust_complete_flag"] = (
    df["trust_answered_n"] == 3
)


# Use complete cases for the primary V1 composite score
df["trust_scoring_eligible"] = (
    df["trust_complete_flag"]
)


# Calculate the mean Social Trust score
df["social_trust_score"] = (
    df[trust_variables]
    .mean(axis=1)
)


# Remove composite scores for incomplete cases
df.loc[
    ~df["trust_scoring_eligible"],
    "social_trust_score"
] = pd.NA


# Save the analytical respondent-level dataset
df.to_parquet(
    output_file,
    index=False
)


print("=" * 100)
print("ESS ROUND 11 SOCIAL TRUST SCORING")
print("=" * 100)

print()
print(f"Respondents:              {len(df):,}")

print(
    f"Complete 3/3 cases:       "
    f"{df['trust_complete_flag'].sum():,}"
)

print(
    f"Scoring eligible:         "
    f"{df['trust_scoring_eligible'].sum():,}"
)

print(
    f"Not scoring eligible:     "
    f"{(~df['trust_scoring_eligible']).sum():,}"
)


print()
print("SOCIAL TRUST SCORE")
print("-" * 100)

print(
    df["social_trust_score"]
    .describe()
    .to_string()
)


print()
print("COMPONENT ITEM MEANS")
print("-" * 100)

for variable in trust_variables:

    print(
        f"{variable:<10} "
        f"{df[variable].mean():.4f}"
    )


print()
print("COUNTRY-LEVEL UNWEIGHTED DIAGNOSTIC")
print("-" * 100)

country_summary = (
    df
    .groupby(
        ["cntry", "country_name"],
        as_index=False
    )
    .agg(
        respondents=("idno", "size"),
        valid_trust_scores=(
            "social_trust_score",
            "count"
        ),
        mean_social_trust=(
            "social_trust_score",
            "mean"
        ),
    )
    .sort_values(
        "mean_social_trust",
        ascending=False
    )
)


print(
    country_summary.to_string(
        index=False,
        formatters={
            "mean_social_trust": "{:.4f}".format
        }
    )
)


print()
print("=" * 100)
print("SOCIAL TRUST VARIABLES CREATED")
print("=" * 100)

print(f"Output: {output_file}")

print()
print(
    "Primary Social Trust scores use complete 3/3 cases."
)

print(
    "Original ppltrst, pplfair, and pplhlp variables were preserved."
)

print(
    "Displayed country means are unweighted diagnostics."
)