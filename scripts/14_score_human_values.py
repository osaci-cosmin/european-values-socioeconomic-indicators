from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_scope.parquet"
)

mapping_file = (
    project_root
    / "config"
    / "human_values_mapping.csv"
)

# Define output file
output_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values.parquet"
)


# Check that input files exist
if not input_file.exists():
    raise FileNotFoundError(
        f"V1 dataset not found: {input_file}"
    )

if not mapping_file.exists():
    raise FileNotFoundError(
        f"Human Values mapping not found: {mapping_file}"
    )

# Protect an existing output file
if output_file.exists():
    raise FileExistsError(
        f"Human Values output already exists: {output_file}"
    )


# Load data and Human Values mapping
df = pd.read_parquet(input_file)
mapping = pd.read_csv(mapping_file)


# Create a safe column-name format
def make_slug(value):
    return (
        value
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# Get the 21 Human Values variables from the validated mapping
human_values = mapping["variable"].tolist()


# Count valid Human Values items per respondent
df["hv_answered_n"] = (
    df[human_values]
    .notna()
    .sum(axis=1)
)

# Define complete Human Values cases
df["hv_complete_flag"] = (
    df["hv_answered_n"] == 21
)

# Use complete cases for primary V1 Human Values scoring
df["hv_scoring_eligible"] = df["hv_complete_flag"]


# ---------------------------------------------------------------------
# Reverse-score the 21 Human Values items
# Original ESS: 1 = Very much like me, 6 = Not like me at all
# Reversed:     6 = Very much like me, 1 = Not like me at all
# ---------------------------------------------------------------------

reversed_variables = []

for variable in human_values:

    reversed_variable = f"{variable}_r"

    df[reversed_variable] = (
        7 - df[variable].astype("Float64")
    )

    reversed_variables.append(
        reversed_variable
    )


# ---------------------------------------------------------------------
# Calculate respondent Mean Rating (MRAT)
# ---------------------------------------------------------------------

df["hv_mrat"] = (
    df[reversed_variables]
    .mean(axis=1)
)

# Keep MRAT only for complete Human Values cases
df.loc[
    ~df["hv_scoring_eligible"],
    "hv_mrat"
] = pd.NA


# ---------------------------------------------------------------------
# Calculate the 10 basic Human Values
# ---------------------------------------------------------------------

for basic_value, group in mapping.groupby("basic_value"):

    score_name = make_slug(basic_value)

    item_variables = [
        f"{variable}_r"
        for variable in group["variable"]
    ]

    raw_column = (
        f"hv_{score_name}_raw"
    )

    centered_column = (
        f"hv_{score_name}_centered"
    )

    # Calculate the raw mean score
    df[raw_column] = (
        df[item_variables]
        .mean(axis=1)
    )

    # Remove scores for incomplete Human Values cases
    df.loc[
        ~df["hv_scoring_eligible"],
        raw_column
    ] = pd.NA

    # Center the value score using respondent MRAT
    df[centered_column] = (
        df[raw_column]
        - df["hv_mrat"]
    )


# ---------------------------------------------------------------------
# Calculate the four higher-order Human Values
# ---------------------------------------------------------------------

for higher_value, group in mapping.groupby(
    "higher_order_value"
):

    score_name = make_slug(higher_value)

    item_variables = [
        f"{variable}_r"
        for variable in group["variable"]
    ]

    raw_column = (
        f"hv_{score_name}_raw"
    )

    centered_column = (
        f"hv_{score_name}_centered"
    )

    # Calculate the raw higher-order score
    df[raw_column] = (
        df[item_variables]
        .mean(axis=1)
    )

    # Remove scores for incomplete Human Values cases
    df.loc[
        ~df["hv_scoring_eligible"],
        raw_column
    ] = pd.NA

    # Center the higher-order score using respondent MRAT
    df[centered_column] = (
        df[raw_column]
        - df["hv_mrat"]
    )


# ---------------------------------------------------------------------
# Add ESS11 Human Values documentation flags
# ---------------------------------------------------------------------

df["hv_country_note_flag"] = (
    df["cntry"].isin(["EE", "GB"])
)


# Save the scored respondent-level dataset
df.to_parquet(
    output_file,
    index=False
)


print("=" * 100)
print("ESS ROUND 11 HUMAN VALUES SCORING")
print("=" * 100)

print()
print(f"Respondents:              {len(df):,}")

print(
    f"Complete 21/21 cases:     "
    f"{df['hv_complete_flag'].sum():,}"
)

print(
    f"Scoring eligible:         "
    f"{df['hv_scoring_eligible'].sum():,}"
)

print(
    f"Not scoring eligible:     "
    f"{(~df['hv_scoring_eligible']).sum():,}"
)


print()
print("MRAT")
print("-" * 100)

print(
    df.loc[
        df["hv_scoring_eligible"],
        "hv_mrat"
    ]
    .describe()
    .to_string()
)


print()
print("HIGHER-ORDER HUMAN VALUES")
print("-" * 100)

higher_order_values = [
    "openness_to_change",
    "conservation",
    "self_transcendence",
    "self_enhancement",
]

for value in higher_order_values:

    raw_column = f"hv_{value}_raw"
    centered_column = f"hv_{value}_centered"

    print()
    print(value.upper())

    print(
        f"Raw mean:      "
        f"{df[raw_column].mean():.4f}"
    )

    print(
        f"Centered mean: "
        f"{df[centered_column].mean():.4f}"
    )

    print(
        f"Valid scores:  "
        f"{df[centered_column].notna().sum():,}"
    )


print()
print("ESS11 HUMAN VALUES DOCUMENTATION FLAGS")
print("-" * 100)

flagged_summary = (
    df.loc[
        df["hv_country_note_flag"],
        ["cntry", "country_name"]
    ]
    .value_counts()
    .reset_index(name="respondents")
)

print(flagged_summary.to_string(index=False))


print()
print("=" * 100)
print("HUMAN VALUES DATASET CREATED")
print("=" * 100)

print(f"Output: {output_file}")

print()
print(
    "Primary Human Values scores use complete 21/21 cases."
)

print(
    "Incomplete respondents were retained in the dataset."
)