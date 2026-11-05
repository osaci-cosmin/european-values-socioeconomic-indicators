from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input file
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values.parquet"
)

# Define output file
output_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values_ideology.parquet"
)


# Check that the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Human Values dataset not found: {input_file}"
    )

# Protect an existing output file
if output_file.exists():
    raise FileExistsError(
        f"Ideology output already exists: {output_file}"
    )


# Load the respondent-level dataset
df = pd.read_parquet(input_file)


# Define valid ideological self-placement responses
df["ideology_valid_flag"] = (
    df["lrscale"].notna()
)


# Create the primary analytical classification
df["ideology_group"] = pd.NA

df.loc[
    df["lrscale"].between(0, 3),
    "ideology_group"
] = "Left"

df.loc[
    df["lrscale"].between(4, 6),
    "ideology_group"
] = "Centre"

df.loc[
    df["lrscale"].between(7, 10),
    "ideology_group"
] = "Right"


# Convert the classification to an ordered categorical variable
df["ideology_group"] = pd.Categorical(
    df["ideology_group"],
    categories=[
        "Left",
        "Centre",
        "Right",
    ],
    ordered=True,
)


# Create binary indicators for aggregation and Power BI
df["ideology_left_flag"] = (
    df["ideology_group"] == "Left"
)

df["ideology_centre_flag"] = (
    df["ideology_group"] == "Centre"
)

df["ideology_right_flag"] = (
    df["ideology_group"] == "Right"
)


# Set binary indicators to missing when lrscale is missing
for variable in [
    "ideology_left_flag",
    "ideology_centre_flag",
    "ideology_right_flag",
]:

    df[variable] = (
        df[variable]
        .astype("boolean")
    )

    df.loc[
        ~df["ideology_valid_flag"],
        variable
    ] = pd.NA


# Save the respondent-level dataset
df.to_parquet(
    output_file,
    index=False
)


print("=" * 100)
print("ESS ROUND 11 IDEOLOGICAL SELF-PLACEMENT")
print("=" * 100)

print()
print(f"Respondents:            {len(df):,}")

valid_n = df["lrscale"].notna().sum()
missing_n = df["lrscale"].isna().sum()

valid_rate = valid_n / len(df) * 100
missing_rate = missing_n / len(df) * 100

print(
    f"Valid lrscale:          "
    f"{valid_n:,} ({valid_rate:.2f}%)"
)

print(
    f"Missing lrscale:        "
    f"{missing_n:,} ({missing_rate:.2f}%)"
)


print()
print("CONTINUOUS LEFT-RIGHT SCALE")
print("-" * 100)

print(
    df["lrscale"]
    .describe()
    .to_string()
)


print()
print("PRIMARY IDEOLOGY CLASSIFICATION")
print("-" * 100)

group_counts = (
    df["ideology_group"]
    .value_counts(
        sort=False,
        dropna=False
    )
)

print(group_counts.to_string())


print()
print("UNWEIGHTED DESCRIPTIVE SHARES")
print("-" * 100)

for group in [
    "Left",
    "Centre",
    "Right",
]:

    count = (
        df["ideology_group"] == group
    ).sum()

    share = (
        count
        / valid_n
        * 100
    )

    print(
        f"{group:<10} "
        f"{count:>7,} "
        f"({share:>6.2f}%)"
    )


print()
print("=" * 100)
print("IDEOLOGY VARIABLES CREATED")
print("=" * 100)

print(f"Output: {output_file}")

print()
print(
    "The continuous ESS Left-Right scale was preserved."
)

print(
    "Left/Centre/Right is a project-defined analytical classification."
)

print(
    "Displayed shares are unweighted diagnostics, not final population estimates."
)