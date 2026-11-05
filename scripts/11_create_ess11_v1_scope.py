from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
interim_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_core_interim.parquet"
)

country_config_file = (
    project_root
    / "config"
    / "country_scope_v1.csv"
)

# Define output file
output_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_scope.parquet"
)


# Check that input files exist
if not interim_file.exists():
    raise FileNotFoundError(
        f"Interim ESS file not found: {interim_file}"
    )

if not country_config_file.exists():
    raise FileNotFoundError(
        f"Country configuration file not found: {country_config_file}"
    )


# Protect an existing output file
if output_file.exists():
    raise FileExistsError(
        f"V1 scope file already exists and will not be overwritten: {output_file}"
    )


# Load data
df = pd.read_parquet(interim_file)

country_config = pd.read_csv(
    country_config_file,
    dtype={"cntry": "string"}
)


# Merge country metadata with respondent-level data
df = df.merge(
    country_config,
    on="cntry",
    how="left",
    validate="many_to_one"
)


# Check that every respondent received country metadata
if df["country_name"].isna().any():
    raise ValueError(
        "Some ESS respondents could not be matched to the country configuration."
    )


# Keep only countries included in the V1 European scope
df_v1 = df[
    df["include_v1"] == 1
].copy()


# Create the four-region analytical variable
df_v1["region_4"] = df_v1["region_m49"].where(
    df_v1["include_regional"] == 1,
    pd.NA
)


# Remove the scope filter after it has been applied
df_v1 = df_v1.drop(
    columns=["include_v1"]
)


# Validate expected V1 scope
expected_rows = 49210
expected_countries = 29
expected_regional_countries = 28

actual_rows = len(df_v1)
actual_countries = df_v1["cntry"].nunique()
actual_regional_countries = (
    df_v1.loc[
        df_v1["include_regional"] == 1,
        "cntry"
    ]
    .nunique()
)


if actual_rows != expected_rows:
    raise ValueError(
        f"Unexpected V1 row count: {actual_rows:,}"
    )

if actual_countries != expected_countries:
    raise ValueError(
        f"Unexpected V1 country count: {actual_countries}"
    )

if actual_regional_countries != expected_regional_countries:
    raise ValueError(
        "Unexpected regional country count: "
        f"{actual_regional_countries}"
    )


# Save the V1 respondent-level scope dataset
df_v1.to_parquet(
    output_file,
    index=False
)


print("=" * 100)
print("ESS ROUND 11 V1 EUROPEAN SCOPE")
print("=" * 100)

print()
print(f"Respondents:             {len(df_v1):,}")
print(f"Countries:               {df_v1['cntry'].nunique()}")
print(
    f"Regional countries:      "
    f"{actual_regional_countries}"
)

print()
print("REGION COVERAGE")
print("-" * 100)

region_summary = (
    df_v1
    .groupby(
        "region_4",
        dropna=False
    )
    .agg(
        countries=("cntry", "nunique"),
        respondents=("idno", "size")
    )
    .reset_index()
)

print(region_summary.to_string(index=False))


print()
print("COUNTRIES WITHOUT FOUR-REGION CLASSIFICATION")
print("-" * 100)

no_region = (
    df_v1.loc[
        df_v1["region_4"].isna(),
        ["cntry", "country_name", "region_m49"]
    ]
    .drop_duplicates()
)

if no_region.empty:
    print("None")
else:
    print(no_region.to_string(index=False))


print()
print("=" * 100)
print("V1 SCOPE DATASET CREATED")
print("=" * 100)

print(f"Output: {output_file}")
print()
print("The RAW and previous INTERIM datasets were not modified.")