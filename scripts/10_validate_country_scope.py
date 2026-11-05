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


# Check that input files exist
if not interim_file.exists():
    raise FileNotFoundError(
        f"Interim ESS file not found: {interim_file}"
    )

if not country_config_file.exists():
    raise FileNotFoundError(
        f"Country configuration file not found: {country_config_file}"
    )


# Load data
df = pd.read_parquet(interim_file)

country_config = pd.read_csv(
    country_config_file,
    dtype={"cntry": "string"}
)


print("=" * 100)
print("ESS ROUND 11 COUNTRY SCOPE VALIDATION")
print("=" * 100)


validation_passed = True


# ---------------------------------------------------------------------
# 1. Compare ESS countries with configuration countries
# ---------------------------------------------------------------------

ess_countries = set(df["cntry"].dropna().unique())
config_countries = set(country_config["cntry"].dropna().unique())

missing_from_config = ess_countries - config_countries
extra_in_config = config_countries - ess_countries


print()
print("COUNTRY CONFIGURATION COVERAGE")
print("-" * 100)

print(f"ESS country codes:       {len(ess_countries)}")
print(f"Configured country codes:{len(config_countries)}")

print(
    f"Missing from config:     "
    f"{sorted(missing_from_config)}"
)

print(
    f"Extra in config:         "
    f"{sorted(extra_in_config)}"
)


if missing_from_config or extra_in_config:
    validation_passed = False
    print("Status:                  FAILED")
else:
    print("Status:                  PASSED")


# ---------------------------------------------------------------------
# 2. Check duplicate country codes
# ---------------------------------------------------------------------

duplicate_codes = country_config["cntry"].duplicated().sum()

print()
print("COUNTRY CODE UNIQUENESS")
print("-" * 100)

print(f"Duplicate country codes: {duplicate_codes}")

if duplicate_codes == 0:
    print("Status:                  PASSED")
else:
    print("Status:                  FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 3. Merge configuration for validation
# ---------------------------------------------------------------------

df_check = df.merge(
    country_config,
    on="cntry",
    how="left",
    validate="many_to_one"
)


# ---------------------------------------------------------------------
# 4. Calculate analytical scope
# ---------------------------------------------------------------------

european_scope = df_check[
    df_check["include_v1"] == 1
]

regional_scope = df_check[
    (df_check["include_v1"] == 1)
    & (df_check["include_regional"] == 1)
]


print()
print("V1 ANALYTICAL SCOPE")
print("-" * 100)

print(f"Original ESS respondents: {len(df_check):,}")

print(
    f"European V1 respondents:  "
    f"{len(european_scope):,}"
)

print(
    f"Regional respondents:     "
    f"{len(regional_scope):,}"
)

print(
    f"V1 countries:             "
    f"{european_scope['cntry'].nunique()}"
)

print(
    f"Regional countries:       "
    f"{regional_scope['cntry'].nunique()}"
)


# ---------------------------------------------------------------------
# 5. Display country-level configuration
# ---------------------------------------------------------------------

print()
print("COUNTRY SCOPE")
print("-" * 100)

country_summary = (
    df_check
    .groupby(
        [
            "cntry",
            "country_name",
            "include_v1",
            "region_m49",
            "include_regional",
        ],
        dropna=False
    )
    .size()
    .reset_index(name="respondents")
    .sort_values("country_name")
)

print(
    country_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if validation_passed:
    print("Country scope validation PASSED.")
else:
    print("Country scope validation requires review.")

print()
print("No respondent records were removed or transformed.")