from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define the V1 respondent-level dataset
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_scope.parquet"
)

# Check that the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"V1 dataset not found: {input_file}"
    )


# Load the V1 dataset
df = pd.read_parquet(input_file)


# Define the 21 Human Values variables
human_values = [
    "ipcrtiva",
    "impricha",
    "ipeqopta",
    "ipshabta",
    "impsafea",
    "impdiffa",
    "ipfrulea",
    "ipudrsta",
    "ipmodsta",
    "ipgdtima",
    "impfreea",
    "iphlppla",
    "ipsucesa",
    "ipstrgva",
    "ipadvnta",
    "ipbhprpa",
    "iprspota",
    "iplylfra",
    "impenva",
    "imptrada",
    "impfuna",
]


# Count valid and missing Human Values items per respondent
hv_answered_n = df[human_values].notna().sum(axis=1)
hv_missing_n = df[human_values].isna().sum(axis=1)


print("=" * 100)
print("ESS ROUND 11 HUMAN VALUES COMPLETENESS DIAGNOSTIC")
print("=" * 100)

print()
print(f"Respondents in V1: {len(df):,}")


# ---------------------------------------------------------------------
# 1. Distribution of answered Human Values items
# ---------------------------------------------------------------------

print()
print("ANSWERED HUMAN VALUES ITEMS")
print("-" * 100)

answered_distribution = (
    hv_answered_n
    .value_counts()
    .sort_index(ascending=False)
)

print(answered_distribution.to_string())


# ---------------------------------------------------------------------
# 2. Complete Human Values cases
# ---------------------------------------------------------------------

complete_cases = (hv_answered_n == 21).sum()
incomplete_cases = (hv_answered_n < 21).sum()

complete_rate = complete_cases / len(df) * 100
incomplete_rate = incomplete_cases / len(df) * 100


print()
print("COMPLETE CASE SUMMARY")
print("-" * 100)

print(
    f"Complete 21/21:       "
    f"{complete_cases:,} ({complete_rate:.2f}%)"
)

print(
    f"Incomplete:           "
    f"{incomplete_cases:,} ({incomplete_rate:.2f}%)"
)


# ---------------------------------------------------------------------
# 3. Missing-item distribution
# ---------------------------------------------------------------------

print()
print("MISSING ITEMS PER RESPONDENT")
print("-" * 100)

missing_distribution = (
    hv_missing_n
    .value_counts()
    .sort_index()
)

print(missing_distribution.to_string())


# ---------------------------------------------------------------------
# 4. Respondents with extensive Human Values missingness
# ---------------------------------------------------------------------

print()
print("EXTENSIVE HUMAN VALUES MISSINGNESS")
print("-" * 100)

for threshold in [1, 2, 3, 5, 10, 21]:

    count = (hv_missing_n >= threshold).sum()
    rate = count / len(df) * 100

    print(
        f"Missing >= {threshold:>2} items: "
        f"{count:>6,} ({rate:>6.2f}%)"
    )


# ---------------------------------------------------------------------
# 5. Country-level completeness
# ---------------------------------------------------------------------

country_check = df[
    ["cntry", "country_name"]
].copy()

country_check["hv_answered_n"] = hv_answered_n
country_check["hv_missing_n"] = hv_missing_n
country_check["hv_complete"] = hv_answered_n == 21


country_summary = (
    country_check
    .groupby(
        ["cntry", "country_name"],
        as_index=False
    )
    .agg(
        respondents=("hv_complete", "size"),
        complete_cases=("hv_complete", "sum"),
        mean_answered_items=("hv_answered_n", "mean"),
        max_missing_items=("hv_missing_n", "max"),
    )
)


country_summary["complete_rate"] = (
    country_summary["complete_cases"]
    / country_summary["respondents"]
    * 100
)


country_summary = country_summary.sort_values(
    "complete_rate"
)


print()
print("COUNTRY-LEVEL COMPLETENESS")
print("-" * 100)

print(
    country_summary.to_string(
        index=False,
        formatters={
            "mean_answered_items": "{:.2f}".format,
            "complete_rate": "{:.2f}%".format,
        }
    )
)


# ---------------------------------------------------------------------
# 6. Inspect countries with known ESS11 Human Values notes
# ---------------------------------------------------------------------

print()
print("COUNTRIES WITH ESS11 HUMAN VALUES DOCUMENTATION NOTES")
print("-" * 100)

flagged_countries = country_summary[
    country_summary["cntry"].isin(
        ["EE", "GB"]
    )
]

print(
    flagged_countries.to_string(
        index=False,
        formatters={
            "mean_answered_items": "{:.2f}".format,
            "complete_rate": "{:.2f}%".format,
        }
    )
)


print()
print("=" * 100)
print("DIAGNOSTIC COMPLETED")
print("=" * 100)

print(
    "No Human Values scores were calculated."
)

print(
    "No respondent records were modified."
)