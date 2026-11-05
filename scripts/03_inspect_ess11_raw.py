from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define the ESS raw file
input_file = (
    project_root
    / "data"
    / "raw"
    / "ess"
    / "ESS11e04_2.parquet"
)

# Check that the raw file exists
if not input_file.exists():
    raise FileNotFoundError(f"ESS raw file not found: {input_file}")

print("=" * 90)
print("ESS ROUND 11 RAW DATA INSPECTION")
print("=" * 90)

# Load the raw ESS dataset without transformations
df = pd.read_parquet(input_file)

# Display basic dataset dimensions
print()
print("DATASET DIMENSIONS")
print("-" * 90)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")

# Display the first variables in the dataset
print()
print("FIRST 30 COLUMNS")
print("-" * 90)

for column in df.columns[:30]:
    print(column)

# Inspect country coverage
print()
print("COUNTRY COVERAGE")
print("-" * 90)

if "cntry" in df.columns:
    country_counts = (
        df["cntry"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(country_counts.to_string())
    print()
    print(f"Number of country codes: {df['cntry'].nunique(dropna=True)}")
else:
    print("Variable 'cntry' was not found.")

# Define expected variables for the project
expected_variables = {
    "Survey structure": [
        "essround",
        "edition",
        "idno",
        "cntry",
    ],

    "Ideology": [
        "lrscale",
    ],

    "Social trust": [
        "ppltrst",
        "pplfair",
        "pplhlp",
    ],

    "Human Values": [
        "ipcrtiv",
        "imprich",
        "ipeqopt",
        "ipshabt",
        "impsafe",
        "impdiff",
        "ipfrule",
        "ipudrst",
        "ipmodst",
        "ipgdtim",
        "impfree",
        "iphlppl",
        "ipsuces",
        "ipstrgv",
        "ipadvnt",
        "ipbhprp",
        "iprspot",
        "iplylfr",
        "impenv",
        "imptrad",
        "impfun",
    ],

    "Weights": [
        "dweight",
        "pspwght",
        "pweight",
        "anweight",
    ],
}

print()
print("EXPECTED VARIABLE CHECK")
print("-" * 90)

for group, variables in expected_variables.items():

    print()
    print(f"[{group}]")

    for variable in variables:
        status = "FOUND" if variable in df.columns else "NOT FOUND"
        print(f"{variable:<12} {status}")

print()
print("=" * 90)
print("RAW INSPECTION COMPLETED")
print("=" * 90)
print("No transformations were applied to the ESS raw dataset.")