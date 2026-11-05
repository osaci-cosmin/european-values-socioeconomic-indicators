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

# Load the raw ESS dataset
df = pd.read_parquet(input_file)


# Define variables that require detailed value inspection
profile_variables = {

    "Ideological Self-Placement": [
        "lrscale",
    ],

    "Generalized Social Trust": [
        "ppltrst",
        "pplfair",
        "pplhlp",
    ],

    "Human Values": [
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
    ],

    "Individual Controls": [
        "agea",
        "gndr",
        "eduyrs",
        "eisced",
        "mnactic",
        "hinctnta",
    ],
}


print("=" * 100)
print("ESS ROUND 11 CORE VARIABLE PROFILING")
print("=" * 100)

print()
print(f"Rows: {len(df):,}")
print()


for group, variables in profile_variables.items():

    print()
    print("=" * 100)
    print(group.upper())
    print("=" * 100)

    for variable in variables:

        series = df[variable]

        print()
        print(f"VARIABLE: {variable}")
        print("-" * 100)

        print(f"Data type:       {series.dtype}")
        print(f"Non-null values: {series.notna().sum():,}")
        print(f"Null values:     {series.isna().sum():,}")
        print(f"Unique values:   {series.nunique(dropna=False):,}")

        # Display detailed frequencies for variables with limited categories
        if series.nunique(dropna=False) <= 30:

            counts = (
                series
                .value_counts(dropna=False)
                .sort_index()
            )

            print()
            print("VALUE COUNTS")
            print(counts.to_string())

        else:

            # Display numerical summary for high-cardinality variables
            if pd.api.types.is_numeric_dtype(series):

                print()
                print("NUMERICAL SUMMARY")
                print(series.describe().to_string())

            else:

                print()
                print("TOP VALUES")
                print(
                    series
                    .value_counts(dropna=False)
                    .head(20)
                    .to_string()
                )


# Inspect survey weights separately
weight_variables = [
    "dweight",
    "pspwght",
    "pweight",
    "anweight",
]

print()
print("=" * 100)
print("SURVEY WEIGHTS")
print("=" * 100)

for variable in weight_variables:

    series = df[variable]

    print()
    print(f"VARIABLE: {variable}")
    print("-" * 100)

    print(series.describe().to_string())

    print(f"Null values: {series.isna().sum():,}")
    print(f"Zero values: {(series == 0).sum():,}")


print()
print("=" * 100)
print("PROFILING COMPLETED")
print("=" * 100)
print("No transformations were applied to the ESS raw dataset.")