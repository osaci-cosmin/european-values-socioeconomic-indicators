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


# Define the core variables required for V1
core_variables = {

    "Identifiers": [
        "essround",
        "edition",
        "idno",
        "cntry",
    ],

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

    "Survey Weights": [
        "dweight",
        "pspwght",
        "pweight",
        "anweight",
    ],

    "Sample Design": [
        "psu",
        "stratum",
    ],
}


print("=" * 90)
print("ESS ROUND 11 CORE VARIABLE VALIDATION")
print("=" * 90)

total_expected = 0
total_found = 0

for group, variables in core_variables.items():

    print()
    print(f"[{group}]")
    print("-" * 90)

    for variable in variables:

        total_expected += 1

        if variable in df.columns:
            status = "FOUND"
            total_found += 1
        else:
            status = "NOT FOUND"

        print(f"{variable:<15} {status}")


print()
print("=" * 90)
print("VALIDATION SUMMARY")
print("=" * 90)

print(f"Expected variables: {total_expected}")
print(f"Variables found:   {total_found}")
print(f"Variables missing: {total_expected - total_found}")

if total_expected == total_found:
    print("Core variable validation PASSED.")
else:
    print("Core variable validation requires review.")

print()
print("No transformations were applied to the ESS raw dataset.")