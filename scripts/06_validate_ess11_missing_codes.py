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


# Define expected valid ranges and special missing codes
validation_rules = {

    "lrscale": {
        "valid_values": list(range(0, 11)),
        "missing_codes": [77, 88, 99],
    },

    "ppltrst": {
        "valid_values": list(range(0, 11)),
        "missing_codes": [77, 88, 99],
    },

    "pplfair": {
        "valid_values": list(range(0, 11)),
        "missing_codes": [77, 88, 99],
    },

    "pplhlp": {
        "valid_values": list(range(0, 11)),
        "missing_codes": [77, 88, 99],
    },

    "agea": {
        "valid_range": (15, 120),
        "missing_codes": [999],
    },

    "eduyrs": {
        "valid_range": (0, 50),
        "missing_codes": [77, 88, 99],
    },

    "eisced": {
        "valid_values": list(range(1, 8)),
        "missing_codes": [55, 77, 88, 99],
    },

    "mnactic": {
        "valid_values": list(range(1, 10)),
        "missing_codes": [77, 88, 99],
    },

    "hinctnta": {
        "valid_values": list(range(1, 11)),
        "missing_codes": [77, 88, 99],
    },
}


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

for variable in human_values:

    validation_rules[variable] = {
        "valid_values": list(range(1, 7)),
        "missing_codes": [66, 77, 88, 99],
    }


print("=" * 100)
print("ESS ROUND 11 MISSING-CODE VALIDATION")
print("=" * 100)


all_passed = True


for variable, rule in validation_rules.items():

    series = df[variable]

    observed_values = set(series.dropna().unique())

    missing_codes = set(rule["missing_codes"])

    # Validate categorical variables
    if "valid_values" in rule:

        valid_values = set(rule["valid_values"])

        allowed_values = valid_values | missing_codes

        unexpected_values = observed_values - allowed_values

    # Validate continuous/range variables
    else:

        minimum, maximum = rule["valid_range"]

        unexpected_values = {
            value
            for value in observed_values
            if value not in missing_codes
            and not (minimum <= value <= maximum)
        }

    missing_count = series.isin(missing_codes).sum()
    missing_rate = (missing_count / len(df)) * 100

    print()
    print(f"VARIABLE: {variable}")
    print("-" * 100)

    print(f"Missing-coded observations: {missing_count:,}")
    print(f"Missing-coded rate:         {missing_rate:.2f}%")

    if unexpected_values:

        all_passed = False

        print(
            "Unexpected values:          "
            f"{sorted(unexpected_values)}"
        )

    else:

        print("Unexpected values:          None")


print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if all_passed:
    print("Missing-code validation PASSED.")
else:
    print("Missing-code validation requires review.")

print()
print("No transformations were applied to the ESS raw dataset.")