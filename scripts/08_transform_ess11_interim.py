from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input and output files
input_file = (
    project_root
    / "data"
    / "raw"
    / "ess"
    / "ESS11e04_2.parquet"
)

output_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_core_interim.parquet"
)


# Load the original ESS dataset
df_raw = pd.read_parquet(input_file)


# Define Human Values variables
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


# Define variables required for V1
core_variables = [
    "essround",
    "edition",
    "idno",
    "cntry",
    "lrscale",
    "ppltrst",
    "pplfair",
    "pplhlp",
    "agea",
    "gndr",
    "eduyrs",
    "eisced",
    "mnactic",
    "hinctnta",
    "dweight",
    "pspwght",
    "pweight",
    "anweight",
    "psu",
    "stratum",
] + human_values


# Create an analytical subset
df = df_raw[core_variables].copy()


# Define official ESS missing codes
missing_codes = {
    "lrscale": [77, 88, 99],
    "ppltrst": [77, 88, 99],
    "pplfair": [77, 88, 99],
    "pplhlp": [77, 88, 99],
    "agea": [999],
    "eduyrs": [77, 88, 99],
    "eisced": [55, 77, 88, 99],
    "mnactic": [77, 88, 99],
    "hinctnta": [77, 88, 99],
}


# Add Human Values missing codes
for variable in human_values:
    missing_codes[variable] = [66, 77, 88, 99]


# Convert official ESS missing codes to NaN
for variable, codes in missing_codes.items():
    df[variable] = df[variable].replace(codes, pd.NA)


# Create education data-quality flags
df["eduyrs_extreme_flag"] = (
    df["eduyrs"].notna()
    & (df["eduyrs"] > 50)
)

df["eduyrs_age_conflict_flag"] = (
    df["eduyrs"].notna()
    & df["agea"].notna()
    & (df["eduyrs"] >= df["agea"])
)

df["eduyrs_quality_flag"] = (
    df["eduyrs_extreme_flag"]
    | df["eduyrs_age_conflict_flag"]
)


# Protect an existing interim file
if output_file.exists():
    raise FileExistsError(
        f"Interim file already exists and will not be overwritten: {output_file}"
    )


# Save the interim dataset
df.to_parquet(
    output_file,
    index=False
)


print("=" * 100)
print("ESS ROUND 11 INTERIM TRANSFORMATION")
print("=" * 100)

print()
print(f"RAW rows:               {len(df_raw):,}")
print(f"Interim rows:           {len(df):,}")
print(f"Interim columns:        {df.shape[1]:,}")

print()
print("EDUCATION QUALITY FLAGS")
print("-" * 100)

print(
    f"Extreme EDUYRS values:  "
    f"{df['eduyrs_extreme_flag'].sum():,}"
)

print(
    f"Age conflicts:          "
    f"{df['eduyrs_age_conflict_flag'].sum():,}"
)

print(
    f"Any EDUYRS issue:       "
    f"{df['eduyrs_quality_flag'].sum():,}"
)

print()
print("MISSING VALUES AFTER RECODING")
print("-" * 100)

for variable in [
    "lrscale",
    "ppltrst",
    "pplfair",
    "pplhlp",
    "agea",
    "eduyrs",
    "eisced",
    "mnactic",
    "hinctnta",
]:

    missing_n = df[variable].isna().sum()
    missing_rate = missing_n / len(df) * 100

    print(
        f"{variable:<12} "
        f"{missing_n:>7,} "
        f"({missing_rate:>6.2f}%)"
    )


print()
print("=" * 100)
print("INTERIM DATASET CREATED")
print("=" * 100)

print(f"Output: {output_file}")
print()
print("The RAW ESS dataset was not modified.")