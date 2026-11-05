from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
raw_file = (
    project_root
    / "data"
    / "raw"
    / "ess"
    / "ESS11e04_2.parquet"
)

interim_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_core_interim.parquet"
)


# Check that both files exist
if not raw_file.exists():
    raise FileNotFoundError(f"RAW file not found: {raw_file}")

if not interim_file.exists():
    raise FileNotFoundError(f"Interim file not found: {interim_file}")


# Load datasets
df_raw = pd.read_parquet(raw_file)
df = pd.read_parquet(interim_file)


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

for variable in human_values:
    missing_codes[variable] = [66, 77, 88, 99]


print("=" * 100)
print("ESS ROUND 11 INTERIM DATA VALIDATION")
print("=" * 100)


validation_passed = True


# ---------------------------------------------------------------------
# 1. Validate row preservation
# ---------------------------------------------------------------------

print()
print("ROW PRESERVATION")
print("-" * 100)

print(f"RAW rows:       {len(df_raw):,}")
print(f"Interim rows:   {len(df):,}")

if len(df_raw) == len(df):
    print("Status:         PASSED")
else:
    print("Status:         FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Validate respondent identifiers
# ---------------------------------------------------------------------

print()
print("RESPONDENT IDENTIFIERS")
print("-" * 100)

duplicate_ids = df.duplicated(
    subset=["cntry", "idno"]
).sum()

print(f"Duplicate country + ID combinations: {duplicate_ids:,}")

if duplicate_ids == 0:
    print("Status:                          PASSED")
else:
    print("Status:                          REQUIRES REVIEW")
    validation_passed = False


# ---------------------------------------------------------------------
# 3. Validate missing-value conversion
# ---------------------------------------------------------------------

print()
print("MISSING-VALUE RECODING")
print("-" * 100)

for variable, codes in missing_codes.items():

    raw_missing_count = df_raw[variable].isin(codes).sum()
    interim_missing_count = df[variable].isna().sum()

    special_codes_remaining = (
        df[variable]
        .isin(codes)
        .sum()
    )

    correct_missing_count = (
        raw_missing_count == interim_missing_count
    )

    no_special_codes = (
        special_codes_remaining == 0
    )

    status = (
        "PASSED"
        if correct_missing_count and no_special_codes
        else "FAILED"
    )

    if status == "FAILED":
        validation_passed = False

    print(
        f"{variable:<12} "
        f"RAW coded missing: {raw_missing_count:>6,} | "
        f"Interim NaN: {interim_missing_count:>6,} | "
        f"Codes remaining: {special_codes_remaining:>3,} | "
        f"{status}"
    )


# ---------------------------------------------------------------------
# 4. Validate ideology and social trust ranges
# ---------------------------------------------------------------------

print()
print("IDEOLOGY AND SOCIAL TRUST RANGES")
print("-" * 100)

scale_variables = [
    "lrscale",
    "ppltrst",
    "pplfair",
    "pplhlp",
]

for variable in scale_variables:

    valid = (
        df[variable]
        .dropna()
        .between(0, 10)
        .all()
    )

    print(
        f"{variable:<12} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate Human Values ranges
# ---------------------------------------------------------------------

print()
print("HUMAN VALUES RANGES")
print("-" * 100)

for variable in human_values:

    valid = (
        df[variable]
        .dropna()
        .between(1, 6)
        .all()
    )

    print(
        f"{variable:<12} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 6. Validate survey weights
# ---------------------------------------------------------------------

print()
print("SURVEY WEIGHTS")
print("-" * 100)

weight_variables = [
    "dweight",
    "pspwght",
    "pweight",
    "anweight",
]

for variable in weight_variables:

    no_missing = df[variable].notna().all()
    all_positive = (df[variable] > 0).all()

    raw_unchanged = df[variable].equals(
        df_raw[variable]
    )

    valid = (
        no_missing
        and all_positive
        and raw_unchanged
    )

    print(
        f"{variable:<12} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 7. Validate education quality flags
# ---------------------------------------------------------------------

print()
print("EDUCATION QUALITY FLAGS")
print("-" * 100)

extreme_count = df["eduyrs_extreme_flag"].sum()
age_conflict_count = df["eduyrs_age_conflict_flag"].sum()
any_issue_count = df["eduyrs_quality_flag"].sum()

print(f"Extreme EDUYRS values: {extreme_count:,}")
print(f"Age conflicts:         {age_conflict_count:,}")
print(f"Any EDUYRS issue:      {any_issue_count:,}")

flag_logic_valid = (
    df["eduyrs_quality_flag"]
    == (
        df["eduyrs_extreme_flag"]
        | df["eduyrs_age_conflict_flag"]
    )
).all()

print(
    f"Flag logic:            "
    f"{'PASSED' if flag_logic_valid else 'FAILED'}"
)

if not flag_logic_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if validation_passed:
    print("ESS interim dataset validation PASSED.")
else:
    print("ESS interim dataset validation requires review.")

print()
print("The RAW ESS dataset was not modified.")