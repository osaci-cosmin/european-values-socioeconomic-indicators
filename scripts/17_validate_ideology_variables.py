from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
source_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values.parquet"
)

ideology_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values_ideology.parquet"
)


# Check that input files exist
if not source_file.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {source_file}"
    )

if not ideology_file.exists():
    raise FileNotFoundError(
        f"Ideology dataset not found: {ideology_file}"
    )


# Load datasets
df_source = pd.read_parquet(source_file)
df = pd.read_parquet(ideology_file)


print("=" * 100)
print("ESS ROUND 11 IDEOLOGY VARIABLE VALIDATION")
print("=" * 100)

validation_passed = True


# ---------------------------------------------------------------------
# 1. Validate row preservation
# ---------------------------------------------------------------------

print()
print("ROW PRESERVATION")
print("-" * 100)

rows_valid = len(df_source) == len(df)

print(f"Source rows:   {len(df_source):,}")
print(f"Ideology rows: {len(df):,}")

print(
    f"Status:        "
    f"{'PASSED' if rows_valid else 'FAILED'}"
)

if not rows_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Validate original Left-Right scale preservation
# ---------------------------------------------------------------------

print()
print("LEFT-RIGHT SCALE PRESERVATION")
print("-" * 100)

lrscale_unchanged = df["lrscale"].equals(
    df_source["lrscale"]
)

print(
    f"lrscale unchanged: "
    f"{'PASSED' if lrscale_unchanged else 'FAILED'}"
)

if not lrscale_unchanged:
    validation_passed = False


# ---------------------------------------------------------------------
# 3. Validate ideology valid flag
# ---------------------------------------------------------------------

print()
print("VALID RESPONSE FLAG")
print("-" * 100)

expected_valid_flag = df["lrscale"].notna()

valid_flag_correct = (
    df["ideology_valid_flag"]
    .equals(expected_valid_flag)
)

print(
    f"ideology_valid_flag: "
    f"{'PASSED' if valid_flag_correct else 'FAILED'}"
)

if not valid_flag_correct:
    validation_passed = False


# ---------------------------------------------------------------------
# 4. Validate Left / Centre / Right classification
# ---------------------------------------------------------------------

print()
print("IDEOLOGY CLASSIFICATION")
print("-" * 100)

expected_group = pd.Series(
    pd.NA,
    index=df.index,
    dtype="object"
)

expected_group.loc[
    df["lrscale"].between(0, 3)
] = "Left"

expected_group.loc[
    df["lrscale"].between(4, 6)
] = "Centre"

expected_group.loc[
    df["lrscale"].between(7, 10)
] = "Right"


actual_group = (
    df["ideology_group"]
    .astype("object")
    .where(
        df["ideology_group"].notna(),
        pd.NA
    )
)


classification_valid = (
    (
        actual_group.fillna("__MISSING__")
        == expected_group.fillna("__MISSING__")
    )
    .all()
)


print(
    f"0-3  = Left:    "
    f"{'PASSED' if classification_valid else 'FAILED'}"
)

print(
    f"4-6  = Centre:  "
    f"{'PASSED' if classification_valid else 'FAILED'}"
)

print(
    f"7-10 = Right:   "
    f"{'PASSED' if classification_valid else 'FAILED'}"
)

if not classification_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate classification coverage
# ---------------------------------------------------------------------

print()
print("CLASSIFICATION COVERAGE")
print("-" * 100)

valid_lrscale_n = df["lrscale"].notna().sum()
classified_n = df["ideology_group"].notna().sum()

coverage_valid = (
    valid_lrscale_n == classified_n
)

print(f"Valid lrscale: {valid_lrscale_n:,}")
print(f"Classified:    {classified_n:,}")

print(
    f"Status:        "
    f"{'PASSED' if coverage_valid else 'FAILED'}"
)

if not coverage_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 6. Validate binary ideology indicators
# ---------------------------------------------------------------------

print()
print("IDEOLOGY BINARY FLAGS")
print("-" * 100)

flag_mapping = {
    "ideology_left_flag": "Left",
    "ideology_centre_flag": "Centre",
    "ideology_right_flag": "Right",
}


for flag, group in flag_mapping.items():

    expected = (
        df["ideology_group"] == group
    ).astype("boolean")

    expected.loc[
        df["lrscale"].isna()
    ] = pd.NA

    valid = (
        (
            df[flag]
            .astype("boolean")
            .fillna(False)
            ==
            expected
            .fillna(False)
        )
        .all()
        and
        (
            df[flag].isna()
            ==
            expected.isna()
        ).all()
    )

    print(
        f"{flag:<25} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 7. Validate mutual exclusivity
# ---------------------------------------------------------------------

print()
print("MUTUAL EXCLUSIVITY")
print("-" * 100)

flag_columns = [
    "ideology_left_flag",
    "ideology_centre_flag",
    "ideology_right_flag",
]

valid_rows = df["lrscale"].notna()

flag_sum = (
    df.loc[
        valid_rows,
        flag_columns
    ]
    .astype("int8")
    .sum(axis=1)
)

valid_one_group = (
    flag_sum == 1
).all()


missing_flags_correct = (
    df.loc[
        ~valid_rows,
        flag_columns
    ]
    .isna()
    .all()
    .all()
)


print(
    f"One group per valid respondent: "
    f"{'PASSED' if valid_one_group else 'FAILED'}"
)

print(
    f"Missing lrscale -> flags NaN:    "
    f"{'PASSED' if missing_flags_correct else 'FAILED'}"
)


if not valid_one_group or not missing_flags_correct:
    validation_passed = False


# ---------------------------------------------------------------------
# 8. Display validated counts
# ---------------------------------------------------------------------

print()
print("VALIDATED CLASSIFICATION COUNTS")
print("-" * 100)

counts = (
    df["ideology_group"]
    .value_counts(
        sort=False,
        dropna=False
    )
)

print(counts.to_string())


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if validation_passed:
    print("Ideology variable validation PASSED.")
else:
    print("Ideology variable validation requires review.")

print()
print("No respondent data were modified.")