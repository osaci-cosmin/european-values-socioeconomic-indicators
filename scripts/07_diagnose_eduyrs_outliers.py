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

# Load the raw ESS dataset
df = pd.read_parquet(input_file)


# Keep only substantive education-year responses
valid_eduyrs = df[
    ~df["eduyrs"].isin([77, 88, 99])
].copy()


print("=" * 100)
print("ESS ROUND 11 EDUYRS OUTLIER DIAGNOSTIC")
print("=" * 100)


# Inspect the highest reported values
print()
print("HIGHEST EDUYRS VALUES")
print("-" * 100)

print(
    valid_eduyrs["eduyrs"]
    .value_counts()
    .sort_index(ascending=False)
    .head(20)
    .to_string()
)


# Inspect observations above the initial plausibility threshold
outliers = valid_eduyrs[
    valid_eduyrs["eduyrs"] > 50
][
    [
        "idno",
        "cntry",
        "agea",
        "eduyrs",
        "eisced",
        "mnactic",
    ]
].sort_values("eduyrs", ascending=False)


print()
print("OBSERVATIONS WITH EDUYRS > 50")
print("-" * 100)

if outliers.empty:
    print("No observations found.")
else:
    print(outliers.to_string(index=False))


# Check whether education years exceed respondent age
age_conflicts = valid_eduyrs[
    (valid_eduyrs["agea"] != 999)
    & (valid_eduyrs["eduyrs"] >= valid_eduyrs["agea"])
][
    [
        "idno",
        "cntry",
        "agea",
        "eduyrs",
        "eisced",
    ]
]


print()
print("LOGICAL CHECK: EDUYRS >= AGE")
print("-" * 100)

if age_conflicts.empty:
    print("No logical conflicts found.")
else:
    print(age_conflicts.to_string(index=False))


print()
print("=" * 100)
print("DIAGNOSTIC COMPLETED")
print("=" * 100)
print("No transformations were applied to the ESS raw dataset.")