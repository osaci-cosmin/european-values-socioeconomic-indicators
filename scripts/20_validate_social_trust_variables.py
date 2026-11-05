from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
source_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values_ideology.parquet"
)

trust_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)


# Check that input files exist
if not source_file.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {source_file}"
    )

if not trust_file.exists():
    raise FileNotFoundError(
        f"Social Trust dataset not found: {trust_file}"
    )


# Load datasets
df_source = pd.read_parquet(source_file)
df = pd.read_parquet(trust_file)


# Define Social Trust items
trust_variables = [
    "ppltrst",
    "pplfair",
    "pplhlp",
]


print("=" * 100)
print("ESS ROUND 11 SOCIAL TRUST VARIABLE VALIDATION")
print("=" * 100)

validation_passed = True


# ---------------------------------------------------------------------
# 1. Validate row preservation
# ---------------------------------------------------------------------

print()
print("ROW PRESERVATION")
print("-" * 100)

rows_valid = len(df_source) == len(df)

print(f"Source rows:       {len(df_source):,}")
print(f"Social Trust rows: {len(df):,}")

print(
    f"Status:            "
    f"{'PASSED' if rows_valid else 'FAILED'}"
)

if not rows_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Validate original Social Trust item preservation
# ---------------------------------------------------------------------

print()
print("ORIGINAL ITEM PRESERVATION")
print("-" * 100)

for variable in trust_variables:

    unchanged = df[variable].equals(
        df_source[variable]
    )

    print(
        f"{variable:<10} "
        f"{'PASSED' if unchanged else 'FAILED'}"
    )

    if not unchanged:
        validation_passed = False


# ---------------------------------------------------------------------
# 3. Validate answered-item count
# ---------------------------------------------------------------------

print()
print("ANSWERED ITEM COUNT")
print("-" * 100)

expected_answered_n = (
    df[trust_variables]
    .notna()
    .sum(axis=1)
)

answered_count_valid = (
    df["trust_answered_n"]
    .equals(expected_answered_n)
)

print(
    f"trust_answered_n: "
    f"{'PASSED' if answered_count_valid else 'FAILED'}"
)

if not answered_count_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 4. Validate complete-case flag
# ---------------------------------------------------------------------

print()
print("COMPLETE-CASE FLAG")
print("-" * 100)

expected_complete_flag = (
    expected_answered_n == 3
)

complete_flag_valid = (
    df["trust_complete_flag"]
    .equals(expected_complete_flag)
)

eligible_flag_valid = (
    df["trust_scoring_eligible"]
    .equals(df["trust_complete_flag"])
)

print(
    f"trust_complete_flag:    "
    f"{'PASSED' if complete_flag_valid else 'FAILED'}"
)

print(
    f"trust_scoring_eligible: "
    f"{'PASSED' if eligible_flag_valid else 'FAILED'}"
)

if not complete_flag_valid or not eligible_flag_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate Social Trust score
# ---------------------------------------------------------------------

print()
print("SOCIAL TRUST SCORE")
print("-" * 100)

eligible = df["trust_scoring_eligible"]

expected_score = (
    df.loc[
        eligible,
        trust_variables
    ]
    .mean(axis=1)
)

stored_score = df.loc[
    eligible,
    "social_trust_score"
]

score_valid = np.allclose(
    stored_score.astype(float),
    expected_score.astype(float),
    equal_nan=True,
)

print(
    f"Composite mean calculation: "
    f"{'PASSED' if score_valid else 'FAILED'}"
)

if not score_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 6. Validate missing score for incomplete respondents
# ---------------------------------------------------------------------

print()
print("INCOMPLETE RESPONDENTS")
print("-" * 100)

incomplete_missing = (
    df.loc[
        ~eligible,
        "social_trust_score"
    ]
    .isna()
    .all()
)

print(
    f"Incomplete cases -> score NaN: "
    f"{'PASSED' if incomplete_missing else 'FAILED'}"
)

if not incomplete_missing:
    validation_passed = False


# ---------------------------------------------------------------------
# 7. Validate score range
# ---------------------------------------------------------------------

print()
print("SCORE RANGE")
print("-" * 100)

score_values = (
    df["social_trust_score"]
    .dropna()
)

range_valid = (
    score_values
    .between(0, 10)
    .all()
)

print(
    f"All valid scores between 0 and 10: "
    f"{'PASSED' if range_valid else 'FAILED'}"
)

print(
    f"Minimum score: {score_values.min():.4f}"
)

print(
    f"Maximum score: {score_values.max():.4f}"
)

if not range_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 8. Validate expected number of complete cases
# ---------------------------------------------------------------------

print()
print("SCORING COVERAGE")
print("-" * 100)

complete_n = df["trust_complete_flag"].sum()
score_n = df["social_trust_score"].notna().sum()

coverage_valid = (
    complete_n == score_n
)

print(f"Complete 3/3 cases: {complete_n:,}")
print(f"Composite scores:   {score_n:,}")

print(
    f"Status:             "
    f"{'PASSED' if coverage_valid else 'FAILED'}"
)

if not coverage_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 9. Validate manual examples
# ---------------------------------------------------------------------

print()
print("MANUAL SCORE CHECKS")
print("-" * 100)

sample = (
    df.loc[
        eligible,
        trust_variables + ["social_trust_score"]
    ]
    .head(5)
    .copy()
)

sample["recalculated_score"] = (
    sample[trust_variables]
    .mean(axis=1)
)

sample["difference"] = (
    sample["social_trust_score"]
    - sample["recalculated_score"]
)

print(sample.to_string(index=False))


manual_valid = np.allclose(
    sample["difference"].astype(float),
    0.0,
    atol=1e-12,
)

print()

print(
    f"Manual recalculation: "
    f"{'PASSED' if manual_valid else 'FAILED'}"
)

if not manual_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if validation_passed:
    print("Social Trust variable validation PASSED.")
else:
    print("Social Trust variable validation requires review.")

print()
print("No respondent data were modified.")