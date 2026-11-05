from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
data_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_values.parquet"
)

mapping_file = (
    project_root
    / "config"
    / "human_values_mapping.csv"
)


# Check that input files exist
if not data_file.exists():
    raise FileNotFoundError(
        f"Human Values dataset not found: {data_file}"
    )

if not mapping_file.exists():
    raise FileNotFoundError(
        f"Human Values mapping not found: {mapping_file}"
    )


# Load data
df = pd.read_parquet(data_file)
mapping = pd.read_csv(mapping_file)


# Create a safe column-name format
def make_slug(value):
    return (
        value
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# Get Human Values variables
human_values = mapping["variable"].tolist()

reversed_variables = [
    f"{variable}_r"
    for variable in human_values
]


print("=" * 100)
print("ESS ROUND 11 HUMAN VALUES SCORE VALIDATION")
print("=" * 100)

validation_passed = True


# ---------------------------------------------------------------------
# 1. Validate scoring eligibility
# ---------------------------------------------------------------------

print()
print("SCORING ELIGIBILITY")
print("-" * 100)

expected_answered_n = (
    df[human_values]
    .notna()
    .sum(axis=1)
)

answered_valid = (
    df["hv_answered_n"]
    .equals(expected_answered_n)
)

expected_complete_flag = (
    expected_answered_n == 21
)

complete_valid = (
    df["hv_complete_flag"]
    .equals(expected_complete_flag)
)

eligible_valid = (
    df["hv_scoring_eligible"]
    .equals(df["hv_complete_flag"])
)

print(
    f"Answered-item count: "
    f"{'PASSED' if answered_valid else 'FAILED'}"
)

print(
    f"Complete-case flag:  "
    f"{'PASSED' if complete_valid else 'FAILED'}"
)

print(
    f"Eligibility flag:    "
    f"{'PASSED' if eligible_valid else 'FAILED'}"
)

if not all([
    answered_valid,
    complete_valid,
    eligible_valid,
]):
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Validate reverse scoring
# ---------------------------------------------------------------------

print()
print("REVERSE SCORING")
print("-" * 100)

reverse_valid = True

for variable in human_values:

    reversed_variable = f"{variable}_r"

    expected = (
        7 - df[variable].astype("Float64")
    )

    valid = np.allclose(
        df[reversed_variable].astype(float),
        expected.astype(float),
        equal_nan=True,
    )

    print(
        f"{reversed_variable:<15} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        reverse_valid = False
        validation_passed = False


# ---------------------------------------------------------------------
# 3. Validate MRAT
# ---------------------------------------------------------------------

print()
print("MRAT")
print("-" * 100)

eligible = df["hv_scoring_eligible"]

expected_mrat = (
    df.loc[
        eligible,
        reversed_variables
    ]
    .mean(axis=1)
)

stored_mrat = df.loc[
    eligible,
    "hv_mrat"
]

mrat_valid = np.allclose(
    stored_mrat.astype(float),
    expected_mrat.astype(float),
    equal_nan=True,
)

ineligible_mrat_missing = (
    df.loc[
        ~eligible,
        "hv_mrat"
    ]
    .isna()
    .all()
)

print(
    f"MRAT calculation:       "
    f"{'PASSED' if mrat_valid else 'FAILED'}"
)

print(
    f"Ineligible MRAT = NaN:  "
    f"{'PASSED' if ineligible_mrat_missing else 'FAILED'}"
)

if not mrat_valid or not ineligible_mrat_missing:
    validation_passed = False


# ---------------------------------------------------------------------
# 4. Validate the 10 basic values
# ---------------------------------------------------------------------

print()
print("BASIC HUMAN VALUES")
print("-" * 100)

for basic_value, group in mapping.groupby("basic_value"):

    score_name = make_slug(basic_value)

    item_variables = [
        f"{variable}_r"
        for variable in group["variable"]
    ]

    raw_column = (
        f"hv_{score_name}_raw"
    )

    centered_column = (
        f"hv_{score_name}_centered"
    )

    expected_raw = (
        df.loc[
            eligible,
            item_variables
        ]
        .mean(axis=1)
    )

    expected_centered = (
        expected_raw
        - df.loc[
            eligible,
            "hv_mrat"
        ]
    )

    raw_valid = np.allclose(
        df.loc[
            eligible,
            raw_column
        ].astype(float),
        expected_raw.astype(float),
        equal_nan=True,
    )

    centered_valid = np.allclose(
        df.loc[
            eligible,
            centered_column
        ].astype(float),
        expected_centered.astype(float),
        equal_nan=True,
    )

    incomplete_missing = (
        df.loc[
            ~eligible,
            [raw_column, centered_column]
        ]
        .isna()
        .all()
        .all()
    )

    valid = (
        raw_valid
        and centered_valid
        and incomplete_missing
    )

    print(
        f"{basic_value:<20} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate the four higher-order values
# ---------------------------------------------------------------------

print()
print("HIGHER-ORDER HUMAN VALUES")
print("-" * 100)

higher_score_info = []

for higher_value, group in mapping.groupby(
    "higher_order_value"
):

    score_name = make_slug(higher_value)

    item_variables = [
        f"{variable}_r"
        for variable in group["variable"]
    ]

    raw_column = (
        f"hv_{score_name}_raw"
    )

    centered_column = (
        f"hv_{score_name}_centered"
    )

    expected_raw = (
        df.loc[
            eligible,
            item_variables
        ]
        .mean(axis=1)
    )

    expected_centered = (
        expected_raw
        - df.loc[
            eligible,
            "hv_mrat"
        ]
    )

    raw_valid = np.allclose(
        df.loc[
            eligible,
            raw_column
        ].astype(float),
        expected_raw.astype(float),
        equal_nan=True,
    )

    centered_valid = np.allclose(
        df.loc[
            eligible,
            centered_column
        ].astype(float),
        expected_centered.astype(float),
        equal_nan=True,
    )

    incomplete_missing = (
        df.loc[
            ~eligible,
            [raw_column, centered_column]
        ]
        .isna()
        .all()
        .all()
    )

    valid = (
        raw_valid
        and centered_valid
        and incomplete_missing
    )

    print(
        f"{higher_value:<22} "
        f"{'PASSED' if valid else 'FAILED'}"
    )

    if not valid:
        validation_passed = False

    higher_score_info.append(
        {
            "column": centered_column,
            "item_count": len(group),
        }
    )


# ---------------------------------------------------------------------
# 6. Validate weighted centering identity
# ---------------------------------------------------------------------

print()
print("CENTERING IDENTITY")
print("-" * 100)

weighted_sum = pd.Series(
    0.0,
    index=df.index,
)

for score in higher_score_info:

    weighted_sum = (
        weighted_sum
        + (
            df[score["column"]]
            * score["item_count"]
        )
    )

weighted_sum_complete = weighted_sum[
    eligible
]

max_absolute_error = (
    weighted_sum_complete
    .abs()
    .max()
)

centering_identity_valid = np.isclose(
    max_absolute_error,
    0.0,
    atol=1e-10,
)

print(
    f"Maximum absolute weighted error: "
    f"{max_absolute_error:.12f}"
)

print(
    f"Weighted centered sum = 0:       "
    f"{'PASSED' if centering_identity_valid else 'FAILED'}"
)

if not centering_identity_valid:
    validation_passed = False


# ---------------------------------------------------------------------
# 7. Validate score ranges
# ---------------------------------------------------------------------

print()
print("SCORE RANGES")
print("-" * 100)

range_valid = True

for score in higher_score_info:

    centered_column = score["column"]

    raw_column = centered_column.replace(
        "_centered",
        "_raw"
    )

    raw_values = df[
        raw_column
    ].dropna()

    raw_ok = (
        raw_values.between(1, 6).all()
    )

    centered_values = df[
        centered_column
    ].dropna()

    centered_ok = (
        centered_values.between(-5, 5).all()
    )

    valid = raw_ok and centered_ok

    print(
        f"{raw_column:<35} "
        f"{'PASSED' if raw_ok else 'FAILED'}"
    )

    print(
        f"{centered_column:<35} "
        f"{'PASSED' if centered_ok else 'FAILED'}"
    )

    if not valid:
        range_valid = False
        validation_passed = False


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if validation_passed:
    print("Human Values score validation PASSED.")
else:
    print("Human Values score validation requires review.")

print()
print("No respondent data were modified.")