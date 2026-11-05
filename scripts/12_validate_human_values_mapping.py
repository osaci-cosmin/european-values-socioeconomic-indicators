from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input files
data_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_scope.parquet"
)

mapping_file = (
    project_root
    / "config"
    / "human_values_mapping.csv"
)


# Check that input files exist
if not data_file.exists():
    raise FileNotFoundError(
        f"V1 dataset not found: {data_file}"
    )

if not mapping_file.exists():
    raise FileNotFoundError(
        f"Human Values mapping not found: {mapping_file}"
    )


# Load data
df = pd.read_parquet(data_file)

mapping = pd.read_csv(mapping_file)


print("=" * 100)
print("ESS ROUND 11 HUMAN VALUES MAPPING VALIDATION")
print("=" * 100)

validation_passed = True


# ---------------------------------------------------------------------
# 1. Validate number of Human Values items
# ---------------------------------------------------------------------

print()
print("ITEM COUNT")
print("-" * 100)

item_count = len(mapping)

print(f"Configured items: {item_count}")

if item_count == 21:
    print("Status:           PASSED")
else:
    print("Status:           FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 2. Validate item numbering
# ---------------------------------------------------------------------

print()
print("ITEM NUMBERING")
print("-" * 100)

expected_numbers = set(range(1, 22))
actual_numbers = set(mapping["item_number"])

missing_numbers = expected_numbers - actual_numbers
extra_numbers = actual_numbers - expected_numbers

print(f"Missing item numbers: {sorted(missing_numbers)}")
print(f"Extra item numbers:   {sorted(extra_numbers)}")

if not missing_numbers and not extra_numbers:
    print("Status:               PASSED")
else:
    print("Status:               FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 3. Validate unique variables
# ---------------------------------------------------------------------

print()
print("VARIABLE UNIQUENESS")
print("-" * 100)

duplicate_variables = mapping["variable"].duplicated().sum()

print(f"Duplicate variables: {duplicate_variables}")

if duplicate_variables == 0:
    print("Status:              PASSED")
else:
    print("Status:              FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 4. Validate variables against ESS dataset
# ---------------------------------------------------------------------

print()
print("DATASET VARIABLE COVERAGE")
print("-" * 100)

configured_variables = set(mapping["variable"])
dataset_variables = set(df.columns)

missing_variables = configured_variables - dataset_variables

print(
    f"Variables missing from dataset: "
    f"{sorted(missing_variables)}"
)

if not missing_variables:
    print("Status:                         PASSED")
else:
    print("Status:                         FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 5. Validate the 10 basic values
# ---------------------------------------------------------------------

print()
print("BASIC VALUES")
print("-" * 100)

basic_counts = (
    mapping
    .groupby("basic_value")
    .size()
    .sort_index()
)

print(basic_counts.to_string())

expected_basic_counts = {
    "Achievement": 2,
    "Benevolence": 2,
    "Conformity": 2,
    "Hedonism": 2,
    "Power": 2,
    "Security": 2,
    "Self-Direction": 2,
    "Stimulation": 2,
    "Tradition": 2,
    "Universalism": 3,
}

basic_valid = (
    basic_counts.to_dict()
    == expected_basic_counts
)

print()

if basic_valid:
    print("Basic-value structure: PASSED")
else:
    print("Basic-value structure: FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 6. Validate higher-order values
# ---------------------------------------------------------------------

print()
print("HIGHER-ORDER VALUES")
print("-" * 100)

higher_counts = (
    mapping
    .groupby("higher_order_value")
    .size()
    .sort_index()
)

print(higher_counts.to_string())

expected_higher_counts = {
    "Conservation": 6,
    "Openness to Change": 6,
    "Self-Enhancement": 4,
    "Self-Transcendence": 5,
}

higher_valid = (
    higher_counts.to_dict()
    == expected_higher_counts
)

print()

if higher_valid:
    print("Higher-order structure: PASSED")
else:
    print("Higher-order structure: FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# 7. Validate reverse-score rule
# ---------------------------------------------------------------------

print()
print("SCORING CONFIGURATION")
print("-" * 100)

reverse_values = set(mapping["reverse_score"])

print(f"Reverse-score values: {sorted(reverse_values)}")

if reverse_values == {1}:
    print("Status:               PASSED")
else:
    print("Status:               FAILED")
    validation_passed = False


# ---------------------------------------------------------------------
# Final validation result
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)

if validation_passed:
    print("Human Values mapping validation PASSED.")
else:
    print("Human Values mapping validation requires review.")

print()
print("No Human Values scores were calculated.")
print("No respondent data were modified.")