
from pathlib import Path

import requests


# ============================================================
# STEP 32 — INSPECT EUROSTAT API DATASETS
# ============================================================

# Purpose:
# Test the official Eurostat Statistics API and inspect
# the dimensions available for our four candidate datasets.
#
# This step helps us identify:
# - Dataset codes and descriptive titles.
# - Dimension names, such as age, sex and unit.
# - Available category codes within each dimension.
# - The structure of the returned JSON-stat response.
#
# We do not yet select the final socioeconomic indicators.
# We do not modify any existing ESS or Eurostat files.


# ============================================================
# 1. DEFINE PROJECT AND API CONFIGURATION
# ============================================================

# Locate the project root for consistency with other scripts.
ROOT = Path(__file__).resolve().parents[1]

# Define the official Eurostat Statistics API endpoint.
BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
)

# These are our candidate Eurostat datasets.
# Each dataset may contain several statistical breakdowns.

DATASETS = {
    "GDP per capita in PPS": "tec00114",
    "Employment": "lfsi_emp_a",
    "Unemployment": "une_rt_a",
    "Income inequality - Gini": "ilc_di12",
}

# Use Germany and 2023 for a small diagnostic request.
#
# Filtering the query avoids retrieving the complete
# historical dataset at this early stage.
#
# This is only an API test. It is not our final country
# coverage or temporal alignment selection.

PARAMS = {
    "lang": "EN",
    "geo": "DE",
    "time": "2023",
    "freq": "A",
}


# ============================================================
# 2. DEFINE API RESPONSE VALIDATION
# ============================================================

def validate_response(data, dataset_code):

    # A JSON-stat statistical response must identify
    # itself as a dataset.

    if data.get("class") != "dataset":
        raise ValueError(
            f"{dataset_code}: Invalid JSON-stat dataset class."
        )

    # JSON-stat describes its dimensions using the
    # id and size arrays.
    #
    # Example:
    # id   = ["freq", "unit", "geo", "time"]
    # size = [1, 1, 1, 1]

    dimension_ids = data.get("id")
    dimension_sizes = data.get("size")

    if not isinstance(dimension_ids, list):
        raise ValueError(
            f"{dataset_code}: Missing dimension IDs."
        )

    if not isinstance(dimension_sizes, list):
        raise ValueError(
            f"{dataset_code}: Missing dimension sizes."
        )

    # Every dimension must have a corresponding size.

    if len(dimension_ids) != len(dimension_sizes):
        raise ValueError(
            f"{dataset_code}: Dimension structure mismatch."
        )

    # The response must contain the dimension metadata.

    if not isinstance(data.get("dimension"), dict):
        raise ValueError(
            f"{dataset_code}: Missing dimension metadata."
        )

    if len(set(dimension_ids)) != len(dimension_ids):
        raise ValueError(
            f"{dataset_code}: Duplicate dimension IDs."
        )

    for dimension_id, dimension_size in zip(
        dimension_ids,
        dimension_sizes,
    ):

        # Check that every declared dimension exists.

        if dimension_id not in data["dimension"]:
            raise ValueError(
                f"{dataset_code}: Missing dimension "
                f"{dimension_id}."
            )

        # The number of positions must be positive.

        if (
            not isinstance(dimension_size, int)
            or isinstance(dimension_size, bool)
            or dimension_size <= 0
        ):
            raise ValueError(
                f"{dataset_code}: Invalid dimension size "
                f"for {dimension_id}."
            )


# ============================================================
# 3. EXTRACT CATEGORY CODES FROM JSON-STAT
# ============================================================

def get_categories(dimension_data):

    # Each JSON-stat dimension contains a category index.
    #
    # The index links category codes to their positions.
    #
    # Example:
    # "T" = Total
    # "M" = Males
    # "F" = Females
    #
    # These are examples, not assumed dataset values.

    category = dimension_data.get("category", {})

    index = category.get("index", {})
    labels = category.get("label", {})

    # JSON-stat can represent category indexes
    # either as a dictionary or as an ordered list.

    if isinstance(index, dict):

        ordered_codes = [
            code
            for code, position in sorted(
                index.items(),
                key=lambda item: item[1],
            )
        ]

    elif isinstance(index, list):

        ordered_codes = index

    else:

        raise ValueError(
            "Unsupported JSON-stat category index."
        )

    # Return each category with its official label.

    return [
        {
            "code": code,
            "label": labels.get(code, code),
        }
        for code in ordered_codes
    ]


# ============================================================
# 4. REQUEST AND INSPECT ONE EUROSTAT DATASET
# ============================================================

def inspect_dataset(name, dataset_code):

    # Construct the official dataset endpoint.
    url = BASE_URL + dataset_code

    print("\n" + "=" * 110)
    print(f"DATASET: {name}")
    print(f"CODE:    {dataset_code}")
    print("=" * 110)

    print(f"\nAPI endpoint: {url}")

    # Send a filtered GET request.
    #
    # timeout=(10, 60):
    # - Wait up to 10 seconds to establish the connection.
    # - Wait up to 60 seconds for the response.
    #
    # A timeout prevents the script from waiting forever.

    try:

        response = requests.get(
            url,
            params=PARAMS,
            timeout=(10, 60),
        )

        # HTTP errors should stop the diagnostic.
        # They must not be interpreted as missing data.

        response.raise_for_status()

        # Decode the JSON-stat response into a dictionary.
        data = response.json()

    except requests.RequestException as error:

        raise RuntimeError(
            f"Eurostat API request failed for "
            f"{dataset_code}: {error}"
        ) from error

    except ValueError as error:

        raise ValueError(
            f"{dataset_code}: Response was not valid JSON."
        ) from error

    print(f"HTTP status: {response.status_code}")
    print(f"Request URL: {response.url}")

    # Check the structural integrity of the response.
    validate_response(data, dataset_code)

    print("\nDATASET METADATA")
    print("-" * 110)

    print(f"Title:   {data.get('label')}")
    print(f"Source:  {data.get('source')}")
    print(f"Updated: {data.get('updated')}")
    print(f"Version: {data.get('version')}")


    # --------------------------------------------------------
    # 4.1 INSPECT DIMENSIONS
    # --------------------------------------------------------

    print("\nDATASET DIMENSIONS")
    print("-" * 110)

    dimension_ids = data["id"]
    dimension_sizes = data["size"]

    print(f"Dimension IDs: {dimension_ids}")
    print(f"Dimension sizes: {dimension_sizes}")

    for dimension_id, expected_size in zip(
        dimension_ids,
        dimension_sizes,
    ):

        dimension_data = data["dimension"][dimension_id]

        dimension_label = dimension_data.get(
            "label",
            dimension_id,
        )

        categories = get_categories(dimension_data)

        # Verify consistency between category metadata
        # and the declared dimension size.

        if len(categories) != expected_size:
            raise ValueError(
                f"{dataset_code}: Category count mismatch "
                f"for dimension {dimension_id}."
            )

        print()
        print(
            f"[{dimension_id}] {dimension_label}"
        )

        print(
            f"Available categories: {len(categories)}"
        )

        # Display up to 15 category codes to keep
        # the diagnostic readable.
        #
        # The full response still contains all categories
        # included by the filtered API request.

        for item in categories[:15]:

            print(
                f"  {item['code']:<20} "
                f"{item['label']}"
            )

        if len(categories) > 15:

            print(
                f"  ... {len(categories) - 15} "
                f"additional categories not displayed"
            )


    # --------------------------------------------------------
    # 4.2 INSPECT DATA AVAILABILITY IN THE TEST RESPONSE
    # --------------------------------------------------------

    print("\nFILTERED RESPONSE DATA")
    print("-" * 110)

    # JSON-stat stores observations as either a list
    # or a dictionary indexed by cell position.
    #
    # We count returned non-missing values.
    #
    # These counts describe only Germany in 2023,
    # not the availability across all V1 countries.

    values = data.get("value", {})

    if isinstance(values, dict):
        observations = list(values.values())

    elif isinstance(values, list):
        observations = values

    elif values is None:
        observations = []

    else:
        raise ValueError(
            f"{dataset_code}: Unsupported value structure."
        )

    available_values = sum(
        value is not None
        for value in observations
    )

    print(
        f"Non-missing observations in test response: "
        f"{available_values}"
    )

    print(
        "Coverage across all V1 countries "
        "has not yet been evaluated."
    )


# ============================================================
# 5. EXECUTE THE DIAGNOSTIC
# ============================================================

def main():

    print("=" * 110)
    print("V1 EUROSTAT API DATASET INSPECTION")
    print("=" * 110)

    print("\nDiagnostic filters:")
    print(PARAMS)

    # Inspect each candidate dataset separately.
    #
    # The script will stop if an API response fails
    # validation, allowing us to investigate the issue.

    for name, dataset_code in DATASETS.items():

        inspect_dataset(
            name,
            dataset_code,
        )

    print("\n" + "=" * 110)
    print("EUROSTAT API INSPECTION COMPLETED")
    print("=" * 110)

    print(
        "All four candidate datasets responded "
        "and passed structural validation."
    )

    print(
        "No files were created or modified."
    )

    print(
        "Final units, reference periods and "
        "country coverage remain to be validated."
    )


if __name__ == "__main__":
    main()
