
import requests


# ============================================================
# STEP 32B — INSPECT TWO ADDITIONAL EUROSTAT INDICATORS
# ============================================================

# Purpose:
# Inspect the official Eurostat API structure for:
# 1. Severe material and social deprivation.
# 2. Annual HICP inflation.
#
# Before using an indicator, we must identify its exact
# dimensions, category codes, units and observation values.
#
# This prevents selecting the wrong statistical measure.
#
# This script only reads data from the Eurostat API.
# No local files are created or modified.


# ============================================================
# 1. DEFINE THE EUROSTAT API ENDPOINT
# ============================================================

# Eurostat provides access to statistical datasets
# through an official REST API.
#
# The response format is JSON-stat 2.0, which stores
# observations alongside their statistical dimensions.

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
)


# ============================================================
# 2. DEFINE THE TWO CANDIDATE DATASETS
# ============================================================

# We use the dataset codes provided by Eurostat.
#
# These are candidates for our socioeconomic V1 layer.
# Their definitive inclusion depends on data coverage
# and methodological comparability.

DATASETS = {
    "Severe material and social deprivation": "ilc_mdsd11",
    "Annual HICP inflation": "tec00118",
}


# ============================================================
# 3. DEFINE A SMALL DIAGNOSTIC REQUEST
# ============================================================

# Request observations for Germany in 2023.
#
# geo = DE: Germany
# time = 2023: calendar year 2023
# freq = A: annual frequency
#
# We deliberately do not specify age, sex or unit yet.
# We first want to discover the available categories.
#
# Germany is a test country, not the final project scope.

PARAMS = {
    "lang": "EN",
    "geo": "DE",
    "time": "2023",
    "freq": "A",
}


# ============================================================
# 4. READ CATEGORY CODES FROM JSON-STAT
# ============================================================

def get_categories(dimension):

    # Every statistical dimension contains categories.
    #
    # For example, a sex dimension may contain:
    # T = Total
    # M = Males
    # F = Females
    #
    # The JSON-stat category index describes the order
    # in which those categories appear in the dataset.

    category = dimension.get("category", {})

    index = category.get("index")
    labels = category.get("label", {})

    # The index can be represented as a dictionary
    # mapping category codes to numeric positions.

    if isinstance(index, dict):

        codes = [
            code
            for code, position in sorted(
                index.items(),
                key=lambda item: item[1],
            )
        ]

    # Alternatively, JSON-stat may provide
    # an already ordered list of category codes.

    elif isinstance(index, list):

        codes = index

    else:

        raise ValueError(
            "Unsupported or missing category index."
        )

    # Create a readable list of code-label pairs.
    #
    # We preserve the exact codes because these
    # will later be used in API filters.

    return [
        {
            "code": code,
            "label": labels.get(code, code),
        }
        for code in codes
    ]


# ============================================================
# 5. REQUEST AND VALIDATE ONE DATASET
# ============================================================

def inspect_dataset(name, dataset_code):

    # Construct the URL for the requested dataset.

    url = BASE_URL + dataset_code

    print("\n" + "=" * 100)
    print(f"INDICATOR: {name}")
    print(f"DATASET:   {dataset_code}")
    print("=" * 100)

    # Send a GET request to the Eurostat API.
    #
    # timeout=(10, 60):
    # - Maximum 10 seconds to establish the connection.
    # - Maximum 60 seconds to receive the response.
    #
    # This avoids waiting indefinitely for an API response.

    try:

        response = requests.get(
            url,
            params=PARAMS,
            timeout=(10, 60),
        )

        # Stop if Eurostat returns an HTTP error.
        # An HTTP failure must not be treated as missing data.

        response.raise_for_status()

        # Convert the JSON response to a Python dictionary.

        data = response.json()

    except requests.RequestException as error:

        raise RuntimeError(
            f"API request failed for {dataset_code}: {error}"
        ) from error

    except ValueError as error:

        raise ValueError(
            f"{dataset_code}: Invalid JSON response."
        ) from error

    print(f"\nHTTP status: {response.status_code}")
    print(f"Request URL: {response.url}")


    # --------------------------------------------------------
    # 5.1 VALIDATE THE RESPONSE STRUCTURE
    # --------------------------------------------------------

    # JSON-stat identifies statistical datasets
    # using the class value "dataset".

    if data.get("class") != "dataset":
        raise ValueError(
            f"{dataset_code}: Unexpected JSON-stat class."
        )

    dimension_ids = data.get("id")
    dimension_sizes = data.get("size")
    dimensions = data.get("dimension")

    # Dimension IDs represent variables such as:
    # freq, geo, time, age, sex or unit.
    #
    # Dimension sizes describe the number of categories.

    if not isinstance(dimension_ids, list):
        raise ValueError("Missing dimension IDs.")

    if not isinstance(dimension_sizes, list):
        raise ValueError("Missing dimension sizes.")

    if not isinstance(dimensions, dict):
        raise ValueError("Missing dimension metadata.")

    # Every dimension must have exactly one declared size.

    if len(dimension_ids) != len(dimension_sizes):
        raise ValueError(
            "Dimension IDs and sizes do not match."
        )

    if len(set(dimension_ids)) != len(dimension_ids):
        raise ValueError("Duplicate dimension IDs.")


    # --------------------------------------------------------
    # 5.2 DISPLAY DATASET METADATA
    # --------------------------------------------------------

    # The dataset title documents what is being measured.
    # The update timestamp helps track data revisions.

    print("\nDATASET METADATA")
    print("-" * 100)

    print(f"Title:   {data.get('label')}")
    print(f"Source:  {data.get('source')}")
    print(f"Updated: {data.get('updated')}")
    print(f"Version: {data.get('version')}")

    print("\nDIMENSION STRUCTURE")
    print("-" * 100)

    print(f"IDs:   {dimension_ids}")
    print(f"Sizes: {dimension_sizes}")


    # --------------------------------------------------------
    # 5.3 INSPECT EACH STATISTICAL DIMENSION
    # --------------------------------------------------------

    # We iterate through every dimension returned
    # by the API rather than assuming its structure.
    #
    # This is important because the two new datasets
    # may have different dimensions from the first four.

    for dimension_id, expected_size in zip(
        dimension_ids,
        dimension_sizes,
    ):

        if dimension_id not in dimensions:
            raise ValueError(
                f"Missing dimension: {dimension_id}"
            )

        dimension = dimensions[dimension_id]

        categories = get_categories(dimension)

        # Check that the number of categories agrees
        # with the JSON-stat dimension size.

        if len(categories) != expected_size:
            raise ValueError(
                f"Category size mismatch: {dimension_id}"
            )

        print()
        print(
            f"[{dimension_id}] "
            f"{dimension.get('label', dimension_id)}"
        )

        print(
            f"Available categories: {len(categories)}"
        )

        # Display category codes and official labels.
        #
        # These are the values that we will use
        # later to filter the correct indicator.

        for item in categories[:25]:

            print(
                f"  {item['code']:<22} "
                f"{item['label']}"
            )

        if len(categories) > 25:

            remaining = len(categories) - 25

            print(
                f"  ... {remaining} additional categories"
            )


    # --------------------------------------------------------
    # 5.4 INSPECT OBSERVATION AVAILABILITY
    # --------------------------------------------------------

    # JSON-stat stores numeric observations in "value".
    #
    # Values may be represented as:
    # - A dictionary indexed by numeric cell position.
    # - A list of values ordered by cell position.
    #
    # Missing values are not the same as zero.

    values = data.get("value")

    if isinstance(values, dict):

        observations = list(values.values())

    elif isinstance(values, list):

        observations = values

    elif values is None:

        observations = []

    else:

        raise ValueError(
            "Unsupported observation value structure."
        )

    non_missing = sum(
        value is not None
        for value in observations
    )

    print("\nOBSERVATION SUMMARY")
    print("-" * 100)

    print(f"Non-missing values: {non_missing}")

    print(
        "Note: this count applies only to "
        "Germany in 2023."
    )

    print(
        "The selected statistical series and "
        "29-country coverage are not yet validated."
    )


# ============================================================
# 6. RUN THE INSPECTION
# ============================================================

def main():

    print("=" * 100)
    print("V1 EUROSTAT ADDITIONAL INDICATOR INSPECTION")
    print("=" * 100)

    print("\nDiagnostic filters:")
    print(PARAMS)

    # Inspect both datasets independently.
    #
    # We do not continue to Step 33 until their
    # exact dimension codes have been reviewed.

    for name, dataset_code in DATASETS.items():

        inspect_dataset(
            name,
            dataset_code,
        )

    print("\n" + "=" * 100)
    print("ADDITIONAL INDICATOR INSPECTION COMPLETED")
    print("=" * 100)

    print("Both API responses passed structural validation.")
    print("No files were created or modified.")
    print("Country coverage remains to be investigated.")


# Execute main() only when running this script directly.

if __name__ == "__main__":
    main()
