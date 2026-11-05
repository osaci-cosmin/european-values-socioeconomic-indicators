
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
import requests


# ============================================================
# STEP 35 — INGEST EUROSTAT RAW DATA
# ============================================================

# Purpose:
# Download the six Eurostat statistical series defined
# in the validated indicator registry from Step 34.
#
# We will:
# 1. Read indicator definitions from a CSV configuration.
# 2. Build reproducible Eurostat API requests.
# 3. Download the original JSON-stat responses.
# 4. Validate their structure and selected dimensions.
# 5. Calculate a SHA-256 hash for each response.
# 6. Save original RAW data and provenance metadata.
#
# Important:
# No ESS files are modified.
# No Eurostat observations are cleaned or transformed.
# No missing values are replaced.
# No ESS-Eurostat joins are performed.


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

# __file__ represents this Python script.
# parents[1] moves from /scripts to the project root.

ROOT = Path(__file__).resolve().parents[1]


# This configuration was created and checked in Step 34.
# It contains the exact dataset codes and API filters.

REGISTRY_PATH = (
    ROOT / "config/eurostat_indicators_v1.csv"
)


# RAW responses are stored separately from ESS data.
#
# Keeping a distinct source folder helps preserve
# data lineage and simplifies later processing.

RAW_DIR = ROOT / "data/raw/eurostat"


# Define the official Eurostat Statistics API endpoint.

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
)


# ============================================================
# 2. LOAD AND VALIDATE THE INDICATOR REGISTRY
# ============================================================

def load_registry():

    # Stop if the registry does not exist.
    # We do not invent missing indicator definitions.

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Indicator registry not found: {REGISTRY_PATH}"
        )

    registry = pd.read_csv(REGISTRY_PATH)

    # Verify the fields needed for API ingestion
    # and provenance documentation.

    required_columns = [
        "indicator_id",
        "dataset_code",
        "label",
        "unit_label",
        "filters_json",
        "api_start_year",
        "api_end_year",
        "reference_year_offset",
        "reference_period_rule",
        "documentation_url",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in registry.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing registry columns: {missing_columns}"
        )

    # V1 currently uses six candidate indicators.
    # An unexpected count needs manual investigation.

    if len(registry) != 6:
        raise ValueError(
            "Expected exactly six indicator definitions."
        )

    # Every indicator must have a unique identifier.
    # This prevents two datasets from being saved
    # under the same logical indicator name.

    if (
        registry["indicator_id"].isna().any()
        or registry["indicator_id"].duplicated().any()
    ):
        raise ValueError(
            "Invalid or duplicate indicator identifiers."
        )

    # Verify that all dataset codes are present.

    if (
        registry["dataset_code"].isna().any()
        or registry["dataset_code"].duplicated().any()
    ):
        raise ValueError(
            "Invalid or duplicate Eurostat dataset codes."
        )

    # Return only the validated configuration structure.

    return registry


# ============================================================
# 3. PARSE AND VALIDATE API FILTERS
# ============================================================

def parse_filters(row):

    indicator_id = row["indicator_id"]

    # The registry stores filters as JSON text.
    #
    # Example:
    # {"freq": "A", "sex": "T", "age": "Y20-64"}
    #
    # json.loads converts this text into a Python
    # dictionary that requests can use as parameters.

    filters = json.loads(row["filters_json"])

    if not isinstance(filters, dict) or not filters:
        raise ValueError(
            f"{indicator_id}: Invalid filter configuration."
        )

    # Our V1 indicators must all be annual series.

    if filters.get("freq") != "A":
        raise ValueError(
            f"{indicator_id}: Expected annual frequency."
        )

    # Prevent accidental inclusion of geographic
    # or temporal filters that conflict with the
    # country-level and year-window selection below.

    forbidden = {
        "geo",
        "geoLevel",
        "time",
        "sinceTimePeriod",
        "untilTimePeriod",
    }

    if forbidden.intersection(filters):
        raise ValueError(
            f"{indicator_id}: Conflicting API filters."
        )

    # Eurostat dimension filters use code strings.
    # Missing or unexpected value types indicate
    # a malformed configuration file.

    for dimension, value in filters.items():

        if (
            not isinstance(dimension, str)
            or not isinstance(value, str)
            or not dimension
            or not value
        ):
            raise ValueError(
                f"{indicator_id}: Invalid filter values."
            )

    return filters


# ============================================================
# 4. READ JSON-STAT CATEGORY CODES
# ============================================================

def get_categories(dimension):

    # JSON-stat represents each statistical dimension
    # using category codes and their numeric positions.
    #
    # We need these positions to validate that the API
    # returned the specific measurement we requested.

    index = dimension["category"]["index"]

    if isinstance(index, dict):

        return [
            code
            for code, position in sorted(
                index.items(),
                key=lambda item: item[1],
            )
        ]

    if isinstance(index, list):
        return index

    raise ValueError(
        "Unsupported JSON-stat category index."
    )


# ============================================================
# 5. VALIDATE THE DOWNLOADED EUROSTAT RESPONSE
# ============================================================

def validate_dataset(data, indicator_id, filters,
                     start_year, end_year):

    # A successful HTTP request does not automatically
    # guarantee that the response contains a valid
    # statistical dataset.
    #
    # Eurostat may also return an API warning or error.
    # Therefore, we inspect the JSON-stat structure.

    if data.get("class") != "dataset":
        raise ValueError(
            f"{indicator_id}: Expected a JSON-stat dataset. "
            f"Response: {str(data)[:300]}"
        )

    if data.get("version") != "2.0":
        raise ValueError(
            f"{indicator_id}: Unexpected JSON-stat version."
        )

    ids = data.get("id")
    sizes = data.get("size")
    dimensions = data.get("dimension")

    if (
        not isinstance(ids, list)
        or not isinstance(sizes, list)
        or not isinstance(dimensions, dict)
    ):
        raise ValueError(
            f"{indicator_id}: Invalid dimension structure."
        )

    if len(ids) != len(sizes):
        raise ValueError(
            f"{indicator_id}: Dimension size mismatch."
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"{indicator_id}: Duplicate dimension IDs."
        )

    # The response should contain precisely:
    # - The selected statistical dimensions.
    # - A geographic dimension.
    # - A time dimension.
    #
    # Unexpected dimensions could cause us to mix
    # multiple measures within one indicator.

    expected_dimensions = (
        set(filters) | {"geo", "time"}
    )

    if set(ids) != expected_dimensions:
        raise ValueError(
            f"{indicator_id}: Unexpected dimensions: {ids}"
        )

    categories = {}

    for dimension_id, expected_size in zip(ids, sizes):

        if dimension_id not in dimensions:
            raise ValueError(
                f"{indicator_id}: Missing {dimension_id}."
            )

        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise ValueError(
                f"{indicator_id}: Invalid dimension size."
            )

        codes = get_categories(
            dimensions[dimension_id]
        )

        # The category count should agree with
        # the size declared in the response.

        if (
            len(codes) != expected_size
            or len(set(codes)) != len(codes)
        ):
            raise ValueError(
                f"{indicator_id}: Invalid category structure "
                f"for {dimension_id}."
            )

        categories[dimension_id] = codes

    # Check each exact indicator filter.
    #
    # Example:
    # employment_rate must have age = Y20-64,
    # rather than another age group.

    for dimension_id, expected_code in filters.items():

        if categories[dimension_id] != [expected_code]:
            raise ValueError(
                f"{indicator_id}: Unexpected categories "
                f"for {dimension_id}."
            )

    # Verify the returned time period.
    #
    # We should never save an unexpected historical
    # period under the requested 2023-2024 label.

    returned_years = [
        int(year)
        for year in categories["time"]
    ]

    if not returned_years:
        raise ValueError(
            f"{indicator_id}: No years returned."
        )

    if any(
        year < start_year or year > end_year
        for year in returned_years
    ):
        raise ValueError(
            f"{indicator_id}: Unexpected years returned."
        )

    # Require country-level geographic categories.

    countries = categories["geo"]

    if not countries:
        raise ValueError(
            f"{indicator_id}: No countries returned."
        )

    # A response may contain missing observations.
    # That is acceptable because missingness is
    # a real property of statistical data.
    #
    # However, at least one numeric value should
    # exist in these previously tested series.

    values = data.get("value")

    if isinstance(values, dict):
        observations = list(values.values())

    elif isinstance(values, list):
        observations = values

    else:
        raise ValueError(
            f"{indicator_id}: Invalid observation container."
        )

    available_n = sum(
        value is not None
        for value in observations
    )

    if available_n == 0:
        raise ValueError(
            f"{indicator_id}: No observations available."
        )

    # Return diagnostic information for the metadata
    # file and the final console output.

    return {
        "countries_n": len(countries),
        "years": returned_years,
        "non_missing_values_n": available_n,
    }


# ============================================================
# 6. PRESERVE RAW DATA AND PROVENANCE METADATA
# ============================================================

def save_snapshot(indicator_id, response_bytes, metadata):

    # SHA-256 creates a fingerprint of the exact
    # bytes returned by the API.
    #
    # If even one byte changes, the hash will normally
    # change, allowing us to detect revised snapshots.

    sha256 = hashlib.sha256(
        response_bytes
    ).hexdigest()

    # Include part of the hash in the filename.
    #
    # This makes the RAW storage version-aware:
    # if Eurostat changes the response later,
    # the new content receives a different filename.

    file_stem = (
        f"{indicator_id}_"
        f"{metadata['api_start_year']}_"
        f"{metadata['api_end_year']}_"
        f"{sha256[:16]}"
    )

    raw_path = RAW_DIR / f"{file_stem}.json"

    metadata_path = (
        RAW_DIR / f"{file_stem}.metadata.json"
    )

    metadata["sha256"] = sha256
    metadata["raw_filename"] = raw_path.name

    # Save the exact downloaded bytes.
    #
    # The "xb" mode creates a new binary file and
    # refuses to overwrite a file that already exists.

    if raw_path.exists():

        existing_hash = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()

        if existing_hash != sha256:
            raise ValueError(
                f"Existing RAW file failed hash validation: "
                f"{raw_path}"
            )

        raw_status = "EXISTING_IDENTICAL"

    else:

        with raw_path.open("xb") as file:
            file.write(response_bytes)

        raw_status = "NEW_SNAPSHOT"

    # A metadata sidecar stores the information
    # needed to understand where the RAW data came from.
    #
    # We also preserve the original retrieval timestamp
    # if a metadata file already exists.

    if metadata_path.exists():

        existing_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if (
            existing_metadata.get("sha256") != sha256
            or existing_metadata.get("indicator_id")
            != indicator_id
        ):
            raise ValueError(
                "Existing metadata does not match "
                "the RAW snapshot."
            )

    else:

        # "x" prevents overwriting existing provenance.
        #
        # This matters for reproducibility because
        # the first retrieval date should be preserved.

        with metadata_path.open(
            "x",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

    return raw_path, metadata_path, raw_status


# ============================================================
# 7. DOWNLOAD ONE INDICATOR
# ============================================================

def ingest_indicator(session, row, registry_hash):

    indicator_id = row["indicator_id"]
    dataset_code = row["dataset_code"]

    filters = parse_filters(row)

    start_year = int(row["api_start_year"])
    end_year = int(row["api_end_year"])

    if start_year > end_year:
        raise ValueError(
            f"{indicator_id}: Invalid year interval."
        )

    # Build the exact API request from the registry.
    #
    # geoLevel=country excludes subnational regions
    # and European aggregates.
    #
    # The two time parameters restrict the response
    # to our relevant statistical years.

    params = {
        "lang": "EN",
        "format": "JSON",
        "geoLevel": "country",
        "sinceTimePeriod": str(start_year),
        "untilTimePeriod": str(end_year),
        **filters,
    }

    url = BASE_URL + dataset_code

    print("\n" + "=" * 100)
    print(f"INDICATOR: {indicator_id}")
    print(f"DATASET:   {dataset_code}")
    print("=" * 100)

    # Download the API response using one shared
    # HTTP session for the six requests.
    #
    # Network failures should stop the script rather
    # than result in incomplete or invented data.

    try:

        response = session.get(
            url,
            params=params,
            timeout=(10, 90),
        )

        response.raise_for_status()

        # Decode JSON only for validation.
        # We save response.content separately
        # to preserve the original response bytes.

        data = response.json()

    except requests.RequestException as error:

        raise RuntimeError(
            f"{indicator_id}: API request failed: {error}"
        ) from error

    except ValueError as error:

        raise ValueError(
            f"{indicator_id}: Invalid JSON response."
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{indicator_id}: Expected a JSON object."
        )

    # Validate the dataset before writing to disk.

    diagnostic = validate_dataset(
        data,
        indicator_id,
        filters,
        start_year,
        end_year,
    )

    # Record the time of retrieval in UTC.
    #
    # This is different from the Eurostat dataset
    # update date and from the ESS fieldwork year.

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    # Preserve source provenance, request filters,
    # dataset metadata and the configuration hash.
    #
    # The metadata is stored next to the RAW file
    # rather than being mixed with statistical values.

    metadata = {
        "indicator_id": indicator_id,
        "indicator_label": row["label"],
        "dataset_code": dataset_code,
        "unit_label": row["unit_label"],
        "source": "Eurostat - European Commission",
        "api_url": response.url,
        "request_filters": params,
        "retrieved_at_utc": retrieved_at,
        "api_dataset_updated": data.get("updated"),
        "api_dataset_title": data.get("label"),
        "api_dataset_source": data.get("source"),
        "jsonstat_version": data.get("version"),
        "http_status": response.status_code,
        "registry_sha256": registry_hash,
        "api_start_year": start_year,
        "api_end_year": end_year,
        "reference_year_offset": int(
            row["reference_year_offset"]
        ),
        "reference_period_rule": row[
            "reference_period_rule"
        ],
        "documentation_url": row[
            "documentation_url"
        ],
        "countries_returned": diagnostic["countries_n"],
        "years_returned": diagnostic["years"],
        "non_missing_values_n": diagnostic[
            "non_missing_values_n"
        ],
    }

    raw_path, metadata_path, status = save_snapshot(
        indicator_id,
        response.content,
        metadata,
    )

    print(f"HTTP status:          {response.status_code}")
    print(f"Countries returned:   {diagnostic['countries_n']}")
    print(f"Years returned:       {diagnostic['years']}")
    print(
        f"Non-missing values:   "
        f"{diagnostic['non_missing_values_n']}"
    )
    print(f"Eurostat updated:     {data.get('updated')}")
    print(f"Snapshot status:      {status}")
    print(f"RAW file:             {raw_path.name}")
    print(f"Metadata file:        {metadata_path.name}")

    return status


# ============================================================
# 8. RUN THE COMPLETE RAW INGESTION
# ============================================================

def main():

    print("=" * 100)
    print("V1 EUROSTAT RAW INGESTION")
    print("=" * 100)

    # First load the registry.
    # No API requests or RAW writes occur before
    # checking its basic structure.

    registry = load_registry()

    # Calculate a hash of the configuration file.
    #
    # This allows us to identify the exact registry
    # version used to generate the RAW snapshots.

    registry_hash = hashlib.sha256(
        REGISTRY_PATH.read_bytes()
    ).hexdigest()

    # Create the Eurostat RAW folder if necessary.

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nIndicators: {len(registry)}")
    print(f"Output folder: {RAW_DIR}")
    print(f"Registry SHA-256: {registry_hash}")

    # Reuse a single HTTP session.
    #
    # This is more efficient than opening an entirely
    # new connection for each of the six datasets.

    statuses = []

    with requests.Session() as session:

        for _, row in registry.iterrows():

            status = ingest_indicator(
                session,
                row,
                registry_hash,
            )

            statuses.append(status)

    # If the script reaches this point, all six
    # indicators passed response validation
    # and their snapshots were stored successfully.

    print("\n" + "=" * 100)
    print("EUROSTAT RAW INGESTION COMPLETED")
    print("=" * 100)

    print(f"Indicators processed: {len(statuses)}")
    print(
        f"New snapshots: "
        f"{statuses.count('NEW_SNAPSHOT')}"
    )
    print(
        f"Existing identical snapshots: "
        f"{statuses.count('EXISTING_IDENTICAL')}"
    )

    print()
    print("RAW responses and provenance metadata preserved.")
    print("No ESS respondent-level files were modified.")
    print("No Eurostat data cleaning or imputation performed.")
    print("No ESS-Eurostat integration performed.")


# ============================================================
# 9. EXECUTE THE INGESTION
# ============================================================

# Python executes main() only when this script
# is launched directly from the terminal.

if __name__ == "__main__":
    main()
