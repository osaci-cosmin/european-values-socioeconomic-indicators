
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, parse_qs
import hashlib
import json
import math

import pandas as pd


# ============================================================
# STEP 36 — VALIDATE EUROSTAT RAW SNAPSHOTS
# ============================================================

# Purpose:
# Independently validate the Eurostat RAW files
# downloaded and saved in Step 35.
#
# We verify:
# 1. Indicator registry integrity.
# 2. Existence of RAW and metadata files.
# 3. SHA-256 integrity of original response bytes.
# 4. Agreement between filenames and file hashes.
# 5. Agreement between registry and provenance metadata.
# 6. Exact API request parameters.
# 7. JSON-stat dimensions and category indexes.
# 8. Valid year ranges and statistical filters.
# 9. Observation counts and numeric value integrity.
#
# Why this is necessary:
# Successful API ingestion does not automatically
# prove that saved files are complete and consistent.
#
# Validation must read files back from disk rather
# than trusting only the Step 35 console output.
#
# This script never changes the original RAW files.
# It does not send new Eurostat API requests.


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

# __file__ identifies this script.
# parents[1] identifies the project root directory.

ROOT = Path(__file__).resolve().parents[1]

# The indicator registry defines our six statistical series.

REGISTRY_PATH = (
    ROOT / "config/eurostat_indicators_v1.csv"
)

# The RAW directory stores downloaded JSON-stat responses
# and their accompanying provenance metadata files.

RAW_DIR = ROOT / "data/raw/eurostat"

# This is the official endpoint used in Step 35.

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
)


# ============================================================
# 2. DEFINE A REUSABLE VALIDATION FUNCTION
# ============================================================

def check(condition, message):

    # Stop execution if a validation rule fails.
    #
    # We prefer an explicit error over accepting
    # potentially corrupted analytical inputs.

    if not bool(condition):
        raise ValueError(message)


# ============================================================
# 3. LOAD AND VALIDATE THE INDICATOR REGISTRY
# ============================================================

def load_registry():

    # The registry is the source of truth for the
    # indicator selection and API extraction settings.

    check(
        REGISTRY_PATH.exists(),
        f"Missing registry: {REGISTRY_PATH}",
    )

    registry = pd.read_csv(REGISTRY_PATH)

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

    # Verify that the configuration includes every
    # field needed for provenance validation.

    missing = [
        column
        for column in required_columns
        if column not in registry.columns
    ]

    check(
        not missing,
        f"Missing registry columns: {missing}",
    )

    # Our V1 registry currently contains six indicators.

    check(
        len(registry) == 6,
        "Expected exactly six indicators.",
    )

    # Indicator IDs and dataset codes must be unique.
    # Duplicate identifiers would make snapshot
    # selection ambiguous.

    for column in ["indicator_id", "dataset_code"]:

        check(
            registry[column].notna().all(),
            f"Missing values in {column}.",
        )

        check(
            not registry[column].duplicated().any(),
            f"Duplicate values in {column}.",
        )

    # Calculate the hash of the registry itself.
    #
    # Step 35 stored this hash in each metadata file.
    # We can therefore check which configuration
    # produced the downloaded snapshots.

    registry_hash = hashlib.sha256(
        REGISTRY_PATH.read_bytes()
    ).hexdigest()

    return registry, registry_hash


# ============================================================
# 4. VALIDATE CATEGORY INDEXES
# ============================================================

def get_category_codes(dimension, expected_size):

    # A JSON-stat dimension contains category codes.
    #
    # Example:
    # time -> 2023, 2024
    # geo  -> DE, FR, EL
    #
    # The category index also identifies the
    # position of every category inside the data cube.

    check(
        isinstance(dimension, dict),
        "Invalid JSON-stat dimension.",
    )

    category = dimension.get("category")

    check(
        isinstance(category, dict),
        "Missing JSON-stat category metadata.",
    )

    index = category.get("index")

    if isinstance(index, dict):

        # Category positions must be integers.
        #
        # Duplicate or missing positions would make
        # multidimensional observation decoding unsafe.

        positions = list(index.values())

        check(
            all(
                isinstance(position, int)
                and not isinstance(position, bool)
                for position in positions
            ),
            "Non-integer category positions.",
        )

        check(
            sorted(positions)
            == list(range(expected_size)),
            "Non-contiguous category positions.",
        )

        # Restore the original JSON-stat category order.

        codes = [
            code
            for code, position in sorted(
                index.items(),
                key=lambda item: item[1],
            )
        ]

    elif isinstance(index, list):

        # The list representation already contains
        # category codes in their correct order.

        codes = index

    else:

        raise ValueError(
            "Unsupported JSON-stat category index."
        )

    # Category count must equal dimension size.
    # Codes must not be duplicated.

    check(
        len(codes) == expected_size,
        "JSON-stat category count mismatch.",
    )

    check(
        all(
            isinstance(code, str)
            for code in codes
        ),
        "Invalid category code type.",
    )

    check(
        len(set(codes)) == len(codes),
        "Duplicate category codes.",
    )

    return codes


# ============================================================
# 5. VALIDATE OBSERVATION CONTAINERS
# ============================================================

def validate_cells(container, total_cells, name):

    # JSON-stat supports two observation structures:
    #
    # 1. Dense list:
    #    Each list position represents a data cube cell.
    #
    # 2. Sparse dictionary:
    #    Only selected cell positions are explicitly stored.
    #
    # Missing observations are not equal to zero.
    #
    # We verify that all stored cell positions are
    # valid for the declared multidimensional structure.

    if container is None:

        # A dataset may not have any status flags.
        # Numeric observations, however, are required
        # by a separate validation below.

        return 0

    if isinstance(container, dict):

        check(
            len(container) <= total_cells,
            f"{name}: Too many stored cells.",
        )

        for key, value in container.items():

            # JSON object keys represent flat positions.
            # The canonical form uses integer strings.

            check(
                isinstance(key, str) and key.isdecimal(),
                f"{name}: Invalid cell position: {key}",
            )

            position = int(key)

            check(
                0 <= position < total_cells,
                f"{name}: Cell position outside data cube.",
            )

    elif isinstance(container, list):

        # A dense list must contain one position
        # for every combination of categories.

        check(
            len(container) == total_cells,
            f"{name}: Incorrect dense array length.",
        )

    else:

        raise ValueError(
            f"{name}: Unsupported JSON-stat container."
        )

    # Numeric values and quality flags have
    # different validation requirements.

    values = (
        container.values()
        if isinstance(container, dict)
        else container
    )

    count = 0

    for value in values:

        if value is None:
            continue

        if name == "value":

            # Numeric observations must be finite.
            #
            # Zero is permitted because it can be
            # a legitimate statistical observation.

            check(
                isinstance(value, (int, float))
                and not isinstance(value, bool),
                "Non-numeric statistical observation.",
            )

            check(
                math.isfinite(float(value)),
                "Non-finite statistical observation.",
            )

        else:

            # Eurostat observation flags should be
            # retained for later quality analysis.

            check(
                isinstance(value, str),
                "Invalid observation status flag.",
            )

        count += 1

    return count


# ============================================================
# 6. VALIDATE THE RAW JSON-STAT RESPONSE
# ============================================================

def validate_raw_dataset(data, row):

    indicator_id = row["indicator_id"]

    # Decode the statistical filters recorded
    # in the validated indicator registry.

    filters = json.loads(row["filters_json"])

    check(
        isinstance(filters, dict),
        f"{indicator_id}: Invalid registry filters.",
    )

    check(
        filters.get("freq") == "A",
        f"{indicator_id}: Expected annual frequency.",
    )

    # Validate fundamental JSON-stat metadata.

    check(
        data.get("class") == "dataset",
        f"{indicator_id}: Invalid JSON-stat class.",
    )

    check(
        data.get("version") == "2.0",
        f"{indicator_id}: Invalid JSON-stat version.",
    )

    ids = data.get("id")
    sizes = data.get("size")
    dimensions = data.get("dimension")

    check(
        isinstance(ids, list)
        and isinstance(sizes, list)
        and isinstance(dimensions, dict),
        f"{indicator_id}: Invalid dimension metadata.",
    )

    check(
        len(ids) == len(sizes),
        f"{indicator_id}: ID/size mismatch.",
    )

    check(
        len(set(ids)) == len(ids),
        f"{indicator_id}: Duplicate dimensions.",
    )

    # Each downloaded series should contain only
    # the selected statistical dimensions plus
    # geography and time.

    expected_dimensions = (
        set(filters) | {"geo", "time"}
    )

    check(
        set(ids) == expected_dimensions,
        f"{indicator_id}: Unexpected dimensions.",
    )

    categories = {}

    for dimension_id, size in zip(ids, sizes):

        check(
            isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0,
            f"{indicator_id}: Invalid dimension size.",
        )

        check(
            dimension_id in dimensions,
            f"{indicator_id}: Missing dimension.",
        )

        categories[dimension_id] = get_category_codes(
            dimensions[dimension_id],
            size,
        )

    # Independently confirm the category filters.
    #
    # This is particularly important for variables
    # with several measurement units or age groups.

    for dimension_id, expected_code in filters.items():

        check(
            categories[dimension_id] == [expected_code],
            f"{indicator_id}: Incorrect statistical filter "
            f"for {dimension_id}.",
        )

    # Verify that every returned year belongs to
    # the period requested in the registry.

    start_year = int(row["api_start_year"])
    end_year = int(row["api_end_year"])

    expected_years = list(
        range(start_year, end_year + 1)
    )

    actual_years = [
        int(year)
        for year in categories["time"]
    ]

    check(
        actual_years == expected_years,
        f"{indicator_id}: Unexpected year coverage.",
    )

    # Every dimension contributes to the number
    # of possible cells in the statistical cube.
    #
    # Example:
    # 40 countries x 2 years = 80 possible cells.
    #
    # Some cells may legitimately have missing values.

    total_cells = math.prod(sizes)

    value_container = data.get("value")
    status_container = data.get("status")

    # A statistical dataset must have an actual
    # observation container.

    check(
        isinstance(value_container, (dict, list)),
        f"{indicator_id}: Missing value container.",
    )

    # Count valid numeric observations directly
    # from the saved RAW data.

    values_n = validate_cells(
        value_container,
        total_cells,
        "value",
    )

    # Validate observation flags separately.
    #
    # Flags do not automatically invalidate
    # numeric observations.

    flags_n = validate_cells(
        status_container,
        total_cells,
        "status",
    )

    check(
        values_n > 0,
        f"{indicator_id}: No numeric observations.",
    )

    return {
        "countries_n": len(categories["geo"]),
        "years": actual_years,
        "total_cells": total_cells,
        "values_n": values_n,
        "flags_n": flags_n,
    }


# ============================================================
# 7. VALIDATE REQUEST PROVENANCE
# ============================================================

def validate_provenance(metadata, row, registry_hash):

    indicator_id = row["indicator_id"]

    # Verify that the sidecar describes the same
    # statistical indicator as the current registry.

    expected_fields = {
        "indicator_id": indicator_id,
        "indicator_label": row["label"],
        "dataset_code": row["dataset_code"],
        "unit_label": row["unit_label"],
        "source": "Eurostat - European Commission",
        "registry_sha256": registry_hash,
        "api_start_year": int(row["api_start_year"]),
        "api_end_year": int(row["api_end_year"]),
        "reference_year_offset": int(
            row["reference_year_offset"]
        ),
        "reference_period_rule": row[
            "reference_period_rule"
        ],
        "documentation_url": row[
            "documentation_url"
        ],
        "jsonstat_version": "2.0",
        "http_status": 200,
    }

    for field, expected_value in expected_fields.items():

        check(
            metadata.get(field) == expected_value,
            f"{indicator_id}: Metadata mismatch: {field}",
        )

    # The recorded API filters must match the
    # current registry and requested year window.

    filters = json.loads(
        row["filters_json"]
    )

    expected_params = {
        "lang": "EN",
        "format": "JSON",
        "geoLevel": "country",
        "sinceTimePeriod": str(
            int(row["api_start_year"])
        ),
        "untilTimePeriod": str(
            int(row["api_end_year"])
        ),
        **filters,
    }

    check(
        metadata.get("request_filters")
        == expected_params,
        f"{indicator_id}: API filter mismatch.",
    )

    # Validate the actual URL recorded by Step 35.
    #
    # This is stronger than checking metadata labels
    # because it also verifies the request parameters.

    api_url = metadata.get("api_url")

    check(
        isinstance(api_url, str),
        f"{indicator_id}: Missing API URL.",
    )

    parsed = urlsplit(api_url)

    expected_url = urlsplit(
        BASE_URL + row["dataset_code"]
    )

    check(
        parsed.scheme == "https"
        and parsed.netloc == expected_url.netloc
        and parsed.path == expected_url.path,
        f"{indicator_id}: Unexpected API endpoint.",
    )

    actual_params = parse_qs(
        parsed.query,
        keep_blank_values=True,
    )

    expected_query = {
        key: [value]
        for key, value in expected_params.items()
    }

    check(
        actual_params == expected_query,
        f"{indicator_id}: Request URL filters mismatch.",
    )

    # Retrieval time must contain timezone information.
    #
    # This is necessary to distinguish the download date
    # from the original fieldwork or economic reference year.

    retrieval_text = metadata.get(
        "retrieved_at_utc"
    )

    check(
        isinstance(retrieval_text, str),
        f"{indicator_id}: Missing retrieval timestamp.",
    )

    retrieval_time = datetime.fromisoformat(
        retrieval_text
    )

    check(
        retrieval_time.tzinfo is not None,
        f"{indicator_id}: Timestamp lacks timezone.",
    )

    check(
        retrieval_time.utcoffset().total_seconds() == 0,
        f"{indicator_id}: Retrieval timestamp is not UTC.",
    )


# ============================================================
# 8. VALIDATE ONE SAVED SNAPSHOT
# ============================================================

def validate_snapshot(sidecar_path, row, registry_hash):

    indicator_id = row["indicator_id"]

    # A sidecar ends with ".metadata.json".
    #
    # Removing that suffix gives the shared filename stem
    # used by the RAW file.

    suffix = ".metadata.json"

    check(
        sidecar_path.name.endswith(suffix),
        "Unexpected metadata filename.",
    )

    file_stem = sidecar_path.name[
        :-len(suffix)
    ]

    raw_path = (
        sidecar_path.parent
        / f"{file_stem}.json"
    )

    # Both files must exist because the sidecar
    # cannot independently prove RAW data integrity.

    check(
        raw_path.exists(),
        f"Missing RAW file: {raw_path.name}",
    )

    # Read the actual original bytes from disk.
    #
    # We do not reconstruct the JSON file using
    # json.dumps because this could change the bytes.

    raw_bytes = raw_path.read_bytes()

    actual_hash = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    # Load provenance information from the sidecar.

    metadata = json.loads(
        sidecar_path.read_text(encoding="utf-8")
    )

    check(
        isinstance(metadata, dict),
        f"{indicator_id}: Invalid sidecar structure.",
    )

    # Compare the independently computed SHA-256
    # with the hash saved during Step 35.

    check(
        metadata.get("sha256") == actual_hash,
        f"{indicator_id}: SHA-256 mismatch.",
    )

    # Verify that the filename also identifies
    # the same content version.

    expected_stem = (
        f"{indicator_id}_"
        f"{int(row['api_start_year'])}_"
        f"{int(row['api_end_year'])}_"
        f"{actual_hash[:16]}"
    )

    check(
        file_stem == expected_stem,
        f"{indicator_id}: Snapshot filename mismatch.",
    )

    check(
        metadata.get("raw_filename") == raw_path.name,
        f"{indicator_id}: RAW filename metadata mismatch.",
    )

    # Independently verify the provenance information
    # against the original indicator registry.

    validate_provenance(
        metadata,
        row,
        registry_hash,
    )

    # Decode RAW JSON only after verifying file integrity.

    data = json.loads(
        raw_bytes.decode("utf-8-sig")
    )

    check(
        isinstance(data, dict),
        f"{indicator_id}: Invalid RAW JSON object.",
    )

    # Validate the statistical structure itself.

    diagnostic = validate_raw_dataset(
        data,
        row,
    )

    # Compare independently counted observations
    # with the counts stored in Step 35 metadata.

    check(
        metadata.get("countries_returned")
        == diagnostic["countries_n"],
        f"{indicator_id}: Country count mismatch.",
    )

    check(
        metadata.get("years_returned")
        == diagnostic["years"],
        f"{indicator_id}: Year coverage mismatch.",
    )

    check(
        metadata.get("non_missing_values_n")
        == diagnostic["values_n"],
        f"{indicator_id}: Observation count mismatch.",
    )

    # Verify that the source labels and update date
    # were correctly preserved by the ingestion script.

    metadata_pairs = [
        ("api_dataset_updated", "updated"),
        ("api_dataset_title", "label"),
        ("api_dataset_source", "source"),
        ("jsonstat_version", "version"),
    ]

    for metadata_field, raw_field in metadata_pairs:

        check(
            metadata.get(metadata_field)
            == data.get(raw_field),
            f"{indicator_id}: RAW/metadata mismatch: "
            f"{metadata_field}",
        )

    print(
        f"{indicator_id:<28} PASSED "
        f"| countries={diagnostic['countries_n']:>2} "
        f"| values={diagnostic['values_n']:>3} "
        f"| flags={diagnostic['flags_n']:>3}"
    )

    return diagnostic


# ============================================================
# 9. RUN VALIDATION FOR ALL SIX INDICATORS
# ============================================================

def main():

    print("=" * 110)
    print("V1 EUROSTAT RAW SNAPSHOT VALIDATION")
    print("=" * 110)

    registry, registry_hash = load_registry()

    check(
        RAW_DIR.exists(),
        f"Missing RAW directory: {RAW_DIR}",
    )

    print("\nREGISTRY")
    print("-" * 110)

    print(f"Indicators:       {len(registry)}")
    print(f"Registry SHA-256: {registry_hash}")

    print("\nINDIVIDUAL SNAPSHOT VALIDATION")
    print("-" * 110)

    results = []

    # Validate every snapshot matching each
    # indicator and its configured year window.
    #
    # If Eurostat data are revised in the future,
    # multiple historical snapshots may exist.
    #
    # We validate all of them instead of silently
    # choosing one version.

    for _, row in registry.iterrows():

        indicator_id = row["indicator_id"]

        start_year = int(row["api_start_year"])
        end_year = int(row["api_end_year"])

        pattern = (
            f"{indicator_id}_"
            f"{start_year}_"
            f"{end_year}_"
            f"*.metadata.json"
        )

        sidecars = sorted(
            RAW_DIR.glob(pattern)
        )

        check(
            len(sidecars) > 0,
            f"No snapshot found for {indicator_id}.",
        )

        for sidecar_path in sidecars:

            diagnostic = validate_snapshot(
                sidecar_path,
                row,
                registry_hash,
            )

            results.append(
                {
                    "indicator_id": indicator_id,
                    "snapshot": sidecar_path.name,
                    **diagnostic,
                }
            )

    # ========================================================
    # 10. PRINT FINAL VALIDATION SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    # Every indicator must have at least one
    # successfully validated RAW snapshot.

    validated_indicators = (
        results_df["indicator_id"].nunique()
    )

    check(
        validated_indicators == 6,
        "Not all six indicators were validated.",
    )

    total_snapshots = len(results_df)

    # Count numeric observations across saved snapshots.
    #
    # With exactly one snapshot per indicator, this is
    # the total number of downloaded numeric observations.
    #
    # If historical revisions exist, the sum includes
    # multiple versions and must not be interpreted
    # as a count of unique country-year observations.

    total_values = int(
        results_df["values_n"].sum()
    )

    print("\n" + "=" * 110)
    print("VALIDATION SUMMARY")
    print("=" * 110)

    print(f"Indicators validated:  {validated_indicators}")
    print(f"Snapshots validated:   {total_snapshots}")
    print(f"Numeric values checked: {total_values}")

    if total_snapshots > 6:

        print()
        print(
            "Multiple historical snapshots exist."
        )

        print(
            "A later pipeline step must explicitly "
            "select which version to use."
        )

    print()
    print("EUROSTAT RAW VALIDATION PASSED.")

    print()
    print("SHA-256 integrity verified.")
    print("Metadata and registry consistency verified.")
    print("JSON-stat dimensions and observations verified.")
    print("Original RAW files were not modified.")
    print("No API requests were made.")
    print("No ESS-Eurostat integration was performed.")


# ============================================================
# 11. EXECUTE THE SCRIPT
# ============================================================

# Run validation only when executing this script
# directly from the VS Code terminal.

if __name__ == "__main__":
    main()
