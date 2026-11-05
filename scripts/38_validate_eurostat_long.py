# Step 38: Independently validate the Eurostat long-format outputs.
# This script only reads files. It does not modify or create datasets.
# It does not import the transformation functions from Step 37.

import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)
from pandas.testing import assert_frame_equal


# Locate the project from this script, saved inside scripts/.
ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "eurostat"
REGISTRY = ROOT / "config" / "eurostat_indicators_v1.csv"

PARQUET = ROOT / "data" / "interim" / (
    "eurostat_country_year_long.parquet"
)

CSV = ROOT / "data" / "interim" / (
    "eurostat_country_year_long.csv"
)


# Use the same fixed source snapshots confirmed in Step 36.
# Counts represent: all cells, numeric values, flagged cells.
SNAPSHOTS = {
    "gdp_pps_index": (
        "gdp_pps_index_2023_2024_4a3bfadae8a14190.json",
        80,
        78,
        9,
    ),
    "employment_rate": (
        "employment_rate_2023_2024_fc6cc1b62db268bf.json",
        70,
        68,
        6,
    ),
    "unemployment_rate": (
        "unemployment_rate_2023_2024_8f8bfd34da9e5911.json",
        70,
        68,
        6,
    ),
    "gini": (
        "gini_2023_2025_19487335ba1c6ec9.json",
        111,
        96,
        7,
    ),
    "material_deprivation_rate": (
        "material_deprivation_rate_2023_2024_56eb615b131b298c.json",
        72,
        67,
        1,
    ),
    "inflation_rate": (
        "inflation_rate_2023_2024_87bd86b4626a3705.json",
        78,
        74,
        0,
    ),
}

REGISTRY_HASH = (
    "8e7d2b06565fc0ba662738fb567f7756af709b85cdb96ab7afeeab8b6a969145"
)


# These columns jointly identify one observation.
KEY = [
    "indicator_id",
    "eurostat_geo",
    "eurostat_year",
]

# Define the complete expected output schema.
COLUMNS = [
    "indicator_id",
    "dataset_code",
    "indicator_label",
    "unit_label",
    "eurostat_geo",
    "eurostat_geo_label",
    "eurostat_year",
    "value",
    "eurostat_status",
    "reference_year_offset",
    "reference_period_rule",
    "source_api_url",
    "api_dataset_updated",
    "retrieved_at_utc",
    "snapshot_sha256",
    "raw_filename",
    "metadata_filename",
    "registry_sha256",
    "is_missing",
    "has_status_flag",
]

INTEGER_COLUMNS = [
    "eurostat_year",
    "reference_year_offset",
]

BOOLEAN_COLUMNS = [
    "is_missing",
    "has_status_flag",
]

# Every remaining field is text.
TEXT_COLUMNS = [
    column
    for column in COLUMNS
    if column not in INTEGER_COLUMNS + BOOLEAN_COLUMNS + ["value"]
]


def require(condition, message):
    # Raise an error immediately when a validation condition fails.
    if not condition:
        raise ValueError(message)


def read_cell(container, position, total, status=False):
    # Read a single RAW cell without expanding the whole array.
    if isinstance(container, dict):
        # Missing dictionary positions represent absent observations.
        return container.get(str(position))

    if isinstance(container, list):
        require(
            len(container) == total,
            "RAW array length differs from the cube size.",
        )
        return container[position]

    if status and isinstance(container, str):
        # A scalar status applies to all observations.
        return container

    raise ValueError("Unsupported RAW value/status structure.")


def position_to_code(category, size):
    # Build a position-to-code lookup independently from Step 37.
    index = category.get("index")

    if isinstance(index, dict):
        require(
            all(type(position) is int for position in index.values()),
            "Invalid RAW category positions.",
        )

        lookup = {
            position: code
            for code, position in index.items()
        }

        # Duplicated positions would disappear when creating the lookup.
        require(
            len(lookup) == len(index),
            "Duplicated RAW category positions.",
        )

    elif isinstance(index, list):
        lookup = dict(enumerate(index))

    else:
        # A singleton dimension may identify its code through its label.
        require(
            index is None and size == 1,
            "Missing RAW category index.",
        )

        labels = category.get("label", {})

        require(
            len(labels) == 1,
            "Invalid singleton RAW dimension.",
        )

        lookup = {0: next(iter(labels))}

    require(
        set(lookup) == set(range(size)),
        "Incomplete RAW category positions.",
    )

    require(
        all(isinstance(code, str) for code in lookup.values()),
        "RAW category codes must be text.",
    )

    require(
        len(set(lookup.values())) == size,
        "Duplicated RAW category codes.",
    )

    return lookup


def expected_from_raw(row, registry_hash):
    indicator = row["indicator_id"]

    (
        filename,
        expected_rows,
        expected_numeric,
        expected_flags,
    ) = SNAPSHOTS[indicator]

    raw_path = RAW / filename
    metadata_path = raw_path.with_suffix(".metadata.json")

    # Read and hash the original source bytes.
    raw_bytes = raw_path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8-sig")
    )

    require(
        raw_hash == metadata["sha256"],
        f"RAW hash mismatch: {indicator}",
    )

    require(
        raw_hash[:16] == raw_path.stem.rsplit("_", 1)[1],
        f"Filename hash mismatch: {indicator}",
    )

    require(
        metadata["raw_filename"] == filename,
        f"RAW filename mismatch: {indicator}",
    )

    require(
        metadata["registry_sha256"] == registry_hash,
        f"Registry provenance mismatch: {indicator}",
    )

    # Check sidecar definitions against the validated registry.
    for field in (
        "indicator_id",
        "dataset_code",
        "unit_label",
        "reference_period_rule",
    ):
        require(
            metadata[field] == row[field],
            f"Metadata mismatch: {indicator} / {field}",
        )

    require(
        metadata["indicator_label"] == row["label"],
        f"Label mismatch: {indicator}",
    )

    for field in (
        "api_start_year",
        "api_end_year",
        "reference_year_offset",
    ):
        require(
            metadata[field] == int(row[field]),
            f"Metadata mismatch: {indicator} / {field}",
        )

    data = json.loads(raw_bytes)

    require(
        data.get("version") == "2.0"
        and data.get("class") == "dataset",
        f"Unexpected RAW format: {indicator}",
    )

    require(
        data["updated"] == metadata["api_dataset_updated"],
        f"Update timestamp mismatch: {indicator}",
    )

    require(
        data["extension"]["id"].lower()
        == row["dataset_code"].lower(),
        f"Dataset mismatch: {indicator}",
    )

    ids = data["id"]
    sizes = data["size"]

    require(
        len(ids) == len(sizes)
        and len(ids) == len(set(ids)),
        f"Invalid dimensions: {indicator}",
    )

    require(
        all(type(size) is int and size > 0 for size in sizes),
        f"Invalid dimension sizes: {indicator}",
    )

    require(
        {"geo", "time"}.issubset(ids),
        f"Missing country/year dimensions: {indicator}",
    )

    require(
        set(ids) == set(data["dimension"]),
        f"Dimension definitions mismatch: {indicator}",
    )

    filters = json.loads(row["filters_json"])

    require(
        isinstance(filters, dict),
        f"Invalid registry filters: {indicator}",
    )

    require(
        set(filters) == set(ids) - {"geo", "time"},
        f"Filter dimensions mismatch: {indicator}",
    )

    require(
        filters.get("freq") == "A",
        f"Non-annual series: {indicator}",
    )

    # Build a lookup for each dimension's category positions.
    lookups = {}

    for dimension, size in zip(ids, sizes):
        category = data["dimension"][dimension]["category"]

        lookups[dimension] = position_to_code(
            category,
            size,
        )

        if dimension not in {"geo", "time"}:
            require(
                size == 1
                and lookups[dimension][0] == filters[dimension],
                f"Unexpected statistical selection: {indicator}",
            )

            require(
                metadata["request_filters"][dimension]
                == filters[dimension],
                f"Request filter mismatch: {indicator}",
            )

    total = math.prod(sizes)

    require(
        total == expected_rows,
        f"Unexpected RAW cube size: {indicator}",
    )

    values = data["value"]
    statuses = data.get("status", {})

    # Reject sparse entries outside the declared cube.
    for name, container in (
        ("value", values),
        ("status", statuses),
    ):
        if isinstance(container, dict):
            for key in container:
                position = int(key)

                require(
                    str(position) == key and 0 <= position < total,
                    f"Invalid RAW {name} position: {indicator}",
                )

    # A stride is the number of cells covered by one dimension step.
    # The last dimension has stride 1.
    strides = [
        math.prod(sizes[index + 1:])
        for index in range(len(sizes))
    ]

    geo_labels = data["dimension"]["geo"]["category"]["label"]
    records = []

    for flat_position in range(total):
        coordinates = {}

        # Decode coordinates using division and remainder.
        # This differs from the product() method used in Step 37.
        for dimension, size, stride in zip(ids, sizes, strides):
            category_position = (
                flat_position // stride
            ) % size

            coordinates[dimension] = (
                lookups[dimension][category_position]
            )

        value = read_cell(
            values,
            flat_position,
            total,
        )

        flag = read_cell(
            statuses,
            flat_position,
            total,
            status=True,
        )

        require(
            value is None
            or (
                type(value) in (int, float)
                and math.isfinite(value)
            ),
            f"Invalid numeric RAW value: {indicator}",
        )

        require(
            flag is None or isinstance(flag, str),
            f"Invalid RAW status: {indicator}",
        )

        geo = coordinates["geo"]
        year = coordinates["time"]

        require(
            year.isdigit() and len(year) == 4,
            f"Invalid RAW year: {indicator}",
        )

        # Build the expected row directly from RAW, metadata and registry.
        records.append(
            {
                "indicator_id": indicator,
                "dataset_code": row["dataset_code"],
                "indicator_label": row["label"],
                "unit_label": row["unit_label"],
                "eurostat_geo": geo,
                "eurostat_geo_label": geo_labels[geo],
                "eurostat_year": int(year),
                "value": value,
                "eurostat_status": (
                    flag if flag is not None else ""
                ),
                "reference_year_offset": int(
                    row["reference_year_offset"]
                ),
                "reference_period_rule": row["reference_period_rule"],
                "source_api_url": metadata["api_url"],
                "api_dataset_updated": metadata["api_dataset_updated"],
                "retrieved_at_utc": metadata["retrieved_at_utc"],
                "snapshot_sha256": raw_hash,
                "raw_filename": filename,
                "metadata_filename": metadata_path.name,
                "registry_sha256": registry_hash,
                "is_missing": value is None,
                "has_status_flag": flag not in (None, ""),
            }
        )

    expected = pd.DataFrame(records, columns=COLUMNS)

    # Check coverage independently from the transformed outputs.
    years = sorted(expected["eurostat_year"].unique().tolist())

    require(
        years == list(
            range(
                int(row["api_start_year"]),
                int(row["api_end_year"]) + 1,
            )
        ),
        f"Unexpected RAW years: {indicator}",
    )

    require(
        years == sorted(metadata["years_returned"]),
        f"Sidecar years mismatch: {indicator}",
    )

    require(
        expected["eurostat_geo"].nunique()
        == metadata["countries_returned"],
        f"Sidecar country count mismatch: {indicator}",
    )

    numeric_count = int(expected["value"].notna().sum())

    require(
        numeric_count
        == expected_numeric
        == metadata["non_missing_values_n"],
        f"RAW numeric count mismatch: {indicator}",
    )

    require(
        int(expected["has_status_flag"].sum()) == expected_flags,
        f"RAW status count mismatch: {indicator}",
    )

    return expected


def check_schema(table, name):
    # Reject missing, additional or duplicated column names.
    require(
        len(table.columns) == len(COLUMNS)
        and set(table.columns) == set(COLUMNS),
        f"{name}: unexpected columns.",
    )

    require(
        not table.duplicated(KEY).any(),
        f"{name}: duplicated indicator-country-year keys.",
    )

    # Only the numeric value column may contain missing values.
    other_columns = [
        column
        for column in COLUMNS
        if column != "value"
    ]

    require(
        not table[other_columns].isna().any().any(),
        f"{name}: missing keys, flags or provenance.",
    )


def check_parquet_types(table):
    # Inspect stored types before any comparison conversions.
    require(
        is_float_dtype(table["value"].dtype),
        "Parquet: value must have a floating-point type.",
    )

    for column in INTEGER_COLUMNS:
        require(
            is_integer_dtype(table[column].dtype),
            f"Parquet: {column} must be an integer.",
        )

    for column in BOOLEAN_COLUMNS:
        require(
            is_bool_dtype(table[column].dtype),
            f"Parquet: {column} must be Boolean.",
        )

    for column in TEXT_COLUMNS:
        require(
            is_string_dtype(table[column].dtype),
            f"Parquet: {column} must have a text-compatible type.",
        )

        # Also inspect values because object dtype can contain other objects.
        require(
            table[column].map(
                lambda value: isinstance(value, str)
            ).all(),
            f"Parquet: non-text content in {column}.",
        )


def load_csv():
    # CSV has no stored schema, so specify how fields must be read.
    types = {
        column: "string"
        for column in TEXT_COLUMNS + BOOLEAN_COLUMNS
    }

    types.update({
        column: "int64"
        for column in INTEGER_COLUMNS
    })

    types["value"] = "float64"

    # Only empty cells in "value" represent missing numeric observations.
    # Empty status strings must remain empty strings.
    table = pd.read_csv(
        CSV,
        encoding="utf-8-sig",
        dtype=types,
        keep_default_na=False,
        na_values={"value": [""]},
        float_precision="round_trip",
    )

    # Read explicit Boolean tokens rather than using astype(bool).
    # astype(bool) would incorrectly treat the text "False" as True.
    for column in BOOLEAN_COLUMNS:
        require(
            table[column].isin(["True", "False"]).all(),
            f"CSV: invalid Boolean token in {column}.",
        )

        table[column] = table[column].map(
            {"True": True, "False": False}
        ).astype("bool")

    return table


def ordered(table):
    # Compare observations by key, not by their current row order.
    return (
        table[COLUMNS]
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def compare_tables(actual, expected, name):
    # Compare every cell, including metadata, flags and missing positions.
    # Dtypes are checked separately because CSV does not preserve them.
    assert_frame_equal(
        ordered(actual),
        ordered(expected),
        check_dtype=False,
        check_exact=True,
        obj=name,
    )

    print(f"{name}: exact cell comparison PASSED.")


def check_indicator_bounds(table):
    # These are broad definition-based limits, not outlier cutoffs.
    # A failed check stops validation; it never deletes observations.
    percentage_indicators = [
        "employment_rate",
        "unemployment_rate",
        "gini",
        "material_deprivation_rate",
    ]

    for indicator in percentage_indicators:
        values = table.loc[
            table["indicator_id"].eq(indicator),
            "value",
        ].dropna()

        require(
            values.between(0, 100).all(),
            f"Values outside 0–100: {indicator}",
        )

    gdp = table.loc[
        table["indicator_id"].eq("gdp_pps_index"),
        "value",
    ].dropna()

    # GDP is an index, so values above 100 are valid.
    require(
        gdp.ge(0).all(),
        "Negative GDP per capita PPS index.",
    )

    inflation = table.loc[
        table["indicator_id"].eq("inflation_rate"),
        "value",
    ].dropna()

    # Negative inflation is valid; do not impose a zero lower bound.
    # A percentage price change cannot fall below -100%.
    require(
        inflation.ge(-100).all(),
        "Inflation below -100%.",
    )

    print("Broad indicator bounds: PASSED.")


def main():
    print("STEP 38 — EUROSTAT LONG-FORMAT VALIDATION")

    # Fail clearly if an expected input file is missing.
    for path in (REGISTRY, PARQUET, CSV):
        require(
            path.is_file(),
            f"Required file not found: {path}",
        )

    registry_bytes = REGISTRY.read_bytes()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()

    require(
        registry_hash == REGISTRY_HASH,
        "Registry differs from the validated V1 version.",
    )

    registry = pd.read_csv(
        io.BytesIO(registry_bytes),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    require(
        not registry["indicator_id"].duplicated().any(),
        "Duplicate registry indicators.",
    )

    require(
        set(registry["indicator_id"]) == set(SNAPSHOTS),
        "Registry must contain the six V1 indicators.",
    )

    expected_parts = []

    # Reconstruct expected observations from all six original snapshots.
    for row in registry.to_dict("records"):
        print(f"Checking RAW: {row['indicator_id']}")

        expected_parts.append(
            expected_from_raw(row, registry_hash)
        )

    expected = pd.concat(
        expected_parts,
        ignore_index=True,
    )

    # Explicit floats represent RAW nulls as NaN for table comparison.
    expected["value"] = expected["value"].astype("float64")

    parquet = pd.read_parquet(
        PARQUET,
        engine="pyarrow",
    )

    csv = load_csv()

    # Check schema and key uniqueness before comparing observations.
    check_schema(expected, "RAW reconstruction")
    check_schema(parquet, "Parquet")
    check_schema(csv, "CSV")

    check_parquet_types(parquet)

    print("\nSchema, unique keys and Parquet types: PASSED.")

    # These comparisons verify actual values, not just totals.
    compare_tables(
        parquet,
        expected,
        "Parquet versus RAW",
    )

    compare_tables(
        csv,
        expected,
        "CSV versus RAW",
    )

    compare_tables(
        csv,
        parquet,
        "CSV versus Parquet",
    )

    # Check that diagnostic flags correctly describe the stored cells.
    require(
        parquet["is_missing"].eq(
            parquet["value"].isna()
        ).all(),
        "is_missing disagrees with the stored values.",
    )

    require(
        parquet["has_status_flag"].eq(
            parquet["eurostat_status"].ne("")
        ).all(),
        "has_status_flag disagrees with the stored statuses.",
    )

    print("Missing-value and status flags: PASSED.")

    check_indicator_bounds(parquet)

    # Report the validated counts per indicator.
    summary = parquet.groupby(
        "indicator_id",
        sort=True,
    ).agg(
        rows=("value", "size"),
        numeric=("value", "count"),
        missing=("is_missing", "sum"),
        status_flags=("has_status_flag", "sum"),
    )

    require(
        tuple(summary.sum()) == (481, 451, 30, 29),
        "Unexpected overall counts.",
    )

    print("\nVALIDATED COUNTS BY INDICATOR")
    print(summary.to_string())

    print("\nTOTALS")
    print(summary.sum().to_string())

    # This message is reached only after all preceding checks succeed.
    print("\nEUROSTAT LONG-FORMAT VALIDATION PASSED.")
    print("6 RAW snapshots checked against Parquet and CSV.")
    print("All values, missing positions, statuses and provenance matched.")
    print("No files were modified.")


# Importing this script must not automatically run the validation.
if __name__ == "__main__":
    main()