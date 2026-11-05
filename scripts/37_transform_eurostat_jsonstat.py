# Step 37: Convert validated Eurostat snapshots into a long table.
# Each row represents one indicator-country-year combination.
# RAW files are only read and are never modified.

import hashlib
import io
import json
import math
from itertools import product
from pathlib import Path

import pandas as pd


# Find the project folder from this script's location.
# This works when the script is saved inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

# Define the input and output locations.
RAW = ROOT / "data" / "raw" / "eurostat"
REGISTRY = ROOT / "config" / "eurostat_indicators_v1.csv"
OUTPUT = ROOT / "data" / "interim"


# Select the exact snapshots already validated in Step 36.
# Each tuple contains:
# filename, total cells, numeric values, cells with status flags.
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

# This hash identifies the validated V1 registry.
# A different hash means that the registry bytes have changed.
REGISTRY_HASH = (
    "8e7d2b06565fc0ba662738fb567f7756af709b85cdb96ab7afeeab8b6a969145"
)


def require(condition, message):
    # Stop execution when a required check fails.
    # This prevents exporting results from unexpected inputs.
    if not condition:
        raise ValueError(message)


def category_codes(category, size):
    # Read the index that links category codes to their positions.
    index = category.get("index")

    if isinstance(index, dict):
        # Dictionary values contain the actual category positions.
        positions = list(index.values())

        require(
            all(type(position) is int for position in positions),
            "Invalid category positions.",
        )

        # Valid positions must run from zero to size minus one.
        require(
            sorted(positions) == list(range(size)),
            "Incomplete category index.",
        )

        # Order codes by position, not alphabetically.
        codes = sorted(index, key=index.get)

    elif isinstance(index, list):
        # A list already defines the category order.
        codes = index

    else:
        # JSON-stat can omit the index for a single-category dimension.
        require(
            size == 1 and index is None,
            "Missing category index.",
        )

        # In this case, obtain the code from the category label mapping.
        codes = list(category.get("label", {}))

    # Check both the number of categories and their uniqueness.
    require(
        len(codes) == size and len(set(codes)) == size,
        "Invalid or duplicated categories.",
    )

    require(
        all(isinstance(code, str) for code in codes),
        "Category codes must be text.",
    )

    return codes


def expand_cells(container, total, status=False):
    # JSON-stat can store cells as a sparse dictionary or a full list.
    if isinstance(container, dict):
        # Start with missing cells; fill only positions present in JSON.
        cells = [None] * total

        for key, value in container.items():
            # JSON object keys are text, so convert the position to int.
            position = int(key)

            # Reject invalid positions instead of silently dropping them.
            require(
                str(position) == key and 0 <= position < total,
                "Invalid cell index.",
            )

            cells[position] = value

    elif isinstance(container, list):
        # A full array must contain one entry per possible cell.
        require(
            len(container) == total,
            "Cell array length differs from cube size.",
        )

        cells = container

    elif status and isinstance(container, str):
        # A single status string applies to every cell in the dataset.
        cells = [container] * total

    else:
        raise ValueError(
            "Unsupported JSON-stat value/status structure."
        )

    return cells


def decode_cube(data, filters):
    # Confirm the format before interpreting its dimensions and values.
    require(
        data.get("version") == "2.0"
        and data.get("class") == "dataset",
        "Expected JSON-stat 2.0 dataset.",
    )

    # IDs define dimension order; sizes define category counts.
    ids = data["id"]
    sizes = data["size"]

    require(
        len(ids) == len(sizes)
        and len(ids) == len(set(ids)),
        "Invalid dimensions.",
    )

    require(
        all(type(size) is int and size > 0 for size in sizes),
        "Invalid dimension sizes.",
    )

    # Country and year are necessary for our long-table key.
    require(
        "geo" in ids and "time" in ids,
        "Missing geo/time dimension.",
    )

    require(
        set(ids) == set(data["dimension"]),
        "Dimension definitions differ from IDs.",
    )

    # All other dimensions must match the registry's statistical filters.
    require(
        set(filters) == set(ids) - {"geo", "time"},
        "Registry filters differ from dimensions.",
    )

    require(
        filters.get("freq") == "A",
        "Expected annual data.",
    )

    # Build an ordered category list for every dimension.
    # zip() pairs each dimension ID with its declared size.
    codes = [
        category_codes(
            data["dimension"][dimension]["category"],
            size,
        )
        for dimension, size in zip(ids, sizes)
    ]

    for dimension, categories in zip(ids, codes):
        if dimension not in {"geo", "time"}:
            # One selected category prevents multiple series per key.
            require(
                categories == [filters[dimension]],
                f"Unexpected selection: {dimension}",
            )

    # Multiply dimension sizes to count all cells, including missing ones.
    total = math.prod(sizes)

    values = expand_cells(
        data["value"],
        total,
    )

    # No status object means no reported status flags.
    statuses = expand_cells(
        data.get("status", {}),
        total,
        status=True,
    )

    # Country labels make the output easier to read.
    geo_labels = data["dimension"]["geo"]["category"]["label"]

    rows = []

    # product() generates every combination of dimension categories.
    # The last dimension changes fastest, matching JSON-stat ordering.
    for position, combination in enumerate(product(*codes)):
        # Link each dimension name to its category for this cell.
        coordinates = dict(zip(ids, combination))

        value = values[position]
        flag = statuses[position]

        # Accept finite numbers or missing values.
        # Reject booleans because True/False are not statistical values.
        require(
            value is None
            or (
                type(value) in (int, float)
                and math.isfinite(value)
            ),
            "Non-numeric or infinite value.",
        )

        require(
            flag is None or isinstance(flag, str),
            "Invalid status flag.",
        )

        geo = coordinates["geo"]
        year = coordinates["time"]

        # Annual periods must use a four-digit year.
        require(
            year.isdigit() and len(year) == 4,
            f"Invalid annual period: {year}",
        )

        # Create one record for this country-year cell.
        rows.append(
            {
                "eurostat_geo": geo,
                "eurostat_geo_label": geo_labels[geo],
                "eurostat_year": int(year),
                "value": value,

                # Preserve combined flags exactly as Eurostat supplied them.
                "eurostat_status": (
                    flag if flag is not None else ""
                ),

                # Zero remains a valid value; only None is missing here.
                "is_missing": value is None,

                # An absent or empty status does not count as a flag.
                "has_status_flag": flag not in (None, ""),
            }
        )

    # Convert the list of records into a pandas table.
    return pd.DataFrame(rows)


def transform_indicator(row, registry_hash):
    indicator = row["indicator_id"]

    # Unpack the snapshot name and its Step 36 baseline counts.
    (
        filename,
        expected_rows,
        expected_values,
        expected_flags,
    ) = SNAPSHOTS[indicator]

    raw_path = RAW / filename

    # Replace .json with .metadata.json to locate the sidecar.
    metadata_path = raw_path.with_suffix(".metadata.json")

    # Hash the original bytes, without reformatting the JSON.
    raw_bytes = raw_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    # utf-8-sig also accepts files containing a UTF-8 BOM.
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8-sig")
    )

    require(
        digest == metadata["sha256"],
        f"RAW hash mismatch: {indicator}",
    )

    # The filename contains the first 16 characters of the RAW hash.
    require(
        digest[:16] == raw_path.stem.rsplit("_", 1)[1],
        "Filename hash mismatch.",
    )

    require(
        metadata["raw_filename"] == filename,
        "Metadata points to another RAW file.",
    )

    require(
        metadata["registry_sha256"] == registry_hash,
        "Registry hash mismatch.",
    )

    # Check that the registry and metadata describe the same series.
    for key in (
        "indicator_id",
        "dataset_code",
        "unit_label",
        "reference_period_rule",
    ):
        require(
            metadata[key] == row[key],
            f"Metadata mismatch: {indicator} / {key}",
        )

    # The registry uses "label"; the sidecar uses "indicator_label".
    require(
        metadata["indicator_label"] == row["label"],
        "Indicator label mismatch.",
    )

    # Convert numeric registry fields from text before comparing them.
    for key in (
        "api_start_year",
        "api_end_year",
        "reference_year_offset",
    ):
        require(
            metadata[key] == int(row[key]),
            f"Metadata mismatch: {key}",
        )

    # Decode the JSON text stored inside the registry's filters column.
    filters = json.loads(row["filters_json"])

    require(
        isinstance(filters, dict),
        "Registry filters must be a JSON object.",
    )

    # Confirm that ingestion used the registered statistical filters.
    for key, value in filters.items():
        require(
            metadata["request_filters"][key] == value,
            f"Request filter mismatch: {key}",
        )

    # Parse the same RAW bytes that passed the hash check.
    data = json.loads(raw_bytes)

    require(
        data["updated"] == metadata["api_dataset_updated"],
        "Update timestamp mismatch.",
    )

    # Compare case-insensitively because Eurostat may use uppercase IDs.
    require(
        data["extension"]["id"].lower()
        == row["dataset_code"].lower(),
        "Dataset code mismatch.",
    )

    # Decode every cell, including cells without numeric values.
    table = decode_cube(data, filters)

    # List the years actually present in the transformed table.
    years = sorted(
        table["eurostat_year"].unique().tolist()
    )

    # range() excludes its upper limit, so add one to include the end year.
    expected_years = list(
        range(
            int(row["api_start_year"]),
            int(row["api_end_year"]) + 1,
        )
    )

    require(
        years
        == expected_years
        == sorted(metadata["years_returned"]),
        "Year coverage mismatch.",
    )

    # nunique() counts distinct countries, not country-year rows.
    require(
        table["eurostat_geo"].nunique()
        == metadata["countries_returned"],
        "Country count mismatch.",
    )

    # notna() identifies numeric observations, including zeros.
    require(
        int(table["value"].notna().sum())
        == metadata["non_missing_values_n"],
        "Numeric count mismatch.",
    )

    # Compare the transformed counts with the validated snapshot baseline.
    counts = (
        len(table),
        int(table["value"].notna().sum()),
        int(table["has_status_flag"].sum()),
    )

    require(
        counts
        == (
            expected_rows,
            expected_values,
            expected_flags,
        ),
        f"Step 36 counts differ: {indicator} / {counts}",
    )

    # Repeat provenance on each row so individual records remain traceable.
    provenance = {
        "indicator_id": indicator,
        "dataset_code": row["dataset_code"],
        "indicator_label": row["label"],
        "unit_label": row["unit_label"],

        # Record the alignment rule without changing the Eurostat year.
        "reference_year_offset": int(
            row["reference_year_offset"]
        ),
        "reference_period_rule": row["reference_period_rule"],

        "source_api_url": metadata["api_url"],
        "api_dataset_updated": metadata["api_dataset_updated"],
        "retrieved_at_utc": metadata["retrieved_at_utc"],
        "snapshot_sha256": digest,
        "raw_filename": filename,
        "metadata_filename": metadata_path.name,
        "registry_sha256": registry_hash,
    }

    # Assigning one value to a column repeats it for every row.
    for column, value in provenance.items():
        table[column] = value

    return table


def main():
    # Define both output filenames before starting the transformation.
    parquet_path = OUTPUT / "eurostat_country_year_long.parquet"
    csv_path = OUTPUT / "eurostat_country_year_long.csv"

    # Existing outputs require inspection; never overwrite them silently.
    for path in (parquet_path, csv_path):
        if path.exists():
            raise FileExistsError(
                "Output already exists; inspect it before rerunning: "
                f"{path}"
            )

    # Check that the registry is the exact validated V1 version.
    registry_bytes = REGISTRY.read_bytes()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()

    require(
        registry_hash == REGISTRY_HASH,
        "Registry differs from the validated V1 version.",
    )

    # BytesIO lets pandas read the same bytes used for the hash.
    # Read fields as text to preserve codes and JSON filter definitions.
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
        "Expected the six V1 indicators.",
    )

    tables = []

    # Each record is a dictionary containing one registry row.
    for row in registry.to_dict("records"):
        print(f"Processing: {row['indicator_id']}")

        # Transform each indicator before combining the results.
        tables.append(
            transform_indicator(row, registry_hash)
        )

    # Stack the six tables and create a fresh sequential row index.
    long = pd.concat(tables, ignore_index=True)

    # These three columns jointly identify one observation.
    key = [
        "indicator_id",
        "eurostat_geo",
        "eurostat_year",
    ]

    require(
        not long.duplicated(key).any(),
        "Duplicate indicator-country-year key.",
    )

    # Float64 stores decimal values and represents missing values as NaN.
    long["value"] = long["value"].astype("float64")

    # Years and reference offsets are whole numbers.
    for column in (
        "eurostat_year",
        "reference_year_offset",
    ):
        long[column] = long[column].astype("int64")

    # Quality indicators remain Boolean: True or False.
    for column in (
        "is_missing",
        "has_status_flag",
    ):
        long[column] = long[column].astype("bool")

    # Select text columns using both older and newer pandas storage types.
    # Explicitly including "string" avoids the reported pandas warning.
    text_columns = long.select_dtypes(
        include=["object", "string"]
    ).columns

    # Give all text columns the same explicit pandas type.
    for column in text_columns:
        # "string" preserves missing values rather than making them text.
        long[column] = long[column].astype("string")

    # Sort by the unique key to make repeated outputs easy to compare.
    long = long.sort_values(key).reset_index(drop=True)

    # Place the indicator identification columns first.
    front = [
        "indicator_id",
        "dataset_code",
        "indicator_label",
        "unit_label",
    ]

    # Append every other column without changing its relative order.
    long = long[
        front
        + [
            column
            for column in long.columns
            if column not in front
        ]
    ]

    # Summarize each indicator separately.
    # size counts all rows; count counts only non-missing values.
    # Summing Boolean flags counts how many rows contain True.
    summary = long.groupby(
        "indicator_id",
        sort=False,
    ).agg(
        rows=("value", "size"),
        numeric=("value", "count"),
        missing=("is_missing", "sum"),
        status_flags=("has_status_flag", "sum"),
    )

    # Confirm the overall counts before preparing either export.
    require(
        tuple(summary.sum()) == (481, 451, 30, 29),
        "Unexpected V1 totals.",
    )

    print("\nCOUNTS BY INDICATOR")
    print(summary.to_string())

    print("\nTOTALS")
    print(summary.sum().to_string())

    # Serialize in memory before creating output files.
    # If pyarrow is missing, execution stops without creating an output.
    parquet_bytes = long.to_parquet(
        index=False,
        engine="pyarrow",
    )

    # Empty CSV cells represent missing values, never artificial zeros.
    # Do not export the DataFrame row index as an extra column.
    csv_text = long.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    )

    # Create the destination folder only if it does not already exist.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # "xb" creates a new binary file and refuses to overwrite an existing one.
    with parquet_path.open("xb") as stream:
        stream.write(parquet_bytes)

    # "x" also refuses to overwrite an existing CSV.
    # UTF-8 with BOM helps Excel display international names correctly.
    with csv_path.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        stream.write(csv_text)

    print(f"\nSaved: {parquet_path}")
    print(f"Saved: {csv_path}")

    print(
        "\nEUROSTAT LONG-FORMAT TRANSFORMATION COMPLETED."
    )

    # Successful transformation does not replace independent validation.
    print(
        "Step 38 independent validation is still pending."
    )


# Execute main() only when this file is run directly.
# Importing its functions does not start the transformation.
if __name__ == "__main__":
    main()