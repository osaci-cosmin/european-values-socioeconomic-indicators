# Step 44: Independently validate the Eurostat temporal matches.
# Reconstruct expected matches with dictionary lookups, not merge().
# Read existing files only; do not modify datasets.

import hashlib
import io
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)
from pandas.testing import assert_frame_equal


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

TARGETS_PATH = (
    INTERIM / "ess_eurostat_indicator_temporal_targets_v1.parquet"
)

SOURCE_PATH = (
    INTERIM / "eurostat_country_year_long.parquet"
)

PARQUET_PATH = (
    INTERIM / "ess_eurostat_temporal_matches_v1.parquet"
)

CSV_PATH = (
    INTERIM / "ess_eurostat_temporal_matches_v1.csv"
)


# Identify one request, separately from its source observation.
REQUEST_KEY = [
    "cntry",
    "indicator_id",
    "alignment_scenario",
]

SOURCE_KEY = [
    "indicator_id",
    "eurostat_geo",
    "eurostat_year",
]

DEFINITION_COLUMNS = [
    "dataset_code",
    "indicator_label",
    "unit_label",
    "reference_year_offset",
    "reference_period_rule",
    "registry_sha256",
]

# Identify the snapshot searched, even when no source cell was found.
LOOKUP_COLUMNS = [
    "snapshot_sha256",
    "raw_filename",
    "metadata_filename",
    "source_api_url",
    "api_dataset_updated",
    "retrieved_at_utc",
]

INTEGER_COLUMNS = [
    "fieldwork_year_min",
    "fieldwork_year_max",
    "dominant_fieldwork_year",
    "target_context_year",
    "alternative_context_year",
    "context_year",
    "api_start_year",
    "api_end_year",
    "reference_year_offset",
    "requested_eurostat_year",
    "eurostat_year",
    "matched_year_gap",
]

# These integer fields can legitimately contain missing values.
NULLABLE_INTEGER_COLUMNS = {
    "alternative_context_year",
    "eurostat_year",
    "matched_year_gap",
}

SHARE_COLUMNS = [
    "dominant_year_weighted_share",
    "weighted_share_2023",
    "weighted_share_2024",
    "context_year_weighted_share",
]

BOOLEAN_COLUMNS = [
    "multi_year_fieldwork_flag",
    "mixed_fieldwork_flag",
    "dominant_year_stable_flag",
    "temporal_sensitivity_flag",
    "temporal_sensitivity_priority_flag",
    "within_extraction_window",
    "source_cell_found",
    "has_numeric_value",
]

# An absent source cell has unknown source flags, not False flags.
NULLABLE_BOOLEAN_COLUMNS = [
    "source_is_missing",
    "source_has_status_flag",
]

MATCH_STATUSES = [
    "available",
    "source_value_missing",
    "source_cell_absent",
]


def require(condition, message):
    # Stop immediately when a required validation fails.
    if not condition:
        raise ValueError(message)


def read_parquet_snapshot(path):
    # Hash the exact bytes used to load the table.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    table = pd.read_parquet(
        io.BytesIO(content),
        engine="pyarrow",
    )

    return table, digest


def build_expected(targets, source, targets_hash, source_hash):
    # Check uniqueness before building dictionaries.
    # Otherwise, duplicated keys could overwrite earlier records.
    require(
        len(targets) == 252
        and not targets.duplicated(REQUEST_KEY).any(),
        "Expected 252 unique temporal requests.",
    )

    require(
        len(source) == 481
        and not source.duplicated(SOURCE_KEY).any(),
        "Expected 481 unique Eurostat source cells.",
    )

    require(
        targets[
            ["indicator_id", "eurostat_geo", "requested_eurostat_year"]
        ].notna().all().all(),
        "Missing target lookup keys.",
    )

    require(
        source[SOURCE_KEY].notna().all().all(),
        "Missing source lookup keys.",
    )

    require(
        targets["alignment_stage"].eq(
            "year_target_created_observation_match_pending"
        ).all(),
        "Unexpected target input stage.",
    )

    # Map each source key directly to its complete observation.
    observations = {}

    for observation in source.to_dict("records"):
        key = (
            observation["indicator_id"],
            observation["eurostat_geo"],
            int(observation["eurostat_year"]),
        )
        observations[key] = observation

    # Build a separate indicator-level snapshot reference.
    metadata_columns = (
        ["indicator_id"]
        + DEFINITION_COLUMNS
        + LOOKUP_COLUMNS
    )

    metadata = source[metadata_columns].drop_duplicates()

    require(
        len(metadata) == 6
        and not metadata["indicator_id"].duplicated().any(),
        "Expected one consistent source snapshot per indicator.",
    )

    require(
        metadata.notna().all().all(),
        "Missing source definitions or snapshot provenance.",
    )

    manifests = {
        row["indicator_id"]: row
        for row in metadata.to_dict("records")
    }

    require(
        set(targets["indicator_id"]) == set(manifests),
        "Target and source indicator sets differ.",
    )

    records = []

    for request in targets.to_dict("records"):
        indicator = request["indicator_id"]
        manifest = manifests[indicator]

        # Check definitions even when the requested source cell is absent.
        for column in DEFINITION_COLUMNS:
            require(
                request[column] == manifest[column],
                f"Target/source mismatch: {indicator} / {column}",
            )

        # Retain all original target fields.
        expected = request.copy()

        expected["input_alignment_stage"] = expected.pop(
            "alignment_stage"
        )

        # Attach the searched snapshot to every request.
        for column in LOOKUP_COLUMNS:
            expected[f"lookup_{column}"] = manifest[column]

        lookup_key = (
            indicator,
            request["eurostat_geo"],
            int(request["requested_eurostat_year"]),
        )

        # Membership checks distinguish an absent cell from a missing value.
        found = lookup_key in observations

        if found:
            observation = observations[lookup_key]
            numeric = not pd.isna(observation["value"])

            expected["eurostat_year"] = int(
                observation["eurostat_year"]
            )
            expected["source_geo_label"] = (
                observation["eurostat_geo_label"]
            )
            expected["value"] = observation["value"]
            expected["eurostat_status"] = (
                observation["eurostat_status"]
            )
            expected["source_is_missing"] = (
                observation["is_missing"]
            )
            expected["source_has_status_flag"] = (
                observation["has_status_flag"]
            )

            expected["observation_match_status"] = (
                "available" if numeric else "source_value_missing"
            )

            expected["matched_year_gap"] = (
                int(observation["eurostat_year"])
                - int(request["requested_eurostat_year"])
            )

        else:
            numeric = False

            # No source cell means no observed year, label or source flags.
            for column in (
                "eurostat_year",
                "source_geo_label",
                "eurostat_status",
                "source_is_missing",
                "source_has_status_flag",
                "matched_year_gap",
            ):
                expected[column] = pd.NA

            expected["value"] = float("nan")
            expected["observation_match_status"] = (
                "source_cell_absent"
            )

        expected["source_cell_found"] = found
        expected["has_numeric_value"] = numeric

        expected["alignment_stage"] = (
            "observation_match_completed_validation_pending"
        )

        expected["temporal_targets_sha256"] = targets_hash
        expected["eurostat_long_sha256"] = source_hash

        records.append(expected)

    return pd.DataFrame(records)


def parse_boolean(value, nullable=False):
    # Missing is allowed only for source flags of absent cells.
    if pd.isna(value) or (
        isinstance(value, str) and value == ""
    ):
        require(nullable, "Unexpected missing Boolean value.")
        return pd.NA

    # Parse text explicitly: bool("False") would incorrectly return True.
    if isinstance(value, str):
        require(
            value in {"True", "False"},
            f"Invalid Boolean token: {value!r}",
        )
        return value == "True"

    require(
        value in (True, False),
        f"Invalid Boolean value: {value!r}",
    )

    return bool(value)


def normalize(table, csv_input=False):
    # Normalize a copy so the original loaded table is not changed.
    result = table.copy()

    for column in INTEGER_COLUMNS:
        values = []

        for value in result[column]:
            missing = pd.isna(value) or (
                isinstance(value, str) and value == ""
            )

            if missing:
                require(
                    column in NULLABLE_INTEGER_COLUMNS,
                    f"Unexpected missing integer: {column}",
                )
                values.append(pd.NA)

            else:
                # Reject fractional numeric values rather than truncating.
                if not isinstance(value, str):
                    require(
                        value == int(value),
                        f"Fractional value in {column}.",
                    )

                values.append(int(value))

        result[column] = pd.array(
            values,
            dtype="Int64",
        )

    for column in SHARE_COLUMNS:
        result[column] = (
            result[column].map(float).astype("float64")
        )

        require(
            result[column].between(0, 1).all(),
            f"Invalid interview share: {column}",
        )

    # Only the economic value column accepts empty numeric CSV cells.
    result["value"] = result["value"].map(
        lambda value: (
            float("nan")
            if pd.isna(value)
            or (isinstance(value, str) and value == "")
            else float(value)
        )
    ).astype("float64")

    for column in BOOLEAN_COLUMNS:
        result[column] = (
            result[column].map(parse_boolean).astype("bool")
        )

    for column in NULLABLE_BOOLEAN_COLUMNS:
        result[column] = result[column].map(
            lambda value: parse_boolean(value, nullable=True)
        ).astype("boolean")

    if csv_input:
        # CSV writes both missing text and empty status strings as blanks.
        # source_cell_found tells us which interpretation is correct.
        absent = ~result["source_cell_found"]

        for column in ("source_geo_label", "eurostat_status"):
            # Check before conversion so unexpected content is not hidden.
            require(
                result.loc[absent, column].eq("").all(),
                f"CSV: unexpected text for absent cells in {column}.",
            )

            result.loc[absent, column] = pd.NA

    non_text_columns = set(
        INTEGER_COLUMNS
        + SHARE_COLUMNS
        + BOOLEAN_COLUMNS
        + NULLABLE_BOOLEAN_COLUMNS
        + ["value"]
    )

    for column in result.columns:
        if column not in non_text_columns:
            result[column] = result[column].astype("string")

    return result


def check_structure(table, expected, name):
    require(
        len(table) == 252,
        f"{name}: expected 252 requests.",
    )

    require(
        len(table.columns) == len(expected.columns)
        and set(table.columns) == set(expected.columns),
        f"{name}: unexpected columns.",
    )

    require(
        not table.duplicated(REQUEST_KEY).any(),
        f"{name}: duplicated request keys.",
    )


def check_parquet_types(table):
    # Inspect stored types before normalizing for comparison.
    for column in INTEGER_COLUMNS:
        require(
            is_integer_dtype(table[column].dtype),
            f"Parquet: invalid integer type in {column}.",
        )

    for column in SHARE_COLUMNS + ["value"]:
        require(
            is_float_dtype(table[column].dtype),
            f"Parquet: invalid floating-point type in {column}.",
        )

    for column in BOOLEAN_COLUMNS + NULLABLE_BOOLEAN_COLUMNS:
        require(
            is_bool_dtype(table[column].dtype),
            f"Parquet: invalid Boolean type in {column}.",
        )

    non_text_columns = set(
        INTEGER_COLUMNS
        + SHARE_COLUMNS
        + BOOLEAN_COLUMNS
        + NULLABLE_BOOLEAN_COLUMNS
        + ["value"]
    )

    for column in table.columns:
        if column not in non_text_columns:
            require(
                is_string_dtype(table[column].dtype),
                f"Parquet: invalid text type in {column}.",
            )

            require(
                table[column].dropna().map(
                    lambda value: isinstance(value, str)
                ).all(),
                f"Parquet: non-text content in {column}.",
            )


def ordered(table):
    # Ignore storage order while comparing every named field and request.
    return (
        table.sort_values(REQUEST_KEY)
        .reset_index(drop=True)
        .reindex(columns=sorted(table.columns))
    )


def compare_tables(actual, expected, name):
    # Step 43 copies values from Parquet inputs without recalculating them.
    # Exact equality is therefore expected, including numeric values.
    assert_frame_equal(
        ordered(actual),
        ordered(expected),
        check_exact=True,
        obj=name,
    )

    print(f"{name}: exact comparison PASSED.")


def main():
    print("STEP 44 — EUROSTAT TEMPORAL MATCH VALIDATION")

    for path in (
        TARGETS_PATH,
        SOURCE_PATH,
        PARQUET_PATH,
        CSV_PATH,
    ):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    targets, targets_hash = read_parquet_snapshot(TARGETS_PATH)
    source, source_hash = read_parquet_snapshot(SOURCE_PATH)

    # Reconstruct expected matches independently from Step 43.
    expected = build_expected(
        targets,
        source,
        targets_hash,
        source_hash,
    )

    parquet = pd.read_parquet(
        PARQUET_PATH,
        engine="pyarrow",
    )

    # Preserve empty strings until their meaning is checked.
    csv = pd.read_csv(
        CSV_PATH,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    check_structure(parquet, expected, "Parquet")
    check_structure(csv, expected, "CSV")
    check_parquet_types(parquet)

    print("\nSchema, unique keys and Parquet types: PASSED.")

    expected = normalize(expected)
    parquet = normalize(parquet)
    csv = normalize(csv, csv_input=True)

    # Compare all fields, not just values and totals.
    compare_tables(
        parquet,
        expected,
        "Parquet versus independent lookup",
    )

    compare_tables(
        csv,
        expected,
        "CSV versus independent lookup",
    )

    compare_tables(
        csv,
        parquet,
        "CSV versus Parquet",
    )

    # Verify the three availability categories separately.
    found = parquet["source_cell_found"]
    numeric = parquet["value"].notna()
    status = parquet["observation_match_status"]

    require(
        parquet["has_numeric_value"].eq(numeric).all(),
        "Numeric availability flags disagree with values.",
    )

    require(
        status.eq("available").eq(found & numeric).all(),
        "Incorrect classification of available observations.",
    )

    require(
        status.eq("source_value_missing").eq(
            found & ~numeric
        ).all(),
        "Incorrect classification of missing source values.",
    )

    require(
        status.eq("source_cell_absent").eq(~found).all(),
        "Incorrect classification of absent source cells.",
    )

    require(
        status.isin(MATCH_STATUSES).all(),
        "Unexpected observation-match category.",
    )

    require(
        parquet.loc[found, "matched_year_gap"].eq(0).all(),
        "Found observations do not use the requested years.",
    )

    require(
        parquet.loc[~found, "matched_year_gap"].isna().all(),
        "Absent observations have an invented year gap.",
    )

    print("Availability categories and exact-year matching: PASSED.")

    # Retained targets must still have the original scenario counts.
    require(
        parquet["alignment_scenario"].value_counts().to_dict()
        == {"primary": 174, "alternative": 78},
        "Unexpected retained scenario counts.",
    )

    # Summarize the independently validated results.
    summary = (
        parquet.groupby(
            [
                "alignment_scenario",
                "indicator_id",
                "observation_match_status",
            ]
        )
        .size()
        .unstack("observation_match_status", fill_value=0)
        .reindex(columns=MATCH_STATUSES, fill_value=0)
    )

    summary["requests"] = summary[MATCH_STATUSES].sum(axis=1)

    print("\nVALIDATED AVAILABILITY")
    print(summary.to_string())

    available_count = int(status.eq("available").sum())
    missing_count = int(status.eq("source_value_missing").sum())
    absent_count = int(status.eq("source_cell_absent").sum())

    # Check agreement with the reported Step 43 output.
    # These totals supplement, rather than replace, the cell comparisons.
    require(
        (available_count, missing_count, absent_count)
        == (235, 9, 8),
        "Availability totals differ from the reported Step 43 result.",
    )

    flagged_count = int(
        parquet["source_has_status_flag"].fillna(False).sum()
    )

    require(
        flagged_count == 16,
        "Flagged-request count differs from Step 43.",
    )

    print("\nSUMMARY")
    print(f"Requests retained: {len(parquet)}")
    print(f"Source cells found: {int(found.sum())}")
    print(f"Available numeric values: {available_count}")
    print(f"Found cells with missing values: {missing_count}")
    print(f"Absent source cells: {absent_count}")
    print(f"Requests matched to flagged cells: {flagged_count}")

    # This message is reached only after every check succeeds.
    print("\nEUROSTAT TEMPORAL MATCH VALIDATION PASSED.")
    print("All request fields, values, statuses and provenance matched.")
    print("No automatic year substitution was detected.")
    print("No files were modified.")


# Importing this file must not run the validation automatically.
if __name__ == "__main__":
    main()