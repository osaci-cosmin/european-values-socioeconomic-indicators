# Step 46: Independently validate the main socioeconomic country table.
# Reconstruct each country record directly from the matched observations.
# Do not import Step 45, use pivot(), or modify any files.

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

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "ess_eurostat_temporal_matches_v1.parquet"
)

PARQUET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ESS11_country_socioeconomic_main_v1.parquet"
)

CSV_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ESS11_country_socioeconomic_main_v1.csv"
)


INDICATORS = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

COUNTRY_COLUMNS = [
    "cntry",
    "eurostat_geo",
    "fieldwork_year_min",
    "fieldwork_year_max",
    "dominant_fieldwork_year",
    "dominant_year_weighted_share",
    "multi_year_fieldwork_flag",
    "mixed_fieldwork_flag",
    "dominant_year_stable_flag",
    "weighted_share_2023",
    "weighted_share_2024",
    "target_context_year",
    "alternative_context_year",
    "alignment_rule",
    "temporal_sensitivity_flag",
    "temporal_sensitivity_priority_flag",
    "alignment_scenario",
    "context_year",
    "context_year_weighted_share",
    "temporal_config_sha256",
    "country_mapping_sha256",
    "registry_sha256",
    "temporal_targets_sha256",
    "eurostat_long_sha256",
]

INDICATOR_METADATA = [
    "dataset_code",
    "indicator_label",
    "unit_label",
    "requested_eurostat_year",
    "eurostat_year",
    "reference_year_offset",
    "reference_period_rule",
    "reference_period_review_status",
    "reference_period_evidence_url",
    "documentation_url",
    "observation_match_status",
    "source_cell_found",
    "has_numeric_value",
    "eurostat_status",
    "source_is_missing",
    "source_has_status_flag",
    "matched_year_gap",
    "lookup_snapshot_sha256",
    "lookup_raw_filename",
    "lookup_metadata_filename",
    "lookup_source_api_url",
    "lookup_api_dataset_updated",
    "lookup_retrieved_at_utc",
]

EXPECTED_AVAILABLE = {
    "gdp_pps_index": 28,
    "employment_rate": 26,
    "unemployment_rate": 26,
    "gini": 25,
    "material_deprivation_rate": 26,
    "inflation_rate": 27,
}


# Define data types explicitly instead of guessing from CSV values.
INTEGER_COLUMNS = [
    "fieldwork_year_min",
    "fieldwork_year_max",
    "dominant_fieldwork_year",
    "target_context_year",
    "alternative_context_year",
    "context_year",
    "available_indicators_n",
    "missing_indicators_n",
]

NULLABLE_INTEGER_COLUMNS = {
    "alternative_context_year",
}

BOOLEAN_COLUMNS = [
    "multi_year_fieldwork_flag",
    "mixed_fieldwork_flag",
    "dominant_year_stable_flag",
    "temporal_sensitivity_flag",
    "temporal_sensitivity_priority_flag",
    "all_six_indicators_available",
]

NULLABLE_BOOLEAN_COLUMNS = []

SHARE_COLUMNS = [
    "dominant_year_weighted_share",
    "weighted_share_2023",
    "weighted_share_2024",
    "context_year_weighted_share",
]

# Add the corresponding metadata types for every indicator.
for indicator in INDICATORS:
    for field in (
        "requested_eurostat_year",
        "eurostat_year",
        "reference_year_offset",
        "matched_year_gap",
    ):
        INTEGER_COLUMNS.append(f"{indicator}__{field}")

    for field in (
        "eurostat_year",
        "matched_year_gap",
    ):
        NULLABLE_INTEGER_COLUMNS.add(f"{indicator}__{field}")

    for field in (
        "source_cell_found",
        "has_numeric_value",
    ):
        BOOLEAN_COLUMNS.append(f"{indicator}__{field}")

    for field in (
        "source_is_missing",
        "source_has_status_flag",
    ):
        NULLABLE_BOOLEAN_COLUMNS.append(f"{indicator}__{field}")


def require(condition, message):
    # Stop immediately when a required check fails.
    if not condition:
        raise ValueError(message)


def build_expected(matches, input_hash):
    # Verify the input scope before selecting the primary scenario.
    require(
        len(matches) == 252,
        "Expected 252 records in the matched input.",
    )

    require(
        matches["alignment_scenario"].value_counts().to_dict()
        == {"primary": 174, "alternative": 78},
        "Unexpected input scenario counts.",
    )

    primary = matches.loc[
        matches["alignment_scenario"].eq("primary")
    ].copy()

    require(
        not primary.duplicated(["cntry", "indicator_id"]).any(),
        "Duplicate primary country-indicator observations.",
    )

    require(
        primary["cntry"].nunique() == 29,
        "Expected 29 primary-scenario countries.",
    )

    records = []

    # Reconstruct one wide record at a time, without pivoting.
    for country, group in primary.groupby("cntry", sort=True):
        require(
            len(group) == 6
            and set(group["indicator_id"]) == set(INDICATORS),
            f"Incomplete indicator requests for {country}.",
        )

        # Country-level metadata must agree across all six observations.
        require(
            group[COUNTRY_COLUMNS].nunique(
                dropna=False
            ).eq(1).all(),
            f"Inconsistent country metadata for {country}.",
        )

        # Once consistency is checked, any row supplies country metadata.
        record = group.iloc[0][COUNTRY_COLUMNS].to_dict()

        observations = {
            row["indicator_id"]: row
            for row in group.to_dict("records")
        }

        available_count = 0

        for indicator in INDICATORS:
            observation = observations[indicator]

            # Copy the actual primary value, including a missing value.
            record[indicator] = observation["value"]

            if not pd.isna(observation["value"]):
                available_count += 1

            # Copy every indicator-specific metadata field directly.
            for field in INDICATOR_METADATA:
                record[f"{indicator}__{field}"] = observation[field]

        record["available_indicators_n"] = available_count
        record["missing_indicators_n"] = 6 - available_count
        record["all_six_indicators_available"] = available_count == 6

        record["matched_input_filename"] = INPUT_PATH.name
        record["matched_input_sha256"] = input_hash

        record["dataset_stage"] = (
            "country_main_created_independent_validation_pending"
        )

        records.append(record)

    return pd.DataFrame(records)


def parse_boolean(value, nullable=False):
    # Source flags can be unknown when the source cell is absent.
    missing = pd.isna(value) or (
        isinstance(value, str) and value == ""
    )

    if missing:
        require(
            nullable,
            "Unexpected missing Boolean value.",
        )
        return pd.NA

    # Parse CSV tokens explicitly instead of using bool("False").
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
    # Normalize a copy while preserving the original loaded data.
    result = table.copy()

    for column in INTEGER_COLUMNS:
        converted = []

        for value in result[column]:
            missing = pd.isna(value) or (
                isinstance(value, str) and value == ""
            )

            if missing:
                require(
                    column in NULLABLE_INTEGER_COLUMNS,
                    f"Unexpected missing integer in {column}.",
                )
                converted.append(pd.NA)

            else:
                # Prevent fractional numeric values from being truncated.
                if not isinstance(value, str):
                    require(
                        value == int(value),
                        f"Fractional value in {column}.",
                    )

                converted.append(int(value))

        result[column] = pd.array(
            converted,
            dtype="Int64",
        )

    for column in SHARE_COLUMNS:
        result[column] = (
            result[column].map(float).astype("float64")
        )

        require(
            result[column].between(0, 1).all(),
            f"Invalid interview share in {column}.",
        )

    # Economic values may be missing; empty CSV cells represent NaN.
    for indicator in INDICATORS:
        result[indicator] = result[indicator].map(
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
        # CSV cannot distinguish an empty status from missing status text.
        # Use the source-cell flag to restore this distinction.
        for indicator in INDICATORS:
            absent = ~result[f"{indicator}__source_cell_found"]
            status_column = f"{indicator}__eurostat_status"

            # Check before conversion so unexpected content is not hidden.
            require(
                result.loc[absent, status_column].eq("").all(),
                f"CSV has status text for absent {indicator} cells.",
            )

            result.loc[absent, status_column] = pd.NA

    non_text_columns = set(
        INTEGER_COLUMNS
        + SHARE_COLUMNS
        + INDICATORS
        + BOOLEAN_COLUMNS
        + NULLABLE_BOOLEAN_COLUMNS
    )

    for column in result.columns:
        if column not in non_text_columns:
            result[column] = result[column].astype("string")

    return result


def check_structure(table, expected, name):
    # Reject additional or missing fields as well as duplicated countries.
    require(
        len(table.columns) == len(expected.columns)
        and set(table.columns) == set(expected.columns),
        f"{name}: unexpected columns.",
    )

    require(
        len(table) == 29
        and table["cntry"].notna().all()
        and not table["cntry"].duplicated().any(),
        f"{name}: expected 29 unique country rows.",
    )

    require(
        set(table["cntry"]) == set(expected["cntry"]),
        f"{name}: country scope differs from the primary input.",
    )


def check_parquet_types(table):
    # Inspect stored data types before comparison conversions.
    for column in INTEGER_COLUMNS:
        require(
            is_integer_dtype(table[column].dtype),
            f"Parquet: invalid integer type in {column}.",
        )

    for column in SHARE_COLUMNS + INDICATORS:
        require(
            is_float_dtype(table[column].dtype),
            f"Parquet: invalid numeric type in {column}.",
        )

    for column in BOOLEAN_COLUMNS + NULLABLE_BOOLEAN_COLUMNS:
        require(
            is_bool_dtype(table[column].dtype),
            f"Parquet: invalid Boolean type in {column}.",
        )

    non_text_columns = set(
        INTEGER_COLUMNS
        + SHARE_COLUMNS
        + INDICATORS
        + BOOLEAN_COLUMNS
        + NULLABLE_BOOLEAN_COLUMNS
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
    # Compare by country and field name rather than storage order.
    return (
        table.sort_values("cntry")
        .reset_index(drop=True)
        .reindex(columns=sorted(table.columns))
    )


def compare_tables(actual, expected, name):
    # Step 45 copied values without recalculating them.
    # Exact equality is expected for all fields.
    assert_frame_equal(
        ordered(actual),
        ordered(expected),
        check_exact=True,
        obj=name,
    )

    print(f"{name}: exact comparison PASSED.")


def main():
    print("STEP 46 — MAIN SOCIOECONOMIC TABLE VALIDATION")

    for path in (INPUT_PATH, PARQUET_PATH, CSV_PATH):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    # Hash the same matched-table bytes used for reconstruction.
    input_bytes = INPUT_PATH.read_bytes()
    input_hash = hashlib.sha256(input_bytes).hexdigest()

    matches = pd.read_parquet(
        io.BytesIO(input_bytes),
        engine="pyarrow",
    )

    expected = build_expected(
        matches,
        input_hash,
    )

    parquet = pd.read_parquet(
        PARQUET_PATH,
        engine="pyarrow",
    )

    # Preserve empty fields until their intended type is known.
    csv = pd.read_csv(
        CSV_PATH,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    check_structure(parquet, expected, "Parquet")
    check_structure(csv, expected, "CSV")
    check_parquet_types(parquet)

    print("\nSchema, country scope and Parquet types: PASSED.")

    expected = normalize(expected)
    parquet = normalize(parquet)
    csv = normalize(csv, csv_input=True)

    # Compare all values and metadata with the independent reconstruction.
    compare_tables(
        parquet,
        expected,
        "Parquet versus primary matched observations",
    )

    compare_tables(
        csv,
        expected,
        "CSV versus primary matched observations",
    )

    compare_tables(
        csv,
        parquet,
        "CSV versus Parquet",
    )

    require(
        parquet["alignment_scenario"].eq("primary").all(),
        "Non-primary scenarios found in the main country table.",
    )

    require(
        parquet["context_year"].eq(
            parquet["target_context_year"]
        ).all(),
        "Country context years differ from the primary target years.",
    )

    # Recalculate country completeness from the six analytical columns.
    actual_available = parquet[INDICATORS].notna().sum(axis=1)

    require(
        parquet["available_indicators_n"].eq(actual_available).all(),
        "Incorrect available-indicator counts.",
    )

    require(
        parquet["missing_indicators_n"].eq(
            6 - actual_available
        ).all(),
        "Incorrect missing-indicator counts.",
    )

    require(
        parquet["all_six_indicators_available"].eq(
            actual_available.eq(6)
        ).all(),
        "Incorrect country-completeness flags.",
    )

    # Check that wide values still agree with their indicator metadata.
    for indicator in INDICATORS:
        numeric = parquet[indicator].notna()

        require(
            parquet[f"{indicator}__has_numeric_value"].eq(
                numeric
            ).all(),
            f"Availability flag mismatch: {indicator}",
        )

        require(
            parquet[f"{indicator}__observation_match_status"].eq(
                "available"
            ).eq(numeric).all(),
            f"Observation-status mismatch: {indicator}",
        )

        require(
            int(numeric.sum()) == EXPECTED_AVAILABLE[indicator],
            f"Unexpected available-country count: {indicator}",
        )

        # Verify the requested-year rule for every country.
        require(
            parquet[f"{indicator}__requested_eurostat_year"].eq(
                parquet["context_year"]
                + parquet[f"{indicator}__reference_year_offset"]
            ).all(),
            f"Requested-year mismatch: {indicator}",
        )

    print("Primary scenario, years and completeness flags: PASSED.")

    # Confirm that incomplete countries were retained with correct counts.
    incomplete_counts = (
        parquet.loc[
            ~parquet["all_six_indicators_available"]
        ]
        .set_index("cntry")["missing_indicators_n"]
        .to_dict()
    )

    require(
        incomplete_counts == {
            "GB": 5,
            "IS": 2,
            "ME": 3,
            "UA": 6,
        },
        "Unexpected missing-indicator distribution by country.",
    )

    available_total = int(actual_available.sum())
    missing_total = int(29 * 6 - available_total)
    complete_countries = int(actual_available.eq(6).sum())

    require(
        (available_total, missing_total, complete_countries)
        == (158, 16, 25),
        "Overall counts differ from the reported Step 45 result.",
    )

    print("\nVALIDATED AVAILABILITY BY INDICATOR")
    print(
        parquet[INDICATORS]
        .notna()
        .sum()
        .rename("available_countries")
        .to_string()
    )

    print("\nCOUNTRIES WITH UNAVAILABLE INDICATORS")
    print(
        parquet.loc[
            ~parquet["all_six_indicators_available"],
            [
                "cntry",
                "available_indicators_n",
                "missing_indicators_n",
            ],
        ].to_string(index=False)
    )

    print("\nSUMMARY")
    print(f"Countries retained: {len(parquet)}")
    print(f"Analytical indicators: {len(INDICATORS)}")
    print(f"Available numeric values: {available_total}")
    print(f"Unavailable values: {missing_total}")
    print(f"Countries with all six indicators: {complete_countries}")

    # Print success only after every comparison and check has passed.
    print("\nMAIN SOCIOECONOMIC TABLE VALIDATION PASSED.")
    print("All country fields, indicator values and metadata matched.")
    print("No country loss, imputation or scenario substitution detected.")
    print("No files were modified.")


# Importing this script does not execute the validation.
if __name__ == "__main__":
    main()