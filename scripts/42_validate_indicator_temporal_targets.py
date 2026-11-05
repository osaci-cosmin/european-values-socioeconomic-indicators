# Step 42: Independently validate the indicator temporal targets.
# Reconstruct expected requests directly from the three configurations.
# Do not import Step 41 or modify any files.

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


# Find the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

TEMPORAL_PATH = (
    ROOT / "config" / "temporal_alignment_targets_v1.csv"
)

MAPPING_PATH = (
    ROOT / "config" / "ess_eurostat_country_mapping_v1.csv"
)

REGISTRY_PATH = (
    ROOT / "config" / "eurostat_indicators_v1.csv"
)

PARQUET_PATH = (
    ROOT / "data" / "interim"
    / "ess_eurostat_indicator_temporal_targets_v1.parquet"
)

CSV_PATH = (
    ROOT / "data" / "interim"
    / "ess_eurostat_indicator_temporal_targets_v1.csv"
)

EXPECTED_REGISTRY_HASH = (
    "8e7d2b06565fc0ba662738fb567f7756af709b85cdb96ab7afeeab8b6a969145"
)

EXPECTED_OFFSETS = {
    "gdp_pps_index": 0,
    "employment_rate": 0,
    "unemployment_rate": 0,
    "gini": 1,
    "material_deprivation_rate": 0,
    "inflation_rate": 0,
}

GINI_REFERENCE_URL = (
    "https://ec.europa.eu/eurostat/"
    "cache/metadata/en/ilc_sieusilc.htm"
)

KEY = [
    "cntry",
    "indicator_id",
    "alignment_scenario",
]

TEMPORAL_COLUMNS = [
    "cntry",
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
    "indicator_alignment_status",
]

INDICATOR_COLUMNS = [
    "indicator_id",
    "dataset_code",
    "label",
    "unit_label",
    "api_start_year",
    "api_end_year",
    "reference_year_offset",
    "reference_period_rule",
    "documentation_url",
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
]

FLOAT_COLUMNS = [
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
]


def require(condition, message):
    # Stop when a required check fails.
    if not condition:
        raise ValueError(message)


def read_configuration(path):
    # Read once so the hash and parsed table refer to the same bytes.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    # Preserve empty cells and literal Boolean tokens for explicit parsing.
    table = pd.read_csv(
        io.BytesIO(content),
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    return table, digest


def parse_boolean(value):
    # Accept Boolean values or their exact CSV representations.
    # Never use bool("False"), which would incorrectly return True.
    if isinstance(value, str):
        require(
            value in {"True", "False"},
            f"Invalid Boolean token: {value!r}",
        )
        return value == "True"

    require(
        not pd.isna(value) and value in (True, False),
        f"Invalid Boolean value: {value!r}",
    )

    return bool(value)


def normalize(table):
    # Work on a copy; leave the loaded source table unchanged.
    result = table.copy()

    for column in INTEGER_COLUMNS:
        # Only the alternative context year may be empty.
        if column == "alternative_context_year":
            values = result[column].replace("", pd.NA)
        else:
            values = result[column]

            require(
                values.notna().all()
                and values.astype("string").ne("").all(),
                f"Missing integer field: {column}",
            )

        # Python int() rejects decimal text such as "2023.5".
        result[column] = pd.array(
            [
                pd.NA if pd.isna(value) else int(value)
                for value in values
            ],
            dtype="Int64",
        )

    for column in FLOAT_COLUMNS:
        # Parse decimal strings without rounding them for display.
        result[column] = result[column].map(float).astype("float64")

        require(
            result[column].between(0, 1).all(),
            f"Invalid proportion in {column}.",
        )

    for column in BOOLEAN_COLUMNS:
        result[column] = (
            result[column].map(parse_boolean).astype("bool")
        )

    non_text_columns = set(
        INTEGER_COLUMNS + FLOAT_COLUMNS + BOOLEAN_COLUMNS
    )

    for column in result.columns:
        if column not in non_text_columns:
            require(
                result[column].notna().all(),
                f"Missing text field: {column}",
            )
            result[column] = result[column].astype("string")

    return result


def build_expected(temporal, mapping, registry, hashes):
    # The configurations were validated in earlier project steps.
    # Check their identity and scope before using them as references.
    require(
        set(temporal.columns) == set(TEMPORAL_COLUMNS),
        "Unexpected temporal configuration schema.",
    )

    require(
        len(temporal) == 29
        and not temporal["cntry"].duplicated().any(),
        "Expected 29 unique temporal country records.",
    )

    require(
        len(mapping) == 29
        and not mapping["ess_cntry"].duplicated().any()
        and not mapping["eurostat_geo"].duplicated().any(),
        "Unexpected country mapping structure.",
    )

    require(
        set(temporal["cntry"]) == set(mapping["ess_cntry"]),
        "Temporal configuration and mapping scopes differ.",
    )

    require(
        len(registry) == 6
        and not registry["indicator_id"].duplicated().any(),
        "Expected six unique registry indicators.",
    )

    country_codes = mapping.set_index(
        "ess_cntry"
    )["eurostat_geo"].to_dict()

    actual_offsets = {
        row["indicator_id"]: int(row["reference_year_offset"])
        for row in registry.to_dict("records")
    }

    require(
        actual_offsets == EXPECTED_OFFSETS,
        "Unexpected indicator reference-year offsets.",
    )

    records = []

    # Reconstruct requests using explicit loops, not Step 41's cross join.
    for country in temporal.to_dict("records"):
        scenarios = [
            ("primary", int(country["target_context_year"]))
        ]

        # An empty alternative year must not generate an alternative request.
        if country["alternative_context_year"] != "":
            scenarios.append(
                (
                    "alternative",
                    int(country["alternative_context_year"]),
                )
            )

        for scenario, context_year in scenarios:
            require(
                context_year in {2023, 2024},
                "Unexpected context year.",
            )

            # Select the share for the scenario's actual context year.
            share_column = f"weighted_share_{context_year}"

            for indicator in registry[INDICATOR_COLUMNS].to_dict(
                "records"
            ):
                offset = int(indicator["reference_year_offset"])
                requested_year = context_year + offset

                # Preserve all original ESS temporal metadata.
                record = country.copy()

                # Keep the original pending status as input history.
                record["input_indicator_alignment_status"] = (
                    record.pop("indicator_alignment_status")
                )

                record["eurostat_geo"] = country_codes[
                    country["cntry"]
                ]

                record["alignment_scenario"] = scenario
                record["context_year"] = context_year

                record["context_year_weighted_share"] = (
                    country[share_column]
                )

                # Copy the registered indicator definitions.
                record.update(indicator)
                record["indicator_label"] = record.pop("label")

                record["requested_eurostat_year"] = requested_year

                record["within_extraction_window"] = (
                    int(indicator["api_start_year"])
                    <= requested_year
                    <= int(indicator["api_end_year"])
                )

                if indicator["indicator_id"] == "gini":
                    record["reference_period_review_status"] = (
                        "general_eu_silc_income_year_rule_supported"
                    )
                    record["reference_period_evidence_url"] = (
                        GINI_REFERENCE_URL
                    )
                else:
                    record["reference_period_review_status"] = (
                        "registry_rule_applied"
                    )
                    record["reference_period_evidence_url"] = (
                        indicator["documentation_url"]
                    )

                # Targets do not yet establish observation availability.
                record["alignment_stage"] = (
                    "year_target_created_observation_match_pending"
                )

                record["temporal_config_sha256"] = hashes["temporal"]
                record["country_mapping_sha256"] = hashes["mapping"]
                record["registry_sha256"] = hashes["registry"]

                records.append(record)

    return normalize(pd.DataFrame(records))


def check_output(table, expected_columns, name):
    # Reject missing or additional columns.
    require(
        len(table.columns) == len(expected_columns)
        and set(table.columns) == set(expected_columns),
        f"{name}: unexpected output schema.",
    )

    require(
        len(table) == 252,
        f"{name}: expected 252 rows.",
    )

    require(
        not table.duplicated(KEY).any(),
        f"{name}: duplicate country-indicator-scenario keys.",
    )


def check_parquet_types(table):
    # Inspect stored types before normalizing them for comparison.
    for column in INTEGER_COLUMNS:
        require(
            is_integer_dtype(table[column].dtype),
            f"Parquet: {column} must be an integer column.",
        )

    for column in FLOAT_COLUMNS:
        require(
            is_float_dtype(table[column].dtype),
            f"Parquet: {column} must be a floating-point column.",
        )

    for column in BOOLEAN_COLUMNS:
        require(
            is_bool_dtype(table[column].dtype),
            f"Parquet: {column} must be Boolean.",
        )

    non_text_columns = set(
        INTEGER_COLUMNS + FLOAT_COLUMNS + BOOLEAN_COLUMNS
    )

    for column in table.columns:
        if column not in non_text_columns:
            require(
                is_string_dtype(table[column].dtype),
                f"Parquet: unexpected text dtype in {column}.",
            )

            require(
                table[column].map(
                    lambda value: isinstance(value, str)
                ).all(),
                f"Parquet: non-text content in {column}.",
            )


def ordered(table):
    # Compare by keys and column names, not by file ordering.
    return (
        table.sort_values(KEY)
        .reset_index(drop=True)
        .reindex(columns=sorted(table.columns))
    )


def compare_with_sources(actual, expected, name):
    actual = ordered(actual)
    expected = ordered(expected)

    # Codes, years, flags, statuses and hashes must match exactly.
    exact_columns = [
        column
        for column in expected.columns
        if column not in FLOAT_COLUMNS
    ]

    assert_frame_equal(
        actual[exact_columns],
        expected[exact_columns],
        check_exact=True,
        obj=f"{name}: exact fields",
    )

    # Decimal parsing can create tiny binary floating-point differences.
    # Allow only an absolute tolerance of 1e-12 for interview shares.
    assert_frame_equal(
        actual[FLOAT_COLUMNS],
        expected[FLOAT_COLUMNS],
        check_exact=False,
        rtol=0,
        atol=1e-12,
        obj=f"{name}: interview shares",
    )

    difference = (
        actual[FLOAT_COLUMNS] - expected[FLOAT_COLUMNS]
    ).abs().max().max()

    print(f"{name}: PASSED.")
    print(f"Maximum share difference: {difference:.3e}")


def main():
    print("STEP 42 — INDICATOR TEMPORAL TARGET VALIDATION")

    for path in (
        TEMPORAL_PATH,
        MAPPING_PATH,
        REGISTRY_PATH,
        PARQUET_PATH,
        CSV_PATH,
    ):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    temporal, temporal_hash = read_configuration(TEMPORAL_PATH)
    mapping, mapping_hash = read_configuration(MAPPING_PATH)
    registry, registry_hash = read_configuration(REGISTRY_PATH)

    require(
        registry_hash == EXPECTED_REGISTRY_HASH,
        "Registry differs from the validated V1 version.",
    )

    hashes = {
        "temporal": temporal_hash,
        "mapping": mapping_hash,
        "registry": registry_hash,
    }

    # Reconstruct every expected request from configuration inputs.
    expected = build_expected(
        temporal,
        mapping,
        registry,
        hashes,
    )

    parquet = pd.read_parquet(
        PARQUET_PATH,
        engine="pyarrow",
    )

    # Read CSV literally before applying its expected data types.
    csv = pd.read_csv(
        CSV_PATH,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    check_output(parquet, expected.columns, "Parquet")
    check_output(csv, expected.columns, "CSV")
    check_parquet_types(parquet)

    print("\nSchema, unique keys and Parquet types: PASSED.")

    parquet = normalize(parquet)
    csv = normalize(csv)

    # Check all fields against the independently reconstructed requests.
    compare_with_sources(
        parquet,
        expected,
        "Parquet versus source configurations",
    )

    compare_with_sources(
        csv,
        expected,
        "CSV versus source configurations",
    )

    # CSV and Parquet should represent exactly the same exported values.
    assert_frame_equal(
        ordered(csv),
        ordered(parquet),
        check_exact=True,
        obj="CSV versus Parquet",
    )

    print("CSV versus Parquet: exact comparison PASSED.")

    counts = parquet["alignment_scenario"].value_counts().to_dict()

    require(
        counts == {"primary": 174, "alternative": 78},
        "Unexpected scenario counts.",
    )

    # Every country-scenario combination must contain all six indicators.
    indicators_per_scenario = parquet.groupby(
        ["cntry", "alignment_scenario"]
    )["indicator_id"].nunique()

    require(
        len(indicators_per_scenario) == 42
        and indicators_per_scenario.eq(6).all(),
        "Incomplete country-scenario indicator coverage.",
    )

    primary = parquet.loc[
        parquet["alignment_scenario"].eq("primary")
    ]

    alternative = parquet.loc[
        parquet["alignment_scenario"].eq("alternative")
    ]

    require(
        primary["cntry"].nunique() == 29,
        "Expected 29 primary-scenario countries.",
    )

    require(
        alternative["cntry"].nunique() == 13,
        "Expected 13 alternative-scenario countries.",
    )

    require(
        primary["context_year"].eq(
            primary["target_context_year"]
        ).all(),
        "Incorrect primary context years.",
    )

    require(
        alternative["context_year"].eq(
            alternative["alternative_context_year"]
        ).all(),
        "Incorrect alternative context years.",
    )

    # Check year arithmetic separately from the full table comparison.
    require(
        parquet["requested_eurostat_year"].sub(
            parquet["context_year"]
        ).eq(parquet["reference_year_offset"]).all(),
        "Requested years do not match the registered offsets.",
    )

    require(
        parquet["within_extraction_window"].all(),
        "Some requests fall outside the extracted year windows.",
    )

    print("Scenario coverage and requested-year rules: PASSED.")

    # Report how requests are distributed across indicator-year pairs.
    summary = parquet.groupby(
        [
            "indicator_id",
            "alignment_scenario",
            "requested_eurostat_year",
        ]
    ).size().rename("requests")

    print("\nVALIDATED REQUESTS")
    print(summary.to_string())

    print("\nSUMMARY")
    print(f"Primary requests: {len(primary)}")
    print(f"Alternative requests: {len(alternative)}")
    print(f"Total requests: {len(parquet)}")

    # Success here concerns temporal targets, not economic data availability.
    print("\nINDICATOR TEMPORAL TARGET VALIDATION PASSED.")
    print("Country codes, years, shares and provenance were checked.")
    print("Economic observation availability has not been checked yet.")
    print("No files were modified.")


# Run only when this script is executed directly.
if __name__ == "__main__":
    main()