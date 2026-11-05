# Step 41: Define which Eurostat year to request for each country and indicator.
# Include the main ESS context year and the available alternative year.
# This step creates targets only; it does not select economic values.

import hashlib
import io
from pathlib import Path

import pandas as pd


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

TEMPORAL_PATH = (
    ROOT
    / "config"
    / "temporal_alignment_targets_v1.csv"
)

MAPPING_PATH = (
    ROOT
    / "config"
    / "ess_eurostat_country_mapping_v1.csv"
)

REGISTRY_PATH = (
    ROOT
    / "config"
    / "eurostat_indicators_v1.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

PARQUET_PATH = (
    OUTPUT_DIR
    / "ess_eurostat_indicator_temporal_targets_v1.parquet"
)

CSV_PATH = (
    OUTPUT_DIR
    / "ess_eurostat_indicator_temporal_targets_v1.csv"
)


# Keep the registry version already used for the validated RAW snapshots.
EXPECTED_REGISTRY_HASH = (
    "8e7d2b06565fc0ba662738fb567f7756af709b85cdb96ab7afeeab8b6a969145"
)

# These expected offsets guard against unintended changes.
# The actual offsets used below are read from the registry.
EXPECTED_OFFSETS = {
    "gdp_pps_index": 0,
    "employment_rate": 0,
    "unemployment_rate": 0,
    "gini": 1,
    "material_deprivation_rate": 0,
    "inflation_rate": 0,
}

# Official evidence supporting the general EU-SILC income-year rule.
GINI_REFERENCE_URL = (
    "https://ec.europa.eu/eurostat/"
    "cache/metadata/en/ilc_sieusilc.htm"
)

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

BOOLEAN_COLUMNS = [
    "multi_year_fieldwork_flag",
    "mixed_fieldwork_flag",
    "dominant_year_stable_flag",
    "temporal_sensitivity_flag",
    "temporal_sensitivity_priority_flag",
]

# Read only the registry fields needed to define temporal requests.
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


def require(condition, message):
    # Stop on unexpected inputs rather than silently creating wrong targets.
    if not condition:
        raise ValueError(message)


def read_csv_snapshot(path):
    # Read the bytes once so the hash describes exactly what pandas reads.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    # Preserve literal strings and empty fields until explicit conversion.
    table = pd.read_csv(
        io.BytesIO(content),
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    return table, digest


def convert_integer(series, nullable=False):
    # Empty alternative-year cells mean no alternative, not year zero.
    values = series.replace("", pd.NA)

    # Reject non-numeric content instead of silently converting it to missing.
    numbers = pd.to_numeric(
        values,
        errors="raise",
    )

    if not nullable:
        require(
            numbers.notna().all(),
            f"Missing integer values in {series.name}.",
        )

    # A year or offset must be a whole number.
    require(
        numbers.dropna().mod(1).eq(0).all(),
        f"Non-integer values in {series.name}.",
    )

    # Nullable Int64 can represent an absent alternative year.
    return numbers.astype("Int64" if nullable else "int64")


def main():
    print("STEP 41 — INDICATOR TEMPORAL TARGETS")

    # Refuse to overwrite either output.
    for path in (PARQUET_PATH, CSV_PATH):
        if path.exists():
            raise FileExistsError(
                "Output already exists; inspect it before rerunning: "
                f"{path}"
            )

    for path in (
        TEMPORAL_PATH,
        MAPPING_PATH,
        REGISTRY_PATH,
    ):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    # Record the versions of all three configuration inputs.
    temporal, temporal_hash = read_csv_snapshot(TEMPORAL_PATH)
    mapping, mapping_hash = read_csv_snapshot(MAPPING_PATH)
    registry, registry_hash = read_csv_snapshot(REGISTRY_PATH)

    require(
        registry_hash == EXPECTED_REGISTRY_HASH,
        "Registry differs from the validated V1 version.",
    )

    # Check the temporal schema without changing column names.
    require(
        len(temporal.columns) == len(TEMPORAL_COLUMNS)
        and set(temporal.columns) == set(TEMPORAL_COLUMNS),
        "Unexpected temporal configuration columns.",
    )

    require(
        len(temporal) == 29
        and not temporal["cntry"].duplicated().any(),
        "Expected one temporal configuration row per ESS country.",
    )

    # Convert the required year columns explicitly.
    for column in (
        "fieldwork_year_min",
        "fieldwork_year_max",
        "dominant_fieldwork_year",
        "target_context_year",
    ):
        temporal[column] = convert_integer(temporal[column])

    temporal["alternative_context_year"] = convert_integer(
        temporal["alternative_context_year"],
        nullable=True,
    )

    # Convert the stored proportions to numeric values.
    for column in (
        "dominant_year_weighted_share",
        "weighted_share_2023",
        "weighted_share_2024",
    ):
        temporal[column] = pd.to_numeric(
            temporal[column],
            errors="raise",
        ).astype("float64")

        require(
            temporal[column].between(0, 1).all(),
            f"Invalid proportion in {column}.",
        )

    # Convert only valid Boolean tokens.
    for column in BOOLEAN_COLUMNS:
        require(
            temporal[column].isin(["True", "False"]).all(),
            f"Invalid Boolean token in {column}.",
        )

        temporal[column] = temporal[column].map(
            {"True": True, "False": False}
        ).astype("bool")

    # The primary context year must retain the established ESS rule.
    require(
        temporal["target_context_year"].eq(
            temporal["dominant_fieldwork_year"]
        ).all(),
        "Primary context year differs from the dominant ESS year.",
    )

    require(
        temporal["target_context_year"].value_counts().to_dict()
        == {2023: 19, 2024: 10},
        "Unexpected primary context-year distribution.",
    )

    alternative_mask = temporal["alternative_context_year"].notna()

    require(
        int(alternative_mask.sum()) == 13,
        "Expected 13 countries with an alternative context year.",
    )

    # Alternative scenarios should exist exactly for multi-year fieldwork.
    require(
        alternative_mask.eq(
            temporal["multi_year_fieldwork_flag"]
        ).all(),
        "Alternative-year availability differs from multi-year fieldwork.",
    )

    # In this V1, the alternative is the other year among 2023 and 2024.
    require(
        (
            temporal.loc[alternative_mask, "target_context_year"]
            + temporal.loc[
                alternative_mask,
                "alternative_context_year",
            ]
        ).eq(2023 + 2024).all(),
        "Unexpected alternative context years.",
    )

    # Retain the original status as historical input metadata.
    # Do not present it as the status of a completed indicator-level review.
    temporal = temporal.rename(
        columns={
            "indicator_alignment_status":
                "input_indicator_alignment_status"
        }
    )

    # Use only country-code pairs from the validated mapping.
    country_mapping = mapping[
        ["ess_cntry", "eurostat_geo"]
    ].copy()

    require(
        len(country_mapping) == 29
        and not country_mapping["ess_cntry"].duplicated().any()
        and not country_mapping["eurostat_geo"].duplicated().any(),
        "Unexpected country mapping structure.",
    )

    require(
        set(temporal["cntry"])
        == set(country_mapping["ess_cntry"]),
        "Temporal configuration and mapping have different countries.",
    )

    # Add Eurostat codes without dropping any ESS country.
    countries = temporal.merge(
        country_mapping,
        how="left",
        left_on="cntry",
        right_on="ess_cntry",
        validate="one_to_one",
    )

    require(
        countries["eurostat_geo"].notna().all(),
        "Some countries have no Eurostat mapping.",
    )

    # cntry already stores the ESS code; remove its redundant copy.
    countries = countries.drop(columns="ess_cntry")

    # Build one primary scenario for every ESS country.
    primary = countries.copy()
    primary["alignment_scenario"] = "primary"
    primary["context_year"] = primary["target_context_year"]

    # Build alternatives only where an alternative fieldwork year exists.
    alternative = countries.loc[
        countries["alternative_context_year"].notna()
    ].copy()

    alternative["alignment_scenario"] = "alternative"
    alternative["context_year"] = (
        alternative["alternative_context_year"].astype("int64")
    )

    # This gives 29 primary plus 13 alternative country scenarios.
    scenarios = pd.concat(
        [primary, alternative],
        ignore_index=True,
    )

    require(
        len(scenarios) == 42,
        "Expected 42 country-scenario combinations.",
    )

    # Preserve the share of interviews associated with each scenario.
    scenarios["context_year_weighted_share"] = (
        scenarios["weighted_share_2023"].where(
            scenarios["context_year"].eq(2023),
            scenarios["weighted_share_2024"],
        )
    )

    # Select indicator definitions from the existing registry.
    indicators = registry[INDICATOR_COLUMNS].copy()

    require(
        len(indicators) == 6
        and not indicators["indicator_id"].duplicated().any(),
        "Expected six unique registry indicators.",
    )

    for column in (
        "api_start_year",
        "api_end_year",
        "reference_year_offset",
    ):
        indicators[column] = convert_integer(indicators[column])

    actual_offsets = indicators.set_index(
        "indicator_id"
    )["reference_year_offset"].to_dict()

    require(
        actual_offsets == EXPECTED_OFFSETS,
        "Registry offsets differ from the expected V1 rules.",
    )

    indicators = indicators.rename(
        columns={"label": "indicator_label"}
    )

    # A cross join pairs every country scenario with every indicator.
    # These rows are requests, not independent respondents or countries.
    targets = scenarios.merge(
        indicators,
        how="cross",
    )

    # Translate the desired context year into the Eurostat year label.
    # For Gini: context 2023 -> Eurostat label 2024.
    targets["requested_eurostat_year"] = (
        targets["context_year"]
        + targets["reference_year_offset"]
    ).astype("int64")

    # Confirm that requested years fall within the extracted API windows.
    targets["within_extraction_window"] = (
        targets["requested_eurostat_year"].ge(
            targets["api_start_year"]
        )
        & targets["requested_eurostat_year"].le(
            targets["api_end_year"]
        )
    )

    require(
        targets["within_extraction_window"].all(),
        "Some requested years fall outside the extracted API windows.",
    )

    # Distinguish general methodological evidence from registry application.
    targets["reference_period_review_status"] = (
        "registry_rule_applied"
    )

    targets["reference_period_evidence_url"] = (
        targets["documentation_url"]
    )

    gini_mask = targets["indicator_id"].eq("gini")

    # This records support for the general income-reference rule.
    # It does not claim a separate audit of every national observation.
    targets.loc[
        gini_mask,
        "reference_period_review_status",
    ] = "general_eu_silc_income_year_rule_supported"

    targets.loc[
        gini_mask,
        "reference_period_evidence_url",
    ] = GINI_REFERENCE_URL

    # No availability claim is made before matching to the Eurostat table.
    targets["alignment_stage"] = (
        "year_target_created_observation_match_pending"
    )

    # Record hashes of the exact configuration files used.
    targets["temporal_config_sha256"] = temporal_hash
    targets["country_mapping_sha256"] = mapping_hash
    targets["registry_sha256"] = registry_hash

    # One row must identify one country-indicator-scenario request.
    key = [
        "cntry",
        "indicator_id",
        "alignment_scenario",
    ]

    require(
        not targets.duplicated(key).any(),
        "Duplicate country-indicator-scenario requests.",
    )

    scenario_counts = (
        targets["alignment_scenario"].value_counts().to_dict()
    )

    require(
        scenario_counts == {
            "primary": 174,
            "alternative": 78,
        },
        "Unexpected scenario counts.",
    )

    require(
        len(targets) == 252,
        "Expected 252 temporal requests.",
    )

    # Standardize text columns using the compatible pandas selection.
    text_columns = targets.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        targets[column] = targets[column].astype("string")

    targets = targets.sort_values(key).reset_index(drop=True)

    print("\nREQUESTS BY SCENARIO")
    print(
        targets.groupby("alignment_scenario").size().to_string()
    )

    print("\nREQUESTS BY INDICATOR AND EUROSTAT YEAR")
    print(
        targets.groupby(
            [
                "indicator_id",
                "alignment_scenario",
                "requested_eurostat_year",
            ]
        ).size().rename("requests").to_string()
    )

    # Show examples where the fieldwork-year choice particularly matters.
    preview = targets.loc[
        targets["indicator_id"].eq("gini")
        & targets["temporal_sensitivity_priority_flag"],
        [
            "cntry",
            "alignment_scenario",
            "context_year",
            "context_year_weighted_share",
            "requested_eurostat_year",
        ],
    ]

    print("\nGINI TARGETS — PRIORITY SENSITIVITY COUNTRIES")
    print(preview.to_string(index=False))

    # Prepare exports before creating files.
    parquet_bytes = targets.to_parquet(
        index=False,
        engine="pyarrow",
    )

    csv_text = targets.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive modes protect existing outputs from overwriting.
    with PARQUET_PATH.open("xb") as stream:
        stream.write(parquet_bytes)

    with CSV_PATH.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        stream.write(csv_text)

    print(f"\nSaved: {PARQUET_PATH}")
    print(f"Saved: {CSV_PATH}")

    print("\nINDICATOR TEMPORAL TARGETS CREATED.")
    print("Primary requests: 174")
    print("Alternative requests: 78")
    print("Total requests: 252")
    print("Economic observations have not been matched yet.")
    print("Independent target validation is still pending.")


# Execute the workflow only when this script is run directly.
if __name__ == "__main__":
    main()