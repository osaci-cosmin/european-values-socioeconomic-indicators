# Step 45: Build the main country-level socioeconomic table.
# Keep one row per ESS country and one value column per indicator.
# Preserve missing values, quality flags and source provenance.

import hashlib
import io
from pathlib import Path

import pandas as pd


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "ess_eurostat_temporal_matches_v1.parquet"
)

OUTPUT_DIR = ROOT / "data" / "processed"

PARQUET_PATH = (
    OUTPUT_DIR
    / "ESS11_country_socioeconomic_main_v1.parquet"
)

CSV_PATH = (
    OUTPUT_DIR
    / "ESS11_country_socioeconomic_main_v1.csv"
)


# Use a fixed order for the six analytical value columns.
INDICATORS = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

# These fields describe the country and its primary ESS context.
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

# Preserve metadata separately for each indicator.
# For example: gini__requested_eurostat_year.
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

# These are the confirmed primary-scenario counts from Step 44.
EXPECTED_AVAILABLE = {
    "gdp_pps_index": 28,
    "employment_rate": 26,
    "unemployment_rate": 26,
    "gini": 25,
    "material_deprivation_rate": 26,
    "inflation_rate": 27,
}


def require(condition, message):
    # Stop before export when a required condition fails.
    if not condition:
        raise ValueError(message)


def main():
    print("STEP 45 — MAIN COUNTRY SOCIOECONOMIC TABLE")

    # Never overwrite an existing processed dataset silently.
    for path in (PARQUET_PATH, CSV_PATH):
        if path.exists():
            raise FileExistsError(
                "Output already exists; inspect it before rerunning: "
                f"{path}"
            )

    require(
        INPUT_PATH.is_file(),
        f"Required input not found: {INPUT_PATH}",
    )

    # Read and hash the same input bytes for reproducible provenance.
    input_bytes = INPUT_PATH.read_bytes()
    input_hash = hashlib.sha256(input_bytes).hexdigest()

    matches = pd.read_parquet(
        io.BytesIO(input_bytes),
        engine="pyarrow",
    )

    required_columns = set(
        COUNTRY_COLUMNS
        + INDICATOR_METADATA
        + ["indicator_id", "value"]
    )

    require(
        required_columns.issubset(matches.columns),
        "The matched table is missing required columns.",
    )

    require(
        len(matches) == 252,
        "Expected 252 requests in the validated matched table.",
    )

    require(
        matches["alignment_scenario"].value_counts().to_dict()
        == {"primary": 174, "alternative": 78},
        "Unexpected scenario counts in the input.",
    )

    # Select the main scenario explicitly.
    # Missing primary values are not replaced with alternative values.
    primary = matches.loc[
        matches["alignment_scenario"].eq("primary")
    ].copy()

    require(
        len(primary) == 174,
        "Expected 174 primary country-indicator records.",
    )

    require(
        primary["cntry"].nunique() == 29,
        "Expected all 29 ESS countries.",
    )

    require(
        set(primary["indicator_id"]) == set(INDICATORS),
        "Unexpected indicator set.",
    )

    # Pivoting requires one value for each country-indicator pair.
    require(
        not primary.duplicated(
            ["cntry", "indicator_id"]
        ).any(),
        "Duplicate primary country-indicator pairs.",
    )

    require(
        primary.groupby("cntry")["indicator_id"].nunique().eq(6).all(),
        "Some countries do not have all six indicator requests.",
    )

    require(
        primary["context_year"].eq(
            primary["target_context_year"]
        ).all(),
        "Primary context years differ from the configured target years.",
    )

    # Verify that the availability flags still describe the actual values.
    require(
        primary["has_numeric_value"].eq(
            primary["value"].notna()
        ).all(),
        "Numeric availability flags disagree with values.",
    )

    require(
        primary["observation_match_status"].eq("available").eq(
            primary["value"].notna()
        ).all(),
        "Available status disagrees with numeric availability.",
    )

    # Country metadata should be identical across the six indicator rows.
    countries = primary[COUNTRY_COLUMNS].drop_duplicates()

    require(
        len(countries) == 29
        and not countries["cntry"].duplicated().any(),
        "Inconsistent country metadata across indicator rows.",
    )

    # Use country codes as row labels while assembling the wide table.
    countries = countries.set_index("cntry").sort_index()

    # pivot() reshapes existing values without calculating averages.
    # It also preserves countries whose six values are all missing.
    values = primary.pivot(
        index="cntry",
        columns="indicator_id",
        values="value",
    )

    # Reorder rows and columns explicitly for stable exports.
    values = values.reindex(
        index=countries.index,
        columns=INDICATORS,
    )

    values.columns.name = None

    # Keep numeric columns as floats so missing values remain NaN.
    values = values.astype("float64")

    # Start with the country metadata and six analytical value columns.
    parts = [countries, values]

    for field in INDICATOR_METADATA:
        # Reshape each metadata field separately to preserve its data type.
        wide_field = primary.pivot(
            index="cntry",
            columns="indicator_id",
            values=field,
        )

        wide_field = wide_field.reindex(
            index=countries.index,
            columns=INDICATORS,
        )

        # Include the indicator name to avoid ambiguous metadata columns.
        wide_field.columns = [
            f"{indicator}__{field}"
            for indicator in INDICATORS
        ]

        parts.append(wide_field)

    # All parts use the same country index, so concatenation aligns safely.
    country_table = pd.concat(
        parts,
        axis=1,
    )

    # Count available values without converting missing values into zeros.
    country_table["available_indicators_n"] = (
        country_table[INDICATORS].notna().sum(axis=1).astype("int64")
    )

    country_table["missing_indicators_n"] = (
        len(INDICATORS) - country_table["available_indicators_n"]
    )

    # This is a diagnostic flag, not a country-selection rule.
    country_table["all_six_indicators_available"] = (
        country_table["available_indicators_n"].eq(6)
    )

    # Link every output row to the exact validated matching table used.
    country_table["matched_input_filename"] = INPUT_PATH.name
    country_table["matched_input_sha256"] = input_hash

    country_table["dataset_stage"] = (
        "country_main_created_independent_validation_pending"
    )

    # Restore cntry as a regular column for CSV and future joins.
    country_table = country_table.reset_index()

    require(
        len(country_table) == 29
        and not country_table["cntry"].duplicated().any(),
        "The wide table must contain one row per ESS country.",
    )

    require(
        set(country_table["cntry"]) == set(primary["cntry"]),
        "The wide transformation changed the country scope.",
    )

    # Check the expected non-missing counts for every indicator.
    for indicator in INDICATORS:
        observed_count = int(
            country_table[indicator].notna().sum()
        )

        require(
            observed_count == EXPECTED_AVAILABLE[indicator],
            f"Availability count changed for {indicator}.",
        )

    require(
        int(country_table["available_indicators_n"].sum()) == 158,
        "Expected 158 numeric values in the primary scenario.",
    )

    require(
        int(country_table["missing_indicators_n"].sum()) == 16,
        "Expected 16 unavailable values in the primary scenario.",
    )

    # Confirm that each wide value still matches its original primary row.
    value_lookup = country_table.set_index("cntry")

    for row in primary[["cntry", "indicator_id", "value"]].itertuples(
        index=False
    ):
        actual = value_lookup.at[row.cntry, row.indicator_id]
        expected = row.value

        if pd.isna(expected):
            require(
                pd.isna(actual),
                f"Missing value changed: {row.cntry} / {row.indicator_id}",
            )
        else:
            require(
                actual == expected,
                f"Numeric value changed: {row.cntry} / {row.indicator_id}",
            )

    # Give all text fields an explicit pandas string type.
    text_columns = country_table.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        country_table[column] = country_table[column].astype("string")

    print("\nMAIN SOCIOECONOMIC VALUES")
    print(
        country_table[
            ["cntry", "context_year"] + INDICATORS
        ].to_string(index=False)
    )

    # Report missingness without deleting incomplete countries.
    incomplete = country_table.loc[
        ~country_table["all_six_indicators_available"],
        [
            "cntry",
            "available_indicators_n",
            "missing_indicators_n",
        ],
    ]

    print("\nCOUNTRIES WITH UNAVAILABLE INDICATORS")
    print(incomplete.to_string(index=False))

    print("\nAVAILABLE VALUES BY INDICATOR")
    print(
        country_table[INDICATORS]
        .notna()
        .sum()
        .rename("available_countries")
        .to_string()
    )

    print("\nSUMMARY")
    print(f"Countries retained: {len(country_table)}")
    print(f"Analytical indicators: {len(INDICATORS)}")
    print(
        "Available numeric values: "
        f"{int(country_table['available_indicators_n'].sum())}"
    )
    print(
        "Unavailable values: "
        f"{int(country_table['missing_indicators_n'].sum())}"
    )
    print(
        "Countries with all six indicators: "
        f"{int(country_table['all_six_indicators_available'].sum())}"
    )

    # Prepare both exports before creating files.
    parquet_bytes = country_table.to_parquet(
        index=False,
        engine="pyarrow",
    )

    csv_text = country_table.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive modes prevent accidental overwriting.
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

    print("\nMAIN COUNTRY SOCIOECONOMIC TABLE CREATED.")
    print("All 29 ESS countries were retained.")
    print("Alternative scenarios remain in the original matched table.")
    print("No imputation or alternative-year substitution was performed.")
    print("Independent country-table validation is still pending.")


# Execute the workflow only when this script is run directly.
if __name__ == "__main__":
    main()