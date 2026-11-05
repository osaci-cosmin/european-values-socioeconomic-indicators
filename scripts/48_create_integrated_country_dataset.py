# Step 48: Integrate ESS survey estimates with socioeconomic indicators.
# Keep one row per country and preserve all analytical values.
# Normalize the country-key dtype before merging.
# Do not modify input files, impute values or recalculate estimates.

import hashlib
import io
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ESS_PATH = (
    PROCESSED / "ESS11_country_survey_estimates.parquet"
)

SOCIOECONOMIC_PATH = (
    PROCESSED / "ESS11_country_socioeconomic_main_v1.parquet"
)

PARQUET_PATH = (
    PROCESSED / "ESS11_country_integrated_main_v1.parquet"
)

CSV_PATH = (
    PROCESSED / "ESS11_country_integrated_main_v1.csv"
)


# These twelve ESS metrics have survey standard errors and confidence intervals.
ESS_METRICS = [
    "mean_left_right",
    "left_share",
    "centre_share",
    "right_share",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
    "mean_social_trust",
    "mean_ppltrst",
    "mean_pplfair",
    "mean_pplhlp",
]

ECONOMIC_INDICATORS = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

# Preserve the established 29-country ESS V1 scope.
EXPECTED_COUNTRIES = {
    "AT", "BE", "BG", "CH", "CY", "DE", "EE",
    "ES", "FI", "FR", "GB", "GR", "HR", "HU",
    "IE", "IS", "IT", "LT", "LV", "ME", "NL",
    "NO", "PL", "PT", "RS", "SE", "SI", "SK",
    "UA",
}


def require(condition, message):
    # Stop immediately when a required check fails.
    if not condition:
        raise ValueError(message)


def read_parquet_snapshot(path):
    # Read the bytes once so the hash describes exactly the loaded file.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    # BytesIO lets pandas read the already-hashed bytes from memory.
    table = pd.read_parquet(
        io.BytesIO(content),
        engine="pyarrow",
    )

    return table, digest


def check_country_key(table, name):
    # Confirm that the join key exists before checking its contents.
    require(
        "cntry" in table.columns,
        f"{name}: country key cntry is missing.",
    )

    require(
        table.columns.is_unique,
        f"{name}: duplicated column names.",
    )

    # Missing country codes must never participate in the join.
    require(
        table["cntry"].notna().all(),
        f"{name}: missing country codes.",
    )

    require(
        table["cntry"].map(
            lambda value: isinstance(value, str)
        ).all(),
        f"{name}: country codes must be text.",
    )

    # One row per country is required for a one-to-one integration.
    require(
        len(table) == 29
        and not table["cntry"].duplicated().any(),
        f"{name}: expected one row for each of 29 countries.",
    )

    require(
        set(table["cntry"]) == EXPECTED_COUNTRIES,
        f"{name}: country scope differs from ESS V1.",
    )


def normalize_country_key(table, name):
    # Keep the original codes to detect any unintended value change.
    original_codes = table["cntry"].tolist()

    # Use the same explicit text type in both join inputs.
    # This avoids merging incompatible pandas string representations.
    table["cntry"] = table["cntry"].astype("string")

    # Only the in-memory dtype may change; the country codes must not.
    require(
        table["cntry"].tolist() == original_codes,
        f"{name}: country codes changed during dtype normalization.",
    )


def compare_preserved_columns(integrated, original, name):
    # Select only the columns belonging to this input.
    # Sort both tables so country order cannot affect the comparison.
    actual = (
        integrated[original.columns]
        .sort_values("cntry")
        .reset_index(drop=True)
    )

    expected = (
        original
        .sort_values("cntry")
        .reset_index(drop=True)
    )

    # Keep strict checks after the explicit country-key normalization.
    # Values, missing positions and remaining dtypes must be preserved.
    assert_frame_equal(
        actual,
        expected,
        check_exact=True,
        check_dtype=True,
        obj=name,
    )

    print(f"{name}: exact preservation PASSED.")


def main():
    print("STEP 48 — INTEGRATED ESS–EUROSTAT COUNTRY DATASET")

    # Refuse to overwrite either integrated output.
    for path in (PARQUET_PATH, CSV_PATH):
        if path.exists():
            raise FileExistsError(
                "Output already exists; inspect it before rerunning: "
                f"{path}"
            )

    # Check both source files before loading them.
    for path in (ESS_PATH, SOCIOECONOMIC_PATH):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    ess, ess_hash = read_parquet_snapshot(ESS_PATH)

    socioeconomic, socioeconomic_hash = read_parquet_snapshot(
        SOCIOECONOMIC_PATH
    )

    # Validate the original keys before changing their in-memory dtype.
    check_country_key(ess, "ESS survey estimates")
    check_country_key(socioeconomic, "Socioeconomic table")

    # Normalize only the join key; leave all other columns unchanged.
    normalize_country_key(ess, "ESS survey estimates")
    normalize_country_key(socioeconomic, "Socioeconomic table")

    require(
        ess["cntry"].dtype == socioeconomic["cntry"].dtype,
        "Country-key dtypes still differ after normalization.",
    )

    print("\nCountry-key dtype normalization: PASSED.")

    # Confirm the input structures inspected in Step 47.
    require(
        len(ess.columns) == 66,
        "Unexpected ESS survey-estimates column count.",
    )

    require(
        len(socioeconomic.columns) == 174,
        "Unexpected socioeconomic column count.",
    )

    # Require the ESS identifiers, sample counts and design information.
    required_survey_columns = {
        "country_name",
        "region_4",
        "respondents_n",
        "survey_psu_n",
        "survey_strata_n",
        "survey_design_df",
        "survey_ci_level",
    }

    # Each metric must retain its estimate, SE and both CI limits.
    for metric in ESS_METRICS:
        required_survey_columns.update(
            {
                metric,
                f"{metric}_se",
                f"{metric}_ci_lower",
                f"{metric}_ci_upper",
            }
        )

    require(
        required_survey_columns.issubset(ess.columns),
        "Missing ESS estimates, uncertainty fields or design metadata.",
    )

    require(
        set(ECONOMIC_INDICATORS).issubset(socioeconomic.columns),
        "Missing socioeconomic indicator columns.",
    )

    # Only the primary socioeconomic scenario belongs in this dataset.
    require(
        socioeconomic["alignment_scenario"].eq("primary").all(),
        "The socioeconomic input contains non-primary scenarios.",
    )

    # Respondent counts describe the samples behind the country estimates.
    # The integrated dataset still has countries as its analysis units.
    require(
        int(ess["respondents_n"].sum()) == 49210,
        "Unexpected total ESS respondent count.",
    )

    # Cyprus remains outside the four European regions in the project.
    missing_region_countries = set(
        ess.loc[ess["region_4"].isna(), "cntry"]
    )

    require(
        missing_region_countries == {"CY"},
        "Unexpected countries without a region_4 classification.",
    )

    # Preserve the socioeconomic processing status as input history.
    require(
        "dataset_stage" in socioeconomic.columns,
        "Missing socioeconomic dataset_stage field.",
    )

    require(
        "socioeconomic_input_stage" not in socioeconomic.columns,
        "The socioeconomic input already contains the renamed stage field.",
    )

    socioeconomic = socioeconomic.rename(
        columns={"dataset_stage": "socioeconomic_input_stage"}
    )

    # Only cntry should be shared between the two input schemas.
    # Reject collisions instead of creating ambiguous _x and _y columns.
    shared_columns = set(ess.columns) & set(socioeconomic.columns)

    require(
        shared_columns == {"cntry"},
        "Unexpected overlapping input columns: "
        f"{sorted(shared_columns - {'cntry'})}",
    )

    # A left join retains every ESS country.
    # one_to_one rejects duplicated country keys in either input.
    integrated = ess.merge(
        socioeconomic,
        how="left",
        on="cntry",
        validate="one_to_one",
        indicator="_integration_match",
    )

    require(
        len(integrated) == 29,
        "The integration changed the country count.",
    )

    # A row can match successfully even when its economic values are missing.
    require(
        integrated["_integration_match"].eq("both").all(),
        "Some ESS countries did not match a socioeconomic row.",
    )

    # Remove the temporary merge diagnostic after checking all matches.
    integrated = integrated.drop(columns="_integration_match")

    # Sort the output to make inspection and later comparisons easier.
    integrated = (
        integrated
        .sort_values("cntry")
        .reset_index(drop=True)
    )

    check_country_key(integrated, "Integrated table")

    # Compare with both inputs after the documented key normalization.
    compare_preserved_columns(
        integrated,
        ess,
        "All ESS fields",
    )

    compare_preserved_columns(
        integrated,
        socioeconomic,
        "All socioeconomic fields",
    )

    # The shared country key appears once in the combined table.
    expected_base_columns = (
        len(ess.columns) + len(socioeconomic.columns) - 1
    )

    require(
        len(integrated.columns) == expected_base_columns,
        "Unexpected column loss or duplication during integration.",
    )

    # Link every output row to the exact two source files used.
    integration_metadata = {
        "ess_survey_input_filename": ESS_PATH.name,
        "ess_survey_input_sha256": ess_hash,
        "socioeconomic_input_filename": SOCIOECONOMIC_PATH.name,
        "socioeconomic_input_sha256": socioeconomic_hash,
        "analysis_unit": "country",
        "integration_stage": (
            "country_integration_created_independent_validation_pending"
        ),
    }

    for column, value in integration_metadata.items():
        require(
            column not in integrated.columns,
            f"Integration metadata would overwrite a column: {column}",
        )

        # Repeat the metadata value without changing any source columns.
        integrated[column] = pd.Series(
            value,
            index=integrated.index,
            dtype="string",
        )

    # Count available economic values without filtering incomplete countries.
    availability = (
        integrated[ECONOMIC_INDICATORS]
        .notna()
        .sum()
        .rename("available_countries")
    )

    require(
        int(availability.sum()) == 158,
        "The integration changed economic data availability.",
    )

    require(
        int(integrated["respondents_n"].sum()) == 49210,
        "The integration changed ESS respondent totals.",
    )

    print("\nINTEGRATED COUNTRY PREVIEW")

    preview_columns = [
        "cntry",
        "country_name",
        "context_year",
        "mean_social_trust",
        "mean_left_right",
        "gdp_pps_index",
        "gini",
        "available_indicators_n",
    ]

    print(
        integrated[preview_columns].to_string(index=False)
    )

    print("\nECONOMIC INDICATOR AVAILABILITY")
    print(availability.to_string())

    print("\nSUMMARY")
    print(f"Integrated country rows: {len(integrated)}")
    print(f"ESS input columns: {len(ess.columns)}")
    print(
        "Socioeconomic input columns: "
        f"{len(socioeconomic.columns)}"
    )
    print(f"Combined source columns: {expected_base_columns}")
    print(
        "Integration metadata columns: "
        f"{len(integration_metadata)}"
    )
    print(f"Total output columns: {len(integrated.columns)}")
    print(f"ESS metrics with SE and CI: {len(ESS_METRICS)}")
    print(f"Economic indicators: {len(ECONOMIC_INDICATORS)}")
    print(
        "Countries with all six economic indicators: "
        f"{int(integrated['all_six_indicators_available'].sum())}"
    )
    print("Unit of analysis: country")

    # Prepare both exports before creating output files.
    parquet_bytes = integrated.to_parquet(
        index=False,
        engine="pyarrow",
    )

    # Preserve missing values as empty CSV cells rather than zeros.
    csv_text = integrated.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Exclusive modes prevent overwriting existing outputs.
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

    print("\nINTEGRATED COUNTRY DATASET CREATED.")
    print("All 29 ESS countries were retained.")
    print(
        "ESS estimates, uncertainty and socioeconomic metadata preserved."
    )
    print("Only the country-key dtype was standardized in memory.")
    print("No input files were modified.")
    print("No imputation or statistical modelling was performed.")
    print("Independent integration validation is still pending.")


# Execute the integration only when this script is run directly.
if __name__ == "__main__":
    main()