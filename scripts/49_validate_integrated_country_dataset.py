# Step 49: Independently validate the integrated country dataset.
# Reconstruct the expected table by country-index alignment, not merge().
# Read files only; do not modify or export datasets.

import hashlib
import io
import re
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


EXPECTED_COUNTRIES = {
    "AT", "BE", "BG", "CH", "CY", "DE", "EE",
    "ES", "FI", "FR", "GB", "GR", "HR", "HU",
    "IE", "IS", "IT", "LT", "LV", "ME", "NL",
    "NO", "PL", "PT", "RS", "SE", "SI", "SK",
    "UA",
}

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

EXPECTED_AVAILABLE = {
    "gdp_pps_index": 28,
    "employment_rate": 26,
    "unemployment_rate": 26,
    "gini": 25,
    "material_deprivation_rate": 26,
    "inflation_rate": 27,
}

ECONOMIC_INDICATORS = list(EXPECTED_AVAILABLE)


def require(condition, message):
    # Stop immediately when a required validation fails.
    if not condition:
        raise ValueError(message)


def read_parquet_snapshot(path):
    # Hash exactly the bytes used to load the source table.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    table = pd.read_parquet(
        io.BytesIO(content),
        engine="pyarrow",
    )

    return table, digest


def check_country_key(table, name):
    require(
        table.columns.is_unique,
        f"{name}: duplicated column names.",
    )

    require(
        "cntry" in table.columns,
        f"{name}: missing country key.",
    )

    require(
        len(table) == 29
        and table["cntry"].notna().all()
        and not table["cntry"].duplicated().any(),
        f"{name}: expected 29 unique, non-missing country codes.",
    )

    require(
        set(table["cntry"]) == EXPECTED_COUNTRIES,
        f"{name}: unexpected country scope.",
    )


def normalize_source_key(table):
    # Reproduce only the documented country-key dtype normalization.
    result = table.copy()
    original_codes = result["cntry"].tolist()

    result["cntry"] = result["cntry"].astype("string")

    require(
        result["cntry"].tolist() == original_codes,
        "Country codes changed during normalization.",
    )

    return result


def build_expected(ess, socioeconomic, ess_hash, socioeconomic_hash):
    # Normalize source keys without changing the source files.
    ess = normalize_source_key(ess)
    socioeconomic = normalize_source_key(socioeconomic)

    require(
        "dataset_stage" in socioeconomic.columns,
        "Missing socioeconomic processing-stage field.",
    )

    require(
        "socioeconomic_input_stage" not in socioeconomic.columns,
        "Unexpected pre-existing socioeconomic_input_stage field.",
    )

    # Preserve the original socioeconomic stage under its integrated name.
    socioeconomic = socioeconomic.rename(
        columns={"dataset_stage": "socioeconomic_input_stage"}
    )

    require(
        set(ess.columns) & set(socioeconomic.columns) == {"cntry"},
        "Unexpected overlapping source columns.",
    )

    # Use country codes as indexes and put both sources in identical order.
    country_order = sorted(EXPECTED_COUNTRIES)

    ess_indexed = ess.set_index("cntry").loc[country_order]
    socioeconomic_indexed = (
        socioeconomic.set_index("cntry").loc[country_order]
    )

    # Concatenate columns by country index instead of repeating Step 48's merge.
    expected = pd.concat(
        [ess_indexed, socioeconomic_indexed],
        axis=1,
        verify_integrity=True,
    ).reset_index()

    expected["cntry"] = expected["cntry"].astype("string")

    require(
        len(expected.columns) == 239,
        "Expected 239 combined source columns.",
    )

    # Recalculate the integration provenance from the actual input files.
    metadata = {
        "ess_survey_input_filename": ESS_PATH.name,
        "ess_survey_input_sha256": ess_hash,
        "socioeconomic_input_filename": SOCIOECONOMIC_PATH.name,
        "socioeconomic_input_sha256": socioeconomic_hash,
        "analysis_unit": "country",
        "integration_stage": (
            "country_integration_created_independent_validation_pending"
        ),
    }

    for column, value in metadata.items():
        require(
            column not in expected.columns,
            f"Metadata column collision: {column}",
        )

        expected[column] = pd.Series(
            value,
            index=expected.index,
            dtype="string",
        )

    return expected


def check_output_schema(table, expected, name):
    check_country_key(table, name)

    require(
        len(table.columns) == 245
        and set(table.columns) == set(expected.columns),
        f"{name}: unexpected integrated output schema.",
    )


def align_rows(table, country_order):
    # Compare countries in a fixed order, regardless of file row order.
    return (
        table.set_index("cntry")
        .loc[country_order]
        .reset_index()
    )


def decode_csv(csv_text, expected):
    # CSV does not store column types.
    # Use the independent reference table to restore those types.

    # Collect columns first to avoid repeated DataFrame insertions.
    decoded_columns = {}

    for column in expected.columns:
        # Read the CSV text and the corresponding reference column.
        raw = csv_text[column]
        reference = expected[column]
        dtype = reference.dtype

        # Identify missing positions from the independent reference.
        missing = reference.isna()

        # Missing source values must appear as empty CSV cells.
        require(
            raw.loc[missing].eq("").all(),
            f"CSV: non-empty content at missing positions in {column}.",
        )

        if is_bool_dtype(dtype):
            # Accept only explicit Boolean text for non-missing values.
            require(
                raw.loc[~missing].isin(["True", "False"]).all(),
                f"CSV: invalid Boolean token in {column}.",
            )

            # Restore Boolean values and preserve missing positions.
            values = [
                pd.NA if is_missing else token == "True"
                for token, is_missing in zip(raw, missing)
            ]

        elif is_integer_dtype(dtype):
            # Integer fields must contain whole-number text.
            # Reject decimal values and malformed tokens.
            require(
                raw.loc[~missing].map(
                    lambda value: re.fullmatch(r"[+-]?\d+", value) is not None
                ).all(),
                f"CSV: invalid integer token in {column}.",
            )

            # Convert valid text to integers without filling missing values.
            values = [
                pd.NA if is_missing else int(token)
                for token, is_missing in zip(raw, missing)
            ]

        elif is_float_dtype(dtype):
            # A valid source number must not become an empty CSV cell.
            require(
                raw.loc[~missing].ne("").all(),
                f"CSV: unexpected empty numeric cells in {column}.",
            )

            # Restore decimal numbers and keep missing values as NaN.
            values = [
                float("nan") if is_missing else float(token)
                for token, is_missing in zip(raw, missing)
            ]

        elif is_string_dtype(dtype):
            # Preserve empty strings when the source contains empty text.
            # Restore missing text only at source-confirmed missing positions.
            values = [
                pd.NA if is_missing else token
                for token, is_missing in zip(raw, missing)
            ]

        else:
            # Stop if the reference contains an unsupported column type.
            raise TypeError(
                f"Unsupported source-derived dtype for {column}: {dtype}"
            )

        # Store each column with its exact reference dtype and row index.
        decoded_columns[column] = pd.Series(
            values,
            index=expected.index,
            dtype=dtype,
        )

    # Build the complete table once to avoid column fragmentation.
    return pd.DataFrame(decoded_columns, index=expected.index)

def compare_tables(actual, expected, name):
    # Column order is not analytically meaningful; compare by column name.
    columns = sorted(expected.columns)

    assert_frame_equal(
        actual[columns],
        expected[columns],
        check_exact=True,
        check_dtype=True,
        obj=name,
    )

    print(f"{name}: exact comparison PASSED.")


def main():
    print("STEP 49 — INTEGRATED COUNTRY DATASET VALIDATION")

    for path in (
        ESS_PATH,
        SOCIOECONOMIC_PATH,
        PARQUET_PATH,
        CSV_PATH,
    ):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    ess, ess_hash = read_parquet_snapshot(ESS_PATH)

    socioeconomic, socioeconomic_hash = read_parquet_snapshot(
        SOCIOECONOMIC_PATH
    )

    check_country_key(ess, "ESS source")
    check_country_key(socioeconomic, "Socioeconomic source")

    require(
        len(ess.columns) == 66
        and len(socioeconomic.columns) == 174,
        "Unexpected source column counts.",
    )

    expected = build_expected(
        ess,
        socioeconomic,
        ess_hash,
        socioeconomic_hash,
    )

    parquet = pd.read_parquet(
        PARQUET_PATH,
        engine="pyarrow",
    )

    # Preserve literal CSV tokens, including empty status strings.
    csv_text = pd.read_csv(
        CSV_PATH,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    check_output_schema(parquet, expected, "Parquet")
    check_output_schema(csv_text, expected, "CSV")

    print("\nSchema, country scope and unique keys: PASSED.")

    country_order = sorted(EXPECTED_COUNTRIES)

    expected = align_rows(expected, country_order)
    parquet = align_rows(parquet, country_order)
    csv_text = align_rows(csv_text, country_order)

    # Do not normalize Parquet here: its stored dtypes must already be correct.
    compare_tables(
        parquet,
        expected,
        "Parquet versus independently aligned sources",
    )

    csv = decode_csv(csv_text, expected)

    compare_tables(
        csv,
        expected,
        "CSV versus independently aligned sources",
    )

    compare_tables(
        csv,
        parquet,
        "CSV versus Parquet",
    )

    # Confirm that all twelve ESS metrics retain their SE and CI columns.
    survey_columns = []

    for metric in ESS_METRICS:
        survey_columns.extend(
            [
                metric,
                f"{metric}_se",
                f"{metric}_ci_lower",
                f"{metric}_ci_upper",
            ]
        )

    require(
        set(survey_columns).issubset(parquet.columns),
        "Missing ESS estimates or uncertainty columns.",
    )

    # These fields had no missing values in the inspected ESS source.
    require(
        parquet[survey_columns].notna().all().all(),
        "Unexpected missing ESS estimates or uncertainty values.",
    )

    require(
        int(parquet["respondents_n"].sum()) == 49210,
        "Unexpected ESS respondent total.",
    )

    # Preserve the project decision to leave Cyprus outside region_4.
    require(
        set(
            parquet.loc[parquet["region_4"].isna(), "cntry"]
        ) == {"CY"},
        "Unexpected regional classification missingness.",
    )

    require(
        parquet["analysis_unit"].eq("country").all(),
        "Incorrect analysis-unit metadata.",
    )

    require(
        parquet["alignment_scenario"].eq("primary").all(),
        "Non-primary scenarios found in the main dataset.",
    )

    require(
        parquet["context_year"].eq(
            parquet["target_context_year"]
        ).all(),
        "Primary context-year mismatch.",
    )

    print("ESS uncertainty fields, sample counts and analysis unit: PASSED.")

    # Check economic coverage without removing incomplete countries.
    availability = parquet[ECONOMIC_INDICATORS].notna().sum()

    require(
        availability.to_dict() == EXPECTED_AVAILABLE,
        "Economic availability differs from the validated source.",
    )

    available_per_country = (
        parquet[ECONOMIC_INDICATORS].notna().sum(axis=1)
    )

    require(
        parquet["available_indicators_n"].eq(
            available_per_country
        ).all(),
        "Incorrect available-indicator counts.",
    )

    require(
        parquet["missing_indicators_n"].eq(
            6 - available_per_country
        ).all(),
        "Incorrect missing-indicator counts.",
    )

    require(
        parquet["all_six_indicators_available"].eq(
            available_per_country.eq(6)
        ).all(),
        "Incorrect completeness flags.",
    )

    require(
        int(available_per_country.eq(6).sum()) == 25,
        "Expected 25 countries with all six economic indicators.",
    )

    print("Economic values, missingness and completeness flags: PASSED.")

    print("\nVALIDATED ECONOMIC AVAILABILITY")
    print(
        availability.rename("available_countries").to_string()
    )

    print("\nSUMMARY")
    print(f"Integrated countries: {len(parquet)}")
    print(f"Integrated columns: {len(parquet.columns)}")
    print(f"ESS metrics with SE and CI: {len(ESS_METRICS)}")
    print(f"Economic indicators: {len(ECONOMIC_INDICATORS)}")
    print(f"Available economic values: {int(availability.sum())}")
    print(f"Unavailable economic values: {int(29 * 6 - availability.sum())}")
    print(
        "Countries with all six economic indicators: "
        f"{int(available_per_country.eq(6).sum())}"
    )
    print("Unit of analysis: country")

    # Success is reported only after every check has passed.
    print("\nINTEGRATED COUNTRY DATASET VALIDATION PASSED.")
    print("All source fields and integration provenance matched.")
    print("The documented country-key dtype normalization was verified.")
    print("No country loss, imputation or scenario substitution detected.")
    print("No files were modified.")


# Run validation only when this script is executed directly.
if __name__ == "__main__":
    main()