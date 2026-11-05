# Step 47: Inspect the country-level datasets before integration.
# Identify their actual columns, data types and row structure.
# Read files only; do not merge or export datasets.

from pathlib import Path

import pandas as pd


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "ESS SURVEY ESTIMATES": (
        PROCESSED / "ESS11_country_survey_estimates.parquet"
    ),
    "ESS WEIGHTED POINT ESTIMATES": (
        PROCESSED / "ESS11_country_weighted_estimates.parquet"
    ),
    "MAIN SOCIOECONOMIC TABLE": (
        PROCESSED / "ESS11_country_socioeconomic_main_v1.parquet"
    ),
}


def inspect_file(label, path):
    print(f"\n{'=' * 80}")
    print(label)
    print(f"File: {path}")

    # Stop clearly if the expected input is unavailable.
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input not found: {path}"
        )

    table = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    print(f"\nRows: {len(table)}")
    print(f"Columns: {len(table.columns)}")

    # ESS column names are needed to design the integration correctly.
    # The socioeconomic table has many metadata columns already known.
    if label != "MAIN SOCIOECONOMIC TABLE":
        schema = pd.DataFrame(
            {
                "column": table.columns,
                "dtype": [
                    str(dtype)
                    for dtype in table.dtypes
                ],
                "missing_n": [
                    int(table[column].isna().sum())
                    for column in table.columns
                ],
            }
        )

        print("\nCOLUMN SCHEMA")
        print(schema.to_string(index=False))

        # Transpose a small preview so wide tables remain readable.
        print("\nFIRST THREE ROWS — TRANSPOSED")
        print(table.head(3).T.to_string())

    else:
        print("\nMAIN VALUE COLUMNS")

        preview_columns = [
            "cntry",
            "context_year",
            "gdp_pps_index",
            "employment_rate",
            "unemployment_rate",
            "gini",
            "material_deprivation_rate",
            "inflation_rate",
        ]

        print(
            table[preview_columns].head(3).to_string(index=False)
        )

    # Inspect possible country identifiers without choosing one silently.
    country_candidates = [
        column
        for column in (
            "cntry",
            "ess_cntry",
            "country_code",
            "country",
        )
        if column in table.columns
    ]

    print("\nCOUNTRY IDENTIFIER CANDIDATES")
    print(country_candidates)

    for column in country_candidates:
        print(f"\nColumn: {column}")
        print(f"Distinct non-missing values: {table[column].nunique()}")
        print(f"Missing values: {int(table[column].isna().sum())}")

        # Repeated countries can be valid in a long-format estimates table.
        # These counts reveal the row structure; they are not an error test.
        print("Rows per country value:")
        print(
            table[column]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    # Look for common metric identifiers to help recognize long format.
    metric_candidates = [
        column
        for column in (
            "metric",
            "metric_id",
            "variable",
            "variable_name",
            "indicator",
            "indicator_id",
            "outcome",
            "measure",
        )
        if column in table.columns
    ]

    print("\nMETRIC IDENTIFIER CANDIDATES")
    print(metric_candidates)

    for column in metric_candidates:
        print(f"\nDistinct values in {column}:")
        print(
            table[column]
            .drop_duplicates()
            .to_string(index=False)
        )


def main():
    print("STEP 47 — COUNTRY INTEGRATION INPUT INSPECTION")

    # Inspect all relevant inputs before deciding how they should be joined.
    for label, path in FILES.items():
        inspect_file(label, path)

    print("\nCOUNTRY INTEGRATION INPUT INSPECTION COMPLETED.")
    print("No datasets were merged.")
    print("No files were modified.")


# Run the inspection only when this script is executed directly.
if __name__ == "__main__":
    main()