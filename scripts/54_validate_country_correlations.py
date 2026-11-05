# Step 54: Independently validate the Step 53 correlation outputs.
# Recalculate from the integrated input without importing Step 53.
# This script reads files only; it does not modify them.

import hashlib
import io
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pandas.testing import assert_frame_equal


# Resolve paths from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

PARQUET_PATH = TABLE_DIR / "ESS11_country_correlations_main_v1.parquet"
CSV_PATH = TABLE_DIR / "ESS11_country_correlations_main_v1.csv"
METADATA_PATH = (
    TABLE_DIR / "ESS11_country_correlations_main_v1.metadata.json"
)

EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

KEY_COLUMNS = ["ess_variable", "economic_variable"]

FLOAT_COLUMNS = [
    "pearson_r",
    "spearman_rho",
    "pearson_minus_spearman",
]

INTEGER_COLUMNS = [
    "countries_n",
    "unavailable_countries_n",
    "context_2023_n",
    "context_2024_n",
    "economic_unique_values_n",
    "ess_unique_values_n",
]

TEXT_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "included_countries",
    "unavailable_countries",
    "analysis_unit",
    "alignment_scenario",
    "across_country_weighting",
    "input_filename",
    "input_sha256",
]

# Allow tiny arithmetic differences, not substantive changes.
ABSOLUTE_TOLERANCE = 1e-12


def require(condition, message):
    # Raise an explicit error if a validation condition fails.
    if not condition:
        raise ValueError(message)


def read_bytes(path):
    require(path.is_file(), f"Required file not found: {path}")
    return path.read_bytes()


def sha256(content):
    # Hash the original file bytes rather than a reconstructed table.
    return hashlib.sha256(content).hexdigest()


def check_source(data):
    require(data.columns.is_unique, "Source has duplicate column names.")

    required = [
        "cntry",
        "context_year",
        "analysis_unit",
        "alignment_scenario",
        *ESS_VARIABLES,
        *ECONOMIC_VARIABLES,
    ]

    missing = sorted(set(required) - set(data.columns))
    require(not missing, f"Missing source columns: {missing}")

    require(len(data) == 29, "Expected 29 source countries.")
    require(data["cntry"].notna().all(), "Missing source country codes.")
    require(data["cntry"].is_unique, "Duplicate source country codes.")
    require(
        set(data["cntry"]) == EXPECTED_COUNTRIES,
        "Source country scope differs from V1.",
    )

    for column, expected in [
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
    ]:
        require(
            data[column].notna().all()
            and data[column].eq(expected).all(),
            f"Unexpected or missing source values in {column}.",
        )

    require(
        data["context_year"].isin([2023, 2024]).all(),
        "Unexpected or missing source context years.",
    )

    for variable in ESS_VARIABLES + ECONOMIC_VARIABLES:
        series = data[variable]

        require(
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype),
            f"{variable}: expected a numeric source column.",
        )

        # Missing values are allowed; infinite values are not.
        require(
            all(math.isfinite(float(value)) for value in series.dropna()),
            f"{variable}: infinite source values found.",
        )


def independent_pearson(x, y):
    # Use the centered-sums formula instead of numpy.corrcoef.
    n = len(x)

    require(n == len(y) and n >= 3, "Invalid correlation sample size.")

    mean_x = math.fsum(x) / n
    mean_y = math.fsum(y) / n

    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]

    sum_xx = math.fsum(value * value for value in centered_x)
    sum_yy = math.fsum(value * value for value in centered_y)

    # Constant variables have no defined correlation.
    require(
        sum_xx > 0 and sum_yy > 0,
        "A correlation input is constant.",
    )

    sum_xy = math.fsum(
        x_value * y_value
        for x_value, y_value in zip(centered_x, centered_y)
    )

    coefficient = sum_xy / (math.sqrt(sum_xx) * math.sqrt(sum_yy))

    require(
        math.isfinite(coefficient)
        and abs(coefficient) <= 1 + ABSOLUTE_TOLERANCE,
        "Invalid independently calculated correlation.",
    )

    return coefficient


def independent_average_ranks(values):
    # Sort positions so we can return ranks in the original country order.
    order = sorted(range(len(values)), key=lambda index: values[index])

    ranks = [0.0] * len(values)
    start = 0

    while start < len(values):
        end = start + 1

        # Consecutive equal values form one tied group.
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1

        # Positions are zero-based, while statistical ranks start at one.
        # The tied group occupies ranks start + 1 through end.
        average_rank = ((start + 1) + end) / 2

        for position in range(start, end):
            ranks[order[position]] = average_rank

        start = end

    # Average ranks preserve the sum of the original ranks 1 through n.
    n = len(values)
    require(
        math.fsum(ranks) == n * (n + 1) / 2,
        "Independent rank-sum check failed.",
    )

    return ranks


def build_reference(data, input_sha256):
    # Use a country lookup and Python lists to reconstruct each sample.
    country_lookup = data.set_index("cntry").to_dict(orient="index")
    country_order = sorted(EXPECTED_COUNTRIES)

    records = []

    for outcome in ESS_VARIABLES:
        for economic in ECONOMIC_VARIABLES:
            included = []
            unavailable = []
            x = []
            y = []
            years = []

            for country in country_order:
                source_row = country_lookup[country]

                x_value = source_row[economic]
                y_value = source_row[outcome]

                # Both values must exist for the country to enter this pair.
                if pd.isna(x_value) or pd.isna(y_value):
                    unavailable.append(country)
                    continue

                included.append(country)
                x.append(float(x_value))
                y.append(float(y_value))
                years.append(int(source_row["context_year"]))

            pearson_r = independent_pearson(x, y)

            # Rank only the countries included in this particular pair.
            x_ranks = independent_average_ranks(x)
            y_ranks = independent_average_ranks(y)
            spearman_rho = independent_pearson(x_ranks, y_ranks)

            records.append(
                {
                    "ess_variable": outcome,
                    "economic_variable": economic,
                    "countries_n": len(included),
                    "unavailable_countries_n": len(unavailable),
                    "pearson_r": pearson_r,
                    "spearman_rho": spearman_rho,
                    "pearson_minus_spearman": pearson_r - spearman_rho,
                    "included_countries": "|".join(included),
                    "unavailable_countries": "|".join(unavailable),
                    "context_2023_n": years.count(2023),
                    "context_2024_n": years.count(2024),
                    "economic_unique_values_n": len(set(x)),
                    "ess_unique_values_n": len(set(y)),
                    "analysis_unit": "country",
                    "alignment_scenario": "primary",
                    "across_country_weighting": "equal",
                    "input_filename": INPUT_PATH.name,
                    "input_sha256": input_sha256,
                }
            )

    reference = pd.DataFrame(records)

    # Define expected types independently of the saved result files.
    expected_types = {
        **{column: "string" for column in TEXT_COLUMNS},
        **{column: "int64" for column in INTEGER_COLUMNS},
        **{column: "float64" for column in FLOAT_COLUMNS},
    }

    return reference.astype(expected_types)


def check_result_structure(table, reference, name):
    require(table.columns.is_unique, f"{name}: duplicate column names.")

    require(
        list(table.columns) == list(reference.columns),
        f"{name}: column schema or column order differs.",
    )

    require(len(table) == 36, f"{name}: expected 36 rows.")

    require(
        not table[KEY_COLUMNS].isna().any().any(),
        f"{name}: missing pair identifiers.",
    )

    require(
        not table.duplicated(KEY_COLUMNS).any(),
        f"{name}: duplicate analytical pairs.",
    )

    actual_keys = set(
        table[KEY_COLUMNS].itertuples(index=False, name=None)
    )
    expected_keys = set(
        reference[KEY_COLUMNS].itertuples(index=False, name=None)
    )

    require(
        actual_keys == expected_keys,
        f"{name}: missing or unexpected analytical pairs.",
    )


def decode_csv(csv_text):
    # CSV stores text, so restore the predefined result schema.
    # Do not infer missing values from country codes or empty strings.
    decoded = {}

    for column in csv_text.columns:
        raw = csv_text[column]

        if column in INTEGER_COLUMNS:
            require(
                raw.str.fullmatch(r"\d+").all(),
                f"CSV: invalid integer text in {column}.",
            )
            values = [int(value) for value in raw]
            dtype = "int64"

        elif column in FLOAT_COLUMNS:
            require(
                raw.ne("").all(),
                f"CSV: empty numerical cells in {column}.",
            )
            values = [float(value) for value in raw]

            require(
                all(math.isfinite(value) for value in values),
                f"CSV: non-finite values in {column}.",
            )
            dtype = "float64"

        else:
            # Preserve country lists and provenance text exactly.
            values = raw.tolist()
            dtype = "string"

        # Collect columns before building the DataFrame.
        decoded[column] = pd.Series(values, dtype=dtype)

    return pd.DataFrame(decoded)


def align_rows(table):
    # Compare matching pairs even if physical row order differs.
    return table.sort_values(KEY_COLUMNS).reset_index(drop=True)


def compare_tables(actual, expected, name):
    actual = align_rows(actual)
    expected = align_rows(expected)

    exact_columns = [
        column
        for column in expected.columns
        if column not in FLOAT_COLUMNS
    ]

    # Country lists, counts, provenance and their types must match exactly.
    assert_frame_equal(
        actual[exact_columns],
        expected[exact_columns],
        check_dtype=True,
        check_exact=True,
        obj=f"{name}: exact fields",
    )

    # Arithmetic implementations can differ at the last few binary digits.
    assert_frame_equal(
        actual[FLOAT_COLUMNS],
        expected[FLOAT_COLUMNS],
        check_dtype=True,
        check_exact=False,
        rtol=0,
        atol=ABSOLUTE_TOLERANCE,
        obj=f"{name}: numerical fields",
    )

    # Report the largest difference across the three numerical result fields.
    maximum_difference = max(
        abs(float(left) - float(right))
        for column in FLOAT_COLUMNS
        for left, right in zip(actual[column], expected[column])
    )

    print(f"{name}: PASSED.")
    print(f"Maximum numerical difference: {maximum_difference:.3e}")

    return maximum_difference


def check_metadata(metadata, input_sha256, parquet_bytes, csv_bytes):
    require(isinstance(metadata, dict), "Metadata must be a JSON object.")

    expected_fields = {
        "step": 53,
        "input_filename": INPUT_PATH.name,
        "input_sha256": input_sha256,
        "analysis_unit": "country",
        "alignment_scenario": "primary",
        "pairs_n": 36,
        "coefficients_n": 72,
        "ess_variables": ESS_VARIABLES,
        "economic_variables": ECONOMIC_VARIABLES,
        "across_country_weighting": "equal",
        "within_country_ess_weighting": "existing survey estimates retained",
        "missing_policy": "complete pairs separately per relationship",
        "country_list_separator": "|",
        "pearson_method": "numpy.corrcoef on observed values",
        "spearman_method": "numpy.corrcoef on scipy average ranks",
        "year_adjustment": "none",
        "survey_uncertainty_propagated": False,
        "p_values_calculated": False,
        "confidence_intervals_calculated": False,
        "extreme_values_removed": False,
        # This is the creation-stage record from Step 53.
        # A read-only validator does not rewrite historical metadata.
        "validation_status": "independent_validation_pending",
    }

    for field, expected in expected_fields.items():
        require(field in metadata, f"Missing metadata field: {field}")

        actual = metadata[field]

        # Preserve the distinction between JSON booleans and integers.
        require(
            type(actual) is type(expected) and actual == expected,
            f"Metadata mismatch: {field}",
        )

    # Validate that the recorded creation timestamp is timezone-aware UTC.
    timestamp = datetime.fromisoformat(metadata["created_at_utc"])

    require(
        timestamp.tzinfo is not None
        and timestamp.utcoffset() == timedelta(0),
        "Metadata creation timestamp must specify UTC.",
    )

    # Historical software versions need not equal the current environment.
    # Check their presence without claiming to reproduce that environment.
    for field in [
        "python_version",
        "numpy_version",
        "pandas_version",
        "scipy_version",
    ]:
        require(
            isinstance(metadata.get(field), str)
            and bool(metadata[field].strip()),
            f"Missing or empty software version: {field}",
        )

    outputs = metadata.get("outputs")

    require(
        isinstance(outputs, list) and len(outputs) == 2,
        "Metadata must identify exactly two result files.",
    )

    recorded_hashes = {}

    for item in outputs:
        require(
            isinstance(item, dict)
            and set(item) == {"filename", "sha256"},
            "Malformed output provenance entry.",
        )
        require(
            item["filename"] not in recorded_hashes,
            "Duplicate output filename in metadata.",
        )
        recorded_hashes[item["filename"]] = item["sha256"]

    expected_hashes = {
        PARQUET_PATH.name: sha256(parquet_bytes),
        CSV_PATH.name: sha256(csv_bytes),
    }

    require(
        recorded_hashes == expected_hashes,
        "Saved result hashes do not match metadata.",
    )


def main():
    print("STEP 54 — INDEPENDENT COUNTRY CORRELATION VALIDATION")

    # Keep exact byte snapshots for both parsing and hash verification.
    input_bytes = read_bytes(INPUT_PATH)
    parquet_bytes = read_bytes(PARQUET_PATH)
    csv_bytes = read_bytes(CSV_PATH)
    metadata_bytes = read_bytes(METADATA_PATH)

    input_sha256 = sha256(input_bytes)

    require(
        input_sha256 == EXPECTED_INPUT_SHA256,
        "Integrated input hash differs from the validated source.",
    )

    source = pd.read_parquet(io.BytesIO(input_bytes), engine="pyarrow")
    check_source(source)

    reference = build_reference(source, input_sha256)

    parquet = pd.read_parquet(
        io.BytesIO(parquet_bytes),
        engine="pyarrow",
    )

    # Read all CSV cells as text before applying the explicit schema.
    csv_text = pd.read_csv(
        io.BytesIO(csv_bytes),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
    )

    check_result_structure(parquet, reference, "Parquet")
    check_result_structure(csv_text, reference, "CSV")

    csv_table = decode_csv(csv_text)

    print("\nSchema, pair coverage and unique keys: PASSED.")

    parquet_difference = compare_tables(
        parquet,
        reference,
        "Parquet versus independent calculations",
    )

    csv_difference = compare_tables(
        csv_table,
        reference,
        "CSV versus independent calculations",
    )

    format_difference = compare_tables(
        csv_table,
        parquet,
        "CSV versus Parquet",
    )

    metadata = json.loads(metadata_bytes.decode("utf-8"))

    check_metadata(
        metadata,
        input_sha256,
        parquet_bytes,
        csv_bytes,
    )

    print("Input/output hashes and provenance metadata: PASSED.")
    print("Country membership, context-year counts and tie counts: PASSED.")

    # Show the sample sizes reconstructed from the integrated source.
    counts = reference.pivot(
        index="ess_variable",
        columns="economic_variable",
        values="countries_n",
    ).reindex(
        index=ESS_VARIABLES,
        columns=ECONOMIC_VARIABLES,
    )

    print("\nVALIDATED COUNTRIES PER PAIR")
    print(counts.to_string())

    print("\nSUMMARY")
    print(f"Validated analytical pairs: {len(reference)}")
    print(f"Validated correlation coefficients: {len(reference) * 2}")
    print(f"Minimum countries per pair: {reference['countries_n'].min()}")
    print(f"Maximum countries per pair: {reference['countries_n'].max()}")
    print(f"Absolute numerical tolerance: {ABSOLUTE_TOLERANCE:.1e}")

    print(
        "Largest numerical difference across comparisons: "
        f"{max(parquet_difference, csv_difference, format_difference):.3e}"
    )

    print("\nCOUNTRY CORRELATION VALIDATION PASSED.")
    print("Pearson was recalculated from centered sums.")
    print("Spearman was recalculated using independently constructed ranks.")
    print("Country samples and provenance matched.")
    print("This validates computation, not causal or inferential claims.")
    print("No files were modified.")


if __name__ == "__main__":
    main()