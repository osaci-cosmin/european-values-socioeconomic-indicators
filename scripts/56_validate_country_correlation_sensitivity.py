# Step 56: Independently validate leave-one-country-out sensitivity.
# Recalculate coefficients directly from the integrated country dataset.
# Read existing files only; do not modify inputs, outputs or metadata.

import hashlib
import io
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


# Resolve paths from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

BASELINE_PATH = TABLE_DIR / "ESS11_country_correlations_main_v1.parquet"

DETAIL_PATH = TABLE_DIR / "ESS11_country_correlations_loo_detail_v1.parquet"
DETAIL_CSV_PATH = DETAIL_PATH.with_suffix(".csv")

SUMMARY_PATH = TABLE_DIR / "ESS11_country_correlations_loo_summary_v1.parquet"
SUMMARY_CSV_PATH = SUMMARY_PATH.with_suffix(".csv")

METADATA_PATH = (
    TABLE_DIR / "ESS11_country_correlations_loo_v1.metadata.json"
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

METHODS = ["pearson", "spearman"]
PAIR_KEYS = ["ess_variable", "economic_variable"]
DETAIL_KEYS = [*PAIR_KEYS, "omitted_country"]

ABSOLUTE_TOLERANCE = 1e-12


# Define expected storage types separately from the saved files.
DETAIL_TYPES = {
    "ess_variable": "string",
    "economic_variable": "string",
    "omitted_country": "string",
    "full_countries_n": "int64",
    "loo_countries_n": "int64",
}

SUMMARY_TYPES = {
    "ess_variable": "string",
    "economic_variable": "string",
    "full_countries_n": "int64",
    "loo_countries_n": "int64",
    "omissions_n": "int64",
}

for method in METHODS:
    DETAIL_TYPES.update(
        {
            f"{method}_full": "float64",
            f"{method}_loo": "float64",
            f"{method}_change": "float64",
            f"{method}_sign_flip": "bool",
        }
    )

    SUMMARY_TYPES.update(
        {
            f"{method}_full": "float64",
            f"{method}_loo_min": "float64",
            f"{method}_loo_max": "float64",
            f"{method}_max_abs_change": "float64",
            f"{method}_largest_change_countries": "string",
            f"{method}_sign_flips_n": "int64",
        }
    )


def require(condition, message):
    # Stop with an explanation when a validation requirement fails.
    if not condition:
        raise ValueError(message)


def file_bytes(path):
    require(path.is_file(), f"Required file not found: {path}")
    return path.read_bytes()


def sha256(content):
    # Hash the original bytes rather than a reconstructed file.
    return hashlib.sha256(content).hexdigest()


def read_parquet(content):
    return pd.read_parquet(io.BytesIO(content), engine="pyarrow")


def pearson_formula(x, y):
    # Use centered sums instead of the Step 55 numpy.corrcoef call.
    n = len(x)

    require(n == len(y) and n >= 3, "Invalid correlation sample size.")
    require(
        all(math.isfinite(value) for value in x + y),
        "Non-finite values in an independent calculation.",
    )

    mean_x = math.fsum(x) / n
    mean_y = math.fsum(y) / n

    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]

    sum_xx = math.fsum(value * value for value in dx)
    sum_yy = math.fsum(value * value for value in dy)
    sum_xy = math.fsum(a * b for a, b in zip(dx, dy))

    require(
        sum_xx > 0 and sum_yy > 0,
        "Correlation is undefined for a constant variable.",
    )

    result = sum_xy / (math.sqrt(sum_xx) * math.sqrt(sum_yy))

    require(
        math.isfinite(result)
        and abs(result) <= 1 + ABSOLUTE_TOLERANCE,
        "Invalid independently calculated coefficient.",
    )

    return result


def average_ranks(values):
    # Independently rank each value by counting smaller and equal values.
    # Example: values occupying ranks 2 and 3 both receive rank 2.5.
    ranks = []

    for value in values:
        smaller_n = sum(other < value for other in values)
        equal_n = sum(other == value for other in values)

        ranks.append(1 + smaller_n + (equal_n - 1) / 2)

    n = len(values)

    require(
        math.fsum(ranks) == n * (n + 1) / 2,
        "Independent average-rank check failed.",
    )

    return ranks


def independent_coefficients(x, y):
    # Rebuild Spearman ranks for this exact sample.
    return {
        "pearson": pearson_formula(x, y),
        "spearman": pearson_formula(
            average_ranks(x),
            average_ranks(y),
        ),
    }


def build_independent_detail(source, baseline, input_sha256):
    # The fixed source hash links this calculation to the validated input.
    require(source.columns.is_unique, "Duplicate source column names.")
    require(len(source) == 29, "Expected 29 source countries.")
    require(source["cntry"].notna().all(), "Missing source country codes.")
    require(source["cntry"].is_unique, "Duplicate source countries.")
    require(
        set(source["cntry"]) == EXPECTED_COUNTRIES,
        "Unexpected source country scope.",
    )

    required_baseline = [
        *PAIR_KEYS,
        "countries_n",
        "included_countries",
        "pearson_r",
        "spearman_rho",
        "input_sha256",
        "analysis_unit",
        "alignment_scenario",
        "across_country_weighting",
    ]

    require(baseline.columns.is_unique, "Duplicate baseline column names.")
    require(
        set(required_baseline).issubset(baseline.columns),
        "Missing required baseline columns.",
    )
    require(
        not baseline[required_baseline].isna().any().any(),
        "Missing required baseline values.",
    )

    expected_pairs = {
        (outcome, economic)
        for outcome in ESS_VARIABLES
        for economic in ECONOMIC_VARIABLES
    }

    require(
        len(baseline) == 36
        and not baseline.duplicated(PAIR_KEYS).any()
        and set(
            baseline[PAIR_KEYS].itertuples(index=False, name=None)
        ) == expected_pairs,
        "Unexpected baseline pair coverage.",
    )

    for column, expected in [
        ("input_sha256", input_sha256),
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
        ("across_country_weighting", "equal"),
    ]:
        require(
            baseline[column].eq(expected).all(),
            f"Baseline mismatch in {column}.",
        )

    source_lookup = source.set_index("cntry").to_dict(orient="index")
    baseline_lookup = baseline.set_index(PAIR_KEYS)
    country_order = sorted(EXPECTED_COUNTRIES)

    records = []
    baseline_max_difference = 0.0

    for outcome in ESS_VARIABLES:
        for economic in ECONOMIC_VARIABLES:
            # Reconstruct membership from source values, not saved lists.
            members = [
                country
                for country in country_order
                if pd.notna(source_lookup[country][outcome])
                and pd.notna(source_lookup[country][economic])
            ]

            x_full = [
                float(source_lookup[country][economic])
                for country in members
            ]
            y_full = [
                float(source_lookup[country][outcome])
                for country in members
            ]

            recalculated_full = independent_coefficients(x_full, y_full)
            saved = baseline_lookup.loc[(outcome, economic)]

            require(
                len(members) == saved["countries_n"]
                and "|".join(members) == saved["included_countries"],
                f"Baseline sample differs for {outcome} / {economic}.",
            )

            full = {
                "pearson": float(saved["pearson_r"]),
                "spearman": float(saved["spearman_rho"]),
            }

            for method in METHODS:
                difference = abs(
                    recalculated_full[method] - full[method]
                )

                require(
                    math.isfinite(full[method])
                    and difference <= ABSOLUTE_TOLERANCE,
                    f"Baseline coefficient differs: {outcome} / {economic}.",
                )

                baseline_max_difference = max(
                    baseline_max_difference, difference
                )

            for omitted_country in members:
                # Select by country code independently of Step 55's mask.
                retained = [
                    country
                    for country in members
                    if country != omitted_country
                ]

                x = [
                    float(source_lookup[country][economic])
                    for country in retained
                ]
                y = [
                    float(source_lookup[country][outcome])
                    for country in retained
                ]

                reduced = independent_coefficients(x, y)

                record = {
                    "ess_variable": outcome,
                    "economic_variable": economic,
                    "omitted_country": omitted_country,
                    "full_countries_n": len(members),
                    "loo_countries_n": len(retained),
                }

                for method in METHODS:
                    record[f"{method}_full"] = full[method]
                    record[f"{method}_loo"] = reduced[method]
                    record[f"{method}_change"] = (
                        reduced[method] - full[method]
                    )

                records.append(record)

    # Sign flags are checked separately against the validated saved numbers.
    # This preserves the exact sign convention at numerical zero.
    reference_types = {
        column: dtype
        for column, dtype in DETAIL_TYPES.items()
        if dtype != "bool"
    }

    reference = pd.DataFrame(records).astype(reference_types)

    require(len(reference) == 948, "Expected 948 country omissions.")

    print("Baseline samples and independent coefficients: PASSED.")
    print(f"Maximum baseline difference: {baseline_max_difference:.3e}")

    return reference


def check_schema(table, schema, keys, expected_rows, name):
    require(table.columns.is_unique, f"{name}: duplicate columns.")
    require(
        list(table.columns) == list(schema),
        f"{name}: unexpected column schema or order.",
    )
    require(len(table) == expected_rows, f"{name}: unexpected row count.")
    require(
        not table.isna().any().any(),
        f"{name}: unexpected missing values.",
    )
    require(
        not table.duplicated(keys).any(),
        f"{name}: duplicate analytical keys.",
    )

    # Check stored types without converting the Parquet table.
    for column, dtype in schema.items():
        expected_dtype = pd.Series([], dtype=dtype).dtype

        require(
            table[column].dtype == expected_dtype,
            f"{name}: unexpected dtype in {column}: "
            f"{table[column].dtype}",
        )

        if dtype == "float64":
            require(
                all(math.isfinite(value) for value in table[column]),
                f"{name}: non-finite values in {column}.",
            )


def decode_csv(content, schema, name):
    # Keep empty text and Boolean tokens unchanged until explicitly checked.
    raw = pd.read_csv(
        io.BytesIO(content),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
    )

    require(
        raw.columns.is_unique and list(raw.columns) == list(schema),
        f"{name}: unexpected CSV column schema.",
    )

    columns = {}

    for column, dtype in schema.items():
        tokens = raw[column]

        if dtype == "int64":
            require(
                tokens.str.fullmatch(r"\d+").all(),
                f"{name}: invalid integer tokens in {column}.",
            )
            values = [int(token) for token in tokens]

        elif dtype == "float64":
            require(
                tokens.ne("").all(),
                f"{name}: empty numerical cells in {column}.",
            )
            values = [float(token) for token in tokens]

        elif dtype == "bool":
            require(
                tokens.isin(["True", "False"]).all(),
                f"{name}: invalid Boolean tokens in {column}.",
            )
            values = [token == "True" for token in tokens]

        else:
            values = tokens.tolist()

        # Build the DataFrame once to avoid column fragmentation.
        columns[column] = pd.Series(values, dtype=dtype)

    return pd.DataFrame(columns)


def compare_numeric_reference(actual, reference):
    columns = list(reference.columns)

    actual = actual.sort_values(DETAIL_KEYS).reset_index(drop=True)
    reference = reference.sort_values(DETAIL_KEYS).reset_index(drop=True)

    floats = [
        column for column in columns
        if DETAIL_TYPES[column] == "float64"
    ]
    exact = [column for column in columns if column not in floats]

    # Membership and sample sizes must match exactly.
    assert_frame_equal(
        actual[exact],
        reference[exact],
        check_dtype=True,
        check_exact=True,
        obj="Independent omission membership",
    )

    # Coefficients and changes permit only tiny arithmetic differences.
    assert_frame_equal(
        actual[floats],
        reference[floats],
        check_dtype=True,
        check_exact=False,
        rtol=0,
        atol=ABSOLUTE_TOLERANCE,
        obj="Independent omission coefficients",
    )

    maximum_difference = max(
        abs(float(left) - float(right))
        for column in floats
        for left, right in zip(actual[column], reference[column])
    )

    print("Detailed results versus independent calculations: PASSED.")
    print(f"Maximum numerical difference: {maximum_difference:.3e}")

    return maximum_difference


def check_sign_flags(detail):
    # Check the strict sign rule using the numerically validated values.
    for method in METHODS:
        expected = (
            detail[f"{method}_full"] * detail[f"{method}_loo"] < 0
        )

        require(
            detail[f"{method}_sign_flip"].eq(expected).all(),
            f"{method}: incorrect sign-reversal flags.",
        )

    print("Strict sign-reversal flags: PASSED.")


def rebuild_summary(detail):
    # Reconstruct summaries with Python operations on validated detail rows.
    # Use saved values to preserve Step 55's exact-equality tie policy.
    records = []

    for (outcome, economic), group in detail.groupby(
        PAIR_KEYS, sort=True
    ):
        rows = group.sort_values("omitted_country").to_dict("records")

        full_n = rows[0]["full_countries_n"]
        reduced_n = rows[0]["loo_countries_n"]

        require(
            all(
                row["full_countries_n"] == full_n
                and row["loo_countries_n"] == reduced_n
                for row in rows
            ),
            "Inconsistent sample counts within a pair.",
        )
        require(
            len(rows) == full_n and reduced_n == full_n - 1,
            "Incorrect omission coverage within a pair.",
        )

        record = {
            "ess_variable": outcome,
            "economic_variable": economic,
            "full_countries_n": full_n,
            "loo_countries_n": reduced_n,
            "omissions_n": len(rows),
        }

        for method in METHODS:
            full = rows[0][f"{method}_full"]

            require(
                all(row[f"{method}_full"] == full for row in rows),
                "Baseline coefficient varies within an omission group.",
            )

            coefficients = [row[f"{method}_loo"] for row in rows]
            changes = [abs(row[f"{method}_change"]) for row in rows]
            maximum_change = max(changes)

            # Retain all countries tied exactly at the saved maximum.
            tied_countries = [
                row["omitted_country"]
                for row, change in zip(rows, changes)
                if change == maximum_change
            ]

            record.update(
                {
                    f"{method}_full": full,
                    f"{method}_loo_min": min(coefficients),
                    f"{method}_loo_max": max(coefficients),
                    f"{method}_max_abs_change": maximum_change,
                    f"{method}_largest_change_countries": "|".join(
                        tied_countries
                    ),
                    f"{method}_sign_flips_n": sum(
                        row[f"{method}_sign_flip"] for row in rows
                    ),
                }
            )

        records.append(record)

    return pd.DataFrame(records).astype(SUMMARY_TYPES)


def compare_exact(actual, expected, keys, name):
    # Serialization and summary aggregation should preserve exact values.
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)

    assert_frame_equal(
        actual,
        expected,
        check_dtype=True,
        check_exact=True,
        obj=name,
    )

    print(f"{name}: exact comparison PASSED.")


def check_metadata(metadata, snapshots):
    expected = {
        "step": 55,
        "input_filename": INPUT_PATH.name,
        "input_sha256": sha256(snapshots[INPUT_PATH]),
        "baseline_filename": BASELINE_PATH.name,
        "baseline_sha256": sha256(snapshots[BASELINE_PATH]),
        "analysis_unit": "country",
        "alignment_scenario": "primary",
        "across_country_weighting": "equal",
        "pairs_n": 36,
        "country_omissions_n": 948,
        "recalculated_coefficients_n": 1896,
        "sample_rule": "omit one country from each baseline pair sample",
        "spearman_rule": "recalculate average ranks after each omission",
        "change_definition": "leave-one-out coefficient minus baseline",
        "sign_flip_definition": "baseline times leave-one-out coefficient < 0",
        "largest_change_ties": "retain all exact ties",
        "country_list_separator": "|",
        "baseline_tolerance": ABSOLUTE_TOLERANCE,
        "main_sample_modified": False,
        "imputation_performed": False,
        "survey_uncertainty_propagated": False,
        "p_values_calculated": False,
        "ranges_are_confidence_intervals": False,
        # Preserve the historical status recorded when Step 55 created files.
        "validation_status": "independent_validation_pending",
    }

    require(isinstance(metadata, dict), "Metadata must be a JSON object.")

    for field, value in expected.items():
        actual = metadata.get(field)

        require(
            type(actual) is type(value) and actual == value,
            f"Metadata mismatch: {field}",
        )

    timestamp = datetime.fromisoformat(metadata["created_at_utc"])

    require(
        timestamp.tzinfo is not None
        and timestamp.utcoffset() == timedelta(0),
        "Metadata timestamp must specify UTC.",
    )

    for field in ["numpy_version", "pandas_version", "scipy_version"]:
        require(
            isinstance(metadata.get(field), str)
            and bool(metadata[field].strip()),
            f"Missing software-version metadata: {field}",
        )

    outputs = metadata.get("outputs")
    require(
        isinstance(outputs, list) and len(outputs) == 4,
        "Metadata must identify exactly four result files.",
    )

    recorded = {}

    for item in outputs:
        require(
            isinstance(item, dict)
            and set(item) == {"filename", "sha256"},
            "Malformed output metadata entry.",
        )
        require(
            item["filename"] not in recorded,
            "Duplicate output filename in metadata.",
        )
        recorded[item["filename"]] = item["sha256"]

    expected_hashes = {
        path.name: sha256(snapshots[path])
        for path in [
            DETAIL_PATH,
            DETAIL_CSV_PATH,
            SUMMARY_PATH,
            SUMMARY_CSV_PATH,
        ]
    }

    require(recorded == expected_hashes, "Output hash mismatch.")
    print("Source hashes, output hashes and provenance metadata: PASSED.")


def main():
    print("STEP 56 — INDEPENDENT CORRELATION SENSITIVITY VALIDATION")

    paths = [
        INPUT_PATH,
        BASELINE_PATH,
        DETAIL_PATH,
        DETAIL_CSV_PATH,
        SUMMARY_PATH,
        SUMMARY_CSV_PATH,
        METADATA_PATH,
    ]

    # Retain exact byte snapshots for reading and provenance checks.
    snapshots = {path: file_bytes(path) for path in paths}

    input_sha256 = sha256(snapshots[INPUT_PATH])

    require(
        input_sha256 == EXPECTED_INPUT_SHA256,
        "Integrated input hash differs from the validated source.",
    )

    source = read_parquet(snapshots[INPUT_PATH])
    baseline = read_parquet(snapshots[BASELINE_PATH])
    detail = read_parquet(snapshots[DETAIL_PATH])
    summary = read_parquet(snapshots[SUMMARY_PATH])

    detail_csv = decode_csv(
        snapshots[DETAIL_CSV_PATH],
        DETAIL_TYPES,
        "Detailed CSV",
    )
    summary_csv = decode_csv(
        snapshots[SUMMARY_CSV_PATH],
        SUMMARY_TYPES,
        "Summary CSV",
    )

    for table, schema, keys, rows, name in [
        (detail, DETAIL_TYPES, DETAIL_KEYS, 948, "Detailed Parquet"),
        (detail_csv, DETAIL_TYPES, DETAIL_KEYS, 948, "Detailed CSV"),
        (summary, SUMMARY_TYPES, PAIR_KEYS, 36, "Summary Parquet"),
        (summary_csv, SUMMARY_TYPES, PAIR_KEYS, 36, "Summary CSV"),
    ]:
        check_schema(table, schema, keys, rows, name)

    print("\nSchemas, row counts, types and unique keys: PASSED.")

    reference = build_independent_detail(
        source, baseline, input_sha256
    )

    maximum_difference = compare_numeric_reference(detail, reference)
    check_sign_flags(detail)

    compare_exact(
        detail_csv,
        detail,
        DETAIL_KEYS,
        "Detailed CSV versus Parquet",
    )

    rebuilt_summary = rebuild_summary(detail)

    compare_exact(
        summary,
        rebuilt_summary,
        PAIR_KEYS,
        "Summary versus independently aggregated detail",
    )

    compare_exact(
        summary_csv,
        summary,
        PAIR_KEYS,
        "Summary CSV versus Parquet",
    )

    metadata = json.loads(
        snapshots[METADATA_PATH].decode("utf-8")
    )
    check_metadata(metadata, snapshots)

    print("\nSUMMARY")
    print(f"Validated analytical pairs: {len(summary)}")
    print(f"Validated country omissions: {len(detail)}")
    print(f"Validated recalculated coefficients: {len(detail) * 2}")
    print(
        "Minimum countries after omission: "
        f"{detail['loo_countries_n'].min()}"
    )
    print(
        "Maximum countries after omission: "
        f"{detail['loo_countries_n'].max()}"
    )

    for method in METHODS:
        pairs_with_flips = int(
            summary[f"{method}_sign_flips_n"].gt(0).sum()
        )
        print(
            f"Pairs with at least one {method.title()} sign reversal: "
            f"{pairs_with_flips}"
        )

    print(f"Absolute numerical tolerance: {ABSOLUTE_TOLERANCE:.1e}")
    print(f"Maximum omission numerical difference: {maximum_difference:.3e}")

    print("\nCOUNTRY CORRELATION SENSITIVITY VALIDATION PASSED.")
    print("All omitted-country samples and coefficients were checked.")
    print("Sensitivity ranges, maximum changes and exact ties matched.")
    print("Sign-reversal flags and counts matched.")
    print("Sensitivity ranges are not confidence intervals.")
    print("No files were modified.")


if __name__ == "__main__":
    main()