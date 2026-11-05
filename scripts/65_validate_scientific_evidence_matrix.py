
# ============================================================
# STEP 65 - INDEPENDENT SCIENTIFIC EVIDENCE MATRIX VALIDATION
# ============================================================
#
# Independently validate the Step 64 evidence matrix.
#
# Verify:
# - Exact 36 x 72 output structure.
# - All 37 original Step 59 columns, values and dtypes.
# - All 35 newly added columns.
# - All 96 Step 62 scenario records.
# - All 72 applicable additional exclusion scenarios.
# - Correct NA semantics for the 12 non-Schwartz pairs.
# - Actual country membership for EE and GB.
# - Numerical deltas and strict sign-reversal flags.
# - All 2,592 Parquet cells and CSV consistency.
# - Source/output hashes and metadata provenance.
#
# All files are read-only.
# No previous calculations are replaced or overwritten.
# ============================================================

import csv
import hashlib
import json
import math
import re

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 1. DEFINE PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

INTEGRATED = (
    ROOT / "data/processed/"
    "ESS11_country_integrated_main_v1.parquet"
)

BASELINE = (
    TABLES / "ESS11_country_correlations_main_v1.parquet"
)

SYNTHESIS = (
    TABLES
    / "ESS11_country_correlation_robustness_synthesis_v1.parquet"
)

SYNTHESIS_METADATA = (
    TABLES
    / "ESS11_country_correlation_robustness_synthesis_v1.metadata.json"
)

SCHWARTZ = (
    TABLES
    / "ESS11_schwartz_country_exclusion_sensitivity_v1.parquet"
)

SCHWARTZ_METADATA = (
    TABLES
    / "ESS11_schwartz_country_exclusion_sensitivity_v1.metadata.json"
)

STEM = "ESS11_scientific_evidence_matrix_v1"

RESULT_PARQUET = TABLES / f"{STEM}.parquet"
RESULT_CSV = TABLES / f"{STEM}.csv"
RESULT_METADATA = TABLES / f"{STEM}.metadata.json"

EXPECTED_INTEGRATED_HASH = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

TOLERANCE = 1e-12


# ------------------------------------------------------------
# 2. DEFINE THE EXPECTED ANALYTICAL UNIVERSE
# ------------------------------------------------------------

ESS_VARIABLES = [
    "mean_social_trust",
    "mean_left_right",
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
]

SCHWARTZ_VARIABLES = {
    "mean_openness",
    "mean_conservation",
    "mean_self_transcendence",
    "mean_self_enhancement",
}

ECONOMIC_VARIABLES = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gini",
    "material_deprivation_rate",
    "inflation_rate",
]

COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

SCENARIOS = {
    "primary_full": set(),
    "exclude_EE": {"EE"},
    "exclude_GB": {"GB"},
    "exclude_EE_GB": {"EE", "GB"},
}

EXCLUSION_SCENARIOS = [
    "exclude_EE",
    "exclude_GB",
    "exclude_EE_GB",
]

# Define all expected ESS-Eurostat analytical pairs.
EXPECTED_PAIRS = {
    (ess, economic)
    for ess in ESS_VARIABLES
    for economic in ECONOMIC_VARIABLES
}

# Define the 11 saved fields for each additional scenario.
SCENARIO_FIELDS = [
    "actual_exclusions",
    "scenario_n",
    "removed_n",
    "pearson_scenario",
    "pearson_change",
    "pearson_abs_change",
    "pearson_sign_flip",
    "spearman_scenario",
    "spearman_change",
    "spearman_abs_change",
    "spearman_sign_flip",
]

SCENARIO_TEXT = {"actual_exclusions"}
SCENARIO_INTEGERS = {"scenario_n", "removed_n"}

SCENARIO_BOOLEANS = {
    "pearson_sign_flip",
    "spearman_sign_flip",
}

SCENARIO_FLOATS = (
    set(SCENARIO_FIELDS)
    - SCENARIO_TEXT
    - SCENARIO_INTEGERS
    - SCENARIO_BOOLEANS
)

# Reconstruct the expected Step 62 column order independently.
STEP62_COLUMNS = [
    "ess_variable",
    "economic_variable",
    "scenario",
    "requested_exclusions",
    "actual_exclusions",
    "primary_n",
    "scenario_n",
    "removed_n",
    "scenario_countries",
] + [
    f"{method}_{suffix}"
    for method in ("pearson", "spearman")
    for suffix in (
        "primary",
        "scenario",
        "change",
        "abs_change",
        "sign_flip",
    )
]


# ------------------------------------------------------------
# 3. GENERAL HELPER FUNCTIONS
# ------------------------------------------------------------

def require(condition, message):

    # Stop at the first unexpected condition.
    if not bool(condition):
        raise AssertionError(message)


def sha256(path):

    # Calculate the exact SHA-256 checksum of a file.
    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def compare_float(actual, expected, label):

    # Missing values must not be interpreted as zero.
    require(
        not pd.isna(actual) and not pd.isna(expected),
        f"{label}: unexpected missing number",
    )

    a = float(actual)
    b = float(expected)

    require(
        math.isfinite(a) and math.isfinite(b),
        f"{label}: non-finite number",
    )

    difference = abs(a - b)

    require(
        difference <= TOLERANCE,
        f"{label}: numerical difference {difference:.3e}",
    )

    return difference


def compare_exact(actual, expected, label):

    # Compare text, integer counts and Boolean flags exactly.
    if pd.isna(expected):

        require(
            pd.isna(actual),
            f"{label}: expected a missing value",
        )

    else:

        require(
            not pd.isna(actual) and actual == expected,
            f"{label}: actual={actual!r}, expected={expected!r}",
        )


def check_required_columns(table, names, label):

    require(
        table.columns.is_unique,
        f"{label}: duplicate column names",
    )

    missing = sorted(set(names) - set(table.columns))

    require(
        not missing,
        f"{label}: missing columns {missing}",
    )


def get_pair_keys(table):

    return list(
        table[
            ["ess_variable", "economic_variable"]
        ].itertuples(index=False, name=None)
    )


def parse_countries(value, label):

    # The project uses pipe-separated country identifiers.
    require(
        isinstance(value, str),
        f"{label}: expected country-list text",
    )

    values = value.split("|") if value else []

    require(
        len(values) == len(set(values)),
        f"{label}: duplicate country",
    )

    require(
        set(values).issubset(COUNTRIES),
        f"{label}: unknown country",
    )

    return set(values)


def check_utc_timestamp(value):

    # Metadata timestamps must identify a UTC instant.
    require(
        isinstance(value, str),
        "Invalid creation timestamp",
    )

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    except ValueError as exc:

        raise AssertionError(
            "Invalid metadata creation timestamp"
        ) from exc

    require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timedelta(0),
        "Creation timestamp is not UTC",
    )


# ------------------------------------------------------------
# 4. READ ALL REQUIRED SOURCES WITHOUT MODIFYING THEM
# ------------------------------------------------------------

def load_sources():

    # Read the historical analytical tables.
    synthesis = pd.read_parquet(
        SYNTHESIS,
        engine="pyarrow",
    )

    sensitivity = pd.read_parquet(
        SCHWARTZ,
        engine="pyarrow",
    )

    baseline = pd.read_parquet(
        BASELINE,
        engine="pyarrow",
    )

    # Only the required columns of the integrated table
    # are needed to verify original country membership.
    integrated = pd.read_parquet(
        INTEGRATED,
        columns=(
            ["cntry"]
            + ESS_VARIABLES
            + ECONOMIC_VARIABLES
        ),
        engine="pyarrow",
    )

    result = pd.read_parquet(
        RESULT_PARQUET,
        engine="pyarrow",
    )

    with SYNTHESIS_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        synthesis_metadata = json.load(handle)

    with SCHWARTZ_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        sensitivity_metadata = json.load(handle)

    with RESULT_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        result_metadata = json.load(handle)

    for name, metadata in (
        ("Step 59", synthesis_metadata),
        ("Step 62", sensitivity_metadata),
        ("Step 64", result_metadata),
    ):

        require(
            isinstance(metadata, dict),
            f"{name}: expected JSON object",
        )

    return (
        integrated,
        baseline,
        synthesis,
        sensitivity,
        result,
        sensitivity_metadata,
        result_metadata,
    )


# ------------------------------------------------------------
# 5. VERIFY SOURCE TABLE STRUCTURES AND COUNTRY COVERAGE
# ------------------------------------------------------------

def validate_sources(integrated, baseline, synthesis, sensitivity):

    require(
        len(integrated) == 29,
        "Integrated dataset must contain 29 countries",
    )

    require(
        integrated["cntry"].notna().all()
        and integrated["cntry"].is_unique,
        "Invalid integrated country identifiers",
    )

    require(
        set(integrated["cntry"]) == COUNTRIES,
        "Unexpected integrated country universe",
    )

    require(
        len(synthesis) == 36
        and len(synthesis.columns) == 37,
        "Unexpected Step 59 synthesis dimensions",
    )

    require(
        len(baseline) == 36,
        "Expected 36 original correlation records",
    )

    check_required_columns(
        synthesis,
        [
            "ess_variable",
            "economic_variable",
            "countries_n",
            "included_countries",
            "pearson_baseline",
            "spearman_baseline",
        ],
        "Step 59",
    )

    check_required_columns(
        baseline,
        [
            "ess_variable",
            "economic_variable",
            "countries_n",
            "included_countries",
            "pearson_r",
            "spearman_rho",
            "input_sha256",
        ],
        "Baseline",
    )

    require(
        baseline["input_sha256"].eq(
            EXPECTED_INTEGRATED_HASH
        ).all(),
        "Baseline input provenance differs",
    )

    require(
        set(get_pair_keys(synthesis)) == EXPECTED_PAIRS
        and len(set(get_pair_keys(synthesis))) == 36,
        "Incorrect Step 59 relationship coverage",
    )

    require(
        set(get_pair_keys(baseline)) == EXPECTED_PAIRS
        and len(set(get_pair_keys(baseline))) == 36,
        "Incorrect baseline relationship coverage",
    )

    require(
        sensitivity.columns.tolist() == STEP62_COLUMNS,
        "Unexpected Step 62 schema",
    )

    require(
        len(sensitivity) == 96,
        "Expected 96 Step 62 scenario rows",
    )

    require(
        not sensitivity.isna().any().any(),
        "Unexpected null in Step 62 results",
    )

    keys = list(
        sensitivity[
            ["ess_variable", "economic_variable", "scenario"]
        ].itertuples(index=False, name=None)
    )

    expected_keys = {
        (ess, economic, scenario)
        for ess in SCHWARTZ_VARIABLES
        for economic in ECONOMIC_VARIABLES
        for scenario in SCENARIOS
    }

    require(
        len(keys) == len(set(keys))
        and set(keys) == expected_keys,
        "Step 62 scenario keys differ",
    )

    print(
        "Original source schemas, 36 pairs, "
        "96 scenarios and country universe: PASSED."
    )


# ------------------------------------------------------------
# 6. VERIFY EXACT OUTPUT COLUMNS AND DATA TYPES
# ------------------------------------------------------------

def validate_output_schema(synthesis, result):

    original_columns = synthesis.columns.tolist()

    additional_columns = [
        "schwartz_sensitivity_applicable",
        "schwartz_sensitivity_scope_note",
    ] + [
        f"{scenario}__{field}"
        for scenario in EXCLUSION_SCENARIOS
        for field in SCENARIO_FIELDS
    ]

    expected_columns = (
        original_columns + additional_columns
    )

    require(
        len(additional_columns) == 35,
        "Expected exactly 35 additional columns",
    )

    require(
        len(expected_columns) == 72,
        "Expected exactly 72 columns",
    )

    require(
        result.columns.tolist() == expected_columns,
        "Step 64 column names or order differ",
    )

    require(
        len(result) == 36,
        "Expected exactly 36 evidence rows",
    )

    require(
        result.columns.is_unique,
        "Duplicate evidence-matrix column names",
    )

    # The first 37 columns must remain exactly unchanged.
    # This comparison also checks the original pandas dtypes.
    pd.testing.assert_frame_equal(
        result[original_columns],
        synthesis.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
        check_index_type=True,
    )

    # Verify the two newly added methodological columns.
    require(
        str(result[
            "schwartz_sensitivity_applicable"
        ].dtype) == "boolean",
        "Invalid nullable applicability dtype",
    )

    require(
        pd.api.types.is_string_dtype(
            result[
                "schwartz_sensitivity_scope_note"
            ].dtype
        ),
        "Invalid scope-note dtype",
    )

    # Check every additional scenario field.
    for scenario in EXCLUSION_SCENARIOS:

        for field in SCENARIO_FIELDS:

            name = f"{scenario}__{field}"
            dtype = result[name].dtype

            if field in SCENARIO_TEXT:

                require(
                    pd.api.types.is_string_dtype(dtype)
                    and not pd.api.types.is_object_dtype(dtype),
                    f"{name}: expected string dtype",
                )

            elif field in SCENARIO_INTEGERS:

                require(
                    str(dtype) == "Int64",
                    f"{name}: expected nullable Int64",
                )

            elif field in SCENARIO_BOOLEANS:

                require(
                    str(dtype) == "boolean",
                    f"{name}: expected nullable boolean",
                )

            else:

                require(
                    str(dtype) == "Float64",
                    f"{name}: expected nullable Float64",
                )

    # Preserve the exact Step 59 analytical pair ordering.
    require(
        get_pair_keys(result) == get_pair_keys(synthesis),
        "Step 64 row order or analytical pairs differ",
    )

    print(
        "Exact 36 x 72 schema, data types and "
        "original 37 columns: PASSED."
    )

    return expected_columns


# ------------------------------------------------------------
# 7. VERIFY BASELINE COEFFICIENTS AND COUNTRY SAMPLES
# ------------------------------------------------------------

def validate_baseline_links(integrated, baseline, synthesis):

    baseline_lookup = baseline.set_index(
        ["ess_variable", "economic_variable"]
    )

    integrated = integrated.set_index(
        "cntry"
    )

    maximum_difference = 0.0

    # Independently verify all 36 original analytical samples.
    for _, row in synthesis.iterrows():

        ess = row["ess_variable"]
        economic = row["economic_variable"]

        key = (ess, economic)

        original = baseline_lookup.loc[key]

        # Derive the sample directly from the integrated values.
        paired = integrated[
            [ess, economic]
        ].dropna()

        countries = set(paired.index)

        stored = parse_countries(
            row["included_countries"],
            f"{key}: synthesis countries",
        )

        original_stored = parse_countries(
            original["included_countries"],
            f"{key}: baseline countries",
        )

        require(
            countries == stored == original_stored,
            f"{key}: baseline country membership differs",
        )

        require(
            len(countries)
            == int(row["countries_n"])
            == int(original["countries_n"]),
            f"{key}: baseline sample size differs",
        )

        # The validated original coefficients must agree.
        maximum_difference = max(
            maximum_difference,
            compare_float(
                row["pearson_baseline"],
                original["pearson_r"],
                f"{key}: Pearson baseline",
            ),
            compare_float(
                row["spearman_baseline"],
                original["spearman_rho"],
                f"{key}: Spearman baseline",
            ),
        )

    print(
        "All 36 baseline country samples and "
        "original coefficients: PASSED."
    )

    return maximum_difference


# ------------------------------------------------------------
# 8. INDEPENDENTLY VALIDATE ALL 96 SOURCE SCENARIOS
# ------------------------------------------------------------

def validate_scenario_sources(synthesis, sensitivity):

    original = synthesis.set_index(
        ["ess_variable", "economic_variable"]
    )

    maximum_difference = 0.0

    checked = 0

    for _, record in sensitivity.iterrows():

        ess = record["ess_variable"]
        economic = record["economic_variable"]
        scenario = record["scenario"]

        key = (ess, economic)

        base = original.loc[key]

        primary_countries = parse_countries(
            base["included_countries"],
            f"{key}: original sample",
        )

        requested = SCENARIOS[scenario]

        removed = primary_countries & requested

        retained = primary_countries - requested

        require(
            record["requested_exclusions"]
            == "|".join(sorted(requested)),
            f"{key}/{scenario}: requested exclusions differ",
        )

        require(
            record["actual_exclusions"]
            == "|".join(sorted(removed)),
            f"{key}/{scenario}: actual exclusions differ",
        )

        require(
            parse_countries(
                record["scenario_countries"],
                f"{key}/{scenario}: retained countries",
            ) == retained,
            f"{key}/{scenario}: retained sample differs",
        )

        require(
            int(record["primary_n"]) == len(primary_countries)
            and int(record["scenario_n"]) == len(retained)
            and int(record["removed_n"]) == len(removed),
            f"{key}/{scenario}: incorrect sample sizes",
        )

        # Check primary coefficient links with Step 59.
        for method in ("pearson", "spearman"):

            primary = float(record[f"{method}_primary"])
            scenario_value = float(
                record[f"{method}_scenario"]
            )

            maximum_difference = max(
                maximum_difference,
                compare_float(
                    primary,
                    base[f"{method}_baseline"],
                    f"{key}/{scenario}: primary {method}",
                ),
            )

            delta = scenario_value - primary

            maximum_difference = max(
                maximum_difference,
                compare_float(
                    record[f"{method}_change"],
                    delta,
                    f"{key}/{scenario}: {method} change",
                ),
                compare_float(
                    record[f"{method}_abs_change"],
                    abs(delta),
                    f"{key}/{scenario}: {method} absolute change",
                ),
            )

            require(
                bool(record[f"{method}_sign_flip"])
                == (primary * scenario_value < 0),
                f"{key}/{scenario}: {method} sign flag differs",
            )

        # Validate documented country availability.
        require(
            "EE" in primary_countries,
            f"{key}: Estonia must be in the original sample",
        )

        require(
            ("GB" in primary_countries)
            == (economic == "gdp_pps_index"),
            f"{key}: GB availability differs",
        )

        # The full scenario must reproduce the primary result.
        if scenario == "primary_full":

            require(
                len(removed) == 0,
                f"{key}: primary scenario removed countries",
            )

            for method in ("pearson", "spearman"):

                maximum_difference = max(
                    maximum_difference,
                    compare_float(
                        record[f"{method}_scenario"],
                        record[f"{method}_primary"],
                        f"{key}: full {method}",
                    ),
                )

        checked += 1

    require(
        checked == 96,
        "Expected 96 checked source scenarios",
    )

    print(
        "All 96 source scenarios, actual country exclusions "
        "and numerical deltas: PASSED."
    )

    return maximum_difference


# ------------------------------------------------------------
# 9. VERIFY ALL NEW STEP 64 VALUES
# ------------------------------------------------------------

def validate_added_columns(result, sensitivity):

    # Build a lookup using all three source identifiers.
    source_lookup = {
        (
            row["ess_variable"],
            row["economic_variable"],
            row["scenario"],
        ): row
        for row in sensitivity.to_dict("records")
    }

    checked_scenarios = 0
    checked_fields = 0
    max_difference = 0.0

    # Check each output row against the appropriate source.
    for _, row in result.iterrows():

        ess = row["ess_variable"]
        economic = row["economic_variable"]

        applicable = ess in SCHWARTZ_VARIABLES

        compare_exact(
            row["schwartz_sensitivity_applicable"],
            applicable,
            f"{ess}/{economic}: applicability",
        )

        # Independently determine the methodological scope.
        if not applicable:

            expected_note = "not_applicable_non_schwartz"

        elif economic == "gdp_pps_index":

            expected_note = "EE_and_GB_both_in_primary_sample"

        else:

            expected_note = (
                "only_EE_in_primary_sample_GB_unavailable"
            )

        compare_exact(
            row["schwartz_sensitivity_scope_note"],
            expected_note,
            f"{ess}/{economic}: scope note",
        )

        # Process all 33 exclusion fields.
        for scenario in EXCLUSION_SCENARIOS:

            for field in SCENARIO_FIELDS:

                column = f"{scenario}__{field}"

                saved_value = row[column]

                # Non-Schwartz pairs must have truly missing values.
                if not applicable:

                    require(
                        pd.isna(saved_value),
                        f"{ess}/{economic}: {column} must be NA",
                    )

                    checked_fields += 1
                    continue

                # Find the independently validated Step 62 source.
                key = (ess, economic, scenario)

                require(
                    key in source_lookup,
                    f"Missing Step 62 record: {key}",
                )

                expected_value = source_lookup[key][field]

                if field in SCENARIO_FLOATS:

                    max_difference = max(
                        max_difference,
                        compare_float(
                            saved_value,
                            expected_value,
                            f"{key}/{field}",
                        ),
                    )

                else:

                    compare_exact(
                        saved_value,
                        expected_value,
                        f"{key}/{field}",
                    )

                checked_fields += 1

            if applicable:
                checked_scenarios += 1

    require(
        checked_scenarios == 72,
        "Expected 72 additional Schwartz scenarios",
    )

    require(
        checked_fields == 36 * 33,
        "Expected every additional scenario field to be checked",
    )

    # Explicitly check the applicability counts.
    applicable = result[
        "schwartz_sensitivity_applicable"
    ]

    require(
        int(applicable.sum()) == 24
        and int((~applicable).sum()) == 12,
        "Incorrect Schwartz/non-Schwartz distribution",
    )

    # Verify GB is actually excluded only from GDP samples.
    schwartz_rows = result.loc[applicable]

    gdp = schwartz_rows.loc[
        schwartz_rows["economic_variable"] == "gdp_pps_index"
    ]

    other = schwartz_rows.loc[
        schwartz_rows["economic_variable"] != "gdp_pps_index"
    ]

    require(
        len(gdp) == 4 and len(other) == 20,
        "Incorrect GDP/non-GDP Schwartz coverage",
    )

    require(
        gdp["exclude_GB__removed_n"].eq(1).all(),
        "GB not removed from all GDP pairs",
    )

    require(
        other["exclude_GB__removed_n"].eq(0).all(),
        "GB incorrectly removed from non-GDP pairs",
    )

    require(
        schwartz_rows["exclude_EE__removed_n"].eq(1).all(),
        "EE not removed from all Schwartz pairs",
    )

    # Confirm no strict sign reversals in the exclusion tests.
    for scenario in EXCLUSION_SCENARIOS:

        for method in ("pearson", "spearman"):

            require(
                not schwartz_rows[
                    f"{scenario}__{method}_sign_flip"
                ].any(),
                f"Unexpected sign reversal: {scenario}/{method}",
            )

    print(
        "All 35 added columns and 72 applicable "
        "Schwartz scenarios: PASSED."
    )

    print(
        "Non-Schwartz NA values, GB/EE coverage "
        "and strict sign flags: PASSED."
    )

    return max_difference


# ------------------------------------------------------------
# 10. VALIDATE EVERY CSV CELL AGAINST PARQUET
# ------------------------------------------------------------

def validate_csv(result, expected_columns):

    with RESULT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        header = next(reader, None)

        require(
            header == expected_columns,
            "CSV header or column ordering differs",
        )

        rows = list(reader)

    require(
        len(rows) == 36,
        "Expected exactly 36 CSV rows",
    )

    max_difference = 0.0
    checked_cells = 0

    # Compare the CSV with every validated Parquet row.
    for row_index, csv_row in enumerate(rows):

        require(
            len(csv_row) == 72,
            f"CSV row {row_index + 1}: expected 72 fields",
        )

        for column_index, column in enumerate(expected_columns):

            text_value = csv_row[column_index]

            parquet_value = result.iloc[
                row_index
            ][column]

            dtype = result[column].dtype

            label = (
                f"CSV row {row_index + 1}, column {column}"
            )

            # CSV represents missing values as empty fields.
            if pd.isna(parquet_value):

                require(
                    text_value == "",
                    f"{label}: missing-value representation differs",
                )

            elif pd.api.types.is_bool_dtype(dtype):

                require(
                    text_value in ("True", "False"),
                    f"{label}: invalid Boolean representation",
                )

                compare_exact(
                    text_value == "True",
                    bool(parquet_value),
                    label,
                )

            elif pd.api.types.is_integer_dtype(dtype):

                require(
                    re.fullmatch(r"-?\d+", text_value) is not None,
                    f"{label}: invalid integer representation",
                )

                compare_exact(
                    int(text_value),
                    int(parquet_value),
                    label,
                )

            elif pd.api.types.is_float_dtype(dtype):

                require(
                    text_value != "",
                    f"{label}: missing floating-point value",
                )

                max_difference = max(
                    max_difference,
                    compare_float(
                        float(text_value),
                        parquet_value,
                        label,
                    ),
                )

            else:

                # Compare preserved text and original metadata fields.
                compare_exact(
                    text_value,
                    str(parquet_value),
                    label,
                )

            checked_cells += 1

    require(
        checked_cells == 2592,
        "Not all 2,592 CSV cells were checked",
    )

    print(
        "All 36 CSV rows and 2,592 cells "
        "match Parquet: PASSED."
    )

    return max_difference


# ------------------------------------------------------------
# 11. VERIFY STEP 62 HASH PROVENANCE
# ------------------------------------------------------------

def validate_step62_provenance(metadata):

    sources = metadata.get("sources")
    outputs = metadata.get("outputs")

    require(
        isinstance(sources, list) and len(sources) == 3,
        "Step 62: invalid source manifest",
    )

    require(
        isinstance(outputs, list) and len(outputs) == 2,
        "Step 62: invalid output manifest",
    )

    # Validate the exact source files used in Step 62.
    for entry, path in zip(
        sources,
        [INTEGRATED, BASELINE, SYNTHESIS],
    ):

        require(
            isinstance(entry, dict),
            "Step 62: invalid source record",
        )

        compare_exact(
            entry.get("filename"),
            path.name,
            "Step 62 source filename",
        )

        compare_exact(
            entry.get("sha256"),
            sha256(path),
            "Step 62 source SHA-256",
        )

    # Step 62's Parquet file must match its own manifest.
    parquet_entry = next(
        (
            entry for entry in outputs
            if entry.get("filename") == SCHWARTZ.name
        ),
        None,
    )

    require(
        parquet_entry is not None,
        "Step 62 Parquet missing from its output manifest",
    )

    compare_exact(
        parquet_entry.get("sha256"),
        sha256(SCHWARTZ),
        "Step 62 Parquet SHA-256",
    )


# ------------------------------------------------------------
# 12. VALIDATE STEP 64 METADATA AND HASHES
# ------------------------------------------------------------

def validate_metadata(metadata, result, max_cross_source):

    expected = {
        "step": 64,
        "dataset_name": "Scientific Evidence Matrix V1",
        "analysis_unit": "country",
        "study_design": "cross-sectional exploratory",
        "country_universe_n": 29,
        "analytical_pairs_n": 36,
        "schwartz_pairs_n": 24,
        "non_schwartz_pairs_n": 12,
        "original_synthesis_columns_n": 37,
        "added_columns_n": 35,
        "output_columns_n": 72,
        "output_rows_n": 36,
        "source_synthesis_step": 59,
        "source_sensitivity_step": 62,
        "sensitivity_validation_step": 63,
        "independent_step_64_validation": "pending",
        "baseline_correlations_recalculated": False,
        "loo_correlations_recalculated": False,
        "temporal_correlations_recalculated": False,
        "causal_analysis": False,
        "measurement_invariance_test": False,
        "statistical_significance_testing": False,
        "arbitrary_robustness_score": False,
    }

    for key, value in expected.items():

        compare_exact(
            metadata.get(key),
            value,
            f"Step 64 metadata: {key}",
        )

    check_utc_timestamp(
        metadata.get("created_at_utc")
    )

    compare_float(
        metadata.get("numeric_comparison_tolerance"),
        TOLERANCE,
        "Metadata numerical tolerance",
    )

    compare_float(
        metadata.get("maximum_cross_source_difference"),
        max_cross_source,
        "Metadata cross-source numerical difference",
    )

    # Independently check the three registered exclusions.
    expected_scenarios = {
        name: sorted(SCENARIOS[name])
        for name in EXCLUSION_SCENARIOS
    }

    require(
        metadata.get("schwartz_exclusion_scenarios")
        == expected_scenarios,
        "Metadata scenario definitions differ",
    )

    notes = metadata.get("interpretation_notes")

    require(
        isinstance(notes, list)
        and len(notes) == 11
        and all(isinstance(value, str) for value in notes),
        "Invalid methodological interpretation notes",
    )

    # --------------------------------------------------------
    # 12.1 VALIDATE THE FULL 72-COLUMN SCHEMA MANIFEST
    # --------------------------------------------------------

    schema = metadata.get("schema")

    require(
        isinstance(schema, list) and len(schema) == 72,
        "Metadata must describe 72 columns",
    )

    for index, entry in enumerate(schema):

        require(
            isinstance(entry, dict),
            f"Invalid metadata schema entry {index}",
        )

        column = result.columns[index]

        compare_exact(
            entry.get("name"),
            column,
            f"Metadata schema column {index}",
        )

        compare_exact(
            entry.get("pandas_dtype"),
            str(result[column].dtype),
            f"Metadata dtype {column}",
        )

    # --------------------------------------------------------
    # 12.2 VALIDATE ALL SIX SOURCE CHECKSUMS
    # --------------------------------------------------------

    expected_sources = [
        INTEGRATED,
        BASELINE,
        SYNTHESIS,
        SYNTHESIS_METADATA,
        SCHWARTZ,
        SCHWARTZ_METADATA,
    ]

    source_records = metadata.get("sources")

    require(
        isinstance(source_records, list)
        and len(source_records) == 6,
        "Metadata must contain six source records",
    )

    for entry, path in zip(
        source_records,
        expected_sources,
    ):

        require(
            isinstance(entry, dict),
            "Invalid source provenance entry",
        )

        relative_path = path.relative_to(ROOT).as_posix()

        compare_exact(
            entry.get("path"),
            relative_path,
            "Source relative path",
        )

        compare_exact(
            entry.get("filename"),
            path.name,
            "Source filename",
        )

        compare_exact(
            entry.get("sha256"),
            sha256(path),
            f"Source SHA-256: {path.name}",
        )

    # --------------------------------------------------------
    # 12.3 VALIDATE THE TWO OUTPUT CHECKSUMS
    # --------------------------------------------------------

    expected_outputs = [
        RESULT_PARQUET,
        RESULT_CSV,
    ]

    output_records = metadata.get("outputs")

    require(
        isinstance(output_records, list)
        and len(output_records) == 2,
        "Metadata must contain two output records",
    )

    for entry, path in zip(
        output_records,
        expected_outputs,
    ):

        require(
            isinstance(entry, dict),
            "Invalid output provenance entry",
        )

        compare_exact(
            entry.get("filename"),
            path.name,
            "Output filename",
        )

        compare_exact(
            entry.get("sha256"),
            sha256(path),
            f"Output SHA-256: {path.name}",
        )

    print(
        "Step 64 metadata, 72-column schema "
        "and eight SHA-256 records: PASSED."
    )


# ------------------------------------------------------------
# 13. CHECK THE DOCUMENTED SENSITIVITY TOTALS
# ------------------------------------------------------------

def validate_summary(result):

    applicable = result.loc[
        result["schwartz_sensitivity_applicable"]
    ]

    expected_changes = {
        "exclude_EE": (0.031088, 0.081344),
        "exclude_GB": (0.006686, 0.022082),
        "exclude_EE_GB": (0.033706, 0.081344),
    }

    for scenario, expected in expected_changes.items():

        maximum_pearson = round(
            float(
                applicable[
                    f"{scenario}__pearson_abs_change"
                ].max()
            ),
            6,
        )

        maximum_spearman = round(
            float(
                applicable[
                    f"{scenario}__spearman_abs_change"
                ].max()
            ),
            6,
        )

        require(
            maximum_pearson == expected[0],
            f"{scenario}: maximum Pearson change differs",
        )

        require(
            maximum_spearman == expected[1],
            f"{scenario}: maximum Spearman change differs",
        )

    print(
        "Scenario maxima and reported sensitivity "
        "totals: PASSED."
    )


# ------------------------------------------------------------
# 14. MAIN INDEPENDENT VALIDATION
# ------------------------------------------------------------

def main():

    print("=" * 90)
    print("STEP 65 - INDEPENDENT SCIENTIFIC EVIDENCE MATRIX VALIDATION")
    print("=" * 90)

    # List all inputs and outputs that must remain unchanged.
    paths = [
        INTEGRATED,
        BASELINE,
        SYNTHESIS,
        SYNTHESIS_METADATA,
        SCHWARTZ,
        SCHWARTZ_METADATA,
        RESULT_PARQUET,
        RESULT_CSV,
        RESULT_METADATA,
    ]

    for path in paths:

        require(
            path.is_file(),
            f"Required file not found: {path}",
        )

    # Snapshot all hashes before reading any analytical table.
    hashes_before = {
        path: sha256(path)
        for path in paths
    }

    require(
        hashes_before[INTEGRATED] == EXPECTED_INTEGRATED_HASH,
        "Integrated dataset SHA-256 does not match V1",
    )

    (
        integrated,
        baseline,
        synthesis,
        sensitivity,
        result,
        sensitivity_metadata,
        result_metadata,
    ) = load_sources()

    # Independently verify input consistency.
    validate_sources(
        integrated,
        baseline,
        synthesis,
        sensitivity,
    )

    # Verify all original columns and expected new fields.
    columns = validate_output_schema(
        synthesis,
        result,
    )

    # Verify all 36 country samples and baseline coefficients.
    baseline_difference = validate_baseline_links(
        integrated,
        baseline,
        synthesis,
    )

    # Independently validate the 96 scenario source records.
    scenario_difference = validate_scenario_sources(
        synthesis,
        sensitivity,
    )

    # Validate the actual integration into Step 64.
    integration_difference = validate_added_columns(
        result,
        sensitivity,
    )

    # Verify that CSV preserves all 2,592 table cells.
    csv_difference = validate_csv(
        result,
        columns,
    )

    # Check historical provenance and the Step 64 manifest.
    validate_step62_provenance(
        sensitivity_metadata
    )

    validate_metadata(
        result_metadata,
        result,
        scenario_difference,
    )

    validate_summary(result)

    # Verify exact file integrity after all operations.
    hashes_after = {
        path: sha256(path)
        for path in paths
    }

    require(
        hashes_before == hashes_after,
        "A source or output file changed during validation",
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("SUMMARY")
    print("-" * 90)

    print("Validated analytical relationships:", len(result))
    print("Validated matrix columns:", len(result.columns))
    print("Validated matrix cells:", result.size)

    print("Original Step 59 columns preserved:", 37)
    print("New Step 64 columns validated:", 35)
    print("Applicable additional Schwartz scenarios:", 72)
    print("Step 62 scenario records checked:", len(sensitivity))

    print(
        "Maximum baseline link difference:",
        f"{baseline_difference:.3e}",
    )

    print(
        "Maximum scenario source difference:",
        f"{scenario_difference:.3e}",
    )

    print(
        "Maximum integration difference:",
        f"{integration_difference:.3e}",
    )

    print(
        "Maximum CSV/Parquet difference:",
        f"{csv_difference:.3e}",
    )

    print()
    print("Source and output file bytes unchanged: PASSED.")

    print()
    print("=" * 90)
    print("INDEPENDENT SCIENTIFIC EVIDENCE MATRIX VALIDATION PASSED.")
    print("=" * 90)

    print("No project files were created or modified.")


if __name__ == "__main__":
    main()
