
from pathlib import Path
import json

import pandas as pd


# ============================================================
# STEP 34 — CREATE EUROSTAT INDICATOR REGISTRY
# ============================================================

# Purpose:
# Store the six selected Eurostat indicator definitions
# in one reproducible configuration file.
#
# Each record specifies:
# - The exact Eurostat dataset code.
# - The dimension filters identifying the statistical series.
# - The measurement unit and interpretation.
# - The requested years for API ingestion.
# - The temporal alignment rule with ESS interviews.
# - The official methodological documentation.
#
# Why this matters:
# A configuration-driven pipeline separates statistical
# decisions from Python data-processing logic.
#
# We will use this registry in later ingestion scripts.
# This step does NOT download Eurostat observations.


# ============================================================
# 1. DEFINE OUTPUT PATH
# ============================================================

# Locate the project root from the scripts folder.

ROOT = Path(__file__).resolve().parents[1]

# Store indicator definitions in the config folder.
# The file can later be tracked and reviewed in Git.

OUTPUT_PATH = (
    ROOT / "config/eurostat_indicators_v1.csv"
)


# ============================================================
# 2. DEFINE THE SIX INDICATORS
# ============================================================

# Filters must match the API dimension codes that
# we verified empirically in Steps 32 and 32B.
#
# reference_year_offset:
# 0 -> Eurostat year equals target context year.
# 1 -> Eurostat year equals target context year + 1.
#
# The offset describes statistical reference periods,
# not the publication date of the dataset.

INDICATORS = [

    {
        "indicator_id": "gdp_pps_index",
        "dataset_code": "tec00114",
        "label": "GDP per capita in PPS index",
        "unit_label": "EU27_2020=100",
        "filters": {
            "freq": "A",
            "indic_ppp": "VI_PPS_EU27_2020_HAB",
            "ppp_cat18": "GDP",
        },
        "api_start_year": 2023,
        "api_end_year": 2024,
        "reference_year_offset": 0,
        "reference_period_rule": "annual_context_year",
        "documentation_url": (
            "https://ec.europa.eu/eurostat/"
            "databrowser/view/tec00114/default/table?lang=en"
        ),
    },

    {
        "indicator_id": "employment_rate",
        "dataset_code": "lfsi_emp_a",
        "label": "Employment rate, ages 20-64",
        "unit_label": "percent_population_20_64",
        "filters": {
            "freq": "A",
            "indic_em": "EMP_LFS",
            "sex": "T",
            "age": "Y20-64",
            "unit": "PC_POP",
        },
        "api_start_year": 2023,
        "api_end_year": 2024,
        "reference_year_offset": 0,
        "reference_period_rule": "annual_context_year",
        "documentation_url": (
            "https://ec.europa.eu/eurostat/"
            "cache/metadata/en/employ_esms.htm"
        ),
    },

    {
        "indicator_id": "unemployment_rate",
        "dataset_code": "une_rt_a",
        "label": "Unemployment rate, ages 15-74",
        "unit_label": "percent_labour_force_15_74",
        "filters": {
            "freq": "A",
            "age": "Y15-74",
            "unit": "PC_ACT",
            "sex": "T",
        },
        "api_start_year": 2023,
        "api_end_year": 2024,
        "reference_year_offset": 0,
        "reference_period_rule": "annual_context_year",
        "documentation_url": (
            "https://ec.europa.eu/eurostat/"
            "cache/metadata/en/employ_esms.htm"
        ),
    },

    {
        "indicator_id": "gini",
        "dataset_code": "ilc_di12",
        "label": "Gini coefficient of disposable income",
        "unit_label": "index_0_100",
        "filters": {
            "freq": "A",
            "age": "TOTAL",
            "statinfo": "GINI_HND",
        },

        # Include 2023 to preserve the same-year
        # series for later temporal sensitivity checks.
        #
        # The primary 2023 income context instead
        # corresponds to EU-SILC survey year 2024.

        "api_start_year": 2023,
        "api_end_year": 2025,
        "reference_year_offset": 1,
        "reference_period_rule": (
            "silc_previous_calendar_income_year"
        ),
        "documentation_url": (
            "https://ec.europa.eu/eurostat/"
            "cache/metadata/en/ilc_sieusilc.htm"
        ),
    },

    {
        "indicator_id": "material_deprivation_rate",
        "dataset_code": "ilc_mdsd11",
        "label": (
            "Severe material and social deprivation rate"
        ),
        "unit_label": "percent_population",
        "filters": {
            "freq": "A",
            "age": "TOTAL",
            "sex": "T",
            "unit": "PC",
        },
        "api_start_year": 2023,
        "api_end_year": 2024,
        "reference_year_offset": 0,
        "reference_period_rule": (
            "silc_current_conditions_survey_year"
        ),
        "documentation_url": (
            "https://ec.europa.eu/eurostat/"
            "cache/metadata/en/ilc_sieusilc.htm"
        ),
    },

    {
        "indicator_id": "inflation_rate",
        "dataset_code": "tec00118",
        "label": "Annual average HICP inflation rate",
        "unit_label": "annual_percentage_change",
        "filters": {
            "freq": "A",
            "unit": "RCH_A_AVG",
            "coicop18": "TOTAL",
        },
        "api_start_year": 2023,
        "api_end_year": 2024,
        "reference_year_offset": 0,
        "reference_period_rule": "annual_context_year",
        "documentation_url": (
            "https://ec.europa.eu/eurostat/"
            "databrowser/view/tec00118/default/table?lang=en"
        ),
    },
]


# ============================================================
# 3. VALIDATE THE REGISTRY DEFINITIONS
# ============================================================

def validate_indicators(indicators):

    # V1 currently contains six approved candidates.
    # An unexpected number may indicate that a
    # statistical definition was accidentally removed.

    if len(indicators) != 6:
        raise ValueError(
            "Expected exactly six indicator definitions."
        )

    ids = [
        item["indicator_id"]
        for item in indicators
    ]

    datasets = [
        item["dataset_code"]
        for item in indicators
    ]

    # Duplicate IDs would make downstream joins ambiguous.

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate indicator IDs.")

    # Each current V1 indicator uses one unique dataset.

    if len(datasets) != len(set(datasets)):
        raise ValueError("Duplicate dataset codes.")

    for item in indicators:

        filters = item["filters"]

        # Every requested statistical series must
        # explicitly use annual frequency.

        if filters.get("freq") != "A":
            raise ValueError(
                f"{item['indicator_id']}: "
                "Expected annual frequency."
            )

        # The API must contain all years required
        # by the main 2023-2024 ESS context targets.

        offset = item["reference_year_offset"]

        required_first_year = 2023 + offset
        required_last_year = 2024 + offset

        if (
            item["api_start_year"] > required_first_year
            or item["api_end_year"] < required_last_year
        ):
            raise ValueError(
                f"{item['indicator_id']}: "
                "API year window is insufficient."
            )

        # Require nonempty documentation and units.
        # These fields support future methodology reports.

        if not item["unit_label"]:
            raise ValueError("Missing measurement unit.")

        if not item["documentation_url"].startswith(
            "https://"
        ):
            raise ValueError(
                "Invalid documentation URL."
            )


# ============================================================
# 4. CONVERT DEFINITIONS TO A TABULAR CONFIGURATION
# ============================================================

def build_registry(indicators):

    records = []

    for item in indicators:

        record = item.copy()

        # CSV cells cannot directly store Python dictionaries.
        #
        # JSON encoding preserves the exact API filters
        # in a portable machine-readable representation.
        #
        # Later scripts can recover the dictionary using:
        # json.loads(record["filters_json"])

        record["filters_json"] = json.dumps(
            record.pop("filters"),
            sort_keys=True,
        )

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 5. SAVE THE REGISTRY SAFELY
# ============================================================

def main():

    print("=" * 100)
    print("V1 EUROSTAT INDICATOR REGISTRY")
    print("=" * 100)

    # Validate before writing anything to disk.
    validate_indicators(INDICATORS)

    registry = build_registry(INDICATORS)

    # Ensure that the configuration directory exists.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build the complete CSV representation in memory.
    #
    # A fixed newline character helps produce
    # reproducible file content across systems.

    csv_content = registry.to_csv(
        index=False,
        lineterminator="\n",
    )

    # Protect existing configuration decisions.
    #
    # If this script is rerun and the file is identical,
    # no changes are necessary.
    #
    # If definitions have changed, stop instead of
    # silently overwriting a reviewed configuration.

    if OUTPUT_PATH.exists():

        existing = OUTPUT_PATH.read_text(
            encoding="utf-8-sig",
        )

        if existing != csv_content:
            raise ValueError(
                "Existing registry differs from the new "
                "definitions. Review changes before replacing."
            )

        print("\nExisting registry is identical.")

    else:

        OUTPUT_PATH.write_text(
            csv_content,
            encoding="utf-8",
        )

        print("\nNew registry created.")


    # --------------------------------------------------------
    # 5.1 DISPLAY INDICATORS AND REFERENCE-YEAR RULES
    # --------------------------------------------------------

    display_columns = [
        "indicator_id",
        "dataset_code",
        "api_start_year",
        "api_end_year",
        "reference_year_offset",
        "reference_period_rule",
    ]

    print("\nREGISTERED INDICATORS")
    print("-" * 100)

    print(
        registry[display_columns].to_string(
            index=False,
        )
    )


    # --------------------------------------------------------
    # 5.2 PRINT FINAL CONFIRMATION
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("REGISTRY VALIDATION PASSED")
    print("=" * 100)

    print(f"Indicators: {len(registry)}")
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("No Eurostat API requests were made.")
    print("No ESS or Eurostat RAW files were modified.")
    print("Missing observations and flags remain unchanged.")


# ============================================================
# 6. RUN THE SCRIPT
# ============================================================

# Execute the workflow only when this file
# is launched directly from the terminal.

if __name__ == "__main__":
    main()
