
from pathlib import Path
from math import isfinite, prod

import pandas as pd
import requests


# ============================================================
# STEP 33 — EUROSTAT COVERAGE DIAGNOSTIC
# ============================================================

# Purpose:
# Check the availability of six socioeconomic indicators
# for the 29 ESS countries included in our V1 project.
#
# We will check:
# 1. Exact Eurostat statistical series.
# 2. Country-code compatibility.
# 3. The main economic context year.
# 4. Alternative years for multi-year ESS fieldwork.
# 5. Missing observations and Eurostat status flags.
# 6. Alternative reference-year choices for Gini.
#
# This script only performs API requests and diagnostics.
# It does not save or modify any files.


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# __file__ is the location of this Python script.
# parents[1] identifies our project root folder.

ROOT = Path(__file__).resolve().parents[1]


# Step 30 created this configuration.
# Step 31 independently validated its contents.
#
# It contains one row per ESS country and identifies
# the main and alternative economic context years.

TARGETS_PATH = (
    ROOT / "config/temporal_alignment_targets_v1.csv"
)


# The official Eurostat Statistics API endpoint.
# Each statistical dataset is accessed using its code.

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
)


# ============================================================
# 2. COUNTRY CODE MAPPING
# ============================================================

# Some country codes differ between ESS and Eurostat.
#
# ESS uses GR for Greece and GB for United Kingdom.
# Eurostat commonly uses EL and UK respectively.
#
# Other country codes remain unchanged.
#
# This mapping changes only API lookup codes.
# It does not modify ESS country identifiers.

COUNTRY_CODE_MAP = {
    "GR": "EL",
    "GB": "UK",
}


# ============================================================
# 3. DEFINE SIX EXACT EUROSTAT INDICATORS
# ============================================================

# A Eurostat dataset can contain multiple measurements.
#
# For example, an employment dataset can contain:
# - Numbers of employed people.
# - Employment rates.
# - Different age groups.
# - Male and female observations.
#
# We must select an exact combination of dimensions
# rather than simply taking the first available value.


INDICATORS = {

    # GDP per capita in PPS, expressed as an index.
    #
    # EU27_2020 = 100.
    # This is not an amount of money in euros.

    "gdp_pps_index": {
        "dataset": "tec00114",
        "filters": {
            "freq": "A",
            "indic_ppp": "VI_PPS_EU27_2020_HAB",
            "ppp_cat18": "GDP",
        },
        "last_year": 2024,
    },

    # Employment rate for people aged 20-64.
    #
    # PC_POP means percentage of the population
    # in the selected age group.
    #
    # EMP_LFS identifies employment measured using
    # the Labour Force Survey concept.

    "employment_rate": {
        "dataset": "lfsi_emp_a",
        "filters": {
            "freq": "A",
            "indic_em": "EMP_LFS",
            "sex": "T",
            "age": "Y20-64",
            "unit": "PC_POP",
        },
        "last_year": 2024,
    },

    # Unemployment rate for people aged 15-74.
    #
    # PC_ACT is important:
    # Unemployment is expressed as a percentage
    # of the labour force, not total population.

    "unemployment_rate": {
        "dataset": "une_rt_a",
        "filters": {
            "freq": "A",
            "age": "Y15-74",
            "unit": "PC_ACT",
            "sex": "T",
        },
        "last_year": 2024,
    },

    # Gini coefficient of equivalised disposable income.
    #
    # The coefficient uses a scale from 0 to 100.
    #
    # For EU-SILC income, the reference year is
    # normally the calendar year before the survey.
    #
    # We request 2023-2025 to investigate both
    # same-year and shifted-year alignment.

    "gini": {
        "dataset": "ilc_di12",
        "filters": {
            "freq": "A",
            "age": "TOTAL",
            "statinfo": "GINI_HND",
        },
        "last_year": 2025,
    },

    # Severe material and social deprivation rate.
    #
    # TOTAL = all ages.
    # T = both sexes.
    # PC = percentage.
    #
    # This measures the share of people experiencing
    # severe material and social deprivation.

    "material_deprivation_rate": {
        "dataset": "ilc_mdsd11",
        "filters": {
            "freq": "A",
            "age": "TOTAL",
            "sex": "T",
            "unit": "PC",
        },
        "last_year": 2024,
    },

    # Annual harmonised consumer price inflation.
    #
    # RCH_A_AVG = annual average rate of change.
    # TOTAL = all consumer expenditure categories.
    #
    # The annual rate describes price changes,
    # not the absolute consumer price level.

    "inflation_rate": {
        "dataset": "tec00118",
        "filters": {
            "freq": "A",
            "unit": "RCH_A_AVG",
            "coicop18": "TOTAL",
        },
        "last_year": 2024,
    },
}


# ============================================================
# 4. READ JSON-STAT CATEGORY CODES
# ============================================================

def get_categories(dimension):

    # Eurostat returns multidimensional statistical data.
    #
    # Each dimension contains category codes such as:
    # geo  -> DE, FR, EL, etc.
    # time -> 2023, 2024, etc.
    #
    # The category index defines the order of those codes.
    #
    # We need the correct order to find a specific value
    # inside the multidimensional JSON-stat structure.

    index = dimension["category"]["index"]

    if isinstance(index, dict):

        # A dictionary maps category codes to positions.
        # Sorting by position restores the correct order.

        ordered = sorted(
            index.items(),
            key=lambda item: item[1],
        )

        return [
            code
            for code, position in ordered
        ]

    if isinstance(index, list):

        # A list already contains the category order.

        return index

    raise ValueError(
        "Unsupported JSON-stat category index."
    )


# ============================================================
# 5. READ AN INDIVIDUAL JSON-STAT CELL
# ============================================================

def get_cell(container, position):

    # JSON-stat may store observations as:
    #
    # 1. A list of values.
    # 2. A sparse dictionary using cell positions.
    #
    # A sparse dictionary may omit missing observations.
    #
    # Therefore, a missing cell must not become zero.

    if container is None:
        return None

    if isinstance(container, dict):

        # JSON object keys are strings.
        return container.get(str(position))

    if isinstance(container, list):

        if position < len(container):
            return container[position]

        return None

    raise ValueError(
        "Unsupported JSON-stat cell container."
    )


# ============================================================
# 6. REQUEST ONE EXACT EUROSTAT SERIES
# ============================================================

def fetch_indicator(indicator_name, specification):

    dataset_code = specification["dataset"]

    # Specify the exact categories validated in
    # Steps 32 and 32B.
    #
    # geoLevel=country excludes EU and regional aggregates.
    #
    # We request all available countries because our
    # diagnostic needs to discover geographical coverage.

    params = {
        "lang": "EN",
        "geoLevel": "country",
        "sinceTimePeriod": "2023",
        "untilTimePeriod": str(
            specification["last_year"]
        ),
        **specification["filters"],
    }

    url = BASE_URL + dataset_code

    print("\n" + "=" * 100)
    print(f"INDICATOR: {indicator_name}")
    print(f"DATASET:   {dataset_code}")
    print("=" * 100)


    # --------------------------------------------------------
    # 6.1 REQUEST DATA FROM EUROSTAT
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            params=params,
            timeout=(10, 90),
        )

        # An HTTP error indicates a failed request.
        # It must not be classified as missing data.

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as error:

        raise RuntimeError(
            f"Eurostat request failed for "
            f"{dataset_code}: {error}"
        ) from error

    except ValueError as error:

        raise ValueError(
            f"{dataset_code}: Invalid JSON response."
        ) from error


    print(f"HTTP status: {response.status_code}")
    print(f"Request URL: {response.url}")
    print(f"Last updated: {data.get('updated')}")


    # --------------------------------------------------------
    # 6.2 VALIDATE JSON-STAT STRUCTURE
    # --------------------------------------------------------

    # We require a proper JSON-stat statistical dataset.
    # The dimension IDs and sizes must be consistent.

    if data.get("class") != "dataset":
        raise ValueError(
            f"{dataset_code}: Invalid dataset class."
        )

    ids = data.get("id")
    sizes = data.get("size")
    dimensions = data.get("dimension")

    if not isinstance(ids, list):
        raise ValueError(
            f"{dataset_code}: Missing dimension IDs."
        )

    if not isinstance(sizes, list):
        raise ValueError(
            f"{dataset_code}: Missing dimension sizes."
        )

    if not isinstance(dimensions, dict):
        raise ValueError(
            f"{dataset_code}: Missing dimension metadata."
        )

    if len(ids) != len(sizes):
        raise ValueError(
            f"{dataset_code}: Dimension length mismatch."
        )

    if len(set(ids)) != len(ids):
        raise ValueError(
            f"{dataset_code}: Duplicate dimensions."
        )


    # --------------------------------------------------------
    # 6.3 VALIDATE FILTERED DIMENSIONS
    # --------------------------------------------------------

    # Read the actual categories returned by the API.
    #
    # The response must contain only our selected
    # non-geographic dimensions, plus geo and time.
    #
    # This prevents mixing different measurements.

    categories = {}

    for dimension_id, expected_size in zip(
        ids,
        sizes,
    ):

        if dimension_id not in dimensions:
            raise ValueError(
                f"{dataset_code}: Missing {dimension_id}."
            )

        codes = get_categories(
            dimensions[dimension_id]
        )

        if len(codes) != expected_size:
            raise ValueError(
                f"{dataset_code}: Category size mismatch "
                f"for {dimension_id}."
            )

        categories[dimension_id] = codes


    allowed_dimensions = (
        set(specification["filters"])
        | {"geo", "time"}
    )

    if set(ids) != allowed_dimensions:
        raise ValueError(
            f"{dataset_code}: Unexpected dimensions: "
            f"{set(ids) - allowed_dimensions}"
        )


    # Each requested statistical dimension should
    # contain exactly the category we specified.

    for dimension_id, expected_code in (
        specification["filters"].items()
    ):

        actual_codes = categories[dimension_id]

        if actual_codes != [expected_code]:
            raise ValueError(
                f"{dataset_code}: Unexpected category "
                f"for {dimension_id}: {actual_codes}"
            )


    # --------------------------------------------------------
    # 6.4 CALCULATE JSON-STAT CELL POSITIONS
    # --------------------------------------------------------

    # JSON-stat represents a multidimensional table
    # as one flattened sequence of values.
    #
    # Example dimensions:
    # [freq, unit, geo, time]
    #
    # The last dimension changes fastest.
    #
    # A stride tells us how many positions to move
    # when one dimension increases by one category.

    positions = {
        dimension_id: {
            code: index
            for index, code in enumerate(codes)
        }
        for dimension_id, codes in categories.items()
    }

    strides = {}

    for index, dimension_id in enumerate(ids):

        # Multiply the sizes of all following dimensions.
        #
        # prod([3, 2]) = 6, for example.

        strides[dimension_id] = prod(
            sizes[index + 1:]
        )


    # --------------------------------------------------------
    # 6.5 EXTRACT COUNTRY-YEAR OBSERVATIONS
    # --------------------------------------------------------

    observations = {}

    for geo_code in categories["geo"]:

        for year_code in categories["time"]:

            cell_positions = {}

            for dimension_id in ids:

                if dimension_id == "geo":
                    category_code = geo_code

                elif dimension_id == "time":
                    category_code = year_code

                else:
                    category_code = (
                        specification["filters"][dimension_id]
                    )

                cell_positions[dimension_id] = (
                    positions[dimension_id][category_code]
                )

            # Calculate the flat position of this cell.

            flat_position = sum(
                cell_positions[dimension_id]
                * strides[dimension_id]
                for dimension_id in ids
            )

            # Read both the value and its Eurostat status.
            #
            # Status codes can indicate characteristics
            # such as provisional or estimated data.
            #
            # We preserve status information because
            # a numeric observation is not necessarily
            # free of methodological qualifications.

            value = get_cell(
                data.get("value"),
                flat_position,
            )

            status = get_cell(
                data.get("status"),
                flat_position,
            )

            observations[
                (geo_code, int(year_code))
            ] = {
                "value": value,
                "status": status,
            }


    print(f"Countries returned: {len(categories['geo'])}")
    print(f"Years returned:     {categories['time']}")

    # Return a structured snapshot that can be reused
    # for every ESS country without additional API calls.

    return {
        "dataset": dataset_code,
        "countries": set(categories["geo"]),
        "years": {
            int(year)
            for year in categories["time"]
        },
        "observations": observations,
    }


# ============================================================
# 7. CLASSIFY DATA AVAILABILITY
# ============================================================

def assess_observation(snapshot, country_code, year):

    # We distinguish several possible situations.
    #
    # NO_GEO:
    # Country is absent from the API response.
    #
    # NO_YEAR:
    # Requested year is absent.
    #
    # MISSING_VALUE:
    # Country and year exist, but no numeric value exists.
    #
    # OK:
    # A valid numeric observation is available.
    #
    # OK[flag]:
    # Numeric observation exists with a Eurostat flag.
    #
    # These distinctions help us diagnose missing data
    # without arbitrarily filling or deleting anything.

    if country_code not in snapshot["countries"]:
        return "NO_GEO"

    if year not in snapshot["years"]:
        return "NO_YEAR"

    observation = snapshot["observations"].get(
        (country_code, year)
    )

    if observation is None:
        return "MISSING_VALUE"

    value = observation["value"]
    status = observation["status"]

    if value is None:
        return (
            f"MISSING_VALUE[{status}]"
            if status
            else "MISSING_VALUE"
        )

    # Numeric values must be finite.
    # Zero is a valid observation and is not missing.

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(
            f"Unexpected numeric value for "
            f"{country_code}, {year}: {value}"
        )

    if status:
        return f"OK[{status}]"

    return "OK"


# ============================================================
# 8. DEFINE TEMPORAL ALIGNMENT RULES
# ============================================================

def get_requested_year(indicator, context_year):

    # For five indicators, the initial diagnostic
    # uses the same year as the ESS economic context.
    #
    # For Gini, we additionally test a shifted year.
    #
    # Example:
    # Economic income year = 2023
    # Candidate EU-SILC survey year = 2024
    #
    # This is provisional until the relevant country
    # and indicator metadata are fully reviewed.

    if indicator == "gini":
        return context_year + 1

    return context_year


# ============================================================
# 9. LOAD THE VALIDATED ESS COUNTRY CONFIGURATION
# ============================================================

def load_targets():

    if not TARGETS_PATH.exists():
        raise FileNotFoundError(
            f"Missing configuration: {TARGETS_PATH}"
        )

    targets = pd.read_csv(TARGETS_PATH)

    # The ESS V1 scope contains 29 countries.
    # All country identifiers must be unique.

    if len(targets) != 29:
        raise ValueError(
            "Expected exactly 29 ESS countries."
        )

    required_columns = [
        "cntry",
        "target_context_year",
        "alternative_context_year",
        "temporal_sensitivity_priority_flag",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in targets.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing target columns: {missing_columns}"
        )

    if (
        targets["cntry"].isna().any()
        or targets["cntry"].duplicated().any()
    ):
        raise ValueError(
            "Invalid ESS country identifiers."
        )

    # The Step 31 validation already established
    # the expected target-year distribution.
    #
    # Here we perform a simple defensive check.

    if not targets[
        "target_context_year"
    ].isin([2023, 2024]).all():
        raise ValueError(
            "Unexpected main context year."
        )

    return targets


# ============================================================
# 10. PERFORM SIX-INDICATOR COVERAGE DIAGNOSTIC
# ============================================================

def main():

    print("=" * 110)
    print("V1 EUROSTAT SIX-INDICATOR COVERAGE DIAGNOSTIC")
    print("=" * 110)

    targets = load_targets()

    print(f"\nESS V1 countries: {len(targets)}")
    print(f"Indicators to inspect: {len(INDICATORS)}")


    # --------------------------------------------------------
    # 10.1 FETCH EACH INDICATOR ONCE
    # --------------------------------------------------------

    # One filtered API request per indicator is enough.
    #
    # We do not need 29 separate requests per dataset.
    # This reduces API traffic and makes the process
    # faster and easier to reproduce.

    snapshots = {}

    for indicator_name, specification in (
        INDICATORS.items()
    ):

        snapshots[indicator_name] = fetch_indicator(
            indicator_name,
            specification,
        )


    # --------------------------------------------------------
    # 10.2 CHECK MAIN AND ALTERNATIVE YEARS
    # --------------------------------------------------------

    primary_rows = []
    alternative_rows = []
    gini_year_comparison = []

    for _, country in targets.iterrows():

        ess_code = country["cntry"]

        # Translate ESS country codes only when needed.

        eurostat_code = COUNTRY_CODE_MAP.get(
            ess_code,
            ess_code,
        )

        context_year = int(
            country["target_context_year"]
        )

        alternative_year = (
            country["alternative_context_year"]
        )

        primary_row = {
            "cntry": ess_code,
            "eurostat_geo": eurostat_code,
            "context_year": context_year,
        }


        # ----------------------------------------------------
        # 10.3 CHECK ALL SIX PRIMARY INDICATORS
        # ----------------------------------------------------

        for indicator_name, snapshot in snapshots.items():

            # Gini uses the shifted year provisionally.
            # All other indicators use the context year.

            requested_year = get_requested_year(
                indicator_name,
                context_year,
            )

            primary_row[indicator_name] = (
                assess_observation(
                    snapshot,
                    eurostat_code,
                    requested_year,
                )
            )


            # ------------------------------------------------
            # 10.4 CHECK THE ALTERNATIVE FIELDWORK YEAR
            # ------------------------------------------------

            # Only the 13 countries with fieldwork
            # in two years have an alternative year.
            #
            # This will later support sensitivity analysis.

            if pd.notna(alternative_year):

                alt_context = int(
                    alternative_year
                )

                alt_requested = get_requested_year(
                    indicator_name,
                    alt_context,
                )

                alternative_rows.append(
                    {
                        "cntry": ess_code,
                        "indicator": indicator_name,
                        "requested_year": alt_requested,
                        "availability": assess_observation(
                            snapshot,
                            eurostat_code,
                            alt_requested,
                        ),
                    }
                )


        # ----------------------------------------------------
        # 10.5 COMPARE TWO POSSIBLE GINI YEAR LABELS
        # ----------------------------------------------------

        # Do not assume that the shifted-year mapping
        # is automatically correct for every analysis.
        #
        # Display availability for:
        # 1. Gini with the same Eurostat year label.
        # 2. Gini with the following survey year label.
        #
        # We will select the final rule only after
        # inspecting the relevant reference periods.

        gini_snapshot = snapshots["gini"]

        gini_year_comparison.append(
            {
                "cntry": ess_code,
                "context_year": context_year,
                "same_year_status": assess_observation(
                    gini_snapshot,
                    eurostat_code,
                    context_year,
                ),
                "shifted_year_status": assess_observation(
                    gini_snapshot,
                    eurostat_code,
                    context_year + 1,
                ),
            }
        )

        primary_rows.append(primary_row)


    # ========================================================
    # 11. PRINT MAIN COUNTRY-LEVEL COVERAGE
    # ========================================================

    primary = pd.DataFrame(primary_rows)
    alternatives = pd.DataFrame(alternative_rows)
    gini_compare = pd.DataFrame(gini_year_comparison)

    primary = primary.sort_values("cntry")

    print("\n" + "=" * 110)
    print("PRIMARY TARGET-YEAR COVERAGE")
    print("=" * 110)

    print(primary.to_string(index=False))


    # --------------------------------------------------------
    # 11.1 SUMMARIZE MAIN-YEAR AVAILABILITY
    # --------------------------------------------------------

    print("\nPRIMARY COVERAGE SUMMARY")
    print("-" * 110)

    for indicator_name in INDICATORS:

        # Both OK and OK[flag] represent observations
        # with available numeric values.
        #
        # Flags are preserved for a later quality review.

        statuses = primary[
            indicator_name
        ].astype(str)

        available_mask = statuses.str.startswith("OK")

        available = int(available_mask.sum())
        missing = len(primary) - available

        print(
            f"{indicator_name:<28} "
            f"Available: {available:>2}/29 | "
            f"Missing: {missing:>2}"
        )

        # Display missing country codes explicitly.
        # Missing values must not be silently discarded.

        if missing > 0:

            missing_countries = primary.loc[
                ~available_mask,
                "cntry",
            ].tolist()

            print(
                "  Missing countries: "
                + ", ".join(missing_countries)
            )


    # ========================================================
    # 12. PRINT ALTERNATIVE-YEAR COVERAGE
    # ========================================================

    print("\n" + "=" * 110)
    print("ALTERNATIVE-YEAR COVERAGE")
    print("=" * 110)

    # This table contains observations for countries
    # whose fieldwork spans both 2023 and 2024.
    #
    # Every indicator should be checked for the
    # alternative economic context year too.

    for indicator_name in INDICATORS:

        group = alternatives.loc[
            alternatives["indicator"] == indicator_name
        ]

        available = int(
            group["availability"]
            .astype(str)
            .str.startswith("OK")
            .sum()
        )

        print(
            f"{indicator_name:<28} "
            f"Available: {available:>2}/{len(group)}"
        )

        missing_group = group.loc[
            ~group["availability"]
            .astype(str)
            .str.startswith("OK")
        ]

        if not missing_group.empty:

            print(
                "  Missing alternatives: "
                + ", ".join(
                    missing_group["cntry"].tolist()
                )
            )


    # ========================================================
    # 13. PRINT GINI REFERENCE-YEAR COMPARISON
    # ========================================================

    print("\n" + "=" * 110)
    print("GINI REFERENCE-YEAR COMPARISON")
    print("=" * 110)

    print(gini_compare.to_string(index=False))

    print()
    print(
        "Same-year and shifted-year availability "
        "are diagnostic alternatives."
    )

    print(
        "Final Gini alignment requires a reference-period "
        "methodology review."
    )


    # ========================================================
    # 14. SUMMARIZE OBSERVATION STATUS FLAGS
    # ========================================================

    print("\n" + "=" * 110)
    print("FLAGGED PRIMARY OBSERVATIONS")
    print("=" * 110)

    flagged_n = 0

    for indicator_name in INDICATORS:

        # A status such as OK[p] means a numeric value
        # exists but carries an additional Eurostat flag.

        flagged_rows = primary.loc[
            primary[indicator_name]
            .astype(str)
            .str.startswith("OK[")
        ]

        for _, row in flagged_rows.iterrows():

            flagged_n += 1

            print(
                f"{row['cntry']:<5} "
                f"{indicator_name:<28} "
                f"{row[indicator_name]}"
            )

    if flagged_n == 0:
        print("No flags detected in primary observations.")


    # ========================================================
    # 15. PRINT FINAL DIAGNOSTIC SUMMARY
    # ========================================================

    print("\n" + "=" * 110)
    print("SIX-INDICATOR COVERAGE DIAGNOSTIC COMPLETED")
    print("=" * 110)

    print(f"ESS countries examined: {len(primary)}")
    print(f"Indicators examined:   {len(INDICATORS)}")
    print(f"Flagged primary cells: {flagged_n}")

    print()
    print("No ESS files were modified.")
    print("No Eurostat RAW files were created.")
    print("No missing values were imputed.")
    print("Gini alignment remains provisional.")


# ============================================================
# 16. EXECUTE THE SCRIPT
# ============================================================

# This standard Python condition prevents main()
# from running automatically when this file is imported.
#
# It runs the workflow only when executing the script
# directly from the terminal.

if __name__ == "__main__":
    main()
