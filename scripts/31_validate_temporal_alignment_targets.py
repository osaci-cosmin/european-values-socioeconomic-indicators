
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 31 — VALIDATE TEMPORAL ALIGNMENT TARGETS
# ============================================================

# Purpose:
# Validate the temporal alignment configuration created
# in Step 30 against the original fieldwork metadata.
#
# We want to ensure that:
# 1. All 29 ESS countries are preserved.
# 2. The selected year is supported by the interview data.
# 3. Weighted year shares are mathematically consistent.
# 4. Alternative years are assigned correctly.
# 5. Countries requiring sensitivity analysis are flagged.
#
# This script performs validation only.
# It does not modify or create any files.


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

# __file__ represents the current Python script.
# parents[1] moves from the scripts folder to the project root.

ROOT = Path(__file__).resolve().parents[1]


# This file contains the fieldwork metadata calculated
# from the original ESS interview timestamps in Step 29.

METADATA_PATH = (
    ROOT
    / "data/processed/ESS11_country_fieldwork_metadata.parquet"
)


# This CSV contains the target economic years selected
# in Step 30 for subsequent ESS-Eurostat integration.

TARGETS_PATH = (
    ROOT
    / "config/temporal_alignment_targets_v1.csv"
)


# ============================================================
# 2. DEFINE VALIDATION HELPER FUNCTIONS
# ============================================================

def check(condition, message):

    # A validation condition must evaluate to True.
    #
    # If it evaluates to False, stop execution immediately.
    # This prevents incorrect metadata from being accepted.

    if not bool(condition):
        raise ValueError(message)


def as_flag(value):

    # Flags can have different representations:
    #
    # Python Boolean: True / False
    # CSV text:       "True" / "False"
    #
    # We convert both representations to actual Booleans.

    if pd.isna(value):
        raise ValueError("Missing boolean flag.")

    if isinstance(value, str):

        normalized = value.strip().lower()

        check(
            normalized in {"true", "false"},
            f"Invalid boolean value: {value}",
        )

        return normalized == "true"

    # Accept actual Boolean values from pandas or NumPy.
    check(
        isinstance(value, (bool, np.bool_)),
        f"Unexpected boolean flag type: {type(value)}",
    )

    return bool(value)


# ============================================================
# 3. MAIN VALIDATION FUNCTION
# ============================================================

def main():

    print("=" * 110)
    print("V1 TEMPORAL ALIGNMENT TARGET VALIDATION")
    print("=" * 110)


    # --------------------------------------------------------
    # 3.1 CHECK THAT BOTH INPUT FILES EXIST
    # --------------------------------------------------------

    # A missing input must produce a clear error.
    # We must not recreate or silently replace the input.

    check(
        METADATA_PATH.exists(),
        f"Missing metadata file: {METADATA_PATH}",
    )

    check(
        TARGETS_PATH.exists(),
        f"Missing target file: {TARGETS_PATH}",
    )


    # --------------------------------------------------------
    # 3.2 LOAD THE TWO DATASETS
    # --------------------------------------------------------

    # Parquet preserves pandas data types efficiently.
    # CSV is used because Step 30 creates a configuration
    # that can also be inspected manually.

    metadata = pd.read_parquet(METADATA_PATH)
    targets = pd.read_csv(TARGETS_PATH)


    print("\nINPUT DATASETS")
    print("-" * 110)

    print(
        f"Fieldwork metadata rows: {len(metadata):,}"
    )

    print(
        f"Temporal target rows:    {len(targets):,}"
    )


    # ========================================================
    # 4. VALIDATE COUNTRY COVERAGE
    # ========================================================

    # V1 includes exactly 29 European ESS countries.
    #
    # One row must represent one country.
    # Duplicate country rows could produce incorrect joins
    # when we later integrate Eurostat observations.

    check(
        len(metadata) == 29,
        "Expected exactly 29 metadata countries.",
    )

    check(
        len(targets) == 29,
        "Expected exactly 29 target countries.",
    )


    for name, frame in [
        ("metadata", metadata),
        ("targets", targets),
    ]:

        # Every observation needs a country identifier.

        check(
            "cntry" in frame.columns,
            f"Missing country identifier in {name}.",
        )

        check(
            frame["cntry"].notna().all(),
            f"Missing country codes in {name}.",
        )

        # Each country must appear exactly once.

        check(
            not frame["cntry"].duplicated().any(),
            f"Duplicate country codes in {name}.",
        )


    # Compare the country sets, not their row positions.
    # This makes validation independent of sorting order.

    check(
        set(metadata["cntry"])
        == set(targets["cntry"]),
        "Country coverage does not match.",
    )


    print("\nCOUNTRY COVERAGE")
    print("-" * 110)
    print("29 countries preserved: PASSED")
    print("Unique country codes:   PASSED")
    print("Country sets aligned:   PASSED")


    # ========================================================
    # 5. ALIGN THE DATASETS USING COUNTRY CODE
    # ========================================================

    # The country code becomes the index.
    #
    # Example:
    # metadata.loc["IT"] retrieves Italy's fieldwork data.
    # targets.loc["IT"] retrieves Italy's target years.

    metadata = metadata.set_index("cntry")
    targets = targets.set_index("cntry")


    # --------------------------------------------------------
    # 5.1 DEFINE REQUIRED METADATA COLUMNS
    # --------------------------------------------------------

    required_metadata_columns = [
        "weighted_share_2023",
        "weighted_share_2024",
        "fieldwork_year_min",
        "fieldwork_year_max",
        "dominant_fieldwork_year",
        "dominant_year_weighted_share",
        "multi_year_fieldwork_flag",
        "mixed_fieldwork_flag",
        "dominant_year_stable_flag",
    ]


    # These columns describe the actual ESS fieldwork.
    # They are our reference for independent validation.

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    check(
        not missing_metadata_columns,
        f"Missing metadata columns: "
        f"{missing_metadata_columns}",
    )


    # --------------------------------------------------------
    # 5.2 DEFINE REQUIRED TARGET COLUMNS
    # --------------------------------------------------------

    required_target_columns = [
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
        "temporal_sensitivity_flag",
        "temporal_sensitivity_priority_flag",
        "alignment_rule",
        "indicator_alignment_status",
    ]


    # These are the columns produced by Step 30.
    #
    # We also check the metadata columns copied into
    # the target configuration, not only the new fields.

    missing_target_columns = [
        column
        for column in required_target_columns
        if column not in targets.columns
    ]

    check(
        not missing_target_columns,
        f"Missing target columns: "
        f"{missing_target_columns}",
    )


    # ========================================================
    # 6. VALIDATE EACH COUNTRY INDEPENDENTLY
    # ========================================================

    print("\nINDEPENDENT COUNTRY VALIDATION")
    print("-" * 110)

    results = []


    # Sort codes so the diagnostic is easy to read.
    # The results do not depend on this ordering.

    for cntry in sorted(targets.index):

        source = metadata.loc[cntry]
        target = targets.loc[cntry]


        # ----------------------------------------------------
        # 6.1 VALIDATE OBSERVED FIELDWORK YEARS
        # ----------------------------------------------------

        # These are the first and last observed years
        # from valid ESS interview start dates.

        check(
            pd.notna(source["fieldwork_year_min"])
            and pd.notna(source["fieldwork_year_max"]),
            f"{cntry}: Missing fieldwork year range.",
        )

        first_year = int(
            source["fieldwork_year_min"]
        )

        last_year = int(
            source["fieldwork_year_max"]
        )


        # V1 ESS11 fieldwork takes place in 2023/2024.
        #
        # We reject unexpected years because they would
        # require a different alignment policy.

        check(
            2023 <= first_year <= last_year <= 2024,
            f"{cntry}: Invalid fieldwork year range.",
        )


        # Verify that Step 30 preserved the year range.

        check(
            int(target["fieldwork_year_min"]) == first_year
            and int(target["fieldwork_year_max"]) == last_year,
            f"{cntry}: Copied year range mismatch.",
        )


        # ----------------------------------------------------
        # 6.2 READ THE WEIGHTED YEAR SHARES
        # ----------------------------------------------------

        # Example:
        #
        # Italy:
        # 2023 approximately 41.7%
        # 2024 approximately 58.3%
        #
        # The shares were calculated in Step 29 using
        # valid interview dates and the ESS analysis weight.

        shares = {}

        for year in [2023, 2024]:

            column = f"weighted_share_{year}"

            value = source[column]

            # A year is observed when it falls inside
            # the actual fieldwork year range.

            observed = (
                first_year <= year <= last_year
            )

            if observed:

                check(
                    pd.notna(value),
                    f"{cntry}: Missing share for {year}.",
                )

                shares[year] = float(value)

            else:

                # An unobserved year may be stored as
                # zero or as a missing value.

                check(
                    pd.isna(value)
                    or np.isclose(float(value), 0.0),
                    f"{cntry}: Unexpected share for {year}.",
                )

                shares[year] = 0.0


            # Check that Step 30 copied the annual share.
            # This detects accidental changes in the CSV.

            copied_value = target[column]

            if pd.isna(value):

                check(
                    pd.isna(copied_value),
                    f"{cntry}: Copied missing share mismatch.",
                )

            else:

                check(
                    pd.notna(copied_value)
                    and np.isclose(
                        float(copied_value),
                        float(value),
                        atol=1e-8,
                    ),
                    f"{cntry}: Copied share mismatch for {year}.",
                )


        # ----------------------------------------------------
        # 6.3 VALIDATE THE WEIGHTED SHARES
        # ----------------------------------------------------

        # A weighted share must be between 0 and 1.
        #
        # Across 2023 and 2024, the shares must sum to 1.
        # Example: 0.417 + 0.583 = 1.

        check(
            all(
                0 <= share <= 1
                for share in shares.values()
            ),
            f"{cntry}: Weighted share outside 0-1.",
        )

        check(
            np.isclose(
                sum(shares.values()),
                1.0,
                atol=1e-8,
            ),
            f"{cntry}: Annual shares do not sum to 1.",
        )


        # ----------------------------------------------------
        # 6.4 RECALCULATE THE DOMINANT FIELDWORK YEAR
        # ----------------------------------------------------

        # We do not blindly trust the year stored by Step 30.
        #
        # Instead, we calculate it again from the actual
        # weighted shares of 2023 and 2024.
        #
        # In the case of an exact tie, prefer 2023.
        # This matches the deterministic tie rule in
        # the previous fieldwork metadata calculations.

        expected_year = (
            2023
            if shares[2023] >= shares[2024]
            else 2024
        )

        expected_share = max(
            shares.values()
        )


        # Verify that the metadata source agrees.

        check(
            int(source["dominant_fieldwork_year"])
            == expected_year,
            f"{cntry}: Metadata dominant year mismatch.",
        )

        check(
            np.isclose(
                float(source["dominant_year_weighted_share"]),
                expected_share,
                atol=1e-8,
            ),
            f"{cntry}: Metadata dominant share mismatch.",
        )


        # ----------------------------------------------------
        # 6.5 VALIDATE THE SELECTED ECONOMIC TARGET YEAR
        # ----------------------------------------------------

        # This is the main objective of Step 30.
        #
        # The target context year must correspond to the
        # weighted dominant fieldwork year.
        #
        # Important:
        # This is NOT yet an actual Eurostat observation year.
        # Indicator reference periods will be checked later.

        check(
            int(target["target_context_year"])
            == expected_year,
            f"{cntry}: Incorrect target context year.",
        )


        # Validate the copied dominant-year metadata.

        check(
            int(target["dominant_fieldwork_year"])
            == expected_year,
            f"{cntry}: Copied dominant year mismatch.",
        )

        check(
            pd.notna(target["dominant_year_weighted_share"])
            and np.isclose(
                float(target["dominant_year_weighted_share"]),
                expected_share,
                atol=1e-8,
            ),
            f"{cntry}: Copied dominant share mismatch.",
        )


        # ----------------------------------------------------
        # 6.6 VALIDATE THE DOMINANT YEAR STABILITY FLAG
        # ----------------------------------------------------

        # Step 29 checked whether the dominant year was
        # considered stable under its diagnostic rule.
        #
        # Step 30 requires every country to have a stable
        # dominant year before producing target years.

        check(
            as_flag(source["dominant_year_stable_flag"]),
            f"{cntry}: Unstable dominant year.",
        )

        check(
            as_flag(target["dominant_year_stable_flag"]),
            f"{cntry}: Target stability flag mismatch.",
        )


        # ----------------------------------------------------
        # 6.7 DETECT MULTI-YEAR FIELDWORK
        # ----------------------------------------------------

        # If first_year and last_year differ, the country
        # contains interviews from both 2023 and 2024.

        expected_multi = (
            first_year != last_year
        )


        # Both the metadata and target configuration
        # must identify multi-year countries correctly.

        check(
            as_flag(source["multi_year_fieldwork_flag"])
            == expected_multi,
            f"{cntry}: Metadata multi-year flag mismatch.",
        )

        check(
            as_flag(target["multi_year_fieldwork_flag"])
            == expected_multi,
            f"{cntry}: Copied multi-year flag mismatch.",
        )

        check(
            as_flag(target["temporal_sensitivity_flag"])
            == expected_multi,
            f"{cntry}: Incorrect temporal sensitivity flag.",
        )


        # ----------------------------------------------------
        # 6.8 IDENTIFY MIXED FIELDWORK PRIORITY COUNTRIES
        # ----------------------------------------------------

        # Our project uses an operational 80% threshold.
        #
        # If no single year contains at least 80% of
        # the weighted interviews, temporal alignment
        # becomes more sensitive to the chosen year.
        #
        # This threshold is a project diagnostic rule,
        # not an official ESS statistical standard.

        expected_priority = (
            expected_multi
            and expected_share < 0.80
        )


        check(
            as_flag(source["mixed_fieldwork_flag"])
            == expected_priority,
            f"{cntry}: Metadata mixed flag mismatch.",
        )

        check(
            as_flag(target["mixed_fieldwork_flag"])
            == expected_priority,
            f"{cntry}: Copied mixed flag mismatch.",
        )

        check(
            as_flag(
                target["temporal_sensitivity_priority_flag"]
            ) == expected_priority,
            f"{cntry}: Incorrect sensitivity priority flag.",
        )


        # ----------------------------------------------------
        # 6.9 VALIDATE THE ALTERNATIVE CONTEXT YEAR
        # ----------------------------------------------------

        # Example:
        #
        # Italy:
        # Main year = 2024
        # Alternative year = 2023
        #
        # Single-year countries have no alternative year.

        alternative = target[
            "alternative_context_year"
        ]


        if expected_multi:

            expected_alternative = (
                2024
                if expected_year == 2023
                else 2023
            )

            check(
                pd.notna(alternative)
                and int(alternative) == expected_alternative,
                f"{cntry}: Incorrect alternative year.",
            )

        else:

            check(
                pd.isna(alternative),
                f"{cntry}: Unexpected alternative year.",
            )


        # ----------------------------------------------------
        # 6.10 VALIDATE METHODOLOGICAL LABELS
        # ----------------------------------------------------

        # The alignment rule must be documented explicitly.
        # This prevents the chosen year from being mistaken
        # for an independently validated Eurostat year.

        expected_rule = (
            "target_context_year_equals_"
            "weighted_dominant_start_year"
        )

        check(
            target["alignment_rule"] == expected_rule,
            f"{cntry}: Incorrect alignment rule.",
        )


        # Eurostat indicator reference periods have not
        # been inspected yet, so the status must remain
        # pending rather than incorrectly marked complete.

        expected_status = (
            "pending_indicator_reference_period_check"
        )

        check(
            target["indicator_alignment_status"]
            == expected_status,
            f"{cntry}: Incorrect indicator status.",
        )


        # ----------------------------------------------------
        # 6.11 STORE THE VALIDATED COUNTRY RESULT
        # ----------------------------------------------------

        results.append(
            {
                "cntry": cntry,
                "target_year": expected_year,
                "multi_year": expected_multi,
                "priority": expected_priority,
            }
        )

        print(
            f"{cntry:<5} PASSED "
            f"| target={expected_year} "
            f"| multi-year={expected_multi} "
            f"| priority={expected_priority}"
        )


    # ========================================================
    # 7. VALIDATE AGGREGATE RESULTS
    # ========================================================

    result = pd.DataFrame(results)


    # Count how many countries target each year.

    target_counts = (
        result["target_year"]
        .value_counts()
    )

    countries_2023 = int(
        target_counts.get(2023, 0)
    )

    countries_2024 = int(
        target_counts.get(2024, 0)
    )


    # Count countries requiring temporal sensitivity
    # checks, and those receiving higher priority.

    multi_n = int(
        result["multi_year"].sum()
    )

    priority_n = int(
        result["priority"].sum()
    )


    # These counts are independently known from the
    # verified V1 Step 27 and Step 29 fieldwork results.

    check(
        countries_2023 == 19,
        "2023 target count mismatch.",
    )

    check(
        countries_2024 == 10,
        "2024 target count mismatch.",
    )

    check(
        multi_n == 13,
        "Multi-year country count mismatch.",
    )

    check(
        priority_n == 5,
        "Priority country count mismatch.",
    )


    # --------------------------------------------------------
    # 7.1 CHECK THE FIVE PRIORITY COUNTRIES
    # --------------------------------------------------------

    # The five countries with less than 80% weighted
    # interviews in the dominant calendar year are:
    #
    # Cyprus, Italy, Latvia, Poland, Portugal.

    expected_priority_countries = {
        "CY",
        "IT",
        "LV",
        "PL",
        "PT",
    }

    actual_priority_countries = set(
        result.loc[
            result["priority"],
            "cntry",
        ]
    )

    check(
        actual_priority_countries
        == expected_priority_countries,
        "Priority country membership mismatch.",
    )


    # ========================================================
    # 8. PRINT FINAL VALIDATION SUMMARY
    # ========================================================

    print("\n" + "=" * 110)
    print("VALIDATION SUMMARY")
    print("=" * 110)

    print(f"Countries validated:        {len(result)}")
    print(f"Target year 2023:           {countries_2023}")
    print(f"Target year 2024:           {countries_2024}")
    print(f"Multi-year countries:       {multi_n}")
    print(f"Priority sensitivity:       {priority_n}")

    print()
    print("Temporal alignment target validation PASSED.")

    print()
    print("No files were modified.")
    print("No Eurostat indicators were selected.")
    print("Indicator-specific reference periods remain pending.")


# ============================================================
# 9. EXECUTE THE SCRIPT
# ============================================================

# This block runs main() only when the script is executed
# directly, rather than when imported into another module.

if __name__ == "__main__":
    main()
