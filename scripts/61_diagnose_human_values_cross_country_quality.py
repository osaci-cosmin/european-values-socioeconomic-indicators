
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 61 - HUMAN VALUES CROSS-COUNTRY DATA QUALITY
# ============================================================

# Resolve the project directory independently of the terminal.
ROOT = Path(__file__).resolve().parents[1]

# Define the validated respondent-level input.
INPUT = ROOT / "data/interim/ESS11_v1_scope.parquet"

# Define the validated Human Values scoring output.
SCORED = ROOT / "data/interim/ESS11_v1_values.parquet"

# Define the original PVQ-21 item mapping.
MAPPING = ROOT / "config/human_values_mapping.csv"

# This diagnostic does not write any files.
# All previously validated files remain unchanged.

COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

EXPECTED_HIGHER_ORDER = {
    "Openness to Change": 6,
    "Conservation": 6,
    "Self-Transcendence": 5,
    "Self-Enhancement": 4,
}


def require(condition, message):

    # Fail explicitly instead of silently fixing source data.
    if not bool(condition):
        raise AssertionError(message)


def main():

    print("=" * 85)
    print("STEP 61 - HUMAN VALUES CROSS-COUNTRY DATA QUALITY DIAGNOSTICS")
    print("=" * 85)

    # --------------------------------------------------------
    # 1. VERIFY INPUT AVAILABILITY
    # --------------------------------------------------------

    for path in (INPUT, SCORED, MAPPING):

        require(
            path.is_file(),
            f"Required file not found: {path}"
        )

    # Read the original validated sample and item mapping.
    df = pd.read_parquet(INPUT, engine="pyarrow")
    mapping = pd.read_csv(MAPPING)

    # --------------------------------------------------------
    # 2. VALIDATE THE PVQ-21 MAPPING
    # --------------------------------------------------------

    required_mapping = [
        "item_number",
        "variable",
        "basic_value",
        "higher_order_value",
        "reverse_score",
    ]

    require(
        required_mapping == mapping.columns.tolist(),
        "Unexpected PVQ-21 mapping schema"
    )

    require(
        len(mapping) == 21,
        "The mapping must contain exactly 21 items"
    )

    require(
        mapping["item_number"].tolist() == list(range(1, 22)),
        "Unexpected PVQ item numbering"
    )

    require(
        mapping["variable"].is_unique,
        "Duplicate PVQ item variables"
    )

    require(
        mapping["reverse_score"].eq(1).all(),
        "Unexpected reverse-score configuration"
    )

    higher_counts = (
        mapping["higher_order_value"]
        .value_counts()
        .to_dict()
    )

    require(
        higher_counts == EXPECTED_HIGHER_ORDER,
        "Higher-order PVQ mapping differs from V1"
    )

    basic_counts = (
        mapping["basic_value"]
        .value_counts()
        .to_dict()
    )

    require(
        len(basic_counts) == 10
        and basic_counts.get("Universalism") == 3
        and all(
            count == 2
            for name, count in basic_counts.items()
            if name != "Universalism"
        ),
        "Unexpected basic Human Values item allocation"
    )

    variables = mapping["variable"].tolist()

    # --------------------------------------------------------
    # 3. CHECK RESPONDENTS, COUNTRIES AND ITEM VALUES
    # --------------------------------------------------------

    required_columns = [
        "cntry",
        "idno",
        "pspwght",
    ] + variables

    require(
        set(required_columns).issubset(df.columns),
        "Missing required respondent-level columns"
    )

    require(
        len(df) == 49210,
        "Unexpected V1 respondent count"
    )

    require(
        df["cntry"].notna().all(),
        "Missing country identifiers"
    )

    require(
        set(df["cntry"]) == COUNTRIES,
        "Country coverage differs from validated V1"
    )

    # Convert the 21 items into numeric form without imputing values.
    items = df[variables].apply(
        pd.to_numeric,
        errors="raise"
    )

    # Check the official ESS valid-response range.
    # Missing observations remain missing.
    for variable in variables:

        valid = items[variable].dropna()

        require(
            valid.between(1, 6).all(),
            f"Invalid PVQ response range: {variable}"
        )

        require(
            valid.mod(1).eq(0).all(),
            f"Non-integer PVQ response: {variable}"
        )

    print("\nMapping, country universe and item domains: PASSED.")

    # --------------------------------------------------------
    # 4. RECONSTRUCT RESPONDENT COMPLETENESS
    # --------------------------------------------------------

    # Count nonmissing PVQ items independently of Step 14.
    answered = items.notna().sum(axis=1)

    missing = 21 - answered

    # A respondent is eligible for V1 scores only with 21/21 items.
    complete = answered.eq(21)

    complete_n = int(complete.sum())
    incomplete_n = int((~complete).sum())

    require(
        complete_n == 45299,
        "Unexpected complete-case count"
    )

    require(
        incomplete_n == 3911,
        "Unexpected incomplete-case count"
    )

    print("\nOVERALL COMPLETENESS")
    print("-" * 85)

    print("Respondents:", len(df))
    print("Complete 21/21:", complete_n)
    print("Incomplete:", incomplete_n)

    print(
        "Complete-case percentage:",
        f"{100 * complete.mean():.2f}%"
    )

    # --------------------------------------------------------
    # 5. VERIFY THE EXISTING STEP 14 SCORING FLAGS
    # --------------------------------------------------------

    # Read only the fields needed for independent comparison.
    score_columns = [
        "cntry",
        "idno",
        "hv_answered_n",
        "hv_complete_flag",
        "hv_scoring_eligible",
        "hv_country_note_flag",
    ]

    scored = pd.read_parquet(
        SCORED,
        columns=score_columns,
        engine="pyarrow"
    )

    require(
        len(scored) == len(df),
        "Step 14 respondent count differs"
    )

    # Step 14 preserves the original row order and identifiers.
    for column in ("cntry", "idno"):

        require(
            df[column].reset_index(drop=True).equals(
                scored[column].reset_index(drop=True)
            ),
            f"Step 14 row alignment differs: {column}"
        )

    # Compare independently calculated completeness information.
    require(
        answered.reset_index(drop=True).astype("int64").equals(
            scored["hv_answered_n"]
            .reset_index(drop=True)
            .astype("int64")
        ),
        "Step 14 answered-item counts differ"
    )

    require(
        complete.reset_index(drop=True).equals(
            scored["hv_complete_flag"]
            .reset_index(drop=True)
            .astype(bool)
        ),
        "Step 14 completeness flags differ"
    )

    require(
        complete.reset_index(drop=True).equals(
            scored["hv_scoring_eligible"]
            .reset_index(drop=True)
            .astype(bool)
        ),
        "Step 14 scoring eligibility differs"
    )

    # Estonia and the United Kingdom are documentation-flagged.
    expected_note = df["cntry"].isin(["EE", "GB"])

    require(
        expected_note.reset_index(drop=True).equals(
            scored["hv_country_note_flag"]
            .reset_index(drop=True)
            .astype(bool)
        ),
        "Step 14 country documentation flags differ"
    )

    print("\nStep 14 completeness and country flags: PASSED.")

    # --------------------------------------------------------
    # 6. RECONSTRUCT COUNTRY-LEVEL COMPLETENESS
    # --------------------------------------------------------

    country_records = []

    # Inspect each of the 29 national samples separately.
    for country, group in df.groupby("cntry", sort=True):

        index = group.index

        n = len(group)

        country_answered = answered.loc[index]
        country_missing = missing.loc[index]
        country_complete = complete.loc[index]

        eligible_n = int(country_complete.sum())

        # Count total missing item cells within this country.
        missing_cells = int(
            country_missing.sum()
        )

        # A valid positive survey weight is required for the
        # weighted diagnostic percentage.
        weight = pd.to_numeric(
            group["pspwght"],
            errors="raise"
        ).astype("float64")

        valid_weight = (
            weight.notna()
            & np.isfinite(weight)
            & weight.gt(0)
        )

        valid_weight_n = int(valid_weight.sum())

        # Calculate a within-country weighted completeness rate.
        # This describes observed eligibility; it is not a
        # correction for missing PVQ responses.
        if valid_weight_n > 0:

            numerator = weight.loc[
                valid_weight
                & country_complete
            ].sum()

            denominator = weight.loc[
                valid_weight
            ].sum()

            weighted_pct = (
                100 * numerator / denominator
            )

        else:

            # Never replace unavailable information with zero.
            weighted_pct = float("nan")

        country_records.append({
            "cntry": country,
            "respondents": n,
            "complete_n": eligible_n,
            "incomplete_n": n - eligible_n,
            "complete_pct":
                100 * eligible_n / n,
            "weighted_complete_pct":
                weighted_pct,
            "valid_weight_n":
                valid_weight_n,
            "missing_cells":
                missing_cells,
            "missing_cell_pct":
                100 * missing_cells / (21 * n),
            "mean_answered_items":
                country_answered.mean(),
            "missing_1_item_n":
                int(country_missing.eq(1).sum()),
            "missing_2_to_5_items_n":
                int(country_missing.between(2, 5).sum()),
            "missing_6_to_20_items_n":
                int(country_missing.between(6, 20).sum()),
            "missing_all_21_items_n":
                int(country_missing.eq(21).sum()),
            "documentation_flag":
                country in ("EE", "GB"),
        })

    country_table = pd.DataFrame(
        country_records
    )

    # Check that national totals reproduce the full dataset.
    require(
        country_table["respondents"].sum() == len(df),
        "Country respondent totals do not reconcile"
    )

    require(
        country_table["complete_n"].sum() == complete_n,
        "Country complete-case totals do not reconcile"
    )

    require(
        country_table["incomplete_n"].sum() == incomplete_n,
        "Country incomplete-case totals do not reconcile"
    )

    require(
        country_table["missing_cells"].sum()
        == int(missing.sum()),
        "Country missing-cell totals do not reconcile"
    )

    # Display all countries, beginning with the lowest completeness.
    print("\nCOMPLETENESS BY COUNTRY")
    print("-" * 85)

    display_columns = [
        "cntry",
        "respondents",
        "complete_n",
        "incomplete_n",
        "complete_pct",
        "weighted_complete_pct",
        "valid_weight_n",
        "missing_cell_pct",
        "documentation_flag",
    ]

    print(
        country_table.sort_values(
            ["complete_pct", "cntry"]
        )[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # 7. EXAMINE RESPONDENT MISSINGNESS PATTERNS
    # --------------------------------------------------------

    print("\nMISSINGNESS PATTERNS BY COUNTRY")
    print("-" * 85)

    pattern_columns = [
        "cntry",
        "missing_1_item_n",
        "missing_2_to_5_items_n",
        "missing_6_to_20_items_n",
        "missing_all_21_items_n",
        "mean_answered_items",
    ]

    print(
        country_table[pattern_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # 8. EXAMINE MISSINGNESS FOR EVERY PVQ ITEM
    # --------------------------------------------------------

    item_records = []

    for _, row in mapping.iterrows():

        variable = row["variable"]

        missing_n = int(
            items[variable].isna().sum()
        )

        item_records.append({
            "variable": variable,
            "basic_value": row["basic_value"],
            "higher_order": row["higher_order_value"],
            "missing_n": missing_n,
            "missing_pct":
                100 * missing_n / len(df),
        })

    item_table = pd.DataFrame(
        item_records
    )

    require(
        item_table["missing_n"].sum()
        == int(missing.sum()),
        "Item missingness totals do not reconcile"
    )

    print("\nMISSINGNESS BY PVQ ITEM")
    print("-" * 85)

    print(
        item_table.sort_values(
            ["missing_n", "variable"],
            ascending=[False, True]
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # 9. FIND THE LARGEST COUNTRY-ITEM MISSINGNESS COUNTS
    # --------------------------------------------------------

    # Count missing responses separately for every country-item pair.
    country_item_missing = (
        items.isna()
        .assign(cntry=df["cntry"])
        .groupby("cntry")
        .sum()
    )

    # Convert the country-by-item matrix into a long table.
    long_missing = (
        country_item_missing
        .stack()
        .reset_index()
    )

    long_missing.columns = [
        "cntry",
        "variable",
        "missing_n",
    ]

    # Add the country sample size for correct percentage denominators.
    sample_sizes = df.groupby("cntry").size()

    long_missing["respondents"] = (
        long_missing["cntry"]
        .map(sample_sizes)
    )

    long_missing["missing_pct"] = (
        100
        * long_missing["missing_n"]
        / long_missing["respondents"]
    )

    # Rank by country-specific missing percentage, not only count.
    # Countries have different sample sizes.
    print("\nTOP 15 COUNTRY-ITEM MISSINGNESS RATES")
    print("-" * 85)

    print(
        long_missing.sort_values(
            ["missing_pct", "cntry", "variable"],
            ascending=[False, True, True]
        ).head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # 10. INSPECT THE TWO DOCUMENTATION-FLAGGED COUNTRIES
    # --------------------------------------------------------

    print("\nESTONIA AND UNITED KINGDOM - DOCUMENTATION FLAGS")
    print("-" * 85)

    print(
        country_table.loc[
            country_table["documentation_flag"],
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # Show the five least-complete items in each flagged country.
    for country in ("EE", "GB"):

        print(f"\n{country} - FIVE HIGHEST ITEM MISSINGNESS RATES")

        print(
            long_missing.loc[
                long_missing["cntry"] == country
            ].sort_values(
                ["missing_pct", "variable"],
                ascending=[False, True]
            ).head(5).to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

    # --------------------------------------------------------
    # 11. FINAL REPORT
    # --------------------------------------------------------

    print("\nSUMMARY")
    print("-" * 85)

    print("Countries examined:", len(country_table))
    print("PVQ items examined:", len(variables))
    print("Respondents:", len(df))
    print("Complete cases:", complete_n)
    print("Incomplete cases:", incomplete_n)

    print(
        "Lowest country completeness:",
        f"{country_table['complete_pct'].min():.2f}%"
    )

    print(
        "Highest country completeness:",
        f"{country_table['complete_pct'].max():.2f}%"
    )

    print(
        "Total missing PVQ item cells:",
        int(missing.sum())
    )

    print(
        "Countries with documentation flags:",
        "EE, GB"
    )

    print("\nHUMAN VALUES CROSS-COUNTRY QUALITY DIAGNOSTICS PASSED.")

    print(
        "This diagnostic does not establish "
        "cross-country measurement invariance."
    )

    print(
        "No project files were created or modified."
    )


if __name__ == "__main__":
    main()
