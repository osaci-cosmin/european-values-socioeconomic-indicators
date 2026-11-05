from pathlib import Path

import pandas as pd


# ============================================================
# 1. Define input and output paths.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data/processed/ESS11_country_fieldwork_metadata.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "config/temporal_alignment_targets_v1.csv"
)


def main():
    # ========================================================
    # 2. Load the country metadata created in Step 29.
    # ========================================================

    metadata = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "cntry",
        "fieldwork_year_min",
        "fieldwork_year_max",
        "dominant_fieldwork_year",
        "dominant_year_weighted_share",
        "multi_year_fieldwork_flag",
        "mixed_fieldwork_flag",
        "dominant_year_stable_flag",
        "weighted_share_2023",
        "weighted_share_2024",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Require one row per country and an identified dominant year.
    if metadata["cntry"].isna().any():
        raise ValueError("Missing country codes.")

    if metadata["cntry"].duplicated().any():
        raise ValueError("Duplicate country codes.")

    if metadata["dominant_fieldwork_year"].isna().any():
        raise ValueError("Unresolved dominant fieldwork years.")

    # This rule relies on the stability result reviewed in Step 29.
    stable = metadata["dominant_year_stable_flag"]

    if stable.isna().any() or not stable.eq(True).all():
        raise ValueError(
            "Review unstable or unresolved dominant years first."
        )

    # ========================================================
    # 3. Define the main target year for economic conditions.
    # ========================================================

    targets = metadata[required_columns].copy()

    # This is the desired economic reference year.
    # It is NOT necessarily the year label in a Eurostat series.
    targets["target_context_year"] = (
        targets["dominant_fieldwork_year"].astype("Int64")
    )

    # ========================================================
    # 4. Retain the other observed year for sensitivity analysis.
    # ========================================================

    targets["alternative_context_year"] = pd.Series(
        pd.NA,
        index=targets.index,
        dtype="Int64",
    )

    for index, row in targets.iterrows():
        first_year = int(row["fieldwork_year_min"])
        last_year = int(row["fieldwork_year_max"])
        main_year = int(row["target_context_year"])

        # This V1 rule is designed for the observed 2023–2024 data.
        if first_year not in {2023, 2024}:
            raise ValueError("Unexpected minimum fieldwork year.")

        if last_year not in {2023, 2024}:
            raise ValueError("Unexpected maximum fieldwork year.")

        if first_year > last_year:
            raise ValueError("Invalid fieldwork year range.")

        if main_year not in {first_year, last_year}:
            raise ValueError("Dominant year outside observed years.")

        # Single-year countries have no alternative observed year.
        if first_year == last_year:
            continue

        # For mixed 2023–2024 fieldwork, retain the other year.
        alternative_year = (
            last_year
            if main_year == first_year
            else first_year
        )

        targets.at[
            index,
            "alternative_context_year",
        ] = alternative_year

    # ========================================================
    # 5. Record the project rules explicitly.
    # ========================================================

    targets["alignment_rule"] = (
        "target_context_year_equals_weighted_dominant_start_year"
    )

    # All multi-year countries can enter temporal sensitivity checks.
    targets["temporal_sensitivity_flag"] = (
        targets["multi_year_fieldwork_flag"]
    )

    # Countries below 80% receive priority in those checks.
    targets["temporal_sensitivity_priority_flag"] = (
        targets["mixed_fieldwork_flag"]
    )

    # Dataset-specific reference periods still need verification.
    targets["indicator_alignment_status"] = (
        "pending_indicator_reference_period_check"
    )

    # ========================================================
    # 6. Save the configuration and display the selected targets.
    # ========================================================

    targets = targets.sort_values("cntry").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Rerunning replaces only this generated configuration file.
    targets.to_csv(OUTPUT_PATH, index=False)

    print("=" * 110)
    print("V1 TEMPORAL ALIGNMENT TARGETS")
    print("=" * 110)

    display_columns = [
        "cntry",
        "target_context_year",
        "dominant_year_weighted_share",
        "alternative_context_year",
        "temporal_sensitivity_priority_flag",
    ]

    print(
        targets[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nCOUNTRIES BY TARGET YEAR")
    print(
        targets["target_context_year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nCountries with an alternative year: "
        f"{int(targets['alternative_context_year'].notna().sum())}"
    )

    print(
        "Priority sensitivity countries: "
        f"{int(targets['temporal_sensitivity_priority_flag'].sum())}"
    )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("No ESS respondent-level files were modified.")
    print("No Eurostat observations were downloaded or selected.")


if __name__ == "__main__":
    main()