from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Define project paths and diagnostic settings.
# ============================================================

# Move from this script's location to the project root folder.
ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/ess/ESS11e04_2.parquet"
V1_PATH = ROOT / "data/interim/ESS11_v1_analytical.parquet"
OUTPUT_DIR = ROOT / "data/processed"

# A respondent is identified by country AND respondent ID.
KEYS = ["cntry", "idno"]

# Project-specific diagnostic threshold, not an ESS standard.
# Below 80%, fieldwork is considered substantially mixed.
MIXED_YEAR_THRESHOLD = 0.80


# ============================================================
# 2. Define a reusable function for the dominant year.
# ============================================================

def dominant_year(frame):
    # Keep only respondents with an observed start year.
    # Missing start years are not replaced with end years.
    valid = frame.loc[frame["fieldwork_year"].notna()]

    # A dominant year cannot be calculated without dates.
    if valid.empty:
        return pd.NA, np.nan

    # Add the analysis weights separately for each year.
    totals = valid.groupby("fieldwork_year")["anweight"].sum()

    # Divide each year's weight by the total valid-date weight.
    shares = totals / totals.sum()

    # Identify the year with the largest weighted share.
    # Treat almost identical maximum shares as a numerical tie.
    winners = shares.index[
        np.isclose(
            shares,
            shares.max(),
            rtol=0,
            atol=1e-12,
        )
    ]

    # Do not choose a year arbitrarily when there is a tie.
    if len(winners) != 1:
        return pd.NA, float(shares.max())

    # Return two results: the dominant year and its share.
    return int(winners[0]), float(shares.max())


def main():
    # ========================================================
    # 3. Load only the columns needed for this task.
    # ========================================================

    # Read interview timestamps from the unchanged RAW file.
    raw = pd.read_parquet(
        RAW_PATH,
        columns=KEYS + ["inwds", "inwde"],
    )

    # Read V1 respondent identifiers and analysis weights.
    v1 = pd.read_parquet(
        V1_PATH,
        columns=KEYS + ["anweight"],
    )

    # ========================================================
    # 4. Validate respondent keys and merge the datasets.
    # ========================================================

    # Check both datasets before merging.
    for name, frame in [("RAW", raw), ("V1", v1)]:
        # Missing keys would make respondent matching unreliable.
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{name}: missing respondent keys.")

        # Duplicate keys could multiply rows during a merge.
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name}: duplicate respondent keys.")

    # A left merge preserves all respondents in the V1 dataset.
    # one_to_one requires unique keys on both sides.
    # indicator adds a column showing whether each match succeeded.
    df = v1.merge(
        raw,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # Require unchanged row count and a RAW match for every V1 row.
    if len(df) != len(v1) or not df["_merge"].eq("both").all():
        raise ValueError("Incomplete or invalid merge.")

    # Weights must be finite and strictly positive.
    if not (
        np.isfinite(df["anweight"]).all()
        and df["anweight"].gt(0).all()
    ):
        raise ValueError("Invalid analysis weights.")

    # ========================================================
    # 5. Prepare timestamps and extract calendar years.
    # ========================================================

    # Stop on unparseable values rather than silently losing them.
    # Existing missing timestamps remain missing.
    for column in ["inwds", "inwde"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="raise",
        )

    # Int64 supports whole years together with missing values.
    df["fieldwork_year"] = (
        df["inwds"].dt.year.astype("Int64")
    )

    end_year = df["inwde"].dt.year.astype("Int64")

    # Step 27 found only 2023 and 2024 among observed start years.
    # Stop if the current input contains an unexpected year.
    observed_years = set(
        df["fieldwork_year"].dropna().astype(int)
    )

    if not observed_years.issubset({2023, 2024}):
        raise ValueError(
            f"Unexpected start years: {observed_years}"
        )

    # ========================================================
    # 6. Create flags for administrative date review.
    # ========================================================

    # Each flag is True when the specified condition is present.
    df["missing_start_flag"] = df["inwds"].isna()
    df["missing_end_flag"] = df["inwde"].isna()

    # Comparing timestamps requires both values to be available.
    both_present = (
        df["inwds"].notna()
        & df["inwde"].notna()
    )

    df["end_before_start_flag"] = (
        both_present
        & df["inwde"].lt(df["inwds"])
    )

    # Different years indicate a review case, not a proven error.
    df["cross_year_flag"] = (
        both_present
        & df["fieldwork_year"].ne(end_year)
    ).fillna(False)

    flag_columns = [
        "missing_start_flag",
        "missing_end_flag",
        "end_before_start_flag",
        "cross_year_flag",
    ]

    # Mark a respondent if at least one review condition is True.
    # axis=1 checks across columns within each respondent's row.
    df["date_review_flag"] = (
        df[flag_columns].any(axis=1)
    )

    # ========================================================
    # 7. Build one metadata record per country.
    # ========================================================

    # Each dictionary added to this list will become one row.
    records = []

    for country, group in df.groupby("cntry", sort=True):
        # group contains all V1 respondents from this country.
        valid = group.loc[
            group["fieldwork_year"].notna()
        ]

        if valid.empty:
            raise ValueError(
                f"{country}: no observed start years."
            )

        # This is the denominator for weighted year shares.
        valid_weight = valid["anweight"].sum()

        # Main calculation: use every available start year,
        # including flagged records when their start year exists.
        main_year, main_share = dominant_year(group)

        # Sensitivity calculation: temporarily omit review cases.
        # This does not remove anyone from the analytical file.
        sensitivity = group.loc[
            ~group["date_review_flag"]
        ]

        sensitivity_year, sensitivity_share = (
            dominant_year(sensitivity)
        )

        # Compare years only when both calculations resolve a year.
        comparable = (
            pd.notna(main_year)
            and pd.notna(sensitivity_year)
        )

        record = {
            "cntry": country,

            # Counts describe the original country sample.
            "respondents": len(group),
            "valid_start_dates": len(valid),

            "missing_start_dates": int(
                group["missing_start_flag"].sum()
            ),
            "missing_end_dates": int(
                group["missing_end_flag"].sum()
            ),
            "end_before_start_n": int(
                group["end_before_start_flag"].sum()
            ),
            "cross_year_n": int(
                group["cross_year_flag"].sum()
            ),
            "date_review_n": int(
                group["date_review_flag"].sum()
            ),

            # These shares use ALL country respondents' weights.
            "date_review_weighted_share": (
                group.loc[
                    group["date_review_flag"],
                    "anweight",
                ].sum()
                / group["anweight"].sum()
            ),
            "valid_start_weighted_share": (
                valid_weight
                / group["anweight"].sum()
            ),

            # The year range is based on observed START dates.
            "fieldwork_year_min": int(
                valid["fieldwork_year"].min()
            ),
            "fieldwork_year_max": int(
                valid["fieldwork_year"].max()
            ),
            "multi_year_fieldwork_flag": (
                valid["fieldwork_year"].nunique() > 1
            ),

            # Main results use weights among valid start dates.
            "dominant_fieldwork_year": main_year,
            "dominant_year_weighted_share": main_share,

            # True means the largest year share is below 80%.
            "mixed_fieldwork_flag": (
                main_share < MIXED_YEAR_THRESHOLD
            ),

            # Store the sensitivity result for comparison.
            "sensitivity_respondents": len(sensitivity),
            "sensitivity_dominant_year": sensitivity_year,
            "sensitivity_dominant_share": sensitivity_share,

            # Missing means the comparison could not be resolved.
            "dominant_year_stable_flag": (
                main_year == sensitivity_year
                if comparable
                else pd.NA
            ),

            # Record the rules so the results are traceable.
            "fieldwork_year_source": "inwds",
            "missing_start_rule": "leave_missing_no_fallback",
            "dominant_year_rule": (
                "largest_anweight_share_valid_start"
            ),
            "mixed_year_threshold": MIXED_YEAR_THRESHOLD,
            "ess_edition": "4.2",
        }

        # Retain counts and weighted shares for BOTH years.
        # This avoids reducing all fieldwork information to one year.
        for year in [2023, 2024]:
            year_mask = valid["fieldwork_year"].eq(year)

            record[f"respondents_{year}"] = int(
                year_mask.sum()
            )

            record[f"weighted_share_{year}"] = (
                valid.loc[year_mask, "anweight"].sum()
                / valid_weight
            )

        records.append(record)

    # Convert the list of country records into a table.
    metadata = pd.DataFrame(records)

    # Preserve missing values without storing years as decimals.
    for column in [
        "dominant_fieldwork_year",
        "sensitivity_dominant_year",
    ]:
        metadata[column] = metadata[column].astype("Int64")

    # Nullable boolean allows True, False, or missing.
    metadata["dominant_year_stable_flag"] = (
        metadata["dominant_year_stable_flag"].astype("boolean")
    )

    # ========================================================
    # 8. Check the generated table before saving.
    # ========================================================

    # Every V1 country must appear in the metadata.
    if set(metadata["cntry"]) != set(v1["cntry"]):
        raise ValueError("Country coverage mismatch.")

    # Country respondent counts must reconstruct the V1 total.
    if metadata["respondents"].sum() != len(v1):
        raise ValueError("Respondent count mismatch.")

    # Each country's valid-date year shares must sum to 100%.
    # allclose allows small floating-point rounding differences.
    if not np.allclose(
        metadata["weighted_share_2023"]
        + metadata["weighted_share_2024"],
        1.0,
    ):
        raise ValueError("Year shares do not sum to one.")

    # ========================================================
    # 9. Save country metadata in two formats.
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        OUTPUT_DIR / "ESS11_country_fieldwork_metadata.csv"
    )
    parquet_path = (
        OUTPUT_DIR / "ESS11_country_fieldwork_metadata.parquet"
    )

    # CSV is easy to inspect; Parquet preserves column types.
    # Rerunning this script replaces these two generated files.
    metadata.to_csv(csv_path, index=False)
    metadata.to_parquet(parquet_path, index=False)

    # ========================================================
    # 10. Print the results for review.
    # ========================================================

    print("=" * 110)
    print("ESS11 COUNTRY FIELDWORK METADATA")
    print("=" * 110)

    print(f"Respondents: {len(df):,}")
    print(f"Countries: {len(metadata)}")
    print(
        f"Missing starts: "
        f"{int(df['missing_start_flag'].sum())}"
    )
    print(
        f"Review cases: "
        f"{int(df['date_review_flag'].sum())}"
    )

    display_columns = [
        "cntry",
        "dominant_fieldwork_year",
        "dominant_year_weighted_share",
        "mixed_fieldwork_flag",
        "date_review_n",
        "date_review_weighted_share",
        "sensitivity_dominant_year",
        "dominant_year_stable_flag",
    ]

    print(
        metadata[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    stable = metadata["dominant_year_stable_flag"]

    print(
        f"\nStable dominant years: "
        f"{int(stable.fillna(False).sum())}"
    )
    print(
        f"Changed dominant years: "
        f"{int(stable.eq(False).sum())}"
    )
    print(
        f"Unresolved comparisons: "
        f"{int(stable.isna().sum())}"
    )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {parquet_path}")

    print(
        "No RAW or respondent-level analytical files "
        "were modified."
    )
    print("No Eurostat year was assigned.")


# Run main() when this file is executed directly.
if __name__ == "__main__":
    main()