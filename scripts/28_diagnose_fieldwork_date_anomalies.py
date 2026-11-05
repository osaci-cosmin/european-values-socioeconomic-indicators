from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/ess/ESS11e04_2.parquet"
V1_PATH = ROOT / "data/interim/ESS11_v1_analytical.parquet"

KEYS = ["cntry", "idno"]


def main():
    # Load only the required columns.
    raw = pd.read_parquet(
        RAW_PATH,
        columns=KEYS + ["inwds", "inwde", "inwtm"],
    )

    v1 = pd.read_parquet(
        V1_PATH,
        columns=KEYS,
    )

    print("=" * 110)
    print("ESS ROUND 11 — FIELDWORK DATE ANOMALIES")
    print("=" * 110)

    # Validate respondent keys.
    for name, frame in [("RAW", raw), ("V1", v1)]:
        missing_keys = frame[KEYS].isna().any(axis=1).sum()
        duplicate_keys = frame.duplicated(KEYS).sum()

        print(
            f"{name}: missing keys = {missing_keys}, "
            f"duplicate keys = {duplicate_keys}"
        )

        if missing_keys > 0 or duplicate_keys > 0:
            raise ValueError(f"{name}: invalid respondent keys.")

    # Keep all V1 respondents and check one-to-one matching.
    df = v1.merge(
        raw,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    merge_passed = (
        len(df) == len(v1)
        and df["_merge"].eq("both").all()
    )

    print(f"\nOriginal rows: {len(v1):,}")
    print(f"Merged rows:   {len(df):,}")
    print(f"Merge passed:  {merge_passed}")

    if not merge_passed:
        raise ValueError("Incomplete or invalid fieldwork merge.")

    # Parse timestamps without silently discarding invalid values.
    for column in ["inwds", "inwde"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="raise",
        )

    # Extract years and calculate the timestamp difference.
    df["start_year"] = df["inwds"].dt.year.astype("Int64")
    df["end_year"] = df["inwde"].dt.year.astype("Int64")

    df["end_minus_start_minutes"] = (
        df["inwde"] - df["inwds"]
    ).dt.total_seconds() / 60

    missing_start = df["inwds"].isna()
    missing_end = df["inwde"].isna()
    both_present = ~missing_start & ~missing_end

    end_before_start = both_present & (
        df["inwde"] < df["inwds"]
    )

    different_years = both_present & (
        df["start_year"] != df["end_year"]
    ).fillna(False)

    # Different years are review cases, not automatic errors.
    review_case = (
        missing_start
        | missing_end
        | end_before_start
        | different_years
    )

    print("\n" + "-" * 110)
    print("DATE COMPLETENESS AND CONSISTENCY")
    print("-" * 110)

    print(f"Respondents:                    {len(df):,}")
    print(f"Countries:                      {df['cntry'].nunique()}")
    print(f"Missing start dates:            {int(missing_start.sum())}")
    print(f"Missing end dates:              {int(missing_end.sum())}")
    print(
        "Both dates missing:             "
        f"{int((missing_start & missing_end).sum())}"
    )
    print(
        "Missing start, end available:    "
        f"{int((missing_start & ~missing_end).sum())}"
    )
    print(f"End before start:               {int(end_before_start.sum())}")
    print(f"Different start/end years:      {int(different_years.sum())}")
    print(
        "End before start across years:  "
        f"{int((end_before_start & different_years).sum())}"
    )
    print(f"Unique respondents to review:   {int(review_case.sum())}")

    display_columns = KEYS + [
        "inwds",
        "inwde",
        "inwtm",
        "start_year",
        "end_year",
        "end_minus_start_minutes",
    ]

    sections = [
        ("MISSING START DATE", missing_start),
        ("MISSING END DATE", missing_end),
        ("END BEFORE START", end_before_start),
        ("DIFFERENT START AND END YEARS", different_years),
    ]

    for title, mask in sections:
        print("\n" + "-" * 110)
        print(title)
        print("-" * 110)

        cases = df.loc[mask, display_columns].sort_values(KEYS)

        if cases.empty:
            print("None.")
        else:
            print(
                cases.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.3f}",
                )
            )

    print("\n" + "=" * 110)
    print("FIELDWORK DATE ANOMALY DIAGNOSTIC COMPLETED")
    print("=" * 110)
    print("Cases may appear in more than one section.")
    print("Timestamp differences are shown separately from inwtm.")
    print("No dates were corrected or imputed.")
    print("No files were modified.")


if __name__ == "__main__":
    main()