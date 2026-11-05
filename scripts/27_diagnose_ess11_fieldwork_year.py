
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# Define project paths
project_root = Path(__file__).resolve().parents[1]

raw_file = (
    project_root
    / "data/raw/ess/ESS11e04_2.parquet"
)

analytical_file = (
    project_root
    / "data/interim/ESS11_v1_analytical.parquet"
)


# Check input files
for file_path in [raw_file, analytical_file]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )


# Inspect available interview columns
raw_columns = pq.ParquetFile(raw_file).schema.names

required = ["cntry", "idno", "inwds"]
optional = ["inwde", "inwtm"]

missing = [
    col for col in required
    if col not in raw_columns
]

if missing:
    raise ValueError(
        f"Required RAW columns missing: {missing}"
    )

selected = required + [
    col for col in optional
    if col in raw_columns
]


# Load only necessary data
raw = pd.read_parquet(
    raw_file,
    columns=selected,
)

analytical = pd.read_parquet(
    analytical_file,
    columns=[
        "cntry",
        "idno",
        "country_name",
        "anweight",
    ],
)


print("=" * 110)
print("ESS ROUND 11 FIELDWORK YEAR DIAGNOSTIC")
print("=" * 110)

print(f"\nRespondents: {len(analytical):,}")
print(f"Countries:   {analytical['cntry'].nunique()}")


# ---------------------------------------------------------
# 1. Validate respondent keys
# ---------------------------------------------------------

print("\nRESPONDENT KEY VALIDATION")
print("-" * 110)

raw_duplicates = raw.duplicated(
    ["cntry", "idno"]
).sum()

v1_duplicates = analytical.duplicated(
    ["cntry", "idno"]
).sum()

print(f"RAW duplicates: {raw_duplicates:,}")
print(f"V1 duplicates:  {v1_duplicates:,}")

if raw_duplicates or v1_duplicates:
    raise ValueError(
        "Duplicate respondent identifiers detected."
    )


# ---------------------------------------------------------
# 2. Merge fieldwork information
# ---------------------------------------------------------

merged = analytical.merge(
    raw,
    on=["cntry", "idno"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

merge_valid = (
    len(merged) == len(analytical)
    and (merged["_merge"] == "both").all()
)

print("\nFIELDWORK MERGE")
print("-" * 110)

print(f"Original rows: {len(analytical):,}")
print(f"Merged rows:   {len(merged):,}")
print(f"Merge passed:  {merge_valid}")

if not merge_valid:
    raise ValueError(
        "Fieldwork merge validation failed."
    )

merged.drop(columns="_merge", inplace=True)


# ---------------------------------------------------------
# 3. Extract interview year and month
# ---------------------------------------------------------

merged["interview_start"] = pd.to_datetime(
    merged["inwds"],
    errors="coerce",
)

merged["fieldwork_year"] = (
    merged["interview_start"]
    .dt.year
    .astype("Int64")
)

merged["fieldwork_month"] = (
    merged["interview_start"]
    .dt.month
    .astype("Int64")
)


# ---------------------------------------------------------
# 4. Check interview date completeness
# ---------------------------------------------------------

print("\nINTERVIEW DATE COMPLETENESS")
print("-" * 110)

valid_dates = merged["interview_start"].notna().sum()
missing_dates = merged["interview_start"].isna().sum()

print(f"Valid start dates:   {valid_dates:,}")
print(f"Missing start dates: {missing_dates:,}")

if "inwde" in merged.columns:

    merged["interview_end"] = pd.to_datetime(
        merged["inwde"],
        errors="coerce",
    )

    invalid_order = (
        merged["interview_start"].notna()
        & merged["interview_end"].notna()
        & (
            merged["interview_end"]
            < merged["interview_start"]
        )
    )

    print(
        f"End before start:    {invalid_order.sum():,}"
    )


# ---------------------------------------------------------
# 5. Overall fieldwork year distribution
# ---------------------------------------------------------

print("\nOVERALL FIELDWORK YEAR DISTRIBUTION")
print("-" * 110)

print(
    merged["fieldwork_year"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# 6. Country x interview year
# ---------------------------------------------------------

print("\nCOUNTRY x FIELDWORK YEAR")
print("-" * 110)

dated = merged.loc[
    merged["fieldwork_year"].notna()
].copy()

country_year = (
    dated
    .groupby(
        ["cntry", "country_name", "fieldwork_year"],
        as_index=False,
    )
    .agg(
        respondents=("idno", "size"),
        weight_sum=("anweight", "sum"),
    )
)

country_totals = (
    country_year
    .groupby(
        ["cntry", "country_name"],
        as_index=False,
    )
    .agg(
        valid_date_respondents=("respondents", "sum"),
        valid_date_weight_sum=("weight_sum", "sum"),
    )
)

country_year = country_year.merge(
    country_totals,
    on=["cntry", "country_name"],
    how="left",
    validate="many_to_one",
)

country_year["respondent_share"] = (
    country_year["respondents"]
    / country_year["valid_date_respondents"]
)

country_year["weighted_share"] = (
    country_year["weight_sum"]
    / country_year["valid_date_weight_sum"]
)

country_year = country_year.sort_values(
    ["country_name", "fieldwork_year"]
)

print(
    country_year.to_string(
        index=False,
        formatters={
            "respondent_share": "{:.4f}".format,
            "weighted_share": "{:.4f}".format,
        },
    )
)


# ---------------------------------------------------------
# 7. Country-level summary
# ---------------------------------------------------------

print("\nCOUNTRY FIELDWORK SUMMARY")
print("-" * 110)

summary_rows = []

for (cntry, country_name), group in merged.groupby(
    ["cntry", "country_name"]
):

    valid_group = group.loc[
        group["fieldwork_year"].notna()
    ]

    distribution = country_year.loc[
        country_year["cntry"] == cntry
    ]

    if distribution.empty:

        dominant_year = pd.NA
        dominant_share = pd.NA
        first_date = pd.NaT
        last_date = pd.NaT
        year_n = 0

    else:

        dominant = distribution.sort_values(
            ["weight_sum", "fieldwork_year"],
            ascending=[False, True],
        ).iloc[0]

        dominant_year = int(
            dominant["fieldwork_year"]
        )

        dominant_share = float(
            dominant["weighted_share"]
        )

        first_date = valid_group[
            "interview_start"
        ].min()

        last_date = valid_group[
            "interview_start"
        ].max()

        year_n = len(distribution)

    summary_rows.append(
        {
            "cntry": cntry,
            "country_name": country_name,
            "respondents": len(group),
            "valid_dates": len(valid_group),
            "missing_dates": (
                len(group) - len(valid_group)
            ),
            "years_n": year_n,
            "first_interview": first_date,
            "last_interview": last_date,
            "dominant_year": dominant_year,
            "dominant_weighted_share": dominant_share,
            "multi_year_flag": year_n > 1,
        }
    )


summary = pd.DataFrame(
    summary_rows
).sort_values("country_name")

print(
    summary.to_string(
        index=False,
        formatters={
            "dominant_weighted_share": (
                lambda x: (
                    "NA" if pd.isna(x)
                    else f"{x:.4f}"
                )
            ),
        },
    )
)


# ---------------------------------------------------------
# 8. Multi-year fieldwork
# ---------------------------------------------------------

print("\nMULTI-YEAR FIELDWORK COUNTRIES")
print("-" * 110)

multi_year = summary.loc[
    summary["multi_year_flag"]
]

if multi_year.empty:
    print("No multi-year fieldwork detected.")
else:
    print(
        multi_year[
            [
                "cntry",
                "country_name",
                "years_n",
                "dominant_year",
                "dominant_weighted_share",
            ]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 110)
print("FIELDWORK YEAR DIAGNOSTIC COMPLETED")
print("=" * 110)

print("No RAW or analytical data were modified.")
print("No Eurostat year was assigned.")
print("Weighted year shares use respondents with valid dates.")