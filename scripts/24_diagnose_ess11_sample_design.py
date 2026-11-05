from pathlib import Path

import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define input file
input_file = (
    project_root
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)

# Check that the input file exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Analytical dataset not found: {input_file}"
    )


# Load respondent-level analytical data
df = pd.read_parquet(input_file)


# Define required sample-design variables
required_columns = [
    "cntry",
    "country_name",
    "idno",
    "psu",
    "stratum",
    "anweight",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("=" * 110)
print("ESS ROUND 11 SAMPLE DESIGN DIAGNOSTIC")
print("=" * 110)

print()
print(f"Respondents: {len(df):,}")
print(f"Countries:   {df['cntry'].nunique()}")


# ---------------------------------------------------------------------
# 1. Missing sample-design information
# ---------------------------------------------------------------------

print()
print("SAMPLE-DESIGN COMPLETENESS")
print("-" * 110)

for variable in [
    "psu",
    "stratum",
    "anweight",
]:

    missing_n = df[variable].isna().sum()

    print(
        f"{variable:<10} "
        f"Missing: {missing_n:,}"
    )


# ---------------------------------------------------------------------
# 2. Check whether PSU identifiers are nested within strata
# ---------------------------------------------------------------------

print()
print("PSU NESTING CHECK")
print("-" * 110)

psu_stratum_count = (
    df
    .groupby(
        ["cntry", "psu"],
        dropna=False
    )["stratum"]
    .nunique()
)

non_nested_psus = (
    psu_stratum_count > 1
).sum()

print(
    f"Country-PSU combinations appearing in >1 stratum: "
    f"{non_nested_psus:,}"
)

if non_nested_psus == 0:
    print(
        "PSU identifiers are nested within one stratum "
        "inside each country: PASSED"
    )
else:
    print(
        "Some PSU identifiers occur in multiple strata."
    )

print(
    "For variance estimation, the safe PSU key will be "
    "country + stratum + PSU."
)


# ---------------------------------------------------------------------
# 3. Number of PSUs in every country-stratum
# ---------------------------------------------------------------------

stratum_structure = (
    df
    .groupby(
        [
            "cntry",
            "country_name",
            "stratum",
        ],
        dropna=False
    )
    .agg(
        respondents=("idno", "size"),
        psu_n=("psu", "nunique"),
    )
    .reset_index()
)


# Identify singleton strata
stratum_structure["singleton_stratum"] = (
    stratum_structure["psu_n"] == 1
)


# ---------------------------------------------------------------------
# 4. Overall design summary
# ---------------------------------------------------------------------

print()
print("OVERALL DESIGN STRUCTURE")
print("-" * 110)

total_strata = len(
    stratum_structure
)

total_psus = (
    stratum_structure["psu_n"]
    .sum()
)

singleton_strata_n = (
    stratum_structure[
        "singleton_stratum"
    ]
    .sum()
)

singleton_respondents_n = (
    stratum_structure.loc[
        stratum_structure[
            "singleton_stratum"
        ],
        "respondents",
    ]
    .sum()
)

singleton_strata_rate = (
    singleton_strata_n
    / total_strata
    * 100
)

singleton_respondent_rate = (
    singleton_respondents_n
    / len(df)
    * 100
)


print(
    f"Country-specific strata:       "
    f"{total_strata:,}"
)

print(
    f"Country-stratum PSUs:          "
    f"{total_psus:,}"
)

print(
    f"Singleton strata:              "
    f"{singleton_strata_n:,} "
    f"({singleton_strata_rate:.2f}%)"
)

print(
    f"Respondents in singleton strata: "
    f"{singleton_respondents_n:,} "
    f"({singleton_respondent_rate:.2f}%)"
)


# ---------------------------------------------------------------------
# 5. Country-level sample-design summary
# ---------------------------------------------------------------------

country_design_rows = []

for (
    cntry,
    country_name
), country_strata in stratum_structure.groupby(
    [
        "cntry",
        "country_name",
    ]
):

    country_data = df[
        df["cntry"] == cntry
    ]

    singleton_rows = country_strata[
        country_strata[
            "singleton_stratum"
        ]
    ]

    country_design_rows.append(
        {
            "cntry": cntry,
            "country_name": country_name,

            "respondents": len(
                country_data
            ),

            "strata_n": len(
                country_strata
            ),

            "psu_n": (
                country_strata[
                    "psu_n"
                ]
                .sum()
            ),

            "min_psu_per_stratum": (
                country_strata[
                    "psu_n"
                ]
                .min()
            ),

            "median_psu_per_stratum": (
                country_strata[
                    "psu_n"
                ]
                .median()
            ),

            "max_psu_per_stratum": (
                country_strata[
                    "psu_n"
                ]
                .max()
            ),

            "singleton_strata_n": (
                len(singleton_rows)
            ),

            "singleton_respondents_n": (
                singleton_rows[
                    "respondents"
                ]
                .sum()
            ),
        }
    )


country_design = pd.DataFrame(
    country_design_rows
)


country_design[
    "singleton_respondent_rate"
] = (
    country_design[
        "singleton_respondents_n"
    ]
    / country_design[
        "respondents"
    ]
)


print()
print("COUNTRY-LEVEL SAMPLE DESIGN")
print("-" * 110)

print(
    country_design.to_string(
        index=False,
        formatters={
            "median_psu_per_stratum": "{:.1f}".format,
            "singleton_respondent_rate": "{:.4f}".format,
        }
    )
)


# ---------------------------------------------------------------------
# 6. Countries containing singleton strata
# ---------------------------------------------------------------------

print()
print("COUNTRIES WITH SINGLETON STRATA")
print("-" * 110)

countries_with_singletons = (
    country_design[
        country_design[
            "singleton_strata_n"
        ] > 0
    ]
    .copy()
)


if countries_with_singletons.empty:

    print(
        "No singleton strata detected."
    )

else:

    print(
        countries_with_singletons[
            [
                "cntry",
                "country_name",
                "strata_n",
                "singleton_strata_n",
                "singleton_respondents_n",
                "singleton_respondent_rate",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "singleton_respondent_rate": "{:.4f}".format,
            }
        )
    )


# ---------------------------------------------------------------------
# 7. Distribution of PSUs per stratum
# ---------------------------------------------------------------------

print()
print("PSU COUNT PER STRATUM")
print("-" * 110)

print(
    stratum_structure[
        "psu_n"
    ]
    .describe()
    .to_string()
)


# ---------------------------------------------------------------------
# 8. Very small strata
# ---------------------------------------------------------------------

print()
print("SMALL STRATA")
print("-" * 110)

for threshold in [
    1,
    2,
    3,
    5,
]:

    count = (
        stratum_structure[
            "psu_n"
        ] <= threshold
    ).sum()

    print(
        f"Strata with <= {threshold} PSU(s): "
        f"{count:,}"
    )


# ---------------------------------------------------------------------
# 9. Check respondent counts inside strata
# ---------------------------------------------------------------------

print()
print("RESPONDENTS PER STRATUM")
print("-" * 110)

print(
    stratum_structure[
        "respondents"
    ]
    .describe()
    .to_string()
)


single_respondent_strata = (
    stratum_structure[
        "respondents"
    ] == 1
).sum()

print()

print(
    f"Strata containing only one respondent: "
    f"{single_respondent_strata:,}"
)


print()
print("=" * 110)
print("SAMPLE DESIGN DIAGNOSTIC COMPLETED")
print("=" * 110)

print(
    "No respondent data were modified."
)

print(
    "No standard errors or confidence intervals were calculated."
)