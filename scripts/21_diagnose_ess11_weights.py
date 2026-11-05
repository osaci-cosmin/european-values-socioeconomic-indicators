from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root directory
project_root = Path(__file__).resolve().parents[1]

# Define the analytical respondent-level dataset
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


# Load the dataset
df = pd.read_parquet(input_file)


# Define ESS weighting variables
weight_variables = [
    "dweight",
    "pspwght",
    "pweight",
    "anweight",
]


print("=" * 100)
print("ESS ROUND 11 SURVEY WEIGHT DIAGNOSTIC")
print("=" * 100)

print()
print(f"Respondents: {len(df):,}")
print(f"Countries:   {df['cntry'].nunique()}")


# ---------------------------------------------------------------------
# 1. Basic weight validation
# ---------------------------------------------------------------------

print()
print("BASIC WEIGHT VALIDATION")
print("-" * 100)

for variable in weight_variables:

    missing_n = df[variable].isna().sum()
    zero_n = (df[variable] == 0).sum()
    negative_n = (df[variable] < 0).sum()

    print(
        f"{variable:<10} "
        f"Missing: {missing_n:>5,} | "
        f"Zero: {zero_n:>5,} | "
        f"Negative: {negative_n:>5,}"
    )


# ---------------------------------------------------------------------
# 2. Validate the analysis-weight identity
# ---------------------------------------------------------------------

print()
print("ANALYSIS WEIGHT IDENTITY")
print("-" * 100)

expected_anweight = (
    df["pspwght"]
    * df["pweight"]
)

identity_valid = np.allclose(
    df["anweight"],
    expected_anweight,
    rtol=1e-10,
    atol=1e-12,
)

max_difference = (
    df["anweight"]
    - expected_anweight
).abs().max()

print(
    f"anweight = pspwght * pweight: "
    f"{'PASSED' if identity_valid else 'FAILED'}"
)

print(
    f"Maximum absolute difference:  "
    f"{max_difference:.12f}"
)


# ---------------------------------------------------------------------
# 3. Check whether population weight is constant within country
# ---------------------------------------------------------------------

print()
print("POPULATION WEIGHT WITHIN COUNTRIES")
print("-" * 100)

pweight_unique = (
    df
    .groupby(
        ["cntry", "country_name"]
    )["pweight"]
    .nunique()
    .reset_index(
        name="unique_pweight_values"
    )
)


print(
    pweight_unique.to_string(
        index=False
    )
)

pweight_constant = (
    pweight_unique[
        "unique_pweight_values"
    ]
    == 1
).all()

print()

print(
    f"One pweight value per country: "
    f"{'PASSED' if pweight_constant else 'FAILED'}"
)


# ---------------------------------------------------------------------
# 4. Country-level weight summary
# ---------------------------------------------------------------------

print()
print("COUNTRY-LEVEL WEIGHT SUMMARY")
print("-" * 100)

country_summary = (
    df
    .groupby(
        ["cntry", "country_name"],
        as_index=False
    )
    .agg(
        respondents=("idno", "size"),

        dweight_mean=("dweight", "mean"),
        dweight_min=("dweight", "min"),
        dweight_max=("dweight", "max"),

        pspwght_mean=("pspwght", "mean"),
        pspwght_min=("pspwght", "min"),
        pspwght_max=("pspwght", "max"),

        pweight=("pweight", "first"),

        anweight_mean=("anweight", "mean"),
        anweight_min=("anweight", "min"),
        anweight_max=("anweight", "max"),
    )
)


print(
    country_summary.to_string(
        index=False,
        formatters={
            "dweight_mean": "{:.4f}".format,
            "dweight_min": "{:.4f}".format,
            "dweight_max": "{:.4f}".format,
            "pspwght_mean": "{:.4f}".format,
            "pspwght_min": "{:.4f}".format,
            "pspwght_max": "{:.4f}".format,
            "pweight": "{:.4f}".format,
            "anweight_mean": "{:.4f}".format,
            "anweight_min": "{:.4f}".format,
            "anweight_max": "{:.4f}".format,
        }
    )
)


# ---------------------------------------------------------------------
# 5. Compare design and post-stratification weights
# ---------------------------------------------------------------------

print()
print("DWEIGHT VS PSPWGHT")
print("-" * 100)

comparison_results = []

for (
    cntry,
    country_name
), group in df.groupby(
    ["cntry", "country_name"]
):

    weights_equal = np.allclose(
        group["dweight"],
        group["pspwght"],
        rtol=1e-10,
        atol=1e-12,
    )

    mean_absolute_difference = (
        group["dweight"]
        - group["pspwght"]
    ).abs().mean()

    comparison_results.append(
        {
            "cntry": cntry,
            "country_name": country_name,
            "dweight_equals_pspwght": weights_equal,
            "mean_absolute_difference": mean_absolute_difference,
        }
    )


weight_comparison = pd.DataFrame(
    comparison_results
)


print(
    weight_comparison.to_string(
        index=False,
        formatters={
            "mean_absolute_difference": "{:.6f}".format
        }
    )
)


# ---------------------------------------------------------------------
# 6. Inspect Ukraine specifically
# ---------------------------------------------------------------------

print()
print("UKRAINE WEIGHT DIAGNOSTIC")
print("-" * 100)

ukraine = df[
    df["cntry"] == "UA"
].copy()


if ukraine.empty:

    print("Ukraine is not present in the V1 dataset.")

else:

    ua_design_equals_psp = np.allclose(
        ukraine["dweight"],
        ukraine["pspwght"],
        rtol=1e-10,
        atol=1e-12,
    )

    ua_analysis_identity = np.allclose(
        ukraine["anweight"],
        (
            ukraine["pspwght"]
            * ukraine["pweight"]
        ),
        rtol=1e-10,
        atol=1e-12,
    )

    print(
        f"dweight = pspwght:             "
        f"{'YES' if ua_design_equals_psp else 'NO'}"
    )

    print(
        f"anweight = pspwght * pweight: "
        f"{'YES' if ua_analysis_identity else 'NO'}"
    )

    print()

    print(
        ukraine[
            weight_variables
        ]
        .describe()
        .to_string()
    )


# ---------------------------------------------------------------------
# 7. Sample-design variables
# ---------------------------------------------------------------------

print()
print("COMPLEX SAMPLE DESIGN")
print("-" * 100)

print(
    f"Missing PSU values:     "
    f"{df['psu'].isna().sum():,}"
)

print(
    f"Missing STRATUM values: "
    f"{df['stratum'].isna().sum():,}"
)

print(
    f"Unique PSUs:            "
    f"{df['psu'].nunique():,}"
)

print(
    f"Unique strata:          "
    f"{df['stratum'].nunique():,}"
)


print()
print("=" * 100)
print("WEIGHT DIAGNOSTIC COMPLETED")
print("=" * 100)

print(
    "No survey weights were modified."
)

print(
    "No weighted estimates were calculated."
)