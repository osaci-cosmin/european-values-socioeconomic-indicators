# Step 39: Create an explicit ESS-to-Eurostat country mapping.
# Keep all 29 ESS countries, including countries without Eurostat data.
# This step does not merge observations or align years.

from pathlib import Path

import pandas as pd


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

# Read the validated ESS respondent-level dataset.
ESS_PATH = (
    ROOT
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)

# Read the Eurostat long table validated in Step 38.
EUROSTAT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "eurostat_country_year_long.parquet"
)

# Save the mapping as a reusable project configuration.
OUTPUT_PATH = (
    ROOT
    / "config"
    / "ess_eurostat_country_mapping_v1.csv"
)


# These are the 29 countries already selected for ESS V1.
EXPECTED_ESS_COUNTRIES = {
    "AT", "BE", "BG", "CH", "CY", "DE", "EE",
    "ES", "FI", "FR", "GB", "GR", "HR", "HU",
    "IE", "IS", "IT", "LT", "LV", "ME", "NL",
    "NO", "PL", "PT", "RS", "SE", "SI", "SK",
    "UA",
}

# Only these two codes differ between the project sources.
CODE_OVERRIDES = {
    "GR": "EL",
    "GB": "UK",
}


def require(condition, message):
    # Stop immediately when an input or mapping check fails.
    if not condition:
        raise ValueError(message)


def main():
    print("STEP 39 — ESS–EUROSTAT COUNTRY MAPPING")

    # Never overwrite an existing configuration silently.
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Mapping already exists; inspect it before rerunning: "
            f"{OUTPUT_PATH}"
        )

    # Check inputs before attempting to read them.
    for path in (ESS_PATH, EUROSTAT_PATH):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    # Load only the country column to avoid reading unnecessary ESS data.
    ess = pd.read_parquet(
        ESS_PATH,
        columns=["cntry"],
        engine="pyarrow",
    )

    require(
        len(ess) == 49210,
        "Unexpected ESS V1 respondent count.",
    )

    require(
        ess["cntry"].notna().all(),
        "ESS contains missing country codes.",
    )

    # unique() extracts country codes without repeating them per respondent.
    observed_ess_countries = set(
        ess["cntry"].unique().tolist()
    )

    # Detect missing countries and unexpected additions to the V1 scope.
    require(
        observed_ess_countries == EXPECTED_ESS_COUNTRIES,
        "ESS country scope differs from the validated 29-country scope. "
        f"Missing: {sorted(EXPECTED_ESS_COUNTRIES - observed_ess_countries)}; "
        f"Unexpected: {sorted(observed_ess_countries - EXPECTED_ESS_COUNTRIES)}",
    )

    # Start from ESS so countries without Eurostat records are retained.
    mapping = pd.DataFrame(
        {
            "ess_cntry": sorted(observed_ess_countries),
        }
    )

    # get(code, code) applies an override or keeps the original code.
    mapping["eurostat_geo"] = mapping["ess_cntry"].map(
        lambda code: CODE_OVERRIDES.get(code, code)
    )

    # Record which country codes were changed.
    mapping["code_changed"] = (
        mapping["ess_cntry"] != mapping["eurostat_geo"]
    )

    # Read country codes from the validated Eurostat table.
    eurostat = pd.read_parquet(
        EUROSTAT_PATH,
        columns=["eurostat_geo"],
        engine="pyarrow",
    )

    require(
        eurostat["eurostat_geo"].notna().all(),
        "Eurostat contains missing country codes.",
    )

    eurostat_codes = set(
        eurostat["eurostat_geo"].unique().tolist()
    )

    # This flag means the code occurs somewhere in the extracted snapshots.
    # It does not mean all indicators or target years are available.
    mapping["eurostat_geo_present_in_snapshots"] = (
        mapping["eurostat_geo"].isin(eurostat_codes)
    )

    # Each ESS country must have exactly one mapping row.
    require(
        len(mapping) == 29
        and not mapping["ess_cntry"].duplicated().any(),
        "Expected one mapping row for each of the 29 ESS countries.",
    )

    # Different ESS countries must not map to the same Eurostat country.
    require(
        not mapping["eurostat_geo"].duplicated().any(),
        "Multiple ESS countries map to the same Eurostat code.",
    )

    require(
        mapping["eurostat_geo"].notna().all(),
        "Some ESS countries have no assigned Eurostat code.",
    )

    # Verify the two exceptions explicitly.
    actual_mapping = mapping.set_index(
        "ess_cntry"
    )["eurostat_geo"].to_dict()

    require(
        actual_mapping["GR"] == "EL",
        "Incorrect mapping for Greece.",
    )

    require(
        actual_mapping["GB"] == "UK",
        "Incorrect mapping for the United Kingdom.",
    )

    # Confirm that every other country retains its original code.
    for code in EXPECTED_ESS_COUNTRIES - set(CODE_OVERRIDES):
        require(
            actual_mapping[code] == code,
            f"Unexpected country-code change: {code}",
        )

    require(
        int(mapping["code_changed"].sum()) == 2,
        "Expected exactly two country-code changes.",
    )

    # The two alternative codes must actually occur in these snapshots.
    require(
        {"EL", "UK"}.issubset(eurostat_codes),
        "Expected Eurostat codes EL and UK were not found.",
    )

    # Assign explicit text types to the two identifier columns.
    for column in ("ess_cntry", "eurostat_geo"):
        mapping[column] = mapping[column].astype("string")

    print("\nCOUNTRY MAPPING")
    print(mapping.to_string(index=False))

    # Report source coverage without removing unmatched countries.
    absent = mapping.loc[
        ~mapping["eurostat_geo_present_in_snapshots"],
        ["ess_cntry", "eurostat_geo"],
    ]

    print("\nCOUNTRIES ABSENT FROM THE EUROSTAT SNAPSHOTS")

    if absent.empty:
        print("None.")
    else:
        print(absent.to_string(index=False))

    print("\nSUMMARY")
    print(f"ESS countries retained: {len(mapping)}")
    print(
        "Country codes changed: "
        f"{int(mapping['code_changed'].sum())}"
    )
    print(
        "Codes present in Eurostat snapshots: "
        f"{int(mapping['eurostat_geo_present_in_snapshots'].sum())}"
    )
    print(
        "Codes absent from Eurostat snapshots: "
        f"{len(absent)}"
    )

    # Create the config folder if necessary.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exclusive mode prevents overwriting an existing mapping file.
    with OUTPUT_PATH.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        mapping.to_csv(
            stream,
            index=False,
            lineterminator="\n",
        )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nESS–EUROSTAT COUNTRY MAPPING CREATED.")
    print("Independent mapping validation is still pending.")


# Run the workflow only when this script is executed directly.
if __name__ == "__main__":
    main()