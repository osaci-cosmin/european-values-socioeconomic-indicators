# Step 40: Independently validate the saved ESS–Eurostat country mapping.
# Read existing files only; do not modify or export datasets.
# Do not import functions from Step 39.

from pathlib import Path

import pandas as pd


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

MAPPING_PATH = (
    ROOT
    / "config"
    / "ess_eurostat_country_mapping_v1.csv"
)

ESS_PATH = (
    ROOT
    / "data"
    / "interim"
    / "ESS11_v1_analytical.parquet"
)

EUROSTAT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "eurostat_country_year_long.parquet"
)


# List every expected correspondence explicitly.
# This checks the saved result without reusing Step 39's override logic.
EXPECTED_MAPPING = {
    "AT": "AT",
    "BE": "BE",
    "BG": "BG",
    "CH": "CH",
    "CY": "CY",
    "DE": "DE",
    "EE": "EE",
    "ES": "ES",
    "FI": "FI",
    "FR": "FR",
    "GB": "UK",
    "GR": "EL",
    "HR": "HR",
    "HU": "HU",
    "IE": "IE",
    "IS": "IS",
    "IT": "IT",
    "LT": "LT",
    "LV": "LV",
    "ME": "ME",
    "NL": "NL",
    "NO": "NO",
    "PL": "PL",
    "PT": "PT",
    "RS": "RS",
    "SE": "SE",
    "SI": "SI",
    "SK": "SK",
    "UA": "UA",
}

EXPECTED_COLUMNS = {
    "ess_cntry",
    "eurostat_geo",
    "code_changed",
    "eurostat_geo_present_in_snapshots",
}

FLAG_COLUMNS = [
    "code_changed",
    "eurostat_geo_present_in_snapshots",
]


def require(condition, message):
    # Stop immediately if a validation condition is not satisfied.
    if not condition:
        raise ValueError(message)


def main():
    print("STEP 40 — ESS–EUROSTAT COUNTRY MAPPING VALIDATION")

    # Check input availability before reading any data.
    for path in (MAPPING_PATH, ESS_PATH, EUROSTAT_PATH):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    # Read all CSV fields as text so invalid tokens remain detectable.
    # Empty fields stay empty strings instead of becoming missing values.
    mapping = pd.read_csv(
        MAPPING_PATH,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    # Reject missing or additional columns.
    require(
        len(mapping.columns) == len(EXPECTED_COLUMNS)
        and set(mapping.columns) == EXPECTED_COLUMNS,
        "Unexpected mapping columns.",
    )

    require(
        len(mapping) == 29,
        "Expected exactly 29 mapping rows.",
    )

    # Empty fields are not valid codes or Boolean flags.
    require(
        not mapping.eq("").any().any(),
        "The mapping contains empty fields.",
    )

    # Do not silently repair spaces during validation.
    # Detect them so the saved configuration can be investigated.
    for column in ("ess_cntry", "eurostat_geo"):
        require(
            mapping[column].eq(
                mapping[column].str.strip()
            ).all(),
            f"Unexpected whitespace in {column}.",
        )

    # Each ESS country must have exactly one mapping row.
    require(
        not mapping["ess_cntry"].duplicated().any(),
        "Duplicated ESS country codes.",
    )

    # Two ESS countries must not point to the same Eurostat country.
    require(
        not mapping["eurostat_geo"].duplicated().any(),
        "Duplicated Eurostat country codes.",
    )

    print("\nSchema, row count and code uniqueness: PASSED.")

    # Convert the saved pairs into a dictionary for exact comparison.
    actual_mapping = mapping.set_index(
        "ess_cntry"
    )["eurostat_geo"].to_dict()

    require(
        actual_mapping == EXPECTED_MAPPING,
        "Saved country pairs differ from the expected V1 mapping.",
    )

    print("All 29 country-code correspondences: PASSED.")

    # CSV stores Boolean values as text, so validate tokens first.
    for column in FLAG_COLUMNS:
        require(
            mapping[column].isin(["True", "False"]).all(),
            f"Invalid Boolean token in {column}.",
        )

        # Explicit mapping avoids treating the text 'False' as True.
        mapping[column] = mapping[column].map(
            {
                "True": True,
                "False": False,
            }
        ).astype("bool")

    # Recalculate code changes directly from the two code columns.
    expected_changes = (
        mapping["ess_cntry"] != mapping["eurostat_geo"]
    )

    require(
        mapping["code_changed"].eq(expected_changes).all(),
        "code_changed does not match the saved country codes.",
    )

    require(
        int(mapping["code_changed"].sum()) == 2,
        "Expected exactly two changed country codes.",
    )

    print("Code-change flags: PASSED.")

    # Load only the ESS column needed for this validation.
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

    # Compare the saved mapping scope with the actual ESS country scope.
    require(
        set(ess["cntry"].unique())
        == set(mapping["ess_cntry"]),
        "Mapping countries do not match the ESS V1 dataset.",
    )

    print("Mapping coverage of the actual ESS dataset: PASSED.")

    # Check presence against the long table validated in Step 38.
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
        eurostat["eurostat_geo"].unique()
    )

    # Presence means the code occurs somewhere in the snapshots.
    # It does not imply numeric availability for every indicator or year.
    expected_presence = mapping["eurostat_geo"].isin(
        eurostat_codes
    )

    require(
        mapping["eurostat_geo_present_in_snapshots"].eq(
            expected_presence
        ).all(),
        "Snapshot-presence flags disagree with the Eurostat table.",
    )

    print("Eurostat snapshot-presence flags: PASSED.")

    # Perform a temporary join to check that the mapping is safe to use.
    # many_to_one allows many respondents but only one mapping per country.
    linked = ess.merge(
        mapping[["ess_cntry", "eurostat_geo"]],
        how="left",
        left_on="cntry",
        right_on="ess_cntry",
        validate="many_to_one",
        indicator=True,
    )

    # A correct mapping must neither lose nor multiply respondents.
    require(
        len(linked) == len(ess),
        "The mapping join changed the respondent count.",
    )

    # Every ESS respondent must match a mapping row, including Ukraine.
    require(
        linked["_merge"].eq("both").all(),
        "Some ESS respondents have no mapping match.",
    )

    require(
        linked["eurostat_geo"].notna().all(),
        "Some respondents have no assigned Eurostat code.",
    )

    # Verify that each country's respondent count is also preserved.
    before = ess["cntry"].value_counts().to_dict()
    after = linked["cntry"].value_counts().to_dict()

    require(
        before == after,
        "Country-level respondent counts changed after the join.",
    )

    print("Temporary join: all 49,210 ESS respondents retained.")
    print("Respondent counts by country: PASSED.")

    # Report the validated exceptions to identical country codes.
    changed = mapping.loc[
        mapping["code_changed"],
        ["ess_cntry", "eurostat_geo"],
    ]

    print("\nVALIDATED CODE CHANGES")
    print(changed.to_string(index=False))

    # Report countries that remain mapped despite absent Eurostat records.
    absent = mapping.loc[
        ~mapping["eurostat_geo_present_in_snapshots"],
        ["ess_cntry", "eurostat_geo"],
    ]

    print("\nCOUNTRIES ABSENT FROM EUROSTAT SNAPSHOTS")

    if absent.empty:
        print("None.")
    else:
        print(absent.to_string(index=False))

    print("\nSUMMARY")
    print(f"Mapping rows: {len(mapping)}")
    print(f"ESS respondents retained in temporary join: {len(linked)}")
    print(f"Changed country codes: {len(changed)}")
    print(
        "Codes present in Eurostat snapshots: "
        f"{int(expected_presence.sum())}"
    )
    print(f"Codes absent from Eurostat snapshots: {len(absent)}")

    # Print success only after every validation has passed.
    print("\nESS–EUROSTAT COUNTRY MAPPING VALIDATION PASSED.")
    print("The temporary join was performed in memory only.")
    print("No files were modified.")


# Run validation only when this file is executed directly.
if __name__ == "__main__":
    main()