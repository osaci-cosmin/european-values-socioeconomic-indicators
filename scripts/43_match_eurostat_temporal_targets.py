# Step 43: Match validated temporal requests to Eurostat observations.
# Keep every request, including requests without available values.
# Do not replace missing observations with another year or scenario.

import hashlib
import io
from pathlib import Path

import pandas as pd


# Locate the project from this script inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

TARGETS_PATH = (
    ROOT
    / "data"
    / "interim"
    / "ess_eurostat_indicator_temporal_targets_v1.parquet"
)

EUROSTAT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "eurostat_country_year_long.parquet"
)

OUTPUT_DIR = ROOT / "data" / "interim"

PARQUET_PATH = (
    OUTPUT_DIR
    / "ess_eurostat_temporal_matches_v1.parquet"
)

CSV_PATH = (
    OUTPUT_DIR
    / "ess_eurostat_temporal_matches_v1.csv"
)


# Identify one analytical request.
REQUEST_KEY = [
    "cntry",
    "indicator_id",
    "alignment_scenario",
]

# Identify one observation in the Eurostat long table.
SOURCE_KEY = [
    "indicator_id",
    "eurostat_geo",
    "eurostat_year",
]

# These definitions must agree between targets and source observations.
DEFINITION_COLUMNS = [
    "dataset_code",
    "indicator_label",
    "unit_label",
    "reference_year_offset",
    "reference_period_rule",
    "registry_sha256",
]

# Record which snapshot was searched, even when a cell is absent.
LOOKUP_COLUMNS = [
    "snapshot_sha256",
    "raw_filename",
    "metadata_filename",
    "source_api_url",
    "api_dataset_updated",
    "retrieved_at_utc",
]

MATCH_STATUSES = [
    "available",
    "source_value_missing",
    "source_cell_absent",
]


def require(condition, message):
    # Stop before export if an assumption required by the join fails.
    if not condition:
        raise ValueError(message)


def read_parquet_snapshot(path):
    # Hash and read the same bytes to identify the exact input version.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    table = pd.read_parquet(
        io.BytesIO(content),
        engine="pyarrow",
    )

    return table, digest


def main():
    print("STEP 43 — EUROSTAT TEMPORAL OBSERVATION MATCHING")

    # Protect existing outputs from accidental overwriting.
    for path in (PARQUET_PATH, CSV_PATH):
        if path.exists():
            raise FileExistsError(
                "Output already exists; inspect it before rerunning: "
                f"{path}"
            )

    for path in (TARGETS_PATH, EUROSTAT_PATH):
        require(
            path.is_file(),
            f"Required input not found: {path}",
        )

    targets, targets_hash = read_parquet_snapshot(TARGETS_PATH)
    source, source_hash = read_parquet_snapshot(EUROSTAT_PATH)

    # Confirm that the expected validated inputs were supplied.
    require(
        len(targets) == 252,
        "Expected 252 temporal requests.",
    )

    require(
        len(source) == 481,
        "Expected 481 Eurostat source cells.",
    )

    require(
        not targets.duplicated(REQUEST_KEY).any(),
        "Duplicate temporal request keys.",
    )

    require(
        not source.duplicated(SOURCE_KEY).any(),
        "Duplicate Eurostat observation keys.",
    )

    require(
        targets["alignment_scenario"].value_counts().to_dict()
        == {"primary": 174, "alternative": 78},
        "Unexpected temporal scenario counts.",
    )

    # Empty join keys must never participate in observation matching.
    require(
        targets[
            ["indicator_id", "eurostat_geo", "requested_eurostat_year"]
        ].notna().all().all(),
        "Missing temporal join keys.",
    )

    require(
        source[SOURCE_KEY].notna().all().all(),
        "Missing Eurostat source keys.",
    )

    require(
        set(targets["indicator_id"]) == set(source["indicator_id"]),
        "Target and source indicator sets differ.",
    )

    require(
        targets["alignment_stage"].eq(
            "year_target_created_observation_match_pending"
        ).all(),
        "Unexpected input alignment stage.",
    )

    # Check the stored source flags before using them.
    require(
        source["is_missing"].eq(source["value"].isna()).all(),
        "Source missing-value flags disagree with source values.",
    )

    require(
        source["eurostat_status"].notna().all(),
        "Source status cells must use empty text for no flag.",
    )

    require(
        source["has_status_flag"].eq(
            source["eurostat_status"].ne("")
        ).all(),
        "Source status flags disagree with source status text.",
    )

    # Build one metadata record for each indicator's searched snapshot.
    metadata_columns = (
        ["indicator_id"]
        + DEFINITION_COLUMNS
        + LOOKUP_COLUMNS
    )

    require(
        source[metadata_columns].notna().all().all(),
        "Missing indicator definitions or snapshot provenance.",
    )

    manifest = source[metadata_columns].drop_duplicates()

    # Multiple metadata records for one indicator would be ambiguous.
    require(
        len(manifest) == 6
        and not manifest["indicator_id"].duplicated().any(),
        "Expected one consistent source snapshot per indicator.",
    )

    # Prefix snapshot fields to distinguish lookup metadata from target fields.
    manifest = manifest.rename(
        columns={
            column: f"lookup_{column}"
            for column in manifest.columns
            if column != "indicator_id"
        }
    )

    # Retain the status of the incoming target table as historical metadata.
    prepared = targets.rename(
        columns={"alignment_stage": "input_alignment_stage"}
    )

    # Attach lookup provenance to every request, even an unmatched request.
    prepared = prepared.merge(
        manifest,
        how="left",
        on="indicator_id",
        validate="many_to_one",
    )

    require(
        prepared["lookup_snapshot_sha256"].notna().all(),
        "Some requests have no identified source snapshot.",
    )

    # Compare definitions for every request, including absent source cells.
    for column in DEFINITION_COLUMNS:
        require(
            prepared[column].eq(
                prepared[f"lookup_{column}"]
            ).all(),
            f"Target/source definition mismatch: {column}",
        )

    # Definitions agree, so keep the target copy and remove duplicate copies.
    prepared = prepared.drop(
        columns=[
            f"lookup_{column}"
            for column in DEFINITION_COLUMNS
        ]
    )

    # Select only observation-level fields for the actual value join.
    observations = source[
        SOURCE_KEY
        + [
            "eurostat_geo_label",
            "value",
            "eurostat_status",
            "is_missing",
            "has_status_flag",
        ]
    ].copy()

    # Prefix source-specific fields so their origin remains explicit.
    observations = observations.rename(
        columns={
            "eurostat_geo_label": "source_geo_label",
            "is_missing": "source_is_missing",
            "has_status_flag": "source_has_status_flag",
        }
    )

    # Nullable Boolean types preserve an unknown flag for an absent cell.
    for column in (
        "source_is_missing",
        "source_has_status_flag",
    ):
        observations[column] = observations[column].astype("boolean")

    # Use nullable integers because unmatched requests have no source year.
    observations["eurostat_year"] = (
        observations["eurostat_year"].astype("Int64")
    )

    # A left join retains all requests.
    # many_to_one prevents duplicate source keys from multiplying requests.
    matched = prepared.merge(
        observations,
        how="left",
        left_on=[
            "indicator_id",
            "eurostat_geo",
            "requested_eurostat_year",
        ],
        right_on=SOURCE_KEY,
        validate="many_to_one",
        indicator="_source_merge",
    )

    require(
        len(matched) == len(targets),
        "The observation join changed the request count.",
    )

    require(
        not matched.duplicated(REQUEST_KEY).any(),
        "The observation join duplicated request keys.",
    )

    # Detect whether the requested cell exists, independently of its value.
    matched["source_cell_found"] = (
        matched["_source_merge"].eq("both")
    )

    found = matched["source_cell_found"]

    # A zero is numeric and available; only missing values are unavailable.
    matched["has_numeric_value"] = matched["value"].notna()

    # Start with absent cells, then classify the two matched-cell outcomes.
    matched["observation_match_status"] = "source_cell_absent"

    matched.loc[
        found & ~matched["has_numeric_value"],
        "observation_match_status",
    ] = "source_value_missing"

    matched.loc[
        found & matched["has_numeric_value"],
        "observation_match_status",
    ] = "available"

    # Every found cell must match the exact requested Eurostat year.
    require(
        matched.loc[found, "eurostat_year"].eq(
            matched.loc[found, "requested_eurostat_year"]
        ).all(),
        "A source observation has a different year from the request.",
    )

    # Check the source missing flags only where a source cell exists.
    require(
        matched.loc[found, "source_is_missing"].notna().all(),
        "Missing source flags for matched cells.",
    )

    require(
        matched.loc[found, "source_is_missing"].eq(
            matched.loc[found, "value"].isna()
        ).all(),
        "Matched source missing flags disagree with values.",
    )

    # Absent cells must not acquire invented observation-level metadata.
    absent_columns = [
        "eurostat_year",
        "source_geo_label",
        "value",
        "eurostat_status",
        "source_is_missing",
        "source_has_status_flag",
    ]

    require(
        matched.loc[~found, absent_columns].isna().all().all(),
        "Absent source cells contain unexpected observation data.",
    )

    require(
        matched["observation_match_status"].isin(
            MATCH_STATUSES
        ).all(),
        "Unexpected observation-match status.",
    )

    # The year gap is only meaningful when an actual source cell was found.
    matched["matched_year_gap"] = (
        matched["eurostat_year"]
        - matched["requested_eurostat_year"]
    ).astype("Int64")

    require(
        matched.loc[found, "matched_year_gap"].eq(0).all(),
        "Non-zero year gaps detected.",
    )

    # Update the processing stage without claiming independent validation.
    matched["alignment_stage"] = (
        "observation_match_completed_validation_pending"
    )

    # Record the exact input table versions used by this matching operation.
    matched["temporal_targets_sha256"] = targets_hash
    matched["eurostat_long_sha256"] = source_hash

    # The merge indicator has served its purpose; keep our explicit flag.
    matched = matched.drop(columns="_source_merge")

    matched["value"] = matched["value"].astype("float64")

    # Standardize text fields without changing missing source information.
    text_columns = matched.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        matched[column] = matched[column].astype("string")

    matched = matched.sort_values(
        REQUEST_KEY
    ).reset_index(drop=True)

    # Count each availability category within each scenario and indicator.
    summary = (
        matched.groupby(
            [
                "alignment_scenario",
                "indicator_id",
                "observation_match_status",
            ]
        )
        .size()
        .unstack("observation_match_status", fill_value=0)
        .reindex(columns=MATCH_STATUSES, fill_value=0)
    )

    summary["requests"] = summary[MATCH_STATUSES].sum(axis=1)

    require(
        int(summary["requests"].sum()) == 252,
        "Availability summary does not account for every request.",
    )

    print("\nAVAILABILITY BY SCENARIO AND INDICATOR")
    print(summary.to_string())

    # Show every unavailable request with its specific reason.
    unavailable = matched.loc[
        ~matched["has_numeric_value"],
        [
            "cntry",
            "eurostat_geo",
            "indicator_id",
            "alignment_scenario",
            "context_year",
            "requested_eurostat_year",
            "observation_match_status",
            "eurostat_status",
        ],
    ]

    print("\nREQUESTS WITHOUT A NUMERIC VALUE")

    if unavailable.empty:
        print("None.")
    else:
        print(unavailable.to_string(index=False))

    # Report status flags separately from numeric availability.
    # A source cell can have a status even when its value is missing.
    flagged = (
        matched["source_has_status_flag"]
        .fillna(False)
        .astype("bool")
    )

    print("\nSUMMARY")
    print(f"Requests retained: {len(matched)}")
    print(
        "Source cells found: "
        f"{int(matched['source_cell_found'].sum())}"
    )
    print(
        "Available numeric values: "
        f"{int(matched['has_numeric_value'].sum())}"
    )
    print(
        "Found cells with missing values: "
        f"{int(matched['observation_match_status'].eq('source_value_missing').sum())}"
    )
    print(
        "Absent source cells: "
        f"{int(matched['observation_match_status'].eq('source_cell_absent').sum())}"
    )
    print(f"Requests matched to flagged cells: {int(flagged.sum())}")

    # Prepare both exports before creating output files.
    parquet_bytes = matched.to_parquet(
        index=False,
        engine="pyarrow",
    )

    csv_text = matched.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive modes refuse to overwrite existing results.
    with PARQUET_PATH.open("xb") as stream:
        stream.write(parquet_bytes)

    with CSV_PATH.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        stream.write(csv_text)

    print(f"\nSaved: {PARQUET_PATH}")
    print(f"Saved: {CSV_PATH}")

    print("\nEUROSTAT TEMPORAL OBSERVATION MATCHING COMPLETED.")
    print("All 252 requests were retained.")
    print("No alternative-year substitution or imputation was performed.")
    print("Independent match validation is still pending.")


# Run the matching workflow only when this script is executed directly.
if __name__ == "__main__":
    main()