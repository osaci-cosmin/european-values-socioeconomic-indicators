# Step 51: Plot distributions of the validated country-level indicators.
# Each point represents one available country.
# Boxplots describe between-country variation, not survey uncertainty.

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        'Install plotting support: python -m pip install "matplotlib>=3.10"'
    ) from exc

# Use a file-based backend so no interactive window is required.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Resolve paths from the script location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

# This hash identifies the input confirmed in Step 50.
EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# A fixed seed makes the visual point separation reproducible.
RANDOM_SEED = 5100

# Each definition contains: column, title, axis label, reference line.
# A reference line provides context; it is not a statistical threshold.
ESS_PANELS = [
    (
        "mean_social_trust",
        "Generalized social trust",
        "Weighted country mean | scale 0–10",
        None,
    ),
    (
        "mean_left_right",
        "Ideological self-placement",
        "Weighted country mean | 0 = left, 10 = right",
        None,
    ),
    (
        "mean_openness",
        "Openness to Change",
        "Weighted country mean | MRAT-centered score",
        0.0,
    ),
    (
        "mean_conservation",
        "Conservation",
        "Weighted country mean | MRAT-centered score",
        0.0,
    ),
    (
        "mean_self_transcendence",
        "Self-Transcendence",
        "Weighted country mean | MRAT-centered score",
        0.0,
    ),
    (
        "mean_self_enhancement",
        "Self-Enhancement",
        "Weighted country mean | MRAT-centered score",
        0.0,
    ),
]

ECONOMIC_PANELS = [
    (
        "gdp_pps_index",
        "GDP per capita in PPS",
        "Index | EU27_2020 = 100",
        100.0,
    ),
    (
        "employment_rate",
        "Employment rate",
        "% of population aged 20–64",
        None,
    ),
    (
        "unemployment_rate",
        "Unemployment rate",
        "% of labour force aged 15–74",
        None,
    ),
    (
        "gini",
        "Income inequality",
        "Gini coefficient | scale 0–100",
        None,
    ),
    (
        "material_deprivation_rate",
        "Severe material and social deprivation",
        "% of population",
        None,
    ),
    (
        "inflation_rate",
        "Annual HICP inflation",
        "Annual average rate of change (%)",
        None,
    ),
]

FIGURE_GROUPS = [
    (
        "51_ess_country_distributions_v1",
        "ESS Round 11 | Country distributions",
        ESS_PANELS,
        "#2166AC",
    ),
    (
        "51_eurostat_country_distributions_v1",
        "Eurostat | Country distributions",
        ECONOMIC_PANELS,
        "#087F6D",
    ),
]

METADATA_PATH = OUTPUT_DIR / "51_country_distributions_v1.metadata.json"


def require(condition, message):
    # Stop explicitly when an input or output condition is not satisfied.
    if not condition:
        raise ValueError(message)


def check_plotting_version():
    # The horizontal orientation argument requires Matplotlib 3.10 or newer.
    version_parts = matplotlib.__version__.split(".")
    major_minor = tuple(int(part) for part in version_parts[:2])

    require(
        major_minor >= (3, 10),
        'Update Matplotlib: python -m pip install "matplotlib>=3.10"',
    )


def check_output_paths():
    # Check every planned output before creating any figure files.
    paths = [
        OUTPUT_DIR / f"{stem}.{extension}"
        for stem, _, _, _ in FIGURE_GROUPS
        for extension in ("png", "pdf")
    ]
    paths.append(METADATA_PATH)

    existing = [str(path) for path in paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Inspect them before rerunning:\n"
            + "\n".join(existing)
        )


def load_input():
    require(INPUT_PATH.is_file(), f"Input file not found: {INPUT_PATH}")

    # Read once so the hash and the DataFrame refer to the same bytes.
    content = INPUT_PATH.read_bytes()
    input_sha256 = hashlib.sha256(content).hexdigest()

    require(
        input_sha256 == EXPECTED_INPUT_SHA256,
        "Input hash differs from Step 50. Investigate before plotting.",
    )

    data = pd.read_parquet(io.BytesIO(content), engine="pyarrow")

    variables = [
        panel[0]
        for panel in ESS_PANELS + ECONOMIC_PANELS
    ]

    required_columns = [
        "cntry",
        "context_year",
        "analysis_unit",
        "alignment_scenario",
        *variables,
    ]

    require(data.columns.is_unique, "Duplicate column names found.")

    missing_columns = sorted(set(required_columns) - set(data.columns))
    require(
        not missing_columns,
        f"Missing required columns: {missing_columns}",
    )

    # Preserve the established one-row-per-country structure.
    require(len(data) == 29, "Expected 29 country rows.")
    require(data["cntry"].notna().all(), "Missing country codes found.")
    require(data["cntry"].is_unique, "Duplicate country codes found.")
    require(
        set(data["cntry"]) == EXPECTED_COUNTRIES,
        "Unexpected V1 country scope.",
    )

    for column, expected in [
        ("analysis_unit", "country"),
        ("alignment_scenario", "primary"),
    ]:
        require(
            data[column].notna().all()
            and data[column].eq(expected).all(),
            f"Unexpected or missing values in {column}.",
        )

    require(
        data["context_year"].isin([2023, 2024]).all(),
        "Unexpected or missing context years.",
    )

    for variable in variables:
        series = data[variable]

        # Reject non-numeric input instead of silently coercing it.
        require(
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype),
            f"{variable}: expected a numeric measurement.",
        )

        observed = series.dropna().to_numpy(dtype=float)

        require(
            len(observed) >= 2,
            f"{variable}: insufficient observations for this plot.",
        )
        require(
            np.isfinite(observed).all(),
            f"{variable}: infinite values found.",
        )

    # Sorting makes country order and point placement reproducible.
    return data.sort_values("cntry").reset_index(drop=True), input_sha256


def draw_panel(ax, data, definition, color, seed):
    variable, title, unit_label, reference = definition

    # Exclude missing values only from this panel.
    # Other panels can still display those countries when data are available.
    observed = data.loc[
        data[variable].notna(),
        ["cntry", variable],
    ]

    values = observed[variable].to_numpy(dtype=float)
    missing_n = len(data) - len(observed)

    # Draw the box above the country points to keep both layers readable.
    ax.boxplot(
        values,
        orientation="horizontal",
        positions=[0.15],
        widths=0.18,
        whis=1.5,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={
            "facecolor": color,
            "edgecolor": color,
            "alpha": 0.22,
        },
        medianprops={"color": "#172B4D", "linewidth": 2},
        whiskerprops={"color": color, "linewidth": 1.3},
        capprops={"color": color, "linewidth": 1.3},
    )

    # All observations appear below, including points beyond the whiskers.
    # Vertical jitter separates overlapping points; it has no data meaning.
    rng = np.random.default_rng(seed)
    point_y = -0.15 + rng.uniform(-0.06, 0.06, size=len(values))

    ax.scatter(
        values,
        point_y,
        s=34,
        color=color,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
        zorder=3,
    )

    minimum = float(values.min())
    maximum = float(values.max())

    # Preserve all country codes when several countries share an extreme.
    minimum_codes = ", ".join(
        observed.loc[observed[variable].eq(minimum), "cntry"].tolist()
    )
    maximum_codes = ", ".join(
        observed.loc[observed[variable].eq(maximum), "cntry"].tolist()
    )

    # Place labels in fixed panel positions to avoid point-label overlap.
    ax.text(
        0.01,
        0.95,
        f"Min: {minimum_codes} | {minimum:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#334155",
    )
    ax.text(
        0.99,
        0.95,
        f"Max: {maximum_codes} | {maximum:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#334155",
    )

    # Include the reference value within the displayed horizontal range.
    limits = [minimum, maximum]
    if reference is not None:
        limits.append(reference)
        ax.axvline(
            reference,
            color="#94A3B8",
            linestyle="--",
            linewidth=1,
            zorder=0,
        )

    low, high = min(limits), max(limits)
    span = high - low
    padding = 0.08 * span if span > 0 else 0.5
    ax.set_xlim(low - padding, high + padding)

    # Each panel uses its own horizontal scale and stated measurement unit.
    ax.set_ylim(-0.36, 0.62)
    ax.set_yticks([])
    ax.set_xlabel(unit_label, fontsize=10)
    ax.set_title(
        f"{title}\nn = {len(observed)} countries | missing = {missing_n}",
        loc="left",
        fontsize=12,
        pad=12,
    )

    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    return {
        "variable": variable,
        "countries_n": len(observed),
        "missing_n": missing_n,
        "minimum": minimum,
        "minimum_countries": minimum_codes,
        "maximum": maximum,
        "maximum_countries": maximum_codes,
        "reference_line": reference,
        "jitter_seed": seed,
    }


def create_figure(data, title, panels, color):
    # Six panels fit into a three-row, two-column layout.
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

    fig.text(
        0.5,
        0.945,
        "Primary alignment | One point per available country | "
        "Equal country weight",
        ha="center",
        fontsize=11,
        color="#475569",
    )

    panel_metadata = []

    for index, (ax, definition) in enumerate(zip(axes.flat, panels)):
        panel_metadata.append(
            draw_panel(
                ax,
                data,
                definition,
                color,
                seed=RANDOM_SEED + index,
            )
        )

    # Explain the plot directly in the exported figure.
    caption = (
        "Box: Q1–Q3; internal line: median; whiskers: observations within "
        "1.5 × IQR of the box. All available countries are shown.\n"
        "Vertical point displacement has no statistical meaning. "
        "Horizontal scales differ by panel. Boxes are not confidence intervals.\n"
        "ESS country estimates retain survey weighting. Context years: "
        "2023 (19 countries) and 2024 (10 countries); Gini label year = context + 1.\n"
        "Sources: ESS Round 11, edition 4.2; Eurostat snapshots. "
        "Dashed lines, where present: centered-score zero or GDP index 100."
    )

    fig.text(
        0.07,
        0.025,
        caption,
        ha="left",
        va="bottom",
        fontsize=9,
        linespacing=1.6,
        color="#475569",
    )

    # Reserve space for the title, subtitles and methodological caption.
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.87,
        bottom=0.19,
        hspace=0.90,
        wspace=0.28,
    )

    return fig, panel_metadata


def save_figure(fig, stem):
    # PNG is convenient for previews; PDF preserves vector graphics.
    saved = []

    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"{stem}.{extension}"

        # Exclusive creation prevents overwriting an existing file.
        with path.open("xb") as handle:
            fig.savefig(
                handle,
                format=extension,
                dpi=220,
                facecolor="white",
            )

        # Record an output hash for provenance.
        saved.append(
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(f"Saved: {path}")

    return saved


def main():
    print("STEP 51 — COUNTRY DISTRIBUTION FIGURES")

    check_plotting_version()
    check_output_paths()
    data, input_sha256 = load_input()

    # Create the output directory only after input checks have passed.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "step": 51,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": INPUT_PATH.name,
        "input_sha256": input_sha256,
        "analysis_unit": "country",
        "alignment_scenario": "primary",
        "across_country_weighting": "equal",
        "ess_within_country_weighting": "existing survey estimates retained",
        "boxplot_whis": 1.5,
        "missing_policy": "available countries separately per variable",
        "extreme_values_removed": False,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "figures": [],
    }

    for stem, title, panels, color in FIGURE_GROUPS:
        print(f"\nCreating: {stem}")

        fig, panel_metadata = create_figure(data, title, panels, color)

        # Close the figure even if saving raises an error.
        try:
            saved_files = save_figure(fig, stem)
        finally:
            plt.close(fig)

        metadata["figures"].append(
            {
                "stem": stem,
                "panels": panel_metadata,
                "files": saved_files,
            }
        )

        for panel in panel_metadata:
            print(
                f"  {panel['variable']}: "
                f"n={panel['countries_n']}, "
                f"missing={panel['missing_n']}"
            )

    # Save methods, software versions and hashes beside the figures.
    with METADATA_PATH.open("x", encoding="utf-8") as handle:
        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    print(f"\nSaved: {METADATA_PATH}")
    print("\nCOUNTRY DISTRIBUTION FIGURES CREATED.")
    print("Figures: 2 PNG files and 2 PDF files.")
    print("Provenance: 1 metadata JSON file.")
    print("No input files were modified.")
    print("No imputation or country exclusion was performed.")
    print("Visual inspection is still pending.")


if __name__ == "__main__":
    main()