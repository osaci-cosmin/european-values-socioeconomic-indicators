# Step 52: Explore country-level ESS–Eurostat relationships.
# Each point represents one country with both measurements available.
# These plots show associations visually; they do not establish causality.

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# Save figures without requiring an interactive plotting window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
except ModuleNotFoundError as exc:
    raise SystemExit(
        'Install label support: python -m pip install "adjustText>=1.3"'
    ) from exc


# Resolve project paths independently of the terminal's current folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ESS11_country_integrated_main_v1.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

METADATA_PATH = OUTPUT_DIR / "52_ess_eurostat_relationships_v1.metadata.json"

# Use the exact input previously inspected in Step 50.
EXPECTED_INPUT_SHA256 = (
    "dc96b7ffe015727adb1bbb3de98ef9562d13cacf23e2d022144984a5bb10db64"
)

EXPECTED_COUNTRIES = set(
    "AT BE BG CH CY DE EE ES FI FR GB GR HR HU IE IS "
    "IT LT LV ME NL NO PL PT RS SE SI SK UA".split()
)

# Each entry contains: source column, display title, axis label.
OUTCOMES = [
    (
        "mean_social_trust",
        "Generalized social trust",
        "Weighted country mean | scale 0–10",
    ),
    (
        "mean_left_right",
        "Ideological self-placement",
        "Weighted country mean | 0 = left, 10 = right",
    ),
    (
        "mean_openness",
        "Openness to Change",
        "Weighted country mean | MRAT-centered score",
    ),
    (
        "mean_conservation",
        "Conservation",
        "Weighted country mean | MRAT-centered score",
    ),
    (
        "mean_self_transcendence",
        "Self-Transcendence",
        "Weighted country mean | MRAT-centered score",
    ),
    (
        "mean_self_enhancement",
        "Self-Enhancement",
        "Weighted country mean | MRAT-centered score",
    ),
]

ECONOMIC_INDICATORS = [
    (
        "gdp_pps_index",
        "GDP per capita in PPS",
        "Index | EU27_2020 = 100",
    ),
    (
        "employment_rate",
        "Employment rate",
        "% of population aged 20–64",
    ),
    (
        "unemployment_rate",
        "Unemployment rate",
        "% of labour force aged 15–74",
    ),
    (
        "gini",
        "Income inequality",
        "Gini coefficient | scale 0–100",
    ),
    (
        "material_deprivation_rate",
        "Severe material and social deprivation",
        "% of population",
    ),
    (
        "inflation_rate",
        "Annual HICP inflation",
        "Annual average rate of change (%)",
    ),
]

# Colors describe context years, not country regions or ideological groups.
YEAR_COLORS = {
    2023: "#2166AC",
    2024: "#C76B16",
}

BASE_SEED = 5200
LABEL_ITERATIONS = 200
FIGURE_DPI = 180


def require(condition, message):
    # Fail explicitly instead of silently changing unexpected input.
    if not condition:
        raise ValueError(message)


def figure_stem(outcome):
    # Give every outcome a stable, descriptive output filename.
    return f"52_{outcome}_vs_eurostat_v1"


def check_outputs():
    # Check all planned files before saving any new output.
    paths = [
        OUTPUT_DIR / f"{figure_stem(outcome)}.{extension}"
        for outcome, _, _ in OUTCOMES
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

    # Calculate provenance from the same bytes that will be analysed.
    content = INPUT_PATH.read_bytes()
    input_sha256 = hashlib.sha256(content).hexdigest()

    require(
        input_sha256 == EXPECTED_INPUT_SHA256,
        "Input differs from Step 50. Investigate before continuing.",
    )

    data = pd.read_parquet(io.BytesIO(content), engine="pyarrow")

    variables = [
        definition[0]
        for definition in OUTCOMES + ECONOMIC_INDICATORS
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

    # Country-level analysis requires one unique row per country.
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
        data["context_year"].isin(YEAR_COLORS).all(),
        "Unexpected or missing context year.",
    )

    for variable in variables:
        series = data[variable]

        # Do not convert text or Boolean fields into numerical measurements.
        require(
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype),
            f"{variable}: expected a numeric measurement.",
        )

        observed = series.dropna().to_numpy(dtype=float)

        require(
            np.isfinite(observed).all(),
            f"{variable}: infinite values found.",
        )

    # Stable ordering helps reproduce label placement.
    return data.sort_values("cntry").reset_index(drop=True), input_sha256


def padded_limits(values):
    # Leave room near the edges for country labels.
    low = float(values.min())
    high = float(values.max())
    span = high - low
    padding = 0.12 * span if span > 0 else 0.5

    return low - padding, high + padding


def create_figure(data, outcome_definition, outcome_index):
    outcome, outcome_title, y_label = outcome_definition

    # Use the same vertical range across the six panels of this figure.
    # This makes the same ESS outcome visually comparable across indicators.
    outcome_values = data[outcome].dropna().to_numpy(dtype=float)
    require(len(outcome_values) >= 2, f"{outcome}: insufficient data.")
    y_limits = padded_limits(outcome_values)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
        dpi=FIGURE_DPI,
    )

    fig.suptitle(
        f"{outcome_title} | Socioeconomic relationships",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.948,
        "ESS Round 11 + Eurostat | Primary alignment | "
        "One point per available country",
        ha="center",
        fontsize=11,
        color="#475569",
    )

    # A single legend applies to all panels.
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=color,
            markersize=7,
            label=f"Context year {year}",
        )
        for year, color in YEAR_COLORS.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.932),
        ncol=2,
        frameon=False,
    )

    caption = (
        "Each point is a country; ESS coordinates are survey-weighted "
        "country estimates. Point size is constant.\n"
        "Only countries with both measurements are plotted in each panel. "
        "Missing values are not imputed; available extremes are retained.\n"
        "Colors indicate the ESS context year, not statistical adjustment "
        "for year. Gini label year = context year + 1.\n"
        "Labels may move for readability; point coordinates remain unchanged. "
        "Survey confidence intervals are not displayed.\n"
        "Sources: ESS Round 11, edition 4.2; Eurostat snapshots. "
        "Exploratory country-level associations do not establish causality."
    )

    fig.text(
        0.08,
        0.025,
        caption,
        ha="left",
        va="bottom",
        fontsize=9,
        linespacing=1.5,
        color="#475569",
    )

    # Finalize the layout before adjusting country labels.
    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.855,
        bottom=0.16,
        hspace=0.52,
        wspace=0.27,
    )

    label_tasks = []
    panel_metadata = []

    for panel_index, (ax, definition) in enumerate(
        zip(axes.flat, ECONOMIC_INDICATORS)
    ):
        economic, economic_title, x_label = definition

        # Select complete pairs separately for each economic indicator.
        available = data[[outcome, economic]].notna().all(axis=1)

        pair = data.loc[
            available,
            ["cntry", "context_year", outcome, economic],
        ].copy()

        require(
            len(pair) >= 3,
            f"{outcome} versus {economic}: insufficient available pairs.",
        )

        x = pair[economic].to_numpy(dtype=float)
        y = pair[outcome].to_numpy(dtype=float)

        colors = [
            YEAR_COLORS[int(year)]
            for year in pair["context_year"]
        ]

        # Draw points at their exact values without jitter.
        ax.scatter(
            x,
            y,
            c=colors,
            s=40,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            zorder=3,
        )

        # Initially place each country code at its own observation.
        texts = [
            ax.text(
                x_value,
                y_value,
                code,
                fontsize=8,
                color="#263445",
                ha="center",
                va="center",
                zorder=4,
            )
            for x_value, y_value, code in zip(x, y, pair["cntry"])
        ]

        ax.set_xlim(*padded_limits(x))
        ax.set_ylim(*y_limits)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=9)

        ax.set_title(
            f"{economic_title}\n"
            f"n = {len(pair)} countries | unavailable pairs = "
            f"{len(data) - len(pair)}",
            loc="left",
            fontsize=12,
            pad=10,
        )

        ax.grid(color="#E2E8F0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        for side in ("bottom", "left"):
            ax.spines[side].set_color("#CBD5E1")

        seed = BASE_SEED + outcome_index * 10 + panel_index
        label_tasks.append((ax, texts, x, y, seed))

        # Record the exact country sample for each displayed relationship.
        panel_metadata.append(
            {
                "ess_variable": outcome,
                "economic_variable": economic,
                "countries_n": len(pair),
                "included_countries": pair["cntry"].tolist(),
                "unavailable_countries": data.loc[
                    ~available, "cntry"
                ].tolist(),
                "context_year_counts": {
                    str(year): int(pair["context_year"].eq(year).sum())
                    for year in YEAR_COLORS
                },
                "label_seed": seed,
            }
        )

        print(f"  {economic}: n={len(pair)}", flush=True)

    # Label adjustment needs the final panel dimensions and axis limits.
    fig.canvas.draw()

    for ax, texts, x, y, seed in label_tasks:
        # adjustText uses randomness when separating coincident labels.
        # Fix its seed and iteration count for repeatable placement.
        np.random.seed(seed)

        adjust_text(
            texts,
            x=x,
            y=y,
            target_x=x,
            target_y=y,
            ax=ax,
            ensure_inside_axes=True,
            expand_axes=False,
            iter_lim=LABEL_ITERATIONS,
            min_arrow_len=4,
            arrowprops={
                "arrowstyle": "-",
                "color": "#94A3B8",
                "linewidth": 0.5,
            },
        )

    return fig, panel_metadata


def save_figure(fig, stem):
    saved_files = []

    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"{stem}.{extension}"

        # Exclusive creation protects any existing output.
        with path.open("xb") as handle:
            fig.savefig(
                handle,
                format=extension,
                dpi=FIGURE_DPI,
                facecolor="white",
            )

        saved_files.append(
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

        print(f"Saved: {path}", flush=True)

    return saved_files


def main():
    print("STEP 52 — ESS–EUROSTAT RELATIONSHIP FIGURES", flush=True)

    check_outputs()
    data, input_sha256 = load_input()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "step": 52,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": INPUT_PATH.name,
        "input_sha256": input_sha256,
        "analysis_unit": "country",
        "alignment_scenario": "primary",
        "missing_policy": "complete pairs separately for each panel",
        "point_size": "constant",
        "point_coordinates_jittered": False,
        "year_colors": YEAR_COLORS,
        "label_iterations": LABEL_ITERATIONS,
        "regression_lines": False,
        "survey_confidence_intervals_displayed": False,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "adjustText_version": version("adjustText"),
        "figures": [],
    }

    for outcome_index, definition in enumerate(OUTCOMES):
        outcome = definition[0]
        stem = figure_stem(outcome)

        print(f"\nCreating: {stem}", flush=True)

        fig, panels = create_figure(data, definition, outcome_index)

        # Release figure memory after exporting both formats.
        try:
            saved_files = save_figure(fig, stem)
        finally:
            plt.close(fig)

        metadata["figures"].append(
            {
                "stem": stem,
                "panels": panels,
                "files": saved_files,
            }
        )

    # Keep sample membership, software versions and hashes with the figures.
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
    print("\nESS–EUROSTAT RELATIONSHIP FIGURES CREATED.")
    print("ESS outcomes: 6")
    print("Economic indicators per outcome: 6")
    print("Scatterplots: 36")
    print("Figure files: 6 PNG and 6 PDF")
    print("Provenance files: 1 metadata JSON")
    print("No input files were modified.")
    print("No imputation or automatic extreme-value removal was performed.")
    print("No correlations, statistical tests or models were calculated.")
    print("Visual inspection is still pending.")


if __name__ == "__main__":
    main()