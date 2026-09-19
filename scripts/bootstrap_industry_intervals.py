#!/usr/bin/env python3
"""Bootstrap confidence intervals for industry mean and median returns."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INDUSTRY_RETURNS_INPUT = Path("data/processed/10_industry_portfolios_returns.csv")
INTERVALS_OUTPUT = Path("data/processed/10_industry_portfolios_bootstrap_intervals.csv")
FIGURE_OUTPUT = Path("reports/figures/industry_bootstrap_intervals_simple.png")

INDUSTRY_COLUMNS = (
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "HiTec",
    "Telcm",
    "Shops",
    "Hlth",
    "Utils",
    "Other",
)

RETURN_TYPES = ("simple", "log", "excess_simple")
STATISTICS = ("mean", "median")
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def values_for_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.array([float(row[column]) for row in rows], dtype=float)


def rounded(value: float) -> float:
    return round(value, 10)


def bootstrap_distribution(
    values: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> np.ndarray:
    sample_size = len(values)
    sample_indices = rng.integers(
        low=0,
        high=sample_size,
        size=(bootstrap_samples, sample_size),
    )
    bootstrap_samples_matrix = values[sample_indices]

    if statistic == "mean":
        return np.mean(bootstrap_samples_matrix, axis=1)
    if statistic == "median":
        return np.median(bootstrap_samples_matrix, axis=1)
    raise ValueError(f"Unsupported statistic: {statistic}")


def percentile_interval(distribution: np.ndarray, confidence_level: float) -> tuple[float, float]:
    tail_probability = (1.0 - confidence_level) / 2.0
    lower_percentile = 100.0 * tail_probability
    upper_percentile = 100.0 * (1.0 - tail_probability)
    lower, upper = np.percentile(distribution, [lower_percentile, upper_percentile])
    return float(lower), float(upper)


def point_estimate(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    raise ValueError(f"Unsupported statistic: {statistic}")


def calculate_intervals(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(RANDOM_SEED)
    interval_rows: list[dict[str, float | int | str]] = []

    for return_type in RETURN_TYPES:
        for industry in INDUSTRY_COLUMNS:
            column = f"{industry}_{return_type}"
            values = values_for_column(rows, column)

            for statistic in STATISTICS:
                bootstrap_values = bootstrap_distribution(
                    values=values,
                    statistic=statistic,
                    rng=rng,
                    bootstrap_samples=BOOTSTRAP_SAMPLES,
                )
                lower, upper = percentile_interval(bootstrap_values, CONFIDENCE_LEVEL)
                estimate = point_estimate(values, statistic)

                interval_rows.append(
                    {
                        "industry": industry,
                        "return_type": return_type,
                        "statistic": statistic,
                        "observations": len(values),
                        "bootstrap_samples": BOOTSTRAP_SAMPLES,
                        "confidence_level": CONFIDENCE_LEVEL,
                        "estimate": rounded(estimate),
                        "ci_lower": rounded(lower),
                        "ci_upper": rounded(upper),
                        "ci_width": rounded(upper - lower),
                        "method": "percentile bootstrap",
                        "random_seed": RANDOM_SEED,
                    }
                )

    return interval_rows


def save_simple_interval_plot(interval_rows: list[dict[str, float | int | str]]) -> Path:
    simple_rows = [
        row for row in interval_rows
        if row["return_type"] == "simple"
    ]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    figure.suptitle("Bootstrap 95% Intervals for Monthly Simple Returns", fontsize=15)

    for axis, statistic in zip(axes, STATISTICS, strict=True):
        rows = [
            row for row in simple_rows
            if row["statistic"] == statistic
        ]
        industries = [str(row["industry"]) for row in rows]
        estimates = np.array([float(row["estimate"]) for row in rows])
        lower = np.array([float(row["ci_lower"]) for row in rows])
        upper = np.array([float(row["ci_upper"]) for row in rows])
        y_positions = np.arange(len(industries))
        x_errors = np.vstack([estimates - lower, upper - estimates])

        axis.errorbar(
            estimates,
            y_positions,
            xerr=x_errors,
            fmt="o",
            color="#2563eb",
            ecolor="#111827",
            elinewidth=1.2,
            capsize=4,
        )
        axis.axvline(0.0, color="#c2410c", linewidth=1.0, linestyle="--")
        axis.set_title(statistic.capitalize())
        axis.set_xlabel("Monthly return")
        axis.set_yticks(y_positions, labels=industries)
        axis.xaxis.set_major_formatter(lambda value, _position: f"{value * 100:.1f}%")
        axis.grid(axis="x", alpha=0.2)

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Industry")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_OUTPUT, dpi=180)
    plt.close(figure)
    return FIGURE_OUTPUT


def print_simple_summary(interval_rows: list[dict[str, float | int | str]]) -> None:
    simple_mean_rows = [
        row for row in interval_rows
        if row["return_type"] == "simple" and row["statistic"] == "mean"
    ]
    simple_median_rows = [
        row for row in interval_rows
        if row["return_type"] == "simple" and row["statistic"] == "median"
    ]

    highest_mean = max(simple_mean_rows, key=lambda row: float(row["estimate"]))
    widest_mean_interval = max(simple_mean_rows, key=lambda row: float(row["ci_width"]))
    highest_median = max(simple_median_rows, key=lambda row: float(row["estimate"]))

    print("Simple monthly return bootstrap summary:")
    print(
        f"highest mean estimate: {highest_mean['industry']} "
        f"{highest_mean['estimate']} "
        f"[{highest_mean['ci_lower']}, {highest_mean['ci_upper']}]"
    )
    print(
        f"widest mean interval: {widest_mean_interval['industry']} "
        f"width={widest_mean_interval['ci_width']}"
    )
    print(
        f"highest median estimate: {highest_median['industry']} "
        f"{highest_median['estimate']} "
        f"[{highest_median['ci_lower']}, {highest_median['ci_upper']}]"
    )


def main() -> int:
    rows = read_csv(INDUSTRY_RETURNS_INPUT)
    interval_rows = calculate_intervals(rows)
    write_csv(INTERVALS_OUTPUT, interval_rows)
    figure_path = save_simple_interval_plot(interval_rows)

    print(f"bootstrap intervals -> {INTERVALS_OUTPUT} ({len(interval_rows)} rows)")
    print(f"bootstrap interval figure -> {figure_path}")
    print(
        f"method: percentile bootstrap, "
        f"samples={BOOTSTRAP_SAMPLES}, confidence={CONFIDENCE_LEVEL}, seed={RANDOM_SEED}"
    )
    print_simple_summary(interval_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
