#!/usr/bin/env python3
"""Plot histograms and Q-Q plots for Ken French industry returns."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


INDUSTRY_RETURNS_INPUT = Path("data/processed/10_industry_portfolios_returns.csv")
FIGURE_DIR = Path("reports/figures")

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

RETURN_TYPE = "simple"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def values_for_industry(rows: list[dict[str, str]], industry: str) -> np.ndarray:
    column = f"{industry}_{RETURN_TYPE}"
    return np.array([float(row[column]) for row in rows], dtype=float)


def fitted_normal(values: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    return mu, sigma


def fitted_student_t(values: np.ndarray) -> tuple[float, float, float]:
    degrees_of_freedom, location, scale = stats.t.fit(values)
    return float(degrees_of_freedom), float(location), float(scale)


def empirical_quantiles(values: np.ndarray) -> np.ndarray:
    return np.sort(values)


def plotting_positions(sample_size: int) -> np.ndarray:
    ranks = np.arange(1, sample_size + 1)
    return (ranks - 0.5) / sample_size


def qq_quantiles_normal(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma = fitted_normal(values)
    probabilities = plotting_positions(len(values))
    theoretical = stats.norm.ppf(probabilities, loc=mu, scale=sigma)
    empirical = empirical_quantiles(values)
    return theoretical, empirical


def qq_quantiles_student_t(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    degrees_of_freedom, location, scale = fitted_student_t(values)
    probabilities = plotting_positions(len(values))
    theoretical = stats.t.ppf(probabilities, df=degrees_of_freedom, loc=location, scale=scale)
    empirical = empirical_quantiles(values)
    return theoretical, empirical, (degrees_of_freedom, location, scale)


def add_reference_line(axis: plt.Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    lower = min(float(np.min(x_values)), float(np.min(y_values)))
    upper = max(float(np.max(x_values)), float(np.max(y_values)))
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="#c2410c",
        linewidth=1.1,
        label="Perfect fit line",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)


def format_percent_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")


def save_histogram_grid(industry_returns: dict[str, np.ndarray]) -> Path:
    output_path = FIGURE_DIR / "industry_simple_return_histograms.png"
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=False, sharey=False)
    figure.suptitle("Monthly Simple Returns by Industry", fontsize=16)

    for axis, industry in zip(axes.flatten(), INDUSTRY_COLUMNS, strict=True):
        values = industry_returns[industry]
        mu, sigma = fitted_normal(values)
        axis.hist(
            values,
            bins=40,
            density=True,
            color="#2563eb",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
            label="Observed returns",
        )

        x_min, x_max = axis.get_xlim()
        x_grid = np.linspace(x_min, x_max, 300)
        axis.plot(
            x_grid,
            stats.norm.pdf(x_grid, loc=mu, scale=sigma),
            color="#111827",
            linewidth=1.2,
            label="Normal fit",
        )
        axis.axvline(mu, color="#c2410c", linewidth=1.0, linestyle="--", label="Mean")
        axis.set_title(industry)
        axis.grid(alpha=0.18)
        format_percent_axis(axis)

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_industry_diagnostic_plot(industry: str, values: np.ndarray) -> Path:
    output_path = FIGURE_DIR / f"{industry.lower()}_distribution_diagnostics.png"
    normal_theoretical, normal_empirical = qq_quantiles_normal(values)
    student_theoretical, student_empirical, student_params = qq_quantiles_student_t(values)
    degrees_of_freedom, location, scale = student_params
    mu, sigma = fitted_normal(values)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle(f"{industry}: Monthly Simple Return Diagnostics", fontsize=15)

    axes[0].hist(
        values,
        bins=45,
        density=True,
        color="#2563eb",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.4,
        label="Observed returns",
    )
    x_min, x_max = axes[0].get_xlim()
    x_grid = np.linspace(x_min, x_max, 400)
    axes[0].plot(
        x_grid,
        stats.norm.pdf(x_grid, loc=mu, scale=sigma),
        color="#111827",
        linewidth=1.2,
        label="Normal fit",
    )
    axes[0].plot(
        x_grid,
        stats.t.pdf(x_grid, df=degrees_of_freedom, loc=location, scale=scale),
        color="#7c3aed",
        linewidth=1.2,
        label="Student-t fit",
    )
    axes[0].axvline(mu, color="#c2410c", linewidth=1.0, linestyle="--", label="Mean")
    axes[0].set_title("Histogram with fitted densities")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.18)
    format_percent_axis(axes[0])
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)

    axes[1].scatter(
        normal_theoretical,
        normal_empirical,
        s=12,
        alpha=0.62,
        color="#2563eb",
        label="Monthly returns",
    )
    add_reference_line(axes[1], normal_theoretical, normal_empirical)
    axes[1].set_title("Q-Q vs fitted Normal")
    axes[1].set_xlabel("Normal theoretical quantiles")
    axes[1].set_ylabel("Empirical quantiles")
    axes[1].grid(alpha=0.18)
    format_percent_axis(axes[1])
    axes[1].yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    axes[1].legend(loc="upper left", fontsize=8, frameon=True)
    axes[1].text(
        0.04,
        0.05,
        "Closer to line = better fit",
        transform=axes[1].transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#d1d5db"},
    )

    axes[2].scatter(
        student_theoretical,
        student_empirical,
        s=12,
        alpha=0.62,
        color="#7c3aed",
        label="Monthly returns",
    )
    add_reference_line(axes[2], student_theoretical, student_empirical)
    axes[2].set_title(f"Q-Q vs fitted Student-t (df={degrees_of_freedom:.1f})")
    axes[2].set_xlabel("Student-t theoretical quantiles")
    axes[2].set_ylabel("Empirical quantiles")
    axes[2].grid(alpha=0.18)
    format_percent_axis(axes[2])
    axes[2].yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    axes[2].legend(loc="upper left", fontsize=8, frameon=True)
    axes[2].text(
        0.04,
        0.05,
        "Closer to line = better fit",
        transform=axes[2].transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#d1d5db"},
    )

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> int:
    rows = read_csv(INDUSTRY_RETURNS_INPUT)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    industry_returns = {
        industry: values_for_industry(rows, industry)
        for industry in INDUSTRY_COLUMNS
    }

    output_paths = [save_histogram_grid(industry_returns)]
    output_paths.extend(
        save_industry_diagnostic_plot(industry, industry_returns[industry])
        for industry in INDUSTRY_COLUMNS
    )

    print("Generated distribution plots:")
    for path in output_paths:
        print(f"- {path}")
    print("Returns are monthly simple decimal returns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
