#!/usr/bin/env python3
"""Compare descriptive statistics for Ken French 10 industry portfolios."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path


INDUSTRY_RETURNS_INPUT = Path("data/processed/10_industry_portfolios_returns.csv")
STATS_OUTPUT = Path("data/processed/10_industry_portfolios_stats.csv")

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


def values_for_column(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def skewness(values: list[float]) -> float:
    mean = statistics.fmean(values)
    volatility = statistics.stdev(values)
    if volatility == 0:
        return 0.0

    n = len(values)
    third_standardized_moment = sum(
        ((value - mean) / volatility) ** 3
        for value in values
    ) / n
    return third_standardized_moment


def excess_kurtosis(values: list[float]) -> float:
    mean = statistics.fmean(values)
    volatility = statistics.stdev(values)
    if volatility == 0:
        return 0.0

    n = len(values)
    fourth_standardized_moment = sum(
        ((value - mean) / volatility) ** 4
        for value in values
    ) / n
    return fourth_standardized_moment - 3.0


def annualize_mean(monthly_mean: float) -> float:
    return (1.0 + monthly_mean) ** 12 - 1.0


def annualize_volatility(monthly_volatility: float) -> float:
    return monthly_volatility * math.sqrt(12.0)


def rounded(value: float) -> float:
    return round(value, 10)


def calculate_stats(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    stats_rows: list[dict[str, float | int | str]] = []

    for return_type in RETURN_TYPES:
        for industry in INDUSTRY_COLUMNS:
            column = f"{industry}_{return_type}"
            values = values_for_column(rows, column)
            mean = statistics.fmean(values)
            volatility = statistics.stdev(values)

            stats_rows.append(
                {
                    "industry": industry,
                    "return_type": return_type,
                    "observations": len(values),
                    "mean_monthly": rounded(mean),
                    "median_monthly": rounded(statistics.median(values)),
                    "volatility_monthly": rounded(volatility),
                    "mean_annualized": rounded(annualize_mean(mean)),
                    "volatility_annualized": rounded(annualize_volatility(volatility)),
                    "skewness": rounded(skewness(values)),
                    "excess_kurtosis": rounded(excess_kurtosis(values)),
                    "min_monthly": rounded(min(values)),
                    "max_monthly": rounded(max(values)),
                }
            )

    return stats_rows


def print_simple_return_ranking(stats_rows: list[dict[str, float | int | str]]) -> None:
    simple_rows = [
        row for row in stats_rows
        if row["return_type"] == "simple"
    ]

    highest_mean = max(simple_rows, key=lambda row: float(row["mean_monthly"]))
    highest_volatility = max(simple_rows, key=lambda row: float(row["volatility_monthly"]))
    lowest_volatility = min(simple_rows, key=lambda row: float(row["volatility_monthly"]))
    most_negative_skew = min(simple_rows, key=lambda row: float(row["skewness"]))
    highest_kurtosis = max(simple_rows, key=lambda row: float(row["excess_kurtosis"]))

    print("Simple monthly return comparison:")
    print(f"highest mean: {highest_mean['industry']} ({highest_mean['mean_monthly']})")
    print(
        f"highest volatility: {highest_volatility['industry']} "
        f"({highest_volatility['volatility_monthly']})"
    )
    print(
        f"lowest volatility: {lowest_volatility['industry']} "
        f"({lowest_volatility['volatility_monthly']})"
    )
    print(f"most negative skewness: {most_negative_skew['industry']} ({most_negative_skew['skewness']})")
    print(f"highest excess kurtosis: {highest_kurtosis['industry']} ({highest_kurtosis['excess_kurtosis']})")


def main() -> int:
    rows = read_csv(INDUSTRY_RETURNS_INPUT)
    stats_rows = calculate_stats(rows)
    write_csv(STATS_OUTPUT, stats_rows)

    print(f"industry stats -> {STATS_OUTPUT} ({len(stats_rows)} rows)")
    print("returns are decimal monthly returns; kurtosis is excess kurtosis")
    print_simple_return_ranking(stats_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
