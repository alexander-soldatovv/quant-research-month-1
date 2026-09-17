#!/usr/bin/env python3
"""Validate cleaned Kenneth French monthly datasets."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path


SENTINEL_MISSING_VALUES = {"-99.99", "-999", "-999.0", "-999.00"}


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    row_count: int
    start_date: str
    end_date: str
    duplicate_dates: list[str]
    missing_months: list[str]
    blank_cells: list[tuple[int, str]]
    non_numeric_cells: list[tuple[int, str, str]]
    sentinel_missing_cells: list[tuple[int, str, str]]
    min_values: dict[str, float]
    max_values: dict[str, float]

    @property
    def passed(self) -> bool:
        return not (
            self.duplicate_dates
            or self.missing_months
            or self.blank_cells
            or self.non_numeric_cells
            or self.sentinel_missing_cells
        )


def monthly_dates(start: date, end: date) -> list[str]:
    dates = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        dates.append(date(year, month, 1).isoformat())
        month += 1
        if month == 13:
            year += 1
            month = 1
    return dates


def validate_csv(path: Path) -> ValidationResult:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"{path} is empty")
    if "date" not in rows[0]:
        raise ValueError(f"{path} does not contain a date column")

    dates = [row["date"] for row in rows]
    parsed_dates = [date.fromisoformat(value) for value in dates]
    numeric_columns = [column for column in rows[0] if column != "date"]

    date_counts = Counter(dates)
    duplicate_dates = sorted(date for date, count in date_counts.items() if count > 1)
    expected_months = monthly_dates(min(parsed_dates), max(parsed_dates))
    missing_months = sorted(set(expected_months) - set(dates))

    blank_cells: list[tuple[int, str]] = []
    non_numeric_cells: list[tuple[int, str, str]] = []
    sentinel_missing_cells: list[tuple[int, str, str]] = []

    for row_number, row in enumerate(rows, start=2):
        for column, value in row.items():
            value = "" if value is None else value.strip()
            if value == "":
                blank_cells.append((row_number, column))
                continue
            if value in SENTINEL_MISSING_VALUES:
                sentinel_missing_cells.append((row_number, column, value))
            if column != "date":
                try:
                    float(value)
                except ValueError:
                    non_numeric_cells.append((row_number, column, value))

    min_values = {
        column: min(float(row[column]) for row in rows)
        for column in numeric_columns
    }
    max_values = {
        column: max(float(row[column]) for row in rows)
        for column in numeric_columns
    }

    return ValidationResult(
        path=path,
        row_count=len(rows),
        start_date=min(dates),
        end_date=max(dates),
        duplicate_dates=duplicate_dates,
        missing_months=missing_months,
        blank_cells=blank_cells,
        non_numeric_cells=non_numeric_cells,
        sentinel_missing_cells=sentinel_missing_cells,
        min_values=min_values,
        max_values=max_values,
    )


def print_result(result: ValidationResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{status} {result.path}")
    print(f"rows: {result.row_count}")
    print(f"date range: {result.start_date} to {result.end_date}")
    print(f"duplicate dates: {len(result.duplicate_dates)}")
    print(f"missing monthly dates: {len(result.missing_months)}")
    print(f"blank cells: {len(result.blank_cells)}")
    print(f"non-numeric cells: {len(result.non_numeric_cells)}")
    print(f"Ken French sentinel missing values: {len(result.sentinel_missing_cells)}")
    print("units: monthly percent returns")
    print(f"min values: {result.min_values}")
    print(f"max values: {result.max_values}")


def main() -> int:
    paths = [
        Path("data/processed/fama_french_3_factors_monthly.csv"),
        Path("data/processed/10_industry_portfolios_monthly_vw.csv"),
    ]

    results = [validate_csv(path) for path in paths]
    for result in results:
        print_result(result)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
