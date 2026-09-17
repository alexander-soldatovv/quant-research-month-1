#!/usr/bin/env python3
"""Calculate simple, log, and excess returns from Ken French monthly data."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


FACTOR_INPUT = Path("data/processed/fama_french_3_factors_monthly.csv")
INDUSTRY_INPUT = Path("data/processed/10_industry_portfolios_monthly_vw.csv")
FACTOR_OUTPUT = Path("data/processed/fama_french_3_factors_returns.csv")
INDUSTRY_OUTPUT = Path("data/processed/10_industry_portfolios_returns.csv")

FACTOR_COLUMNS = ("Mkt-RF", "SMB", "HML", "RF")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percent_to_decimal(value: str) -> float:
    return float(value) / 100.0


def simple_to_log(simple_return: float) -> float:
    if simple_return <= -1.0:
        raise ValueError(f"Cannot compute log return for simple return <= -100%: {simple_return}")
    return math.log1p(simple_return)


def rounded(value: float) -> float:
    return round(value, 10)


def calculate_factor_returns(factor_rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    output_rows: list[dict[str, float | str]] = []

    for row in factor_rows:
        rf = percent_to_decimal(row["RF"])
        mkt_excess = percent_to_decimal(row["Mkt-RF"])
        mkt_simple = mkt_excess + rf

        output_row: dict[str, float | str] = {
            "date": row["date"],
            "RF_simple": rounded(rf),
            "RF_log": rounded(simple_to_log(rf)),
            "Mkt_simple": rounded(mkt_simple),
            "Mkt_log": rounded(simple_to_log(mkt_simple)),
            "Mkt_excess_simple": rounded(mkt_excess),
            "Mkt_excess_log": rounded(simple_to_log(mkt_simple) - simple_to_log(rf)),
        }

        for factor in ("SMB", "HML"):
            simple_return = percent_to_decimal(row[factor])
            output_row[f"{factor}_simple"] = rounded(simple_return)
            output_row[f"{factor}_log"] = rounded(simple_to_log(simple_return))

        output_rows.append(output_row)

    return output_rows


def calculate_industry_returns(
    industry_rows: list[dict[str, str]],
    factor_rows: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    rf_by_date = {
        row["date"]: percent_to_decimal(row["RF"])
        for row in factor_rows
    }
    output_rows: list[dict[str, float | str]] = []

    for row in industry_rows:
        rf = rf_by_date[row["date"]]
        rf_log = simple_to_log(rf)
        output_row: dict[str, float | str] = {
            "date": row["date"],
            "RF_simple": rounded(rf),
            "RF_log": rounded(rf_log),
        }

        for industry in INDUSTRY_COLUMNS:
            simple_return = percent_to_decimal(row[industry])
            log_return = simple_to_log(simple_return)
            excess_simple = simple_return - rf
            excess_log = log_return - rf_log

            output_row[f"{industry}_simple"] = rounded(simple_return)
            output_row[f"{industry}_log"] = rounded(log_return)
            output_row[f"{industry}_excess_simple"] = rounded(excess_simple)
            output_row[f"{industry}_excess_log"] = rounded(excess_log)

        output_rows.append(output_row)

    return output_rows


def ensure_matching_dates(
    factor_rows: list[dict[str, str]],
    industry_rows: list[dict[str, str]],
) -> None:
    factor_dates = {row["date"] for row in factor_rows}
    industry_dates = {row["date"] for row in industry_rows}
    missing_factor_dates = sorted(industry_dates - factor_dates)
    missing_industry_dates = sorted(factor_dates - industry_dates)

    if missing_factor_dates or missing_industry_dates:
        raise ValueError(
            "Factor and industry dates do not match. "
            f"Missing factor dates: {missing_factor_dates[:5]}; "
            f"missing industry dates: {missing_industry_dates[:5]}"
        )


def main() -> int:
    factor_rows = read_csv(FACTOR_INPUT)
    industry_rows = read_csv(INDUSTRY_INPUT)
    ensure_matching_dates(factor_rows, industry_rows)

    factor_output_rows = calculate_factor_returns(factor_rows)
    industry_output_rows = calculate_industry_returns(industry_rows, factor_rows)

    write_csv(FACTOR_OUTPUT, factor_output_rows)
    write_csv(INDUSTRY_OUTPUT, industry_output_rows)

    print(f"factor returns -> {FACTOR_OUTPUT} ({len(factor_output_rows)} rows)")
    print(f"industry returns -> {INDUSTRY_OUTPUT} ({len(industry_output_rows)} rows)")
    print("simple/excess returns are decimal returns; log returns use log(1 + simple_return)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
