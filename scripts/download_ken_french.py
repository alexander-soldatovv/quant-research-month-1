#!/usr/bin/env python3
"""Download and clean selected Kenneth French Data Library datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import ssl
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import certifi
except ImportError:  # pragma: no cover - optional local dependency
    certifi = None


BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


@dataclass(frozen=True)
class Dataset:
    key: str
    url: str
    raw_csv_name: str
    cleaned_csv_name: str
    expected_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


DATASETS = (
    Dataset(
        key="fama_french_3_factors",
        url=f"{BASE_URL}/F-F_Research_Data_Factors_CSV.zip",
        raw_csv_name="F-F_Research_Data_Factors.csv",
        cleaned_csv_name="fama_french_3_factors_monthly.csv",
        expected_columns=("date", "Mkt-RF", "SMB", "HML", "RF"),
        numeric_columns=("Mkt-RF", "SMB", "HML", "RF"),
    ),
    Dataset(
        key="10_industry_portfolios",
        url=f"{BASE_URL}/10_Industry_Portfolios_CSV.zip",
        raw_csv_name="10_Industry_Portfolios.csv",
        cleaned_csv_name="10_industry_portfolios_monthly_vw.csv",
        expected_columns=(
            "date",
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
        ),
        numeric_columns=(
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
        ),
    ),
)


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "quant-research-month-1/1.0"})
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    try:
        with urlopen(request, timeout=60, context=context) as response:
            return response.read()
    except URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error) or not url.startswith("https://"):
            raise
        fallback_url = "http://" + url.removeprefix("https://")
        fallback_request = Request(
            fallback_url,
            headers={"User-Agent": "quant-research-month-1/1.0"},
        )
        with urlopen(fallback_request, timeout=60) as response:
            return response.read()


def extract_single_csv(zip_bytes: bytes) -> tuple[str, str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in archive, found: {names}")
        name = names[0]
        return name, archive.read(name).decode("utf-8-sig")


def is_monthly_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", value.strip()))


def parse_monthly_table(raw_csv: str, expected_columns: Iterable[str]) -> list[dict[str, str]]:
    expected_columns = tuple(expected_columns)
    rows = list(csv.reader(io.StringIO(raw_csv)))

    header_index = None
    for index, row in enumerate(rows):
        if row and row[0].strip() == "":
            cleaned_header = ["date"] + [cell.strip() for cell in row[1:] if cell.strip()]
            if cleaned_header == list(expected_columns):
                header_index = index
                break

    if header_index is None:
        raise ValueError(f"Could not find monthly table header: {expected_columns}")

    parsed_rows: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not row or not is_monthly_date(row[0]):
            if parsed_rows:
                break
            continue

        padded = row[: len(expected_columns)]
        record = {
            column: value.strip()
            for column, value in zip(expected_columns, padded, strict=False)
        }
        record["date"] = f"{record['date'][:4]}-{record['date'][4:]}-01"
        parsed_rows.append(record)

    if not parsed_rows:
        raise ValueError(f"No monthly rows parsed for columns: {expected_columns}")

    return parsed_rows


def write_clean_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_metadata(dataset: Dataset, source_file_name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "key": dataset.key,
        "source_url": dataset.url,
        "source_file_name": source_file_name,
        "cleaned_file_name": dataset.cleaned_csv_name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": len(rows),
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "columns": list(dataset.expected_columns),
        "units": "percent returns",
    }


def download_dataset(dataset: Dataset, raw_dir: Path, processed_dir: Path) -> dict[str, object]:
    zip_bytes = download_bytes(dataset.url)
    source_file_name, raw_csv = extract_single_csv(zip_bytes)

    raw_zip_path = raw_dir / Path(dataset.url).name
    raw_csv_path = raw_dir / dataset.raw_csv_name
    cleaned_csv_path = processed_dir / dataset.cleaned_csv_name

    raw_zip_path.write_bytes(zip_bytes)
    raw_csv_path.write_text(raw_csv)

    rows = parse_monthly_table(raw_csv, dataset.expected_columns)
    write_clean_csv(cleaned_csv_path, dataset.expected_columns, rows)

    metadata = build_metadata(dataset, source_file_name, rows)
    metadata["raw_zip_path"] = raw_zip_path.as_posix()
    metadata["raw_csv_path"] = raw_csv_path.as_posix()
    metadata["cleaned_csv_path"] = cleaned_csv_path.as_posix()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Fama/French 3 factors and 10 industry portfolios."
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = [
        download_dataset(dataset, args.raw_dir, args.processed_dir)
        for dataset in DATASETS
    ]

    metadata_path = args.processed_dir / "ken_french_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for item in metadata:
        print(
            f"{item['key']}: {item['row_count']} rows, "
            f"{item['start_date']} to {item['end_date']} -> {item['cleaned_csv_path']}"
        )
    print(f"metadata -> {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
