"""Get the raw data: download one monthly file and read it."""

from pathlib import Path

import pandas as pd
import requests

from taxi_ml.config import BASE_URL, RAW_DIR


def raw_filename(year: int, month: int) -> str:
    return f"yellow_tripdata_{year}-{month:02d}.parquet"


def download_month(year: int, month: int, raw_dir: Path = RAW_DIR, base_url: str = BASE_URL) -> Path:
    """Download the monthly file into raw_dir. An existing file is not downloaded again."""
    path = raw_dir / raw_filename(year, month)
    if path.exists():
        return path

    raw_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")  # a failed download never looks like a good file
    with requests.get(f"{base_url}/{path.name}", stream=True, timeout=60) as response:
        response.raise_for_status()
        with part.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    part.rename(path)
    return path


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the pipeline to download it first.")
    return pd.read_parquet(path)
