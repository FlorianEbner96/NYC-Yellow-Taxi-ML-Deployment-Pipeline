"""Entry point: download -> load -> clean -> features -> data/processed/*.parquet.

Run with: uv run python -m taxi_ml.pipeline
"""

from pathlib import Path

import pandas as pd

from taxi_ml.clean import clean
from taxi_ml.config import MONTH, PROCESSED_DIR, PROCESSED_NAME, TARGET, YEAR
from taxi_ml.data import download_month, load_raw
from taxi_ml.features import build_features


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw trips -> (features X, target y). Pure: no file access."""
    cleaned = clean(raw)
    return build_features(cleaned), cleaned[TARGET]


def run(raw_path: Path | None = None, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Run all stages and write features + target to one parquet file."""
    raw_path = raw_path or download_month(YEAR, MONTH)
    raw = load_raw(raw_path)
    X, y = build_dataset(raw)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / PROCESSED_NAME
    X.assign(**{TARGET: y}).reset_index(drop=True).to_parquet(out)
    print(f"{len(raw):,} raw rows -> {len(X):,} rows kept ({len(X) / len(raw):.1%}) -> {out}")
    return out


if __name__ == "__main__":
    run()
