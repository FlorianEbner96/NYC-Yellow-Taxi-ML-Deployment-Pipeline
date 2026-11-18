"""Integration tests: the pipeline stages working together on a real parquet file."""

import pandas as pd
import pytest

from taxi_ml.config import FEATURES, TARGET
from taxi_ml.data import download_month, load_raw
from taxi_ml.pipeline import build_dataset, run


@pytest.fixture
def raw_file(tmp_path, raw_trips):
    path = tmp_path / "raw" / "yellow_tripdata_2025-01.parquet"
    path.parent.mkdir()
    raw_trips.to_parquet(path)
    return path


def test_load_raw_reads_the_parquet_file(raw_file, raw_trips):
    assert len(load_raw(raw_file)) == len(raw_trips)


def test_load_raw_gives_a_clear_error_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_raw(tmp_path / "nope.parquet")


def test_download_month_skips_files_that_already_exist(raw_file):
    # base_url is unreachable on purpose: if it were used, this test would fail
    path = download_month(2025, 1, raw_file.parent, base_url="http://127.0.0.1:1/none")
    assert path == raw_file


def test_build_dataset_returns_features_and_target_for_valid_rows_only(raw_file):
    X, y = build_dataset(load_raw(raw_file))
    assert X.columns.tolist() == FEATURES
    assert len(X) == len(y) == 3
    assert y.name == TARGET
    assert y.tolist() == [10, 30, 15]


def test_build_dataset_features_have_no_missing_values(raw_file):
    X, _ = build_dataset(load_raw(raw_file))
    assert not X.isna().any().any()


def test_run_writes_the_processed_file(raw_file, tmp_path):
    out_dir = tmp_path / "processed"
    out = run(raw_path=raw_file, processed_dir=out_dir)
    assert out.exists()
    df = pd.read_parquet(out)
    assert df.columns.tolist() == FEATURES + [TARGET]
    assert len(df) == 3
