import shutil

import pandas as pd
import pytest

from generator import simulate_day, validate
from generator.validate import NOT_SAME, SAME


@pytest.fixture
def origin(tmp_path):
    folder = tmp_path / "origin"
    for _ in range(10):
        simulate_day(folder, seed=5)
    return folder


def test_exact_copy_is_same(origin, tmp_path, capsys):
    shutil.copytree(origin, tmp_path / "destination")

    assert validate(origin, tmp_path / "destination")
    assert capsys.readouterr().out.strip() == SAME


def test_any_layout_and_parquet_is_same(origin, tmp_path):
    everything = pd.concat(pd.read_csv(path) for path in origin.glob("*.csv"))
    (tmp_path / "destination" / "nested").mkdir(parents=True)
    everything.to_parquet(tmp_path / "destination" / "nested" / "orders.parquet")

    assert validate(origin, tmp_path / "destination")


def test_one_changed_amount_is_not_same(origin, tmp_path, capsys):
    destination = shutil.copytree(origin, tmp_path / "destination")
    path = sorted(destination.glob("*.csv"))[0]
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, "amount"] = "0.01"
    frame.to_csv(path, index=False)

    assert not validate(origin, destination)
    assert capsys.readouterr().out.strip() == NOT_SAME


def test_missing_destination_is_not_same(origin, tmp_path):
    assert not validate(origin, tmp_path / "nothing_here")
