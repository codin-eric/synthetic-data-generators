from datetime import date

import pandas as pd

from generator import simulate_day


def _simulate(origin, days):
    return [simulate_day(origin, seed=3) for _ in range(days)]


def test_one_file_per_date(tmp_path):
    written = _simulate(tmp_path, 10)

    assert written[0] == date(2025, 1, 1)
    assert [d.isoformat() for d in written] == sorted(
        path.stem for path in tmp_path.glob("*.csv")
    )


def test_resumes_after_newest_file(tmp_path):
    _simulate(tmp_path, 3)

    assert simulate_day(tmp_path, seed=3, start_date="2030-06-01") == date(2025, 1, 4)


def test_columns_and_types(tmp_path):
    _simulate(tmp_path, 10)

    for path in tmp_path.glob("*.csv"):
        frame = pd.read_csv(path, dtype={"amount": str})
        assert list(frame.columns) == ["order_id", "created_at", "amount", "status"]
        assert frame["order_id"].dtype == "int64"
        created = pd.to_datetime(frame["created_at"], format="%Y-%m-%d %H:%M:%S")
        assert (created.dt.strftime("%Y-%m-%d") == path.stem).all()
        assert frame["amount"].str.fullmatch(r"\d+\.\d{2}").all()
        assert frame["status"].isin(["placed", "shipped", "returned"]).all()


def test_order_ids_are_unique(tmp_path):
    _simulate(tmp_path, 30)

    ids = pd.concat(pd.read_csv(path) for path in tmp_path.glob("*.csv"))["order_id"]
    assert ids.is_unique


def test_orders_per_day(tmp_path):
    simulate_day(tmp_path, seed=3, orders_per_day=25)

    assert len(pd.read_csv(tmp_path / "2025-01-01.csv")) == 25
