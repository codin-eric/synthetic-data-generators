"""Check whether your destination holds the same data as the database."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from generator.database import COLUMNS

SAME = "origin and destination are the same"
NOT_SAME = "origin and destination are not the same"


def validate(
    origin: str | Path = "origin", destination: str | Path = "destination"
) -> bool:
    """Compare every order in ``origin`` with every order in ``destination``.

    ``destination`` may hold CSV or Parquet files, in any folder layout. Prints
    one line saying whether the two are the same.
    """
    same = _same(_read(Path(origin), "*.csv"), _read(Path(destination), "**/*"))
    print(SAME if same else NOT_SAME)
    return same


def _read(folder: Path, pattern: str) -> pd.DataFrame | None:
    frames = []
    for path in sorted(folder.glob(pattern)):
        if path.suffix == ".csv":
            frames.append(pd.read_csv(path))
        elif path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    if not set(COLUMNS) <= set(frame.columns):
        return None
    return _normalize(frame[COLUMNS])


def _normalize(frame: pd.DataFrame) -> pd.DataFrame | None:
    try:
        normalized = pd.DataFrame(
            {
                "order_id": frame["order_id"].astype("int64"),
                "created_at": pd.to_datetime(frame["created_at"]).dt.floor("s"),
                "amount_cents": (frame["amount"].astype(float) * 100)
                .round()
                .astype("int64"),
                "status": frame["status"].astype(str),
            }
        )
    except (ValueError, TypeError):
        return None
    return normalized.sort_values(list(normalized.columns)).reset_index(drop=True)


def _same(origin: pd.DataFrame | None, destination: pd.DataFrame | None) -> bool:
    if origin is None or destination is None:
        return False
    return origin.equals(destination)
