"""A live orders database, simulated as a folder of daily dumps.

The ``origin`` folder plays the database: one CSV per date, holding the orders
created on that date. Each call to :func:`simulate_day` moves the database
forward by one day.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["order_id", "created_at", "amount", "status"]
STATUSES = ["placed", "shipped", "returned"]

DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

ORDERS_PER_DAY = 100
START_DATE = "2025-01-01"

# Orders arrive in a morning and an evening rush.
RUSH_HOURS = [9, 17]
RUSH_SIGMA = 2.0

# Order amounts: median around 33, long right tail.
AMOUNT_MU = 3.5
AMOUNT_SIGMA = 0.6

DATES_REOPENED_PER_DAY = 5
UPDATE_RATE = 0.02
DELETE_RATE = 0.002

# Sigmoid over a date's position in history: recent dates are reopened far
# more often than old ones, but no date ever reaches zero.
SIGMOID_STEEPNESS = 0.2
SIGMOID_MIDPOINT_DAYS_BACK = 30
SIGMOID_MIN_HISTORY = 60


def simulate_day(
    origin: str | Path = "origin",
    seed: int | None = None,
    start_date: str = START_DATE,
    orders_per_day: int = ORDERS_PER_DAY,
) -> date:
    """Advance the database by one day and write that day's dump.

    The first call writes ``start_date``. Later calls write the day after the
    newest file already in ``origin``, so the database resumes where it left
    off and ``start_date`` is ignored.

    Args:
        origin: folder holding the database's daily dumps.
        seed: makes the run reproducible. Randomness is drawn from
            ``(seed, day index)``, so seven single calls write the same files
            as one seven-day run.
        start_date: first day, ``YYYY-MM-DD``, used only on an empty origin.
        orders_per_day: new orders created per day.

    Returns:
        The date that was written.
    """
    origin = Path(origin)
    origin.mkdir(parents=True, exist_ok=True)

    history = _load(origin)
    first_day = (
        datetime.strptime(min(history), DATE_FORMAT).date()
        if history
        else datetime.strptime(start_date, DATE_FORMAT).date()
    )
    today = (
        datetime.strptime(max(history), DATE_FORMAT).date() + timedelta(days=1)
        if history
        else first_day
    )
    day_index = (today - first_day).days
    rng = np.random.default_rng(None if seed is None else [seed, day_index])

    touched = _reopen(history, rng)
    history[today.strftime(DATE_FORMAT)] = _new_orders(
        today, _next_order_id(history), orders_per_day, rng
    )
    touched.add(today.strftime(DATE_FORMAT))

    for day_key in sorted(touched):
        _write(history[day_key], origin / f"{day_key}.csv")
    return today


def _load(origin: Path) -> dict[str, pd.DataFrame]:
    return {
        path.stem: pd.read_csv(path, dtype=str).astype({"order_id": int})
        for path in sorted(origin.glob("*.csv"))
    }


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame[COLUMNS].to_csv(path, index=False, lineterminator="\n")


def _next_order_id(history: dict[str, pd.DataFrame]) -> int:
    # The newest order is never deleted (see _reopen), so the max id survives
    # and ids are never reused.
    ids = [int(frame["order_id"].max()) for frame in history.values() if len(frame)]
    return max(ids, default=0) + 1


def _rush_weights() -> np.ndarray:
    weights = np.array(
        [
            sum(
                math.exp(-((hour - peak) ** 2) / (2 * RUSH_SIGMA**2))
                for peak in RUSH_HOURS
            )
            for hour in range(24)
        ]
    )
    return weights / weights.sum()


def _new_orders(
    day: date, first_id: int, count: int, rng: np.random.Generator
) -> pd.DataFrame:
    seconds = rng.choice(24, size=count, p=_rush_weights()) * 3600 + rng.integers(
        0, 3600, size=count
    )
    created = pd.Timestamp(day) + pd.to_timedelta(np.sort(seconds), unit="s")
    return pd.DataFrame(
        {
            "order_id": np.arange(first_id, first_id + count),
            "created_at": created.strftime(TIMESTAMP_FORMAT),
            "amount": [_format_amount(a) for a in _draw_amounts(rng, count)],
            "status": STATUSES[0],
        }
    )


def _draw_amounts(rng: np.random.Generator, count: int) -> np.ndarray:
    return np.maximum(rng.lognormal(AMOUNT_MU, AMOUNT_SIGMA, size=count), 1.0)


def _format_amount(amount: float) -> str:
    return f"{amount:.2f}"


def _pick_dates(days: list[str], rng: np.random.Generator) -> list[str]:
    span = len(days)
    if span <= DATES_REOPENED_PER_DAY:
        return days

    midpoint = (
        span - SIGMOID_MIDPOINT_DAYS_BACK if span >= SIGMOID_MIN_HISTORY else span / 2
    )
    weights = np.array(
        [
            1 / (1 + math.exp(-SIGMOID_STEEPNESS * (position - midpoint)))
            for position in range(span)
        ]
    )
    picked = rng.choice(
        span, size=DATES_REOPENED_PER_DAY, replace=False, p=weights / weights.sum()
    )
    return [days[position] for position in sorted(picked)]


def _reopen(history: dict[str, pd.DataFrame], rng: np.random.Generator) -> set[str]:
    """Update and delete some orders on past dates. Returns the dates touched."""
    if not history:
        return set()

    newest_id = _next_order_id(history) - 1
    touched = set()
    for day_key in _pick_dates(sorted(history), rng):
        frame = history[day_key].copy()
        if frame.empty:
            continue

        roll = rng.random(len(frame))
        update = roll < UPDATE_RATE
        delete = (
            (roll >= UPDATE_RATE)
            & (roll < UPDATE_RATE + DELETE_RATE)
            & (frame["order_id"] != newest_id).to_numpy()
        )

        for label in frame.index[update]:
            frame.at[label, "status"], frame.at[label, "amount"] = _change(
                frame.at[label, "status"], frame.at[label, "amount"], rng
            )

        if update.any() or delete.any():
            history[day_key] = frame.loc[~delete].reset_index(drop=True)
            touched.add(day_key)
    return touched


def _change(status: str, amount: str, rng: np.random.Generator) -> tuple[str, str]:
    """Move the status forward, correct the amount, or both."""
    can_advance = status != STATUSES[-1]
    advance = can_advance and rng.random() < 0.6
    correct = not advance or rng.random() < 0.3

    if advance:
        status = STATUSES[STATUSES.index(status) + 1]
    if correct:
        new_amount = _format_amount(_draw_amounts(rng, 1)[0])
        if new_amount == amount:
            new_amount = _format_amount(float(amount) + 0.01)
        amount = new_amount
    return status, amount
