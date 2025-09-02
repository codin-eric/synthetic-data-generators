"""
Daily data change generator creates a daily dataset that simulates the first day of data, changes delta days in the past
and consolidates everything into one file so you can validate against your etl

design choices:
- I'm using pandas to consolidate and psudo-randomly change the data. If the dataset gets too large this will be a problem.
- The historic change uses a log distribution to change the count of records so older days dont change too much while newer days can change more. 
- I had the idea of using a gaussian distribution to simulate the time of day when the data is created but atm this doesnt add any value.
"""
from datetime import datetime, timedelta
from pathlib import Path
import random
import math
import pandas as pd
import numpy as np
import click


RECORDS_PER_DAY = 100  # Number of records to generate per day
MAX_COUNT_RECORD = 100  # Maximum count value for each record

GAUSEAN_PEAKS = [9, 17]  # Example peaks for Gaussian distribution
GAUSEAN_SIGMA = 2.0  # Standard deviation for Gaussian distribution

SIGMOID_L = 1  # Maximum value of the sigmoid function
SIGMOID_K = 0.1  # Steepness of the curve

FOLDER_NAME = "data"
FILE_PREFIX = "data_"

ROOT_DIR = Path(__file__).parent
DATA_FOLDER = ROOT_DIR / FOLDER_NAME


def gaussean_weight(h):
    """Gaussean weight function to simulate time of day distribution

    Args:
        h (int|float): point in the curve

    Returns:
        flaot : weight at point h
    """
    weight = sum(
        math.exp(-((h - peak) ** 2) / (2 * GAUSEAN_SIGMA**2))
        for peak in GAUSEAN_PEAKS
    )
    return weight


def sigmoid_weight(x, middle_point, max_value=SIGMOID_L, steepness=SIGMOID_K):
    """Sigmoid function

    Args:
        x (int|float): point
        middle_point (int|float, optional): middle point of the curve.
        max_value (int, optional): curves maximum value. Defaults to SIGMOID_L.
        stepness (int, optional): How fast the curve increases. Defaults to SIGMOID_K.

    Returns:
        float: sigmoid curve value at x
    """
    weight = max_value / (1 + math.exp(-steepness * (x - middle_point)))
    return weight


click.command()


@click.option(
    "--start-date",
    default=datetime.today().strftime("%Y-%m-%d"),
    help="Start date of the simulation in YYYY-MM-DD format",
)
@click.option(
    "--records-per-day",
    default=RECORDS_PER_DAY,
    help="Number of records to generate per day",
)
def simulate_daily_transactions(
    start_date=datetime.today().strftime("%Y-%m-%d"), records_per_day=RECORDS_PER_DAY
):  # Casting date to str to avoid multiple types:
    """Simulate daily transactions and save to CSV files. If historic files exist, gather max id and max date to
    continue the sequence and randomly with log distribution update counts in historic files.

    Args:
        start_date (str, optional): Start date of the simulation. Defaults to today.
        records_per_day (int, optional): Number of records to generate per day. Defaults to RECORDS_PER_DAY.

    Returns:
        str: Path to the generated CSV file.
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    dfs = []

    base_id = 0
    # Check if there are historic files
    existing_files = list(DATA_FOLDER.glob(f"{FILE_PREFIX}*.csv"))
    if existing_files:
        dfs_history = []
        for historic_file in existing_files:
            dfs_history.append(pd.read_csv(historic_file))
        df_history = pd.concat(dfs_history, ignore_index=True)
        df_history["date"] = pd.to_datetime(
            df_history["date"]
        )  # TODO: Probably this could be casted on read

        base_id = df_history["id"].max()
        start_date = df_history["date"].max() + timedelta(
            days=1
        )  # next day after the last record in history
        start_date = start_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )  # reset HMS to not have issues with the random time generation
        print(f"new max_date: {start_date}")
        # change counts in the historic df with a Sigmoid distribution
        days = (df_history["date"] - df_history["date"].min()).dt.days
        middle_point = df_history["date"].dt.day.unique().size / 2
        weights = [sigmoid_weight(day, middle_point=middle_point) for day in days]

        rng = np.random.default_rng()
        mask = rng.random(size=len(df_history)) < weights

        df_history.loc[mask, "count"] = df_history.loc[mask, "count"].apply(
            lambda x: max(1, random.randint(1, MAX_COUNT_RECORD))
        )

        # save each historic file with the new counts
        history_date = df_history["date"].dt.strftime("%Y-%m-%d").unique()
        for date in history_date:
            print(f"Updating historic file for date: {date}")
            df_date = df_history[df_history["date"].dt.strftime("%Y-%m-%d") == date]
            df_date.to_csv(DATA_FOLDER / f"{FILE_PREFIX}{date}.csv", index=False)
        dfs.append(df_history)

    data_file = (
        ROOT_DIR / "data" / f"{FILE_PREFIX}{start_date.strftime('%Y-%m-%d')}.csv"
    )
    # Create the data for the current day
    records = []
    for i in range(records_per_day):
        # using a gaussian distribution to simulate the time of day when the record is created
        weights = [gaussean_weight(h) for h in range(24)]
        hour = random.choices(population=list(range(24)), weights=weights, k=1)[0]

        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        record_time = start_date + timedelta(hours=hour, minutes=minute, seconds=second)

        records.append(
            {
                "date": record_time.strftime("%Y-%m-%d %H:%M:%S"),
                "id": 1 + i + base_id,
                "count": random.randint(1, MAX_COUNT_RECORD),
            }
        )

    # Save current day data
    df_current_day = pd.DataFrame(records)
    df_current_day.to_csv(data_file, index=False)

    return data_file


if __name__ == "__main__":
    simulate_daily_transactions()
