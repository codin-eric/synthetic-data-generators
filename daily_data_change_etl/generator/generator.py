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

ID_FILE_POSITION = 1  # Position of the ID column in the CSV file (0-indexed)

GAUSEAN_PEAKS = [9, 17]  # Example peaks for Gaussian distribution
GAUSEAN_SIGMA = 2.0  # Standard deviation for Gaussian distribution

SIGMOID_L = 1  # Maximum value of the sigmoid function
SIGMOID_K = 0.2  # Steepness of the curve
SIGMOID_MIDDLE_POINT_DAYS = 30  # Ammount of days before max days to calculate Sigmoids middle point. This should modify aprox 2 months in the past
SIGMOID_MIDDLE_POINT_MIN_DAYS = (
    60  # Minimum days to set Sigmoids middle point. Before is SUM(days)/2
)

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
        math.exp(-((h - peak) ** 2) / (2 * GAUSEAN_SIGMA**2)) for peak in GAUSEAN_PEAKS
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
@click.option(
    "--simulate-historic-changes",
    default=True,
    help="Whether to simulate changes in historic data",
)
def simulate_daily_transactions(
    start_date=datetime.today().strftime("%Y-%m-%d"),
    records_per_day=RECORDS_PER_DAY,
    simulate_historic_changes=True,
):  # Casting date to str to avoid multiple types:
    """Simulate daily transactions and save to CSV files. If historic files exist, gather max id and max date to
    continue the sequence and randomly with log distribution update counts in historic files.

    Args:
        start_date (str, optional): Start date of the simulation. Defaults to today.
        records_per_day (int, optional): Number of records to generate per day. Defaults to RECORDS_PER_DAY.
        simulate_historic_changes (bool, optional): Whether to simulate changes in historic data. Defaults to True.

    Returns:
        str: Path to the generated CSV file.
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    base_id = 0
    # Check if there are historic files
    existing_files = list(DATA_FOLDER.glob(f"{FILE_PREFIX}*.csv"))
    if existing_files:
        existing_files.sort()
        first_file = existing_files[0]
        last_file = existing_files[-1]

        # Get the max id by reading the last line of the last file
        with open(last_file, "rb") as f:
            f.seek(-2, 2)  # Jump to the second last byte.
            while f.read(1) != b"\n":  # Until EOL is found...
                f.seek(-2, 1)
            last_line = f.readline().decode()  # Read last line.
        base_id = int(last_line.split(",")[ID_FILE_POSITION]) + 1

        first_date_str = first_file.stem.replace(FILE_PREFIX, "")
        first_date = datetime.strptime(first_date_str, "%Y-%m-%d")
        last_date_str = last_file.stem.replace(FILE_PREFIX, "")
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")

        # Start date for the new data
        start_date = last_date + timedelta(
            days=1
        )  # next day after the last record in history
        start_date = start_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )  # reset HMS to not have issues with the random time generation

        # change counts in the historic df with a Sigmoid distribution
        if simulate_historic_changes:
            days_ammount = (last_date - first_date).days + 1
            days = range(1, days_ammount)
            if days_ammount < SIGMOID_MIDDLE_POINT_MIN_DAYS:
                middle_point = days_ammount / 2
            else:
                middle_point = days_ammount - SIGMOID_MIDDLE_POINT_DAYS

            weights = [sigmoid_weight(day, middle_point=middle_point) for day in days]
            change_days = set(
                random.choices(population=days, weights=weights, k=int(middle_point))
            )

            for date_number in change_days:
                date = first_date + timedelta(days=date_number)
                history_fn = f"{FILE_PREFIX}{date.strftime('%Y-%m-%d')}.csv"
                df_history = pd.read_csv(
                    DATA_FOLDER / history_fn,
                    parse_dates=["date"],
                )
                # apply a mask to change aprox 10% of the records # TODO: this should also be a distribution
                mask = np.random.rand(len(df_history)) < 0.1
                df_history.loc[mask, "count"] = df_history.loc[mask, "count"].apply(
                    lambda x: random.randint(1, MAX_COUNT_RECORD)
                )

                df_history.to_csv(DATA_FOLDER / history_fn, index=False)

    # Create the data for the current day
    data_file = (
        ROOT_DIR / "data" / f"{FILE_PREFIX}{start_date.strftime('%Y-%m-%d')}.csv"
    )

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
