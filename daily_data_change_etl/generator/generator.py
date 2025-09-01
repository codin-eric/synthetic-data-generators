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

FOLDER_NAME = "data"
FILE_PREFIX = "data_"

ROOT_DIR = Path(__file__).parent
DATA_FOLDER = ROOT_DIR / FOLDER_NAME 


def gaussean_weight(h):
    weight = sum(math.exp(-((h - peak) ** 2) / (2 * GAUSEAN_SIGMA ** 2)) for peak in GAUSEAN_PEAKS)
    return weight


def log_weight(days):  # TODO: it would be nice to be able to change the shape of the log distribution
    return np.log1p(days) / np.log1p(days.max())


click.command()
@click.option('--start_date', default=datetime.today().strftime('%Y-%m-%d'), help='Start date of the simulation in YYYY-MM-DD format')
def simulate_daily_transactions(start_date=datetime.today().strftime('%Y-%m-%d')):  # Casting date to str to avoid multiple types:
    """Simulate daily transactions and save to CSV files. If historic files exist, gather max id and max date to
    continue the sequence and randomly with log distribution update counts in historic files.

    Args:
        start_date (str, optional): Start date of the simulation. Defaults to today.

    Returns:
        str: Path to the generated CSV file. 
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    # folder and file definition 
    consolidated_file = DATA_FOLDER / "consolidated.csv"

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
        df_history["date"] = pd.to_datetime(df_history["date"])  #TODO: Probably this could be casted on read

        base_id = df_history['id'].max() 
        start_date = df_history['date'].max() + timedelta(days=1) # next day after the last record in history
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0) # reset HMS to not have issues with the random time generation
        print(f"new max_date: {start_date}")
        # change counts in the historic df with a log distribution
        days = (df_history['date'] - df_history['date'].min()).dt.days + 1  # avoid log zero
        weights = log_weight(days) 

        rng = np.random.default_rng()
        mask = rng.random(size=len(df_history)) < weights

        df_history.loc[mask, 'count'] = df_history.loc[mask, 'count'].apply(lambda x: max(1, random.randint(1, MAX_COUNT_RECORD)))

        # save each historic file with the new counts
        history_date = df_history['date'].dt.strftime('%Y-%m-%d').unique()
        for date in history_date:
            print(f"Updating historic file for date: {date}")
            df_date = df_history[df_history['date'].dt.strftime('%Y-%m-%d') == date]
            df_date.to_csv(DATA_FOLDER / f"{FILE_PREFIX}{date}.csv", index=False)
        dfs.append(df_history)


    data_file = ROOT_DIR / "data" / f"{FILE_PREFIX}{start_date.strftime('%Y-%m-%d')}.csv"
    # Create the data for the current day
    records = []
    for i in range(RECORDS_PER_DAY):
        # using a gaussian distribution to simulate the time of day when the record is created
        weights = [gaussean_weight(h) for h in range(24)]
        hour = random.choices(population=list(range(24)), weights=weights, k=1)[0]

        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        record_time = start_date + timedelta(hours=hour, minutes=minute, seconds=second)

        records.append(
            {
                "date": record_time.strftime("%Y-%m-%d %H:%M:%S"),
                "id": 1+i+base_id, 
                "count": random.randint(1, MAX_COUNT_RECORD)
            }
        )

    # Save current day data
    df_current_day = pd.DataFrame(records)
    df_current_day.to_csv(data_file, index=False)

    #consolidate all dfs in one
    dfs.append(df_current_day)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(consolidated_file, index=False)

    return data_file


if __name__ == "__main__":
    simulate_daily_transactions()