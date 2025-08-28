import pandas as pd
from generator import simulate_daily_transactions, ROOT_DIR


ETL_FOLDER = ROOT_DIR / "etl"


if __name__ == "__main__":
    # create a list of dates

    ETL_FOLDER.mkdir(parents=True, exist_ok=True)

    for _ in range(1):
        # simulate daily transaction
        file_name = simulate_daily_transactions()
        # move the daily record
        df = pd.read_csv(file_name)
        df.to_csv(ETL_FOLDER / file_name.name, index=False)