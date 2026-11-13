# Daily data change ETL

An orders database grows by one day of orders every day. Every night you
extract it into your own storage. The goal is simple: your copy must always
match the database.

All you get is its tables, one file per day of orders.

## The data

`origin/` plays the database. It holds one CSV per day, named after the date
the orders were created, such as `origin/2025-01-01.csv`:

| column | type | meaning |
|---|---|---|
| `order_id` | int | unique, increasing across days |
| `created_at` | timestamp | when the order was placed, within the file's date |
| `amount` | decimal(10,2) | order amount |
| `status` | text | `placed`, `shipped` or `returned` |

Each run of the generator advances the database by one or more days.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
cd daily_data_change_etl
uv sync
uv run python -m generator simulate --seed 42             # advance one day
uv run python -m generator simulate --days 30 --seed 42   # advance 30 days
uv run python -m generator validate                       # compare origin/ and destination/
```

`simulate` options: `--days` (default 1), `--seed`, `--origin` (default
`origin`), `--start-date` (default `2025-01-01`) and `--orders-per-day`
(default 100). The same `--seed` always produces the same database, and the
database resumes from where `origin/` left off. Delete `origin/` to start
over.

`validate` prints one line, either `origin and destination are the same` or
`origin and destination are not the same`.

## The challenge

Simulate a 7-day run. Each day, run `simulate`, then copy that day's new file
from `origin/` to `destination/`. After day 7, run `validate` to compare
`origin/` with `destination/`.

Find out why they differ. Then write an extraction that runs once after every
simulated day and keeps them the same, for any number of days.

What a solution must produce:

- `destination/` holds CSV or Parquet files, in any folder layout, with the
  columns `order_id`, `created_at`, `amount` and `status`.
- `validate` reads **every** `.csv` and `.parquet` file under `destination/`,
  at any depth. Every row it finds counts, so leave only the current rows
  there.
- After each simulated day, `validate` says `origin and destination are the
  same`.

`intro.ipynb` runs the 7-day simulation step by step, with a plot:

```bash
uv sync --group notebook
uv run jupyter lab intro.ipynb
```

## Use any tool

`pandas` and `cp` are only a starting point. Solve it with the tool you want
to practice: Spark, DuckDB, Polars, Delta Lake, dbt or plain SQL.

## Tests

The generator's own tests (determinism, schema, `validate`) run offline:

```bash
uv run pytest -q
```

## Solutions

Community solutions live in `solutions/`, one folder per author. They contain spoilers: try the challenge first. To add yours, read
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Community

Questions and discussion go on the
[Discord](https://discord.com/invite/GREKfe7), a Spanish-speaking community.
