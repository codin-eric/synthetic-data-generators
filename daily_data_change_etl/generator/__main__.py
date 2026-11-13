"""Command line: ``python -m generator simulate`` / ``python -m generator validate``."""

from __future__ import annotations

import argparse
import sys

from generator.database import ORDERS_PER_DAY, START_DATE, simulate_day
from generator.validate import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m generator")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="advance the database")
    simulate.add_argument("--days", type=int, default=1)
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument("--origin", default="origin")
    simulate.add_argument("--start-date", default=START_DATE)
    simulate.add_argument("--orders-per-day", type=int, default=ORDERS_PER_DAY)

    check = commands.add_parser("validate", help="compare origin and destination")
    check.add_argument("--origin", default="origin")
    check.add_argument("--destination", default="destination")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        for _ in range(args.days):
            written = simulate_day(
                args.origin, args.seed, args.start_date, args.orders_per_day
            )
        print(f"database is now at {written}")
        return 0
    return 0 if validate(args.origin, args.destination) else 1


if __name__ == "__main__":
    sys.exit(main())
