from pathlib import Path

from generator import simulate_day
from generator.__main__ import main


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.glob("*.csv"))}


def test_same_seed_gives_identical_files(tmp_path):
    for run in ("a", "b"):
        main(
            ["simulate", "--days", "30", "--seed", "7", "--origin", str(tmp_path / run)]
        )

    assert _snapshot(tmp_path / "a") == _snapshot(tmp_path / "b")


def test_single_days_match_one_multi_day_run(tmp_path):
    for _ in range(7):
        simulate_day(tmp_path / "single", seed=7)
    main(
        ["simulate", "--days", "7", "--seed", "7", "--origin", str(tmp_path / "batch")]
    )

    assert _snapshot(tmp_path / "single") == _snapshot(tmp_path / "batch")


def test_different_seeds_differ(tmp_path):
    simulate_day(tmp_path / "a", seed=1)
    simulate_day(tmp_path / "b", seed=2)

    assert _snapshot(tmp_path / "a") != _snapshot(tmp_path / "b")
