from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

from .results import DEFAULT_RESULTS_LOG_PATH


METRIC_KEYS = [
    ("Wall time difference, B - A, seconds", "B_minus_A_wall_time_s"),
    ("Token difference, B - A", "B_minus_A_total_tokens"),
    ("Test-run difference, B - A", "B_minus_A_test_runs"),
    ("Agent-run difference, B - A", "B_minus_A_agent_runs"),
    ("Repeated-diagnosis difference, B - A", "B_minus_A_repeated_diagnoses"),
    ("File-write difference, B - A", "B_minus_A_file_writes"),
]


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _values(rows: Iterable[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _summarize(values: list[float]) -> tuple[int, float, float | None, float | None, float | None, float | None]:
    n = len(values)
    if n == 0:
        raise ValueError("No values to summarize.")
    mean = statistics.mean(values)
    if n == 1:
        return n, mean, None, None, None, None

    sd = statistics.stdev(values)
    if sd == 0:
        return n, mean, sd, mean, mean, None

    sem = sd / math.sqrt(n)

    try:
        from scipy import stats  # type: ignore[import-not-found]

        t_critical = stats.t.ppf(0.975, n - 1)
        p_value = 2 * stats.t.sf(abs(mean / sem), n - 1)
    except Exception:
        t_critical = 1.96
        p_value = None

    ci_low = mean - t_critical * sem
    ci_high = mean + t_critical * sem
    return n, mean, sd, ci_low, ci_high, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize debug-flow A/B JSONL results.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_LOG_PATH)
    args = parser.parse_args()

    rows = _load_rows(args.results)
    if not rows:
        raise SystemExit(f"No result rows found in {args.results}")

    print(f"Results file: {args.results}")
    print(f"rows = {len(rows)}")
    print()

    for label, key in METRIC_KEYS:
        values = _values(rows, key)
        if not values:
            continue
        n, mean, sd, ci_low, ci_high, p_value = _summarize(values)
        print(label)
        print(f"n = {n}")
        print(f"mean = {mean:.2f}")
        if sd is not None:
            print(f"standard deviation = {sd:.2f}")
            print(f"95% CI = [{ci_low:.2f}, {ci_high:.2f}]")
            if p_value is not None:
                print(f"paired t-test p = {p_value:.3g}")
            elif sd == 0:
                print("paired t-test p = unavailable (no variation across paired differences)")
            else:
                print("paired t-test p = unavailable (scipy not installed)")
        else:
            print("standard deviation = unavailable for n=1")
        print()


if __name__ == "__main__":
    main()
