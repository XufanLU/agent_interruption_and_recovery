import json
import math
from pathlib import Path
import statistics

from scipy import stats  # type: ignore[reportMissingImports]


RESULTS_PATH = Path(__file__).with_name("live_ab_results.jsonl")
ALPHA = 0.05

wall_time_diffs = []
token_diffs = []

with RESULTS_PATH.open("r") as f:
    for line in f:
        row = json.loads(line)

        # B - A: checkpointing agent minus restart-from-zero baseline.
        wall_time_diffs.append(row["B_minus_A_wall_time_s"])
        token_diffs.append(row["B_minus_A_total_tokens"])


def summarize(values):
    n = len(values)
    if n < 2:
        raise ValueError("Need at least two paired observations for a t-test.")

    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(n)

    # Paired t-test on the per-pair B - A differences.
    degrees_of_freedom = n - 1
    t_statistic = mean / sem
    p_value = 2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
    t_critical = stats.t.ppf(1 - ALPHA / 2, degrees_of_freedom)
    ci_low = mean - t_critical * sem
    ci_high = mean + t_critical * sem

    return n, mean, sd, ci_low, ci_high, t_statistic, p_value


for name, values in [
    ("Wall time difference, B - A, seconds", wall_time_diffs),
    ("Token difference, B - A", token_diffs),
]:
    n, mean, sd, ci_low, ci_high, t_statistic, p_value = summarize(values)

    print(name)
    print(f"n = {n}")
    print(f"mean = {mean:.2f}")
    print(f"standard deviation = {sd:.2f}")
    print(f"95% t CI = [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"paired t-test: t = {t_statistic:.2f}, p = {p_value:.3g}")
    print(f"significant at alpha={ALPHA}: {p_value < ALPHA}")
    print()
