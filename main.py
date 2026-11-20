"""gta6-trophy-rarity-forecast-01: forecast each GTA VI (PS5) trophy's Exophase
earn rate at the case's target, as five quantiles, from the house's recorded
reads of the trophy page.

Method (standard library only, deterministic, no network, no randomness)
------------------------------------------------------------------------
For every trophy, fit a straight line to logit(rate) against ln(1 + hours
since 2026-11-19T00:00:00Z) over the case's history, then read the line at
the target.  The log-time axis is the house curve baseline's: trophy
progress decelerates as the launch cohort ages, so rates move roughly
linearly in log time, not in clock time.  Three things make this more than
the house curve baseline:

1.  Refresh states, not reads.  Exophase refreshes a game's counts on its
    own schedule, hours apart, so most reads repeat the previous page
    byte for byte.  Fitting every read would let long stale stretches
    dominate the fit and understate the noise.  The history is first
    reduced to its distinct states (a read counts only when its rate
    vector or its players-tracked count changed), and the line is fitted
    on those.
2.  Shrinkage.  Each trophy's slope is shrunk toward its tier's median
    slope (platinum / gold / silver / bronze), with weight
    n / (n + 4) on the trophy's own slope, n the number of distinct
    states, and the intercept refitted given the shrunk slope.  A trophy
    with a noisy or short history borrows strength from its tier; a
    trophy with a long, clean history mostly speaks for itself.  The
    extrapolated median is clamped to at most 5 logit units from the
    trophy's last rate, so no extrapolation runs away.  The line is
    fitted on the last 24 hours of distinct states, as the house curve
    baseline does: the launch's early regime (a flood of new players
    diluting every rate) is not the regime that extrapolates, and a
    trophy's rate curve is concave in log time, so a line through the
    whole history sits below its recent points.  With fewer than 3
    distinct states in the window the fit falls back to all states.
3.  An honest spread.  The logit spread s combines, in quadrature: the
    house random-walk term (0.5 * sqrt(horizon in days)); the fit's
    prediction interval (residual scatter times the usual leverage
    factor for extrapolating to the target); a 0.35 floor for the
    refresh jumps Exophase makes between reads (launch night's first
    refresh moved every rate but the 100% one, and the practice cases
    exist precisely because a flat history can hide one); and a
    dilution term from the tracked population's growth over the last
    24 h of history, since a fast-growing player base dilutes every
    rate.  s is floored at 0.5 and capped at 2.0 logit units.

Quantiles are inv_logit(median_logit + z_tau * s) with
z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals half-up,
exactly as the house baseline rounds them.

What it does on the practice cases: their histories are one flat page
(638 players, every rate unchanged), so every slope is 0 and the median
is the last rate, as the house baseline says -- but with a wider spread
(0.5 instead of 0.125 logit), because a flat history is exactly when a
refresh is most likely to be hiding.  On the final case, whose history
covers two days of real reads, the trend -- not persistence -- carries
the median, which is where this entry expects to beat the house.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there):
    JSON {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
    "horizon_minutes", "target", "levels", "game", "trophies",
    "history", "notes"}, where `trophies` lists the case's trophies
    {id, name, description, tier, points, secret} and `history` every
    read before `at`, oldest first, each {slot, at, players_tracked,
    trophies_earned, rates}, `rates` in the order of `trophies` (null
    where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1,
    "q": [q05, q25, q50, q75, q95]}, ...]}, one row for each of the
    case's trophies, each id once, no other key, each q five finite
    numbers from 0 to 100 (percent), non-decreasing; at most 65,536
    bytes.  Anything else is an invalid case.

Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release day begins; the log-time axis is measured from here, as the
# house curve baseline measures it.
ANCHOR = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread pieces, in logit units.
SIGMA_RW = 0.5        # random-walk term per sqrt(day), as the house baseline
REFRESH_FLOOR = 0.35  # unmodelled refresh-jump / staleness uncertainty
SPREAD_MIN = 0.5
SPREAD_MAX = 2.0
MAX_LOGIT_MOVE = 5.0  # median may move at most this far from the last rate
MAX_SLOPE = 2.5       # shrunk slope cap, logit units per log-time unit
SHRINK_K = 4.0        # pseudo-count pulling a slope toward its tier median


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(at: datetime) -> float:
    """ln(1 + hours since the anchor). Guarded so pre-anchor reads stay defined."""
    hours = (at - ANCHOR).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd, xbar, sxx, n)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, xbar, sxx, n
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        sd = math.sqrt(resid / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, xbar, sxx, n


def distinct_states(history):
    """Collapse the history to its distinct states: a read counts only when
    its rate vector or its players-tracked count changed. Exophase
    refreshes in jumps, so the runs of identical reads carry no new
    information and would otherwise overweight stale stretches."""
    states = []
    seen = set()
    for read in history:
        key = (tuple(read.get("rates") or []), read.get("players_tracked"))
        if key in seen:
            continue
        seen.add(key)
        states.append(read)
    return states


def dilution_term(states):
    """Extra spread from the tracked population's growth over the last 24 h
    of history: fast growth dilutes every rate, and the next refresh can
    bring a jump the trend has not seen yet."""
    if len(states) < 2:
        return 0.0
    last = states[-1]
    n_last = last.get("players_tracked")
    if not n_last or n_last <= 0:
        return 0.0
    t_last = parse_time(last["at"])
    # The earliest state at least 24 h before the last read, else the first.
    base_n = None
    for read in states:
        if (t_last - parse_time(read["at"])).total_seconds() >= 24 * 3600:
            base_n = read.get("players_tracked")
    if base_n is None:
        base_n = states[0].get("players_tracked")
    if not base_n or base_n <= 0:
        return 0.0
    growth = math.log(n_last / base_n)
    return min(0.30, 0.20 * max(0.0, growth))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    states = distinct_states(case["history"])
    last = case["history"][-1]
    last_rates = last["rates"]

    # The fit window: the last 24 h of distinct states, falling back to
    # all states when the window holds fewer than 3 distinct times.
    t_last = parse_time(last["at"])
    window = [r for r in states
              if (t_last - parse_time(r["at"])).total_seconds() <= 24 * 3600]
    fit_states = window if len({log_time(parse_time(r["at"])) for r in window}) >= 3 else states

    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = log_time(target)
    base_spread = SIGMA_RW * math.sqrt(horizon_days)
    dil = dilution_term(states)

    # Per-trophy fits on the fit window's distinct states.
    fits = []  # (slope, resid_sd, n, xbar, sxx, last_logit)
    for i in range(n_tr):
        xs, ys = [], []
        for read in fit_states:
            rate = read["rates"][i]
            if rate is None:
                continue
            xs.append(log_time(parse_time(read["at"])))
            ys.append(logit(rate))
        if len(xs) >= 2 and len(set(xs)) >= 2:
            a, b, sd, xbar, sxx, n = ols(xs, ys)
        elif ys:
            a, b, sd, xbar, sxx, n = ys[-1], 0.0, 0.0, xs[-1], 0.0, len(xs)
        else:
            a, b, sd, xbar, sxx, n = logit(1.0), 0.0, 0.0, 0.0, 0.0, 0
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = next(
                (r["rates"][i] for r in reversed(case["history"]) if r["rates"][i] is not None),
                1.0,
            )
        fits.append([b, sd, n, xbar, sxx, logit(last_rate), a])

    # Tier slope priors: the median fitted slope within each tier.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f[0] for f, t in zip(fits, trophies) if t["tier"] == tier and f[2] >= 2]
        if slopes:
            tier_slopes[tier] = median(slopes)
    global_slope = median([f[0] for f in fits if f[2] >= 2]) if any(f[2] >= 2 for f in fits) else 0.0
    tier_resid = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        sds = [f[1] for f, t in zip(fits, trophies) if t["tier"] == tier and f[2] >= 3 and f[1] > 0]
        if sds:
            tier_resid[tier] = median(sds)

    rows = []
    for i, trophy in enumerate(trophies):
        b, sd, n, xbar, sxx, last_lg, _a = fits[i]
        prior = tier_slopes.get(trophy["tier"], global_slope)
        # Shrink the slope toward the tier median; refit the intercept
        # given the shrunk slope so the line still passes through the data.
        w = n / (n + SHRINK_K) if n > 0 else 0.0
        b_star = w * b + (1.0 - w) * prior
        b_star = max(-MAX_SLOPE, min(MAX_SLOPE, b_star))
        # Refit intercept from the fit window's distinct states with the shrunk slope.
        xs, ys = [], []
        for read in fit_states:
            rate = read["rates"][i]
            if rate is None:
                continue
            xs.append(log_time(parse_time(read["at"])))
            ys.append(logit(rate))
        if xs:
            a_star = sum(y - b_star * x for x, y in zip(xs, ys)) / len(xs)
        else:
            a_star = last_lg
        med_lg = a_star + b_star * x_target
        # Near the 0 / 100 boundary a logit is mostly dilution noise, not
        # signal: a near-certain trophy may not fall far, a near-impossible
        # one may not rise far, whatever the fitted line says.
        move_cap = 1.5 if (last_rates[i] is not None and (last_rates[i] >= 95 or last_rates[i] <= 0.02)) else MAX_LOGIT_MOVE
        med_lg = max(last_lg - move_cap, min(last_lg + move_cap, med_lg))
        med_lg = max(logit(0.01), min(logit(99.99), med_lg))

        # Spread: random walk + prediction interval + refresh floor + dilution.
        resid = sd if (n >= 3 and sd > 0) else tier_resid.get(trophy["tier"], 0.15)
        if n >= 3 and sxx > 1e-12:
            lev = 1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx
            pred = resid * math.sqrt(min(lev, 9.0))
        else:
            pred = resid
        pred = min(pred, 1.0)
        spread = math.sqrt(base_spread ** 2 + pred ** 2 + REFRESH_FLOOR ** 2 + dil ** 2)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))

        q = [round4(inv_logit(med_lg + z * spread)) for z in Z]
        # Rounding can only break monotonicity by a hair; enforce it.
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
