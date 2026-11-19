"""GTA VI trophy rarity forecaster for gta6-trophy-rarity-forecast-01.

Idea in one paragraph: an earn rate is a ratio, earners over players tracked,
and the two sides move for different reasons, so the forecast models them
separately and divides.  Players tracked grows as more people link their
accounts to Exophase, in jumps whenever Exophase refreshes a game's counts;
earners grow as the tracked players play through the game.  Both are fitted
as straight lines of log count against log time since release (launch
dynamics are power laws, not straight lines in clock time), the earners
fitted per trophy and the players once for the whole game, and each is read
at the target and divided to give the median rate.  A second, direct fit --
the logit of the rate itself against log time over the last day -- is blended
in half and half, so neither view can run away with the forecast, and the
move from the last read is damped by a quarter, because a 4.5-day
extrapolation of launch week deserves humility.  The spread adds two honest
sources of doubt in quadrature: a random walk that widens with the square
root of the horizon, at half the house baseline's rate (a trend forecast
with a fitted median earns a tighter spread than a no-change one), and the
residual wobble of the direct fit, which is where the refresh jumps the
history has shown already live -- the fitted reads are post-refresh values,
so counting the jumps again on top would double-count them.  The spread is
floored so the forecast never claims false precision and capped so it never
spreads without bound.
When the history is perfectly flat -- every read the same page, as on launch
night -- there is no trend to fit, and the forecast falls back to what the
rules themselves publish: Exophase had not refreshed the page for hours, so
a refresh over the horizon is the base case, not a tail event.  The first
refresh took tracked players from 638 to 3,868, a sixfold dilution, while
earners grew only modestly in those hours, so the median rate is cut to
about a fifth of the last read, with easier trophies (the ones the launch
night's own rates show new players earn first) cut less and late, grinding
trophies cut more.  A trophy already at 100% stays at 100%: dilution cannot
take it down, since everyone tracked holds it.

Everything here is standard library only, deterministic, and offline: no
randomness, no clock, no network.  The same input bytes always produce the
same output bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# The standard normal's quantile at each level.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# --- flat-history (no refresh seen yet) branch ---------------------------
FLAT_REFRESH_HOURS = 5.0     # Exophase refreshes a game's counts hours apart
FLAT_MEDIAN_RATIO = 0.20     # one refresh: ~6x more players, ~1.2x earners
FLAT_SIGMA = 0.5             # logit spread around the diluted median
FLAT_SIGMA_PER_LOG = 0.15    # spread grows slowly with a longer horizon
FLAT_SHIFT_PER_LOG = 0.35    # later refreshes dilute less; shift grows slowly
FLAT_SHIFT_CAP = 3.0         # never shift the median more than this (logits)
FLAT_SIGMA_CAP = 1.2
FLAT_MODULATION = 0.5        # how strongly the launch-night rate moves the cut
FLAT_RATIO_LO = 0.10         # per-trophy median ratio is clamped to this band
FLAT_RATIO_HI = 0.45

# --- trend branch ---------------------------------------------------------
DAMP = 0.75                  # shrink the extrapolated logit move toward 0
BLEND_TREND = 0.5            # weight on the direct logit-rate trend (rest: split)
MAX_LOGIT_MOVE = 2.5         # never move the median more than this (logits)
FIT_WINDOW_HOURS = 36.0      # earners/players fits use the last 36 hours
TREND_WINDOW_HOURS = 24.0    # direct logit-rate fit uses the last 24 hours
RW_SIGMA_PER_DAY = 0.25      # random-walk spread, logit units per sqrt day
SIGMA_FLOOR = 0.35           # never claim more precision than this (logits)
SIGMA_CAP = 1.0              # never spread wider than this (logits)
MIN_FIT_POINTS = 3           # a line needs at least this many distinct reads

# A trophy the last read puts at (essentially) 100% cannot be diluted.
SATURATED = 99.99
SATURATED_SIGMA = 0.15

# Degenerate fallback when a case carries no usable history at all.
TIER_PRIOR = {"platinum": 0.10, "gold": 0.30, "silver": 0.50, "bronze": 1.00}


def _parse_time(text):
    """ISO-8601 timestamp -> aware datetime, tolerating a trailing Z."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def _logit(pct):
    p = _clamp(pct)
    return math.log(p / (100.0 - p))


def _inv_logit(lo):
    if lo >= 40.0:
        return 100.0
    if lo <= -40.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lo))


def _round4(value):
    """Round to 4 decimals, halves away from zero, as a JSON number."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def _z_for(level):
    """Standard normal quantile at a level; exact for the contest's five."""
    for known, z in zip(LEVELS, Z):
        if level == known:
            return z
    # Any other level: invert the normal CDF by bisection on erf.
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _ols(xs, ys):
    """Least-squares line; returns (slope, intercept, residual sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, intercept, resid


def _quantile_row(median_rate, sigma, levels):
    """Five quantiles around a median rate, normal on the logit scale."""
    centre = _logit(median_rate)
    raw = []
    for level in levels:
        value = _inv_logit(centre + _z_for(level) * sigma)
        raw.append(_round4(_clamp(value, 0.0, 100.0)))
    row = []
    floor = 0.0
    for value in raw:
        value = max(value, floor)
        row.append(value)
        floor = value
    return row


def _distinct_reads(history):
    """Collapse consecutive identical reads: a refresh is a new page.

    Returns a list of (time, players, rates) with at least the time parsed;
    reads that carry no usable data are dropped.
    """
    kept = []
    previous = None
    for read in history or []:
        if not isinstance(read, dict):
            continue
        when = _parse_time(read.get("at")) or _parse_time(read.get("slot"))
        players = read.get("players_tracked")
        rates = read.get("rates")
        if when is None or not isinstance(players, (int, float)):
            continue
        if players <= 0 or not isinstance(rates, list) or not rates:
            continue
        key = (players, tuple(rates))
        if key == previous:
            continue
        previous = key
        kept.append((when, float(players), rates))
    return kept


def _last_rates(history, count):
    """Each trophy's rate in the latest read that lists it (None if never)."""
    found = [None] * count
    for read in reversed(history or []):
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        for i in range(min(count, len(rates))):
            if found[i] is None and isinstance(rates[i], (int, float)):
                found[i] = float(rates[i])
        if all(v is not None for v in found):
            break
    return found


def _flat_forecast(case, trophies, last, horizon_minutes):
    """No refresh seen yet: expect one, cut the median, spread wide.

    The page has sat unchanged for hours, so a refresh over the horizon is
    the base case.  One refresh multiplies the players tracked severalfold
    while earners grow only modestly, so the median rate falls to about a
    fifth of the last read; the cut is shallower for trophies the launch
    night's own rates show new players earn first, deeper for late ones.
    """
    usable = [r for r in last if r is not None and r < SATURATED and r > 0]
    mean_rate = sum(usable) / len(usable) if usable else 1.0
    horizon_hours = max(horizon_minutes, 0.0) / 60.0
    refreshes = horizon_hours / FLAT_REFRESH_HOURS
    stretch = math.log(max(1.0, refreshes))
    shift = min(FLAT_SHIFT_CAP, -math.log(FLAT_MEDIAN_RATIO)
                * (1.0 + FLAT_SHIFT_PER_LOG * stretch))
    sigma = min(FLAT_SIGMA_CAP, FLAT_SIGMA + FLAT_SIGMA_PER_LOG * stretch)
    levels = case.get("levels") or list(LEVELS)
    rows = []
    for trophy, rate in zip(trophies, last):
        tid = trophy.get("id")
        if rate is None:
            rate = TIER_PRIOR.get(trophy.get("tier"), 1.0)
        if rate >= SATURATED:
            rows.append({"id": tid,
                         "q": _quantile_row(SATURATED, SATURATED_SIGMA, levels)})
            continue
        base = max(rate, 0.05)
        ratio = FLAT_MEDIAN_RATIO * (base / mean_rate) ** FLAT_MODULATION
        ratio = min(FLAT_RATIO_HI, max(FLAT_RATIO_LO, ratio))
        # The ratio already carries one refresh's dilution; the stretch
        # adds only the extra downward shift a longer horizon's further
        # refreshes bring, which is zero at the practice horizon.
        extra = -(shift + math.log(FLAT_MEDIAN_RATIO))
        median_rate = _clamp(base * ratio * math.exp(extra))
        rows.append({"id": tid, "q": _quantile_row(median_rate, sigma, levels)})
    return {"trophies": rows}


def _trend_forecast(case, trophies, last, distinct, horizon_minutes,
                    checkpoint, target, release_start):
    """Rich history: fit earners and players separately, divide, blend.

    log earners and log players tracked are each fitted against log time
    since release; the direct logit-rate trend over the last day is blended
    in half and half, the move from the last read damped, and the spread
    combines the random walk, the fit residuals and the refresh jumps the
    history has shown.
    """
    horizon_days = max(horizon_minutes, 0.0) / 1440.0
    levels = case.get("levels") or list(LEVELS)
    x_target = math.log1p(max(0.0, (target - release_start).total_seconds() / 3600.0)) \
        if target is not None and release_start is not None else None

    def hours_of(when):
        if release_start is None:
            return 0.0
        return max(0.0, (when - release_start).total_seconds() / 3600.0)

    # The fits use the recent window; widen it when the window is too thin.
    window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0 \
        if checkpoint is not None else None
    recent = [r for r in distinct
              if window_start is None or r[0].timestamp() >= window_start]
    if len(recent) < MIN_FIT_POINTS:
        recent = distinct
    xs = [math.log1p(hours_of(when)) for when, _, _ in recent]

    # Players tracked: one line for the whole game.
    log_n = [math.log(players) for _, players, _ in recent]
    n_fit = _ols(xs, log_n)
    last_players = distinct[-1][1]
    if n_fit is not None and x_target is not None:
        n_target = math.exp(n_fit[1] + n_fit[0] * x_target)
    else:
        n_target = last_players
    n_target = max(n_target, last_players)
    rw_sigma = RW_SIGMA_PER_DAY * math.sqrt(horizon_days)

    # Direct logit-rate trend over the last day, for the blend.
    trend_start = checkpoint.timestamp() - TREND_WINDOW_HOURS * 3600.0 \
        if checkpoint is not None else None
    trend_reads = [r for r in distinct
                   if trend_start is None or r[0].timestamp() >= trend_start]
    if len(trend_reads) < MIN_FIT_POINTS:
        trend_reads = distinct
    txs = [math.log1p(hours_of(when)) for when, _, _ in trend_reads]

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id")
        rate = last[idx] if idx < len(last) else None
        if rate is None:
            rate = TIER_PRIOR.get(trophy.get("tier"), 1.0)
        if rate >= SATURATED:
            rows.append({"id": tid,
                         "q": _quantile_row(SATURATED, SATURATED_SIGMA, levels)})
            continue
        # Earners per read, from the page's own rate and players tracked.
        ex, ey = [], []
        for (when, players, rates), x in zip(recent, xs):
            if idx < len(rates) and isinstance(rates[idx], (int, float)):
                earners = rates[idx] * players / 100.0
                if earners > 0:
                    ex.append(x)
                    ey.append(math.log(earners))
        e_fit = _ols(ex, ey) if len(ex) >= MIN_FIT_POINTS else None
        if e_fit is not None and x_target is not None:
            slope = max(0.0, e_fit[0])
            earners_target = math.exp(e_fit[1] + slope * x_target)
        else:
            earners_target = None
        # Direct trend for this trophy.
        lx, ly = [], []
        for (when, _, rates), x in zip(trend_reads, txs):
            if idx < len(rates) and isinstance(rates[idx], (int, float)):
                lx.append(x)
                ly.append(_logit(rates[idx]))
        t_fit = _ols(lx, ly) if len(lx) >= MIN_FIT_POINTS else None
        resid = t_fit[2] if t_fit is not None else 0.25

        last_earners = max(rate, 0.01) * last_players / 100.0
        if earners_target is None:
            # No usable earners line: hold the earners, let players dilute.
            earners_target = last_earners
        earners_target = max(earners_target, last_earners)
        earners_target = min(earners_target, n_target)
        rate_split = _clamp(100.0 * earners_target / n_target)
        if t_fit is not None and x_target is not None:
            rate_trend = _clamp(_inv_logit(t_fit[1] + t_fit[0] * x_target))
            centre = ((1.0 - BLEND_TREND) * _logit(rate_split)
                      + BLEND_TREND * _logit(rate_trend))
        else:
            centre = _logit(rate_split)
        centre = _logit(rate) + DAMP * (centre - _logit(rate))
        centre = _logit(rate) + max(-MAX_LOGIT_MOVE,
                                    min(MAX_LOGIT_MOVE, centre - _logit(rate)))
        median_rate = _clamp(_inv_logit(centre))
        sigma = math.sqrt(rw_sigma ** 2 + resid ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        rows.append({"id": tid, "q": _quantile_row(median_rate, sigma, levels)})

    return {"trophies": rows}


def _fallback_rows(trophies, last):
    """Valid-shaped output when the main forecast cannot run at all."""
    levels = list(LEVELS)
    rows = []
    for trophy, rate in zip(trophies, last):
        if rate is None:
            rate = TIER_PRIOR.get(trophy.get("tier"), 1.0)
        if rate >= SATURATED:
            rows.append({"id": trophy.get("id"),
                         "q": _quantile_row(SATURATED, SATURATED_SIGMA, levels)})
        else:
            rows.append({"id": trophy.get("id"),
                         "q": _quantile_row(_clamp(rate), 1.0, levels)})
    return {"trophies": rows}


def forecast_case(case):
    """Forecast one case: five quantiles per trophy at the target."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    try:
        horizon_minutes = float(horizon_minutes)
    except (TypeError, ValueError):
        horizon_minutes = 0.0
    last = _last_rates(history, len(trophies))
    distinct = _distinct_reads(history)
    if not trophies:
        return {"trophies": []}
    if not distinct or all(r is None for r in last):
        return _fallback_rows(trophies, last)
    try:
        checkpoint = _parse_time(case.get("at"))
        target = _parse_time(case.get("target"))
        game = case.get("game") or {}
        release_start = None
        release_date = game.get("release_date")
        if isinstance(release_date, str):
            release_start = _parse_time(release_date + "T00:00:00Z")
        if target is None and checkpoint is not None:
            target = datetime.fromtimestamp(
                checkpoint.timestamp() + horizon_minutes * 60.0,
                tz=timezone.utc)
        if len(distinct) <= 2:
            return _flat_forecast(case, trophies, last, horizon_minutes)
        return _trend_forecast(case, trophies, last, distinct,
                               horizon_minutes, checkpoint, target,
                               release_start)
    except Exception:
        return _fallback_rows(trophies, last)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        result = forecast_case(case)
        target = OUTPUT_DIR / (path.stem + ".out")
        target.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                          encoding="utf-8")


if __name__ == "__main__":
    main()
