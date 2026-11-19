"""gta6-trophy-rarity-forecast-01 entry: hybrid saturation + dilution forecaster.

Forecasts, for every trophy in a case, the percentage of Exophase-tracked
players who will have earned it at the case's target, as five quantiles
(5/25/50/75/95%). Scored by pinball loss on the logit scale.

Model (written live in-session, 20 Nov 2026). The rate is a ratio, so the
model projects its two parts separately and divides them again:

    rate(t) = earners(t) / players(t)

  - players N(t): the tracked base grows by refreshes. Its log-growth rate
    is fitted robustly (Theil-Sen) on log N over the SAME post-refresh
    window as the earners (revision v4, review finding W1): a refresh jump
    is a level shift, not a growth rate, and fitting over the whole
    history lets the jump contaminate the slope, making every rate fall
    too fast. Only if the window has fewer than 2 usable player counts
    does it fall back to the whole history.
  - earners E_i(t) per trophy: fitted robustly (Theil-Sen on log E) over the
    reads since the last refresh, i.e. inside the current regime, where a
    refresh jump cannot masquerade as ongoing earning. Slopes are shrunk
    toward the tier median (weight grows with the number of usable reads),
    so platinum (1 trophy) and gold (2) borrow strength from the field.
  - the median rate follows the RATE's own log-growth (earners' slope
    minus players' slope), capped at 5%/h: a sustained 5%/h move over the
    effective horizon is already a 16x change, so a steeper fitted slope
    is short-window noise, not a credible sustained rate. Projecting E
    and N separately would let asymmetric overflow caps distort the
    ratio; the difference is the honest quantity. The projection uses a
    saturating time factor: the fitted rate decays with a 72-hour time
    constant, so a 24-hour rate is not extrapolated linearly over 4.5 days.
  - the median rate is capped by a per-trophy ceiling read from the tier
    and the description (story chapters reach most players; 100% completion
    and big grinds reach few), and floored so a dilution fall cannot go
    below 5% of the last rate.
  - the 100% trophy ("Just Like Paradise", id 2) can only fall: new
    players mostly have not fished yet, so its earners are projected as
    the old earners plus 40% of the new players. It is identified by id
    or by the "Gone Fishing" description AND by having been >=99.5% at
    some point in the history (revision v4, review finding W2): the last
    rate must not be the trigger, because by the deadline it is already
    diluted below 99.5% and the rule would never fire.
  - on flat history (no movement at all) every slope is zero and the
    median reduces to the last rate: near-persistence, which is the honest
    answer when the data shows nothing.

Spread: quantiles are built on the logit scale, median +/- z * sigma, with
sigma = the kit's own random-walk scale (0.5 * sqrt(horizon in days)) times
a confidence multiplier: wider where the fit is weak (few reads, no
signal, the 100% trophy), never narrower than the kit's own scale (the
floor is 1.0; revision v4, review finding W3: a strong fit does not
reduce regime risk). Over 90 minutes rates barely move, so the spread is
tight; over 4.5 days they move a lot, so it is wide. The spread is the
honest uncertainty, not a guess at the answer.

Guarantees: single file, Python 3.13 standard library only, no network,
no randomness, no clock reads (every parameter comes from the case or
from the constants above), so two runs on the same input are
byte-identical. Any unexpected input shape falls back to the persistence
baseline (with a stderr warning) rather than crashing, because an invalid
case takes no prize.

Usage (grader): python3 -P -s /bot/main.py   (reads /input, writes /output)
Locally: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at the five levels, exactly as the kit states them.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The fitted per-hour log-growth rate decays with this time constant, so a
# rate measured over ~1 day is not extrapolated linearly over ~4.5 days.
TAU_HOURS = 72.0
# A sustained 5%/h change in a rate (or in the player base) over the
# effective horizon is already a 16x move; steeper fitted slopes are
# short-window noise, so they are capped here, not extrapolated.
RATE_SLOPE_CAP = 0.05
PLAYER_SLOPE_CAP = 0.05
# Sanity caps: the player base will not grow more than 20x over the horizon,
# and a median never falls below 5% of the last observed rate.
N_GROWTH_CAP = 20.0
RATE_FLOOR_FRAC = 0.05
# For the 100% trophy: share of new players assumed to have earned it by
# the target (fishing is an early side activity; most new players have not
# done it yet, so the rate must fall as the base grows).
NEW_PLAYER_EARN_FRAC = 0.4
# Shrinkage: a trophy's own slope gets weight m / (m + 3), m = usable reads.
SHRINK_PRIOR_READS = 3

# Week-one ceilings, reasoned from the trophy list itself (19 Nov data):
# the platinum needs every trophy including the grinds (already 0.63% on
# day one, so its ceiling sits just above that); gold is story completion
# and 100% completion; silver is mostly counted grinds; bronze ranges from
# early story chapters (most players reach them) to collectibles.
TIER_BASE_CEILING = {"platinum": 2.0, "gold": 10.0, "silver": 12.0, "bronze": 35.0}
# Fallback rate when a trophy was never observed in any read.
TIER_PRIOR_RATE = {"platinum": 0.5, "gold": 0.8, "silver": 1.0, "bronze": 1.0}
# Description overrides, first match wins (checked against the lowercase
# description). A ceiling is a cap on the rise, never a target.
CEILING_RULES = [
    ("100% completion", 3.0),
    ("chapter 1", 60.0),
    ("chapter 2", 55.0),
    ("chapter 3", 50.0),
    # Chapter 4 tightened 45.0 -> 20.0 in revision v4 (Agent 07 rising
    # regime: synthetic truth 12%, counterfactual 20% cut the loss
    # 1.135 -> 0.303; Agent 06 concurred with modest tightening).
    ("chapter 4", 20.0),
    ("final mission.", 15.0),
    ("final mission for", 25.0),
    ("250 stunts", 5.0),
    ("3 collectible card sets", 4.0),
    ("gold medal in all", 4.0),
    ("50 challenges", 6.0),
    ("8 mission strands", 8.0),
    ("5 bank vaults", 10.0),
    ("100 locations", 12.0),
    ("100 vehicles", 15.0),
    ("20 different races", 12.0),
    # Tightened in revision v4 after the Agent 07 rising-regime audit
    # (synthetic truths 10%/9%; counterfactual ceilings 15% cut the loss
    # 0.921 -> 0.274 and 1.038 -> 0.297). Agent 06 concurred with modest
    # tightening only; the bronze base 35% stays for the rest.
    ("child's play", 15.0),
    ("headliners", 15.0),
]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the kit's convention)."""
    value = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(value) if value == int(value) else value


def z_for_level(level):
    for known, z in zip(LEVELS, Z):
        if level == known:
            return z
    return NormalDist().inv_cdf(level)


def parse_time(text):
    """ISO timestamp to epoch seconds; None when unparseable. A naive
    timestamp is read as UTC (revision v4, Agent 03 minor 2): the local
    timezone must never shift the time differences the slopes use."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def theil_sen(xs, ys):
    """Median pairwise slope of ys against xs; None when it cannot be fit."""
    slopes = []
    n = len(xs)
    for a in range(n):
        for b in range(a + 1, n):
            dx = xs[b] - xs[a]
            if dx != 0:
                slopes.append((ys[b] - ys[a]) / dx)
    if not slopes:
        return None
    return median(slopes)


def ceiling_for(trophy, last_rate):
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    cap = TIER_BASE_CEILING.get(tier, 35.0)
    for needle, value in CEILING_RULES:
        if needle in desc:
            cap = value
            break
    # A ceiling never sits below where the trophy already is.
    return max(cap, last_rate)


def fit_window(reads, last_t):
    """Reads to fit slopes on: since the last refresh if that leaves >=3
    reads, else the last 24h, else everything. Capped to the newest 120.
    The window needs >=3 reads: a shorter window cannot fit a stable
    slope, and a clean 1-read segment is worse than a longer window,
    because with no fittable slope every earner slope falls back to the
    tier median (0 when none fit) while the player slope still fits, and
    the difference then invents a fall. A window that straddles a refresh
    is acceptable here: both slopes are fitted on the same window, so the
    refresh step cancels in their difference (revision v4, review
    finding W1 follow-up). This supersedes the earlier idea of preferring
    a sub-3-read segment (Agent 03 minor 3): that created an unfittable
    window and broke the synthetic growth test, so it was reverted."""
    players = [r["_n"] for r in reads]
    start = 0
    for j in range(1, len(reads)):
        if players[j] and players[j - 1] and players[j] > 1.10 * players[j - 1]:
            start = j
    segment = reads[start:]
    if len(segment) >= 3:
        window = segment
    else:
        recent = [r for r in reads if last_t - r["_t"] <= 24.0 * 3600.0]
        window = recent if len(recent) >= 3 else reads
    return window[-120:]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    zs = [z_for_level(level) for level in levels]
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_h = max(0.0, float(case.get("horizon_minutes") or 0) / 60.0)
    # Saturating time factor: the fitted rate decays with a 72h constant,
    # so a short-horizon forecast is barely damped and a 4.5-day one is
    # damped to ~56 effective hours.
    eff_h = TAU_HOURS * (1.0 - math.exp(-horizon_h / TAU_HOURS)) if horizon_h > 0 else 0.0
    # The kit's own random-walk spread scale for this horizon.
    sigma_scale = 0.5 * math.sqrt(max(horizon_h, 0.0) / 24.0)

    # Normalise the history: epoch times and known player counts.
    reads = []
    for read in history:
        t = parse_time(read.get("at"))
        if t is None:
            continue
        n = read.get("players_tracked")
        reads.append({"_t": t, "_n": n if isinstance(n, (int, float)) and n > 0 else None,
                      "_rates": read.get("rates") or []})
    if not reads:
        # No history at all: tier priors, wide spread, never crash.
        rows = []
        for trophy in trophies:
            prior = TIER_PRIOR_RATE.get((trophy.get("tier") or "").lower(), 1.0)
            rows.append({"id": trophy["id"], "q": quantiles(prior, 1.5, zs)})
        return {"trophies": rows}

    reads.sort(key=lambda r: r["_t"])
    t0 = reads[0]["_t"]
    for r in reads:
        r["_h"] = (r["_t"] - t0) / 3600.0
    last_t = reads[-1]["_t"]
    known_n = [r for r in reads if r["_n"] is not None]
    n_last = known_n[-1]["_n"] if known_n else None

    window = fit_window(reads, last_t)

    # Player-base growth: Theil-Sen on log N over the SAME window as the
    # earners (revision v4, review finding W1). A refresh jump is a level
    # shift, not a growth rate: fitting over the whole history lets the
    # jump contaminate the slope and makes every rate fall too fast.
    # Fall back to the whole history only if the window has fewer than
    # 2 usable player counts.
    g_n = 0.0
    window_n = [r for r in window if r["_n"] is not None]
    if len(window_n) >= 2:
        xs = [r["_h"] for r in window_n]
        ys = [math.log(r["_n"]) for r in window_n]
        slope = theil_sen(xs, ys)
        g_n = slope if slope is not None else 0.0
    elif len(known_n) >= 2:
        xs = [r["_h"] for r in known_n[-200:]]
        ys = [math.log(r["_n"]) for r in known_n[-200:]]
        slope = theil_sen(xs, ys)
        g_n = slope if slope is not None else 0.0
    if n_last is None:
        # Rates without a player base: plain persistence, starter spread.
        return persistence_case(case, name, levels, zs, horizon_h)

    # Player-base projection (needed for the 100% trophy's dilution):
    # the fitted log-growth, capped, so a short-window slope cannot
    # explode the projected base. The base never shrinks.
    g_n_capped = max(0.0, min(PLAYER_SLOPE_CAP, g_n))
    n_target = n_last * math.exp(g_n_capped * eff_h)
    n_target = min(n_target, N_GROWTH_CAP * n_last)
    n_target = max(n_target, n_last)

    # First pass: per-trophy slopes on log earners inside the fit window.
    slopes = []
    per_trophy = []
    for i, trophy in enumerate(trophies):
        points = []
        for r in window:
            rates = r["_rates"]
            if i >= len(rates) or rates[i] is None or r["_n"] is None:
                continue
            earners = float(rates[i]) / 100.0 * r["_n"]
            if earners > 0:
                points.append((r["_h"], math.log(earners)))
        slope = theil_sen([p[0] for p in points], [p[1] for p in points]) if len(points) >= 2 else None
        slopes.append(slope)
        per_trophy.append({"points": len(points), "slope": slope})

    # Tier medians for shrinkage (global median when a tier is too thin).
    all_slopes = [s for s in slopes if s is not None]
    global_median = median(all_slopes) if all_slopes else 0.0
    tier_medians = {}
    for tier in TIER_BASE_CEILING:
        vals = [slopes[k] for k, t in enumerate(trophies)
                if (t.get("tier") or "").lower() == tier and slopes[k] is not None]
        tier_medians[tier] = median(vals) if len(vals) >= 2 else global_median

    rows = []
    for i, trophy in enumerate(trophies):
        tier = (trophy.get("tier") or "").lower()
        # Last observed rate (whole history, not just the window).
        last_rate = None
        for r in reversed(reads):
            rates = r["_rates"]
            if i < len(rates) and rates[i] is not None:
                last_rate = float(rates[i])
                break
        if last_rate is None:
            last_rate = TIER_PRIOR_RATE.get(tier, 1.0)
        e_last = last_rate / 100.0 * n_last

        info = per_trophy[i]
        shrink_to = tier_medians.get(tier, global_median)
        if info["slope"] is None:
            g_e = shrink_to
        else:
            w = info["points"] / (info["points"] + SHRINK_PRIOR_READS)
            g_e = w * info["slope"] + (1.0 - w) * shrink_to

        # The median follows the RATE's own log-growth (earners minus
        # players), capped: projecting E and N separately would let
        # asymmetric overflow caps distort the ratio.
        g_rate = max(-RATE_SLOPE_CAP, min(RATE_SLOPE_CAP, g_e - g_n))
        raw_rate = last_rate * math.exp(g_rate * eff_h)
        cap = ceiling_for(trophy, last_rate)
        med = min(raw_rate, cap)
        med = max(med, RATE_FLOOR_FRAC * last_rate, 0.01)
        med = min(med, 99.99)

        # The 100% trophy ("Just Like Paradise"): identified by id or by
        # the "Gone Fishing" description, AND by having been >=99.5% at
        # some point in the history (revision v4, review finding W2).
        # The last rate must not be the trigger: by the deadline it is
        # already diluted below 99.5% and the rule would never fire.
        desc = (trophy.get("description") or "").lower()
        is_fishing = (str(trophy.get("id")) == "2") or ("gone fishing" in desc)
        was_universal = False
        for r in reads:
            rates = r["_rates"]
            if i < len(rates) and rates[i] is not None and float(rates[i]) >= 99.5:
                was_universal = True
                break
        is_universal = is_fishing and was_universal

        flat = (g_n == 0.0 and g_e == 0.0)
        if is_universal:
            # The 100% trophy can only fall: its earners grow only by the
            # share of new players who have earned it by the target.
            e_target = e_last + NEW_PLAYER_EARN_FRAC * max(0.0, n_target - n_last)
            med = min(99.99, max(0.01, 100.0 * e_target / n_target))
            mult = 1.5
        elif flat or info["slope"] is None or info["points"] < 3:
            mult = 1.3
        else:
            # Never narrower than the kit's own scale (revision v4,
            # review finding W3): a strong fit does not reduce regime
            # risk, so the floor is 1.0 and the old 0.85 multiplier is gone.
            mult = 1.0
        # The spread grows with the horizon: over 90 minutes rates barely
        # move, over 4.5 days they do. The multiplier is the confidence.
        sigma = min(1.6, max(0.15, sigma_scale * mult))

        rows.append({"id": trophy["id"], "q": quantiles(med, sigma, zs)})
    return {"trophies": rows}


def quantiles(med, sigma, zs):
    base = logit(med)
    out = []
    for z in zs:
        value = round4(clamp_pct(inv_logit(base + z * sigma)))
        out.append(value)
    # Rounding can only tie, never invert, but enforce the order anyway.
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def persistence_case(case, name, levels, zs, horizon_h):
    """Starter-equivalent fallback: last rate, random-walk spread."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    spread = 0.5 * math.sqrt(max(0.0, horizon_h) / 24.0)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = float(rates[i])
                break
        if last_rate is None:
            last_rate = TIER_PRIOR_RATE.get((trophy.get("tier") or "").lower(), 1.0)
        rows.append({"id": trophy["id"], "q": quantiles(last_rate, spread, zs)})
    return {"trophies": rows}


def last_resort_case(case, name):
    """Minimal valid output when even persistence fails: every trophy at
    its tier prior with the starter spread (revision v4, Agent 03 minor
    1). Defensive throughout, so it never raises on a dict-shaped case;
    a non-dict case yields an empty (still valid) output."""
    if not isinstance(case, dict):
        return {"trophies": []}
    levels = case.get("levels") or LEVELS
    try:
        zs = [z_for_level(level) for level in levels]
    except Exception:
        zs = [z_for_level(level) for level in LEVELS]
    try:
        horizon_h = max(0.0, float(case.get("horizon_minutes") or 0) / 60.0)
    except (TypeError, ValueError):
        horizon_h = 0.0
    spread = 0.5 * math.sqrt(max(0.0, horizon_h) / 24.0)
    trophies = case.get("trophies") or []
    if not isinstance(trophies, list):
        return {"trophies": []}
    rows = []
    for trophy in trophies:
        if not isinstance(trophy, dict) or "id" not in trophy:
            continue
        prior = TIER_PRIOR_RATE.get((trophy.get("tier") or "").lower(), 1.0)
        rows.append({"id": trophy["id"], "q": quantiles(prior, spread, zs)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            # An unreadable file must not kill the other cases.
            print(f"warning: {path.name}: unreadable ({exc}); skipping",
                  file=sys.stderr)
            continue
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never crash a case: persistence instead
            print(f"warning: {path.name}: {exc}; falling back to persistence",
                  file=sys.stderr)
            try:
                levels = case.get("levels") or LEVELS
                zs = [z_for_level(level) for level in levels]
                horizon_h = max(0.0, float(case.get("horizon_minutes") or 0) / 60.0)
                forecast = persistence_case(case, path.name, levels, zs, horizon_h)
            except Exception as exc2:
                print(f"warning: {path.name}: persistence fallback failed "
                      f"({exc2}); emitting tier priors", file=sys.stderr)
                try:
                    forecast = last_resort_case(case, path.name)
                except Exception:
                    forecast = {"trophies": []}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
