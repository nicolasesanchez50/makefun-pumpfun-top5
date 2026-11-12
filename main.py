#!/usr/bin/env python3
"""Forecast the five largest pump.fun coins at each of the house's answer reads.

Reads the house's 10-minute snapshots at $MAKEFUN_INPUT_DIR (default /input) and
writes one forecast to $MAKEFUN_OUTPUT_DIR/forecast.json (default /output):

  probabilities  a probability for each coin named at each of the five places
  top5           the five mints most likely to hold a top-5 place, in order
  sizes          for each place, a probability for each of the 82 log10(cap) bins

Standard library only, no network, deterministic: the same history always
produces the same bytes.

The model, in short.  A coin's log10 market cap in SOL is a mean-reverting
random walk: it relaxes toward the launch floor with time, with a volatility
that grows as the square root of the horizon, because a coin whose market cap
is only observed every ten minutes can move a long way in the seven to thirteen
hours between the cut and the answer reads.  On top of the coins the history
names, a population of coins the history never ranked (born after the cut, or
too small to make the 200-row snapshots) competes for the chart: most sit at
the floor, a few are runners, and a very few break out.  Monte Carlo over that
population gives, for every read, the probability each named coin holds each
place and the distribution of each place's market cap; the forecast averages
those over the reads, since one forecast is scored at all of them.

A place's probability mass left unnamed is shared by the scorer among the
answer chart's coins the forecast does not name.  The history's coins are named
-- up to the chart's size -- so that the unnamed mass covers the coins nobody
could see, and the coins picked are the ones the simulation puts on the chart
most often.
"""

import json
import math
import os
import random
import sys
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- model constants -------------------------------------------------------
FLOOR = 1.45          # log10 SOL at which a fresh pump.fun coin starts
TAU = 3.0             # hours: relaxation time of a coin's log cap toward the floor
SIGMA = 0.55          # log10 dex per sqrt(hour) of a named coin's future path
SCENARIOS = 360       # Monte Carlo paths per read
UNKNOWN = 46          # coins the history never named that sit on a 100-coin chart
MAX_ROWS = 100        # the most rows a place may carry
P_FLOOR_ROW = 5e-5    # a named coin's minimum probability at a place
EXTRA_SIZE_SD = 0.22  # extra spread on a place's cap, for what the model misses
SEED = 0x5EED1234


def parse_time(text):
    """Seconds since the epoch for an ISO 8601 UTC string."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def logcap(row):
    """log10 of a row's market cap in SOL (lamports / 1e9)."""
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def bin_of(value, edges):
    """The size bin a log10 cap falls in: 0 below the first edge, i above edges[i-1]."""
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if value >= edges[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def normal(rng):
    """A standard normal draw (Box-Muller, deterministic given the generator)."""
    u = 0.0
    while u <= 0.0:
        u = rng.random()
    v = rng.random()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def load_inputs():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                mark = json.loads(line)
            except ValueError:
                continue
            if isinstance(mark, dict) and mark.get("ok") and isinstance(mark.get("rows"), list):
                marks.append(mark)
    return task, marks


def coin_states(marks, last_ts, floor_level):
    """One state per coin the history names: its last cap and the cap it fell to."""
    series = {}
    for mark in marks:
        ts = parse_time(mark["at"])
        for row in mark["rows"]:
            series.setdefault(row["mint"], []).append(
                (ts, logcap(row), row.get("rank", 10 ** 6), row.get("venue", "pool"))
            )
    states = []
    for mint, points in series.items():
        points.sort()
        last_seen, last_level, last_rank, venue = points[-1]
        absent = max(0.0, (last_ts - last_seen) / 3600.0)
        level = last_level
        if absent > 0.2:
            # It dropped out of the 200-row snapshots: its cap is below the last
            # snapshot's 200th row and has had time to fall further.
            level = min(last_level - 0.30 * absent, floor_level - 0.05)
        states.append({
            "mint": mint,
            "level": level,
            "rank": last_rank,
            "venue": venue,
            "seen": (last_seen - last_ts) / 3600.0,
        })
    return states


def unknown_level(rng, cohort_hours, top_level):
    """A log10 cap drawn for a coin the history never named.

    Most such coins sit on the launch floor; a tenth are runners; two in a
    hundred break out, and the breakout level rises with the age of the day's
    cohort (the day's biggest coin keeps growing, then plateaus) and with how
    high the day's leader already stands: a day whose cut leader is large is
    a day whose answer-read leader is larger still.
    """
    u = rng.random()
    if u < 0.02:
        centre = 4.45 + 0.055 * min(cohort_hours, 26.0) + 0.25 * max(0.0, top_level - 3.0)
        return centre + 0.55 * normal(rng)
    if u < 0.14:
        centre = 2.55 + 0.020 * min(cohort_hours, 26.0)
        return centre + 0.45 * normal(rng)
    return 1.60 + 0.25 * normal(rng)


def main():
    task, marks = load_inputs()
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    bins = len(edges) + 1
    chart_size = int(task.get("chart_size", 100))
    t0 = parse_time(task["t0"])
    reads = sorted(parse_time(t) for t in task["reads"])

    if not marks:
        write_forecast({"probabilities": [], "top5": [], "sizes": []}, task)
        return

    last_ts = max(parse_time(m["at"]) for m in marks)
    last_mark = max(marks, key=lambda m: parse_time(m["at"]))
    last_rows = sorted(last_mark["rows"], key=lambda r: r.get("rank", 10 ** 6))
    top_level = logcap(last_rows[0]) if last_rows else 3.0
    floor_level = logcap(last_rows[min(len(last_rows) - 1, chart_size * 2 - 1)]) if last_rows else FLOOR

    states = coin_states(marks, last_ts, floor_level)
    # Only coins that could still reach a 100-coin chart are worth simulating.
    states = [s for s in states if s["level"] >= floor_level - 2.5]
    if len(states) > 1400:
        states.sort(key=lambda s: -s["level"])
        states = states[:1400]

    rng = random.Random(SEED)
    n = len(states)
    place_hits = [[0.0] * places for _ in range(n)]      # per coin, per place, summed over reads
    presence = [0.0] * n                                  # in the chart, summed over reads
    size_hits = [[0.0] * bins for _ in range(places)]     # per place, per bin, summed over reads
    unknown_place = [0.0] * places
    n_reads = max(1, len(reads))

    # Precompute each coin's relaxation factor once per read.
    for read_ts in reads:
        h = max(0.0, (read_ts - last_ts) / 3600.0)
        cohort_hours = max(0.0, (read_ts - t0) / 3600.0)
        decay = math.exp(-h / TAU)
        spread = SIGMA * math.sqrt(h)
        base = [s["level"] for s in states]
        means = [FLOOR + (lv - FLOOR) * decay for lv in base]
        for scenario in range(SCENARIOS):
            caps = []
            for i in range(n):
                caps.append((means[i] + spread * normal(rng), i))
            for j in range(UNKNOWN):
                caps.append((unknown_level(rng, cohort_hours, top_level), -1 - j))
            caps.sort(key=lambda pair: (-pair[0], pair[1]))
            chart = caps[:chart_size]
            for k in range(min(places, len(chart))):
                level, idx = chart[k]
                size_hits[k][bin_of(level, edges)] += 1.0
                if idx >= 0:
                    place_hits[idx][k] += 1.0
                else:
                    unknown_place[k] += 1.0
            for level, idx in chart:
                if idx >= 0:
                    presence[idx] += 1.0

    norm = float(SCENARIOS * n_reads)
    # Named set: the coins the simulation puts on the chart most often.
    order = sorted(range(n), key=lambda i: (-presence[i], -place_hits[i][0], states[i]["mint"]))
    named = order[:MAX_ROWS]

    probabilities = []
    named_total = [0.0] * places
    for i in named:
        for k in range(places):
            q = place_hits[i][k] / norm
            if q <= 0.0:
                q = P_FLOOR_ROW
            q = max(q, P_FLOOR_ROW)
            named_total[k] += q
            probabilities.append({"mint": states[i]["mint"], "place": k + 1, "p": round(q, 9)})

    # Keep each place's named mass below one: scale down if the model overshoots.
    for k in range(places):
        if named_total[k] > 0.995:
            factor = 0.995 / named_total[k]
            for row in probabilities:
                if row["place"] == k + 1:
                    row["p"] = round(row["p"] * factor, 9)

    # top5: the five coins with the most mass across the places, place 1 weighted most.
    score = []
    for i in range(n):
        s = sum(place_hits[i][k] / norm * (places - k) for k in range(places))
        score.append((s, states[i]["mint"]))
    score.sort(key=lambda pair: (-pair[0], pair[1]))
    top5 = [mint for _, mint in score[:places]]

    sizes = []
    for k in range(places):
        total = sum(size_hits[k])
        if total <= 0:
            continue
        probs = [c / total for c in size_hits[k]]
        for b, p in enumerate(probs):
            if p > 0.0:
                sizes.append({"place": k + 1, "bin": b, "p": round(p, 9)})

    write_forecast({"probabilities": probabilities, "top5": top5, "sizes": sizes}, task)


def write_forecast(doc, task):
    name = task.get("output") or "forecast.json"
    path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    sys.stderr.write("wrote %s (%d bytes)\n" % (path, len(text) + 1))


if __name__ == "__main__":
    main()
