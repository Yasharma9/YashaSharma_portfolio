"""
make_sample_data.py
Makes a sample weekly_records.csv for yield_forecast.py.

All values are generated with a seeded random generator. They are realistic in shape
(crop cycles, seasons, weather effects, pest events, missing records) but do not come
from any farm or company.

What is in the file
  - 20 polyhouses (A to T) on one farm, 1/4, 1/2 or 1 acre each
  - Tomato (about 36 to 44 week cycles) and cucumber (about 14 to 18 week cycles)
  - One row per polyhouse per week while a crop is in the ground, with a 2 to 4 week
    gap for cleaning between cycles
  - Weekly weather inside the polyhouse: day and night temperature, humidity, solar
    radiation and VPD, following Indian seasons (hot dry summer, humid monsoon, mild winter)
  - Weekly growth records: fruits per cluster, harvested clusters, plant height, nodes

How yield is made
  yield = crop curve for the crop age (ramp up, peak, decline)
        x variety and polyhouse effects
        x light in the last 2 weeks
        x fruit set quality some weeks earlier (temperature, night temperature and VPD
          when the fruit was set: 6 weeks before harvest for tomato, 2 for cucumber)
        x pest or disease dips (random, about 1 in 4 cycles)
        x week-to-week noise

Run
  python make_sample_data.py --out weekly_records.csv

Needs: python 3.9+, numpy, pandas
"""
import argparse

import numpy as np
import pandas as pd

CROPS = {
    #        harvest start, cycle length, peak kg/m2/week, ramp weeks, decline from age, decline per week, set lag
    "TOM": dict(start=(12, 0.8), length=(36, 44), peak=1.25, ramp=6, decl_from=28, decl=0.025, lag=6,
                fpc=5.0, clusters=1.0, height=(30, 9.0), nodes=(4, 1.0),
                varieties={"TO-0001": 1.00, "TO-0002": 1.08, "TO-0003": 0.93}),
    "CUC": dict(start=(5.5, 0.6), length=(14, 18), peak=2.0, ramp=3, decl_from=11, decl=0.06, lag=2,
                fpc=1.2, clusters=2.2, height=(25, 14.0), nodes=(3, 1.6),
                varieties={"CU-0001": 1.00, "CU-0002": 1.06}),
}
AREAS = [1012, 2024, 4047]  # quarter, half and one acre, in m2


def bump(woy, a, b):
    """Smooth 0..1 bump between week a and week b of the year."""
    mid, half = (a + b) / 2, (b - a) / 2
    return np.clip(np.cos(np.pi / 2 * (woy - mid) / half), 0, None) ** 1.5 * ((woy >= a - 2) & (woy <= b + 2))


def farm_weather(dates, rng):
    """Weekly farm-level weather inside the polyhouses."""
    woy = np.minimum(dates.isocalendar().week.values.astype(float), 52)
    mons, summer = bump(woy, 26, 39), bump(woy, 10, 22)
    n = len(dates)

    def ar(sd, phi=0.6):
        e = rng.normal(0, sd, n)
        for i in range(1, n):
            e[i] += phi * e[i - 1]
        return e * np.sqrt(1 - phi ** 2)

    tday = 27.5 + 4.5 * np.cos(2 * np.pi * (woy - 20) / 52) - 3.0 * mons + ar(1.1)
    tnight = tday - 9.5 + 3.5 * mons + 1.0 * summer + ar(0.7)
    rh = 58 + 24 * mons - 14 * summer + ar(4.0)
    tsr = 14.5 + 4.5 * np.cos(2 * np.pi * (woy - 15) / 52) - 5.5 * mons + ar(1.2)
    return pd.DataFrame(dict(date=dates, tday=tday, tnight=tnight, rh=np.clip(rh, 30, 96), tsr=np.clip(tsr, 5, 26)))


def vpd(tday, tnight, rh):
    t = 0.6 * tday + 0.4 * tnight
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100)


def crop_curve(age, c, start, length):
    """Potential yield (kg/m2/week) for a crop age, before any effects."""
    if age < start:
        return 0.0
    k = age - start + 1
    up = 1 / (1 + np.exp(-(k - c["ramp"] / 2) * 4 / c["ramp"]))
    down = max(0.25, 1 - c["decl"] * max(0, age - c["decl_from"]))
    end = 0.6 if age >= length - 1 else 1.0           # last week: final strip-out harvest
    return c["peak"] * up * down * end


def set_quality(tday, tnight, v):
    """How well fruit sets in a week, 0.5..1.05. Heat, cold nights and dry or damp air reduce it."""
    f = (1 - 0.045 * max(0, tday - 30) - 0.03 * max(0, 23 - tday)) \
        * (1 - 0.05 * max(0, tnight - 21) - 0.05 * max(0, 14 - tnight)) \
        * (1 - 0.18 * max(0, v - 1.5) - 0.2 * max(0, 0.55 - v))
    return float(np.clip(f, 0.5, 1.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weekly_records.csv")
    ap.add_argument("--end", default="2026-11-23", help="Monday of the last week in the file")
    ap.add_argument("--weeks", type=int, default=157, help="number of weeks in the file (157 = about 3 years)")
    ap.add_argument("--polyhouses", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    end = pd.Timestamp(a.end)
    first = end - pd.Timedelta(weeks=a.weeks - 1)
    sim_start = first - pd.Timedelta(weeks=60)                  # simulate earlier so cycles are staggered
    dates = pd.date_range(sim_start, end, freq="7D")
    wx = farm_weather(dates, rng).set_index("date")

    rows = []
    for p in range(a.polyhouses):
        ph = chr(ord("A") + p)
        main_crop = "TOM" if p < 12 else "CUC"
        area = float(rng.choice(AREAS, p=[0.3, 0.5, 0.2]))
        ph_eff = rng.normal(1, 0.06)
        offs = dict(tday=rng.normal(0, 0.6), tnight=rng.normal(0, 0.4), rh=rng.normal(0, 2.5), tsr=rng.normal(0, 0.6))
        t = sim_start + pd.Timedelta(weeks=int(rng.integers(0, 30)))
        cycle = 0
        while t <= end:
            crop = main_crop if rng.random() < 0.88 else ("CUC" if main_crop == "TOM" else "TOM")
            c = CROPS[crop]
            cycle += 1
            start = max(3, int(round(rng.normal(*c["start"]))))
            length = int(rng.integers(c["length"][0], c["length"][1] + 1))
            var = rng.choice(list(c["varieties"]))
            # pest or disease event in about 1 cycle in 4
            pest = np.ones(length + 1)
            if rng.random() < 0.25:
                at, dur, depth = int(rng.integers(start, max(start + 1, length - 2))), int(rng.integers(3, 7)), rng.uniform(0.2, 0.45)
                for k in range(dur):
                    if at + k <= length:
                        pest[at + k] = 1 - depth * np.sin(np.pi * (k + 0.5) / dur)
            noise = 0.0
            sq_hist = []
            for age in range(1, length + 1):
                d = t + pd.Timedelta(weeks=age - 1)
                if d > end:
                    break
                w = wx.loc[d]
                td, tn = w.tday + offs["tday"], w.tnight + offs["tnight"]
                rh, tsr = float(np.clip(w.rh + offs["rh"], 30, 97)), w.tsr + offs["tsr"]
                vp = vpd(td, tn, rh)
                sq = set_quality(td, tn, vp)
                sq_hist.append(sq)
                set_then = sq_hist[-1 - c["lag"]] if len(sq_hist) > c["lag"] else np.mean(sq_hist)
                light = 0.72 + 0.28 * min(np.mean([wx.loc[d - pd.Timedelta(weeks=j)].tsr + offs["tsr"] for j in (1, 2)]) / 17, 1.15)
                noise = 0.55 * noise + rng.normal(0, 0.075)
                pot = crop_curve(age, c, start, length)
                y = max(0.0, pot * c["varieties"][var] * ph_eff * light * set_then * pest[age] * (1 + noise))
                rel = pot / c["peak"]
                fruits = c["fpc"] * sq * rng.normal(1, 0.05)
                clusters = c["clusters"] * rel * set_then * pest[age] * rng.normal(1, 0.08) if pot > 0 else 0.0
                height = c["height"][0] + c["height"][1] * age * rng.normal(1, 0.03)
                nodes = c["nodes"][0] + c["nodes"][1] * age * rng.normal(1, 0.03)
                if d >= first:
                    rows.append(dict(week_start=d.date().isoformat(), polyhouse=ph, cycle_id=cycle, crop_code=crop, variety_code=var,
                                     crop_age_weeks=age, area_m2=area, harvest_kg=round(y * area, 1),
                                     day_temp_c=round(td, 1), night_temp_c=round(tn, 1), rh_pct=round(rh, 1),
                                     solar_radiation_mj_m2=round(tsr, 2), vpd_kpa=round(vp, 2),
                                     fruits_per_cluster=round(fruits, 2), harvested_clusters=round(clusters, 2),
                                     plant_height_cm=round(height, 1), nodes_per_plant=round(nodes, 1)))
            t = t + pd.Timedelta(weeks=length + int(rng.integers(2, 5)))   # cleaning gap before the next crop

    df = pd.DataFrame(rows)
    # real data is never complete: drop some weeks and blank some growth records
    keep = rng.random(len(df)) > 0.015
    df = df[keep | (df["crop_age_weeks"] == 1)].copy()
    for col in ["fruits_per_cluster", "harvested_clusters", "plant_height_cm", "nodes_per_plant"]:
        df.loc[rng.random(len(df)) < 0.04, col] = np.nan
    df = df.sort_values(["polyhouse", "week_start"]).reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows, {df['polyhouse'].nunique()} polyhouses, {df['week_start'].min()} to {df['week_start'].max()} -> {a.out}")


if __name__ == "__main__":
    main()
