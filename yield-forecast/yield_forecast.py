"""
yield_forecast.py
Weekly yield forecasting for polyhouse crops.

What it does
  1. Reads weekly records (one row per polyhouse per week), built from about 3 years of history.
  2. Builds features: recent yield, crop age, crop and variety, weather, plant growth, season.
  3. Trains one model per forecast distance (1, 2, ... N weeks ahead) with several algorithms.
  4. Tests them on the most recent weeks that were held out (a time-based backtest).
  5. Refits the best model for each distance and forecasts the running crops.
  6. Writes forecast_output.json (for the dashboard) and CSV files.

Run
  python yield_forecast.py --data weekly_records.csv --out results

Options
  --horizon 8        weeks ahead to forecast (default 8)
  --test-weeks 52    weeks held out for the backtest (default 52)
  --weather file.csv optional weather forecast, see README section "Weather forecast file"
  --tune             small hyper-parameter search with time-ordered cross-validation (slower)

Needs: python 3.9+, numpy, pandas, scikit-learn
"""
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Settings. Change the right-hand side of COLUMNS to match your file.
# ---------------------------------------------------------------------------
COLUMNS = {
    "date": "week_start",          # Monday of the week, any date format pandas can read
    "polyhouse": "polyhouse",      # polyhouse id
    "crop": "crop_code",           # crop code, e.g. TOM
    "variety": "variety_code",     # variety code, e.g. TO-0001 (several varieties joined by |)
    "yield_kg": "harvest_kg",      # total harvest of the polyhouse in the week, kg
    "area_m2": "area_m2",          # growing area of the polyhouse, m2
}
# Optional columns. Any that are missing from your file are skipped.
OPTIONAL = {
    "cycle": "cycle_id",                       # crop cycle id; inferred from gaps if absent
    "crop_age": "crop_age_weeks",              # weeks since transplant; computed if absent
    "tday": "day_temp_c",                      # weekly mean day temperature
    "tnight": "night_temp_c",                  # weekly mean night temperature
    "rh": "rh_pct",                            # weekly mean relative humidity
    "tsr": "solar_radiation_mj_m2",            # weekly mean daily solar radiation
    "vpd": "vpd_kpa",                          # weekly mean VPD
    "fruits": "fruits_per_cluster",            # growth records
    "clusters": "harvested_clusters",
    "height": "plant_height_cm",
    "nodes": "nodes_per_plant",
}
WEATHER_COLS = ["tday", "tnight", "rh", "tsr", "vpd"]
GROWTH_COLS = ["fruits", "clusters", "height", "nodes"]
INTERVAL = (0.10, 0.90)  # 80% interval
CYCLE_GAP_DAYS = 21      # if your file has no cycle column, a gap longer than this starts a new crop cycle
CYCLE_END_Q = 0.9        # forecasts stop after the crop age that 90% of finished cycles of that crop did not pass
SEED = 42
INFO_COLS = ("y", "target_date", "origin_date", "polyhouse", "crop", "variety")


# ---------------------------------------------------------------------------
# 2. Read and prepare the weekly records
# ---------------------------------------------------------------------------
def load(path):
    raw = pd.read_csv(path)
    ren = {}
    for key, col in {**COLUMNS, **OPTIONAL}.items():
        if col in raw.columns:
            ren[col] = key
        elif key in COLUMNS:
            raise SystemExit(f"Column '{col}' (needed as '{key}') is not in {path}. Edit COLUMNS at the top of the script.")
    df = raw.rename(columns=ren)[list(ren.values())].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["polyhouse"] = df["polyhouse"].astype(str)
    df["variety"] = df["variety"].fillna("unknown").astype(str)
    df = df[df["area_m2"] > 0]
    df = df.sort_values(["polyhouse", "date"]).drop_duplicates(["polyhouse", "date"]).reset_index(drop=True)
    df["y"] = df["yield_kg"] / df["area_m2"]                       # kg per m2 per week
    return df


def add_cycles(df):
    """Give each crop cycle an id. Uses the file's own cycle column if it has one."""
    if "cycle" not in df.columns:
        prev_crop = df.groupby("polyhouse")["crop"].shift()
        gap = df.groupby("polyhouse")["date"].diff().dt.days.fillna(0)
        newc = (gap > CYCLE_GAP_DAYS) | ((df["crop"] != prev_crop) & prev_crop.notna())
        df["cycle"] = newc.groupby(df["polyhouse"]).cumsum()
    df["cycle_key"] = df["polyhouse"] + "_" + df["cycle"].astype(str)
    if "crop_age" not in df.columns:
        df["crop_age"] = df.groupby("cycle_key").cumcount() + 1
    df["crop_age"] = df["crop_age"].astype(float)
    for c in WEATHER_COLS + GROWTH_COLS:
        if c in df.columns:
            # fill short gaps inside a crop cycle only; anything left is filled by the model's imputer
            df[c] = df.groupby("cycle_key")[c].transform(lambda s: s.interpolate(limit_direction="both", limit=3))
    return df


def climatology(df, cols):
    """Average weather by week of the year, from the history. Used as the expected weather for future weeks."""
    woy = df["date"].dt.isocalendar().week.clip(upper=52).astype(int)
    out = {}
    for c in cols:
        m = df.groupby(woy)[c].mean().reindex(range(1, 53))
        m = m.interpolate(limit_direction="both")
        sm = pd.concat([m.iloc[-2:], m, m.iloc[:2]]).rolling(3, center=True).mean().iloc[2:-2]
        out[c] = sm.values
    return out


def first_harvest_age(df):
    """First crop age with harvest, for each crop. Weeks before that are not forecast."""
    return df[df["y"] > 0].groupby("crop")["crop_age"].min().to_dict()


def cycle_end_age(df, last_date):
    """Typical last crop age of each crop, from cycles that have finished."""
    g = df.groupby("cycle_key").agg(crop=("crop", "first"), end=("date", "max"), age=("crop_age", "max"))
    done = g[g["end"] < last_date - pd.Timedelta(weeks=2)]
    return done.groupby("crop")["age"].quantile(CYCLE_END_Q).round().to_dict()


# ---------------------------------------------------------------------------
# 3. Features for "h weeks ahead"
# ---------------------------------------------------------------------------
def season(date):
    w = date.isocalendar().week
    a = 2 * np.pi * min(w, 52) / 52
    return np.sin(a), np.cos(a)


def feature_row(rec, hist, h, clim, crops, varieties, weather=None):
    """One feature row from the latest record `rec` and the polyhouse history `hist` (up to and including rec)."""
    tdate = rec["date"] + pd.Timedelta(weeks=h)
    row = {"age": rec["crop_age"], "target_age": rec["crop_age"] + h}
    ys = hist["y"].values
    for k in range(4):
        row[f"y_lag{k}"] = ys[-1 - k] if len(ys) > k else 0.0
    row["y_roll4"] = float(np.mean(ys[-4:]))
    row["season_sin"], row["season_cos"] = season(tdate)
    two = hist.tail(2)
    for c in WEATHER_COLS:
        if c in hist.columns:
            row[c] = float(two[c].mean())
    for c in GROWTH_COLS:
        if c in hist.columns:
            row[c] = float(rec[c])
    woy = min(tdate.isocalendar().week, 52) - 1
    for c in WEATHER_COLS:
        if c in clim:
            val = clim[c][woy]
            if weather is not None and (rec["polyhouse"], tdate) in weather:
                w = weather[(rec["polyhouse"], tdate)].get(c)
                if w is not None and not pd.isna(w):
                    val = w
            row[f"fc_{c}"] = float(val)
    for cr in crops:
        row[f"crop_{cr}"] = 1.0 if rec["crop"] == cr else 0.0
    for v in varieties:
        row[f"variety_{v}"] = 1.0 if rec["variety"] == v else 0.0
    return row


def set_expected_weather(tab, clim):
    """(Re)write the expected-weather columns from a climatology, using each row's target week."""
    tab = tab.copy()
    woy = tab["target_date"].dt.isocalendar().week.clip(upper=52).astype(int).values - 1
    for c, vals in clim.items():
        tab[f"fc_{c}"] = np.asarray(vals)[woy]
    return tab


def build_table(df, h, clim, crops, varieties, min_age):
    rows = []
    for _, g in df.groupby("cycle_key"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(len(g) - h):
            rec = g.iloc[i]
            tgt = g.iloc[i + h]
            if (tgt["date"] - rec["date"]).days != 7 * h:      # a week is missing in between
                continue
            if tgt["crop_age"] < min_age.get(rec["crop"], 1):  # harvest not started yet in the target week
                continue
            r = feature_row(rec, g.iloc[: i + 1], h, clim, crops, varieties)
            r.update(y=tgt["y"], target_date=tgt["date"], origin_date=rec["date"], polyhouse=rec["polyhouse"], crop=rec["crop"], variety=rec["variety"])
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Models and scoring
# ---------------------------------------------------------------------------
def make_models():
    """Every model starts with a median imputer, so a missing growth or weather record never stops a run."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Naive": None,   # repeat this week's yield
        "Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=2.0)),
        "RandomForest": make_pipeline(imp(), RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
        "GradientBoosting": make_pipeline(imp(), GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)),
    }


def tune_gb(X, y):
    """Small grid search for Gradient Boosting. Rows must be in time order so each fold only trains on the past."""
    grid = {"gradientboostingregressor__n_estimators": [200, 400],
            "gradientboostingregressor__max_depth": [2, 3, 4],
            "gradientboostingregressor__learning_rate": [0.03, 0.06]}
    gs = GridSearchCV(make_models()["GradientBoosting"], grid, cv=TimeSeriesSplit(4), scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, {k.split("__")[1]: v for k, v in gs.best_params_.items()}


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0.05
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mape": float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan"),
        "wape": float(np.sum(np.abs(y - p)) / max(np.sum(y), 1e-9) * 100),
        "r2": float(1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
    }


def ratio_quantiles(y, p):
    r = (np.asarray(y) - np.asarray(p)) / np.maximum(np.asarray(p), 0.05)
    return float(np.quantile(r, INTERVAL[0])), float(np.quantile(r, INTERVAL[1]))


GROUPS = [  # related features are shuffled together, so credit is not split between near-copies
    ("Recent harvest", lambda f: f.startswith("y_")),
    ("Crop stage (age, height, nodes)", lambda f: f in ("age", "target_age", "height", "nodes")),
    ("Fruit set (fruits per cluster)", lambda f: f == "fruits"),
    ("Harvested clusters", lambda f: f == "clusters"),
    ("Weather, last 2 weeks", lambda f: f in WEATHER_COLS),
    ("Expected weather", lambda f: f.startswith("fc_")),
    ("Season", lambda f: f.startswith("season_")),
    ("Crop and variety", lambda f: f.startswith(("crop_", "variety_"))),
]


def importance(model, X, y, feats, repeats=8):
    """Grouped permutation importance: how much the error (MAE) grows when a group of features is shuffled."""
    rng = np.random.default_rng(SEED)
    X = X.reset_index(drop=True)
    y = np.asarray(y, float)
    base = np.mean(np.abs(y - model.predict(X[feats])))
    out = []
    for name, test in GROUPS:
        cols = [f for f in feats if test(f)]
        if not cols:
            continue
        rise = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].values[rng.permutation(len(X))]   # same shuffle for the whole group
            rise.append(np.mean(np.abs(y - model.predict(Xp[feats]))) - base)
        out.append([name, max(0.0, float(np.mean(rise)))])
    tot = sum(v for _, v in out) or 1
    return sorted(([k, round(v / tot * 100, 2)] for k, v in out), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def read_weather(path):
    """Optional weather forecast: columns polyhouse (or ALL), week_start, and any of day_temp_c, night_temp_c, rh_pct, solar_radiation_mj_m2, vpd_kpa."""
    w = pd.read_csv(path)
    w["week_start"] = pd.to_datetime(w["week_start"]).dt.normalize()
    ren = {OPTIONAL[k]: k for k in WEATHER_COLS if OPTIONAL[k] in w.columns}
    return w.rename(columns=ren)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--test-weeks", type=int, default=52)
    ap.add_argument("--weather", default=None)
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_cycles(load(a.data))
    present_w = [c for c in WEATHER_COLS if c in df.columns]
    crops = sorted(df["crop"].unique())
    varieties = sorted(df["variety"].unique())
    min_age = first_harvest_age(df)
    last_date = df["date"].max()
    end_age = cycle_end_age(df, last_date)
    cutoff = last_date - pd.Timedelta(weeks=a.test_weeks)
    mid_test = cutoff + pd.Timedelta(weeks=a.test_weeks // 2)
    # expected weather for the backtest only uses history up to the cutoff; the live forecast uses all of it
    clim_train = climatology(df[df["date"] <= cutoff], present_w) if present_w else {}
    clim_all = climatology(df, present_w) if present_w else {}
    print(f"{len(df)} weekly records, {df['polyhouse'].nunique()} polyhouses, {df['date'].min().date()} to {last_date.date()}")
    print(f"Training on target weeks up to {cutoff.date()}, testing on the {a.test_weeks} weeks after it")

    weather = None
    if a.weather:
        wf = read_weather(a.weather)
        weather = {}
        for _, r in wf.iterrows():
            ph = df["polyhouse"].unique() if str(r.get("polyhouse", "ALL")) == "ALL" else [str(r["polyhouse"])]
            for p in ph:
                weather[(p, r["week_start"])] = {c: r[c] for c in present_w if c in wf.columns}

    area_by_ph = df.groupby("polyhouse")["area_m2"].last().to_dict()
    models_all = ["Naive", "Ridge", "RandomForest", "GradientBoosting"]
    metrics = {m: {} for m in models_all}
    by_crop, best, skill, coverage, tuned = {}, {}, {}, {}, {}
    backtest_rows, scatter_rows = [], []
    importance_by_h = {}
    fitted_best, ratio_q = {}, {}

    for h in range(1, a.horizon + 1):
        tab = build_table(df, h, clim_train, crops, varieties, min_age)
        if tab.empty:
            raise SystemExit("No usable rows. Check that weeks are consecutive within each crop cycle.")
        feats = [c for c in tab.columns if c not in INFO_COLS]
        tr = tab[tab["target_date"] <= cutoff].sort_values(["target_date", "polyhouse"])
        te = tab[tab["target_date"] > cutoff]
        if len(tr) < 100 or len(te) < 20:
            raise SystemExit(f"Too little data for horizon {h}: {len(tr)} training rows, {len(te)} test rows.")
        models = make_models()
        if a.tune:
            models["GradientBoosting"], tuned[h] = tune_gb(tr[feats], tr["y"])
        preds = {}
        for name, mdl in models.items():
            if name == "Naive":
                p = te["y_lag0"].values
            else:
                if not (a.tune and name == "GradientBoosting"):   # the tuned model is already fitted on tr
                    mdl.fit(tr[feats], tr["y"])
                p = np.clip(mdl.predict(te[feats]), 0, None)
            preds[name] = p
            metrics[name][h] = score(te["y"], p)
        best[h] = min([m for m in models_all if m != "Naive"], key=lambda m: metrics[m][h]["mae"])
        skill[h] = round((1 - metrics[best[h]][h]["mae"] / metrics["Naive"][h]["mae"]) * 100, 1)
        te = te.assign(pred=preds[best[h]])

        # 80% range: learn the error spread on the first half of the test year, check it on the second half,
        # then use the whole test year for the live forecast
        first, second = te[te["target_date"] <= mid_test], te[te["target_date"] > mid_test]
        lo, hi = ratio_quantiles(first["y"], first["pred"])
        inside = (second["y"] >= second["pred"] * (1 + lo)) & (second["y"] <= second["pred"] * (1 + hi))
        coverage[h] = round(float(inside.mean()) * 100, 1)
        ratio_q[h] = ratio_quantiles(te["y"], te["pred"])

        for cr, g in te.groupby("crop"):
            by_crop.setdefault(cr, {})[h] = score(g["y"], g["pred"])
        if h == 1:
            for ph, g in te.sort_values("target_date").groupby("polyhouse"):
                backtest_rows.append((ph, g["crop"].tolist(), [d.date().isoformat() for d in g["target_date"]], g["y"].round(4).tolist(), g["pred"].round(4).tolist()))
            smp = te.sample(min(400, len(te)), random_state=SEED)
            scatter_rows = [[round(float(y), 3), round(float(p), 3), c] for y, p, c in zip(smp["y"], smp["pred"], smp["crop"])]
        if h in (1, a.horizon):
            importance_by_h[h] = importance(models[best[h]], te[feats], te["y"], feats)

        # refit the best model (with tuned settings, if any) on all rows for the live forecast
        full = set_expected_weather(tab, clim_all).sort_values(["target_date", "polyhouse"])
        final = clone(models[best[h]])
        final.fit(full[feats], full["y"])
        fitted_best[h] = (final, feats)
        print(f"  h={h}: best {best[h]:17s} MAE {metrics[best[h]][h]['mae']:.3f}  MAPE {metrics[best[h]][h]['mape']:.1f}%  "
              f"(naive MAPE {metrics['Naive'][h]['mape']:.1f}%, {skill[h]:+.0f}% better)  80% range held {coverage[h]:.0f}%")

    # ---- forecast the running crops --------------------------------------
    forecast = {}
    for ph, g in df.groupby("polyhouse"):
        g = g.sort_values("date")
        rec = g.iloc[-1]
        if (last_date - rec["date"]).days > 14:
            continue                                           # no record in the last two weeks: crop not running
        cyc = g[g["cycle_key"] == rec["cycle_key"]].reset_index(drop=True)
        dates, ys, los, his = [], [], [], []
        stop = end_age.get(rec["crop"], np.inf)
        for h in range(1, a.horizon + 1):
            tdate = rec["date"] + pd.Timedelta(weeks=h)
            tage = rec["crop_age"] + h
            if tage < min_age.get(rec["crop"], 1) or tage > stop:
                val = lo = hi = 0.0                            # before first harvest, or after the crop normally ends
            else:
                model, feats = fitted_best[h]
                row = feature_row(rec, cyc, h, clim_all, crops, varieties, weather)
                val = float(max(0.0, model.predict(pd.DataFrame([row])[feats])[0]))
                lo, hi = val * (1 + ratio_q[h][0]), val * (1 + ratio_q[h][1])
            dates.append(tdate.date().isoformat())
            ys.append(round(val, 4)); los.append(round(max(0.0, lo), 4)); his.append(round(hi, 4))
        hist = cyc.tail(16)
        forecast[ph] = dict(crop=rec["crop"], variety=rec["variety"], area_m2=float(rec["area_m2"]), crop_age=float(rec["crop_age"]),
                            expected_end_age=None if np.isinf(stop) else float(stop),
                            last_actual_date=rec["date"].date().isoformat(), last_actual=round(float(rec["y"]), 4),
                            history_dates=[d.date().isoformat() for d in hist["date"]], history=hist["y"].round(4).tolist(),
                            dates=dates, y=ys, lo=los, hi=his)

    # ---- write results -----------------------------------------------------
    result = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_start": df["date"].min().date().isoformat(), "data_end": last_date.date().isoformat(),
            "n_rows": int(len(df)), "n_polyhouses": int(df["polyhouse"].nunique()), "n_cycles": int(df["cycle_key"].nunique()),
            "crops": crops, "varieties": varieties,
            "first_harvest_age": {k: float(v) for k, v in min_age.items()}, "cycle_end_age": {k: float(v) for k, v in end_age.items()},
            "horizon_weeks": a.horizon, "test_start": (cutoff + pd.Timedelta(weeks=1)).date().isoformat(), "test_weeks": a.test_weeks,
            "models": models_all, "best_model": {str(h): m for h, m in best.items()},
            "skill_vs_naive_pct": {str(h): v for h, v in skill.items()},
            "interval_coverage_pct": {str(h): v for h, v in coverage.items()},
            "weather": "weather forecast file" if a.weather else "seasonal normals from history",
            "interval": "80%, from backtest errors at each distance ahead",
            "unit": "kg per m2 per week", "tuned": {str(h): v for h, v in tuned.items()} if a.tune else False,
        },
        "metrics": {"overall": {m: {str(h): v for h, v in d.items()} for m, d in metrics.items()},
                    "by_crop": {c: {str(h): v for h, v in d.items()} for c, d in by_crop.items()}},
        "backtest": {ph: dict(crop=cr[-1], crops=cr, area_m2=float(area_by_ph[ph]), dates=d, actual=ya, pred=yp) for ph, cr, d, ya, yp in backtest_rows},
        "forecast": forecast,
        "importance": importance_by_h.get(1),
        "importance_by_h": {str(h): v for h, v in importance_by_h.items()},
        "scatter": scatter_rows,
    }
    (out / "forecast_output.json").write_text(json.dumps(result, separators=(",", ":")))
    pd.DataFrame([dict(model=m, weeks_ahead=h, **v) for m, d in metrics.items() for h, v in d.items()]).to_csv(out / "model_metrics.csv", index=False)
    pd.DataFrame([dict(polyhouse=ph, crop=f["crop"], variety=f["variety"], week_start=d, yield_kg_m2=y, low_kg_m2=lo, high_kg_m2=hi,
                       yield_kg=round(y * f["area_m2"], 1), low_kg=round(lo * f["area_m2"], 1), high_kg=round(hi * f["area_m2"], 1))
                  for ph, f in forecast.items() for d, y, lo, hi in zip(f["dates"], f["y"], f["lo"], f["hi"])]).to_csv(out / "forecast.csv", index=False)
    print(f"Done. Results are in {out.resolve()}  (forecast_output.json, forecast.csv, model_metrics.csv)")


if __name__ == "__main__":
    main()
