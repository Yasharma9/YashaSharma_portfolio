"""
yield_forecast.py
Weekly yield forecasting for polyhouse crops.

What it does
  1. Reads weekly records (one row per polyhouse per week), built from about 3 years of history.
  2. Builds features: recent yield, crop stage and the crop's typical yield curve, fruit set in
     earlier weeks, weather, season, crop and variety.
  3. For each crop and each forecast distance (1, 2, ... N weeks ahead) it trains several algorithms.
  4. Picks the best algorithm per crop and distance on a validation period, then tests all of them
     on the most recent weeks, which were held out (a time-based backtest).
  5. Refits the chosen model on all data and forecasts the running crops, with an 80% range.
  6. Writes forecast_output.json (for the dashboard) and CSV files.

Run
  python yield_forecast.py --data weekly_records.csv --out results

Options
  --horizon 8        weeks ahead to forecast (default 8)
  --test-weeks 52    weeks held out for the backtest (default 52)
  --val-weeks 39     weeks before the test period used to pick the best model (default 39, in 3 rolling folds)
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
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
    "planned_end": "planned_end_week",         # crop age (weeks) at which removal is planned
    "fruits": "fruits_per_cluster",            # growth records
    "set": "set_clusters",                     # clusters setting fruit this week (harvested some weeks later)
    "clusters": "harvested_clusters",
    "height": "plant_height_cm",
    "nodes": "nodes_per_plant",
}
WEATHER_COLS = ["tday", "tnight", "rh", "tsr", "vpd"]
GROWTH_COLS = ["fruits", "set", "clusters", "height", "nodes"]
SET_LAGS = 8             # fruit set this week and up to 7 weeks earlier: fruit set then is harvested later
INTERVAL = (0.10, 0.90)  # 80% interval
CYCLE_GAP_DAYS = 21      # if your file has no cycle column, a gap longer than this starts a new crop cycle
CYCLE_END_Q = 0.9        # forecasts stop after the crop age that 90% of finished cycles of that crop did not pass
MAPE_MIN = 0.05          # weeks with less than this (kg/m2) are left out of MAPE, which explodes near zero
SEED = 42
MODELS = ["Naive", "Ridge", "RandomForest", "GradientBoosting", "Ensemble"]
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
    # weeks since this cycle's first harvest (negative before it): lines cycles up by when harvest really began
    first = df[df["y"] > 0].groupby("cycle_key")["crop_age"].min()
    df["hw"] = df["crop_age"] - df["cycle_key"].map(first)
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


def crop_profile(df):
    """Typical yield of each crop at each crop age (smoothed mean over past cycles)."""
    out = {}
    for crop, g in df.groupby("crop"):
        m = g.groupby(g["crop_age"].astype(int))["y"].mean()
        m = m.reindex(range(1, int(m.index.max()) + 1)).interpolate(limit_direction="both")
        out[crop] = m.rolling(3, center=True, min_periods=1).mean().to_dict()
    return out


def harvest_profile(df):
    """Typical yield of each crop by weeks since the cycle's first harvest."""
    out = {}
    for crop, g in df[df["hw"] >= 0].groupby("crop"):
        m = g.groupby(g["hw"].astype(int))["y"].mean()
        m = m.reindex(range(0, int(m.index.max()) + 1)).interpolate(limit_direction="both")
        out[crop] = m.rolling(3, center=True, min_periods=1).mean().to_dict()
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
    """One feature row from the latest record `rec` and the cycle history `hist` (up to and including rec)."""
    tdate = rec["date"] + pd.Timedelta(weeks=h)
    row = {"age": rec["crop_age"], "target_age": rec["crop_age"] + h}
    ys = hist["y"].values
    for k in range(4):
        row[f"y_lag{k}"] = ys[-1 - k] if len(ys) > k else 0.0
    row["y_roll4"] = float(np.mean(ys[-4:]))
    hs = np.flatnonzero(ys > 0)
    row["hw"] = float(len(ys) - 1 - hs[0]) if len(hs) else -1.0
    row["y_trend"] = float(np.mean(ys[-2:]) - np.mean(ys[-4:-2])) if len(ys) >= 4 else 0.0
    row["season_sin"], row["season_cos"] = season(tdate)
    two = hist.tail(2)
    for c in WEATHER_COLS:
        if c in hist.columns:
            row[c] = float(two[c].mean())
    for c in GROWTH_COLS:
        if c in hist.columns:
            row[c] = float(rec[c])
    for c in ("fruits", "set"):
        if c in hist.columns:
            v = hist[c].values
            for k in range(1, SET_LAGS):
                row[f"{c}_lag{k}"] = float(v[-1 - k]) if len(v) > k else np.nan
            if c == "set":   # fruit set in the week that is harvested in the target week, for each lag
                row["set_sum"] = float(np.nansum(v[-SET_LAGS:]))
    if "planned_end" in hist.columns and not pd.isna(rec["planned_end"]):
        row["to_end"] = float(rec["planned_end"] - (rec["crop_age"] + h))   # weeks between target week and planned removal
        row["last_week"] = 1.0 if row["to_end"] == 0 else 0.0
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


def set_profile(tab, prof):
    """Typical yield for the crop in the target week and now, and how this crop is doing against it.
    Once harvest has started, the curve by weeks since first harvest is used; before that, the curve by crop age."""
    tab = tab.copy()
    by_age, by_hw = prof
    look = lambda d, crop, k: d.get(crop, {}).get(int(k), np.nan)
    def pick(c, age, hw, h):
        return look(by_hw, c, hw + h) if hw >= 0 else look(by_age, c, age + h)
    tab["prof_target"] = [pick(c, a, w, t - a) for c, a, w, t in zip(tab["crop"], tab["age"], tab["hw"], tab["target_age"])]
    tab["prof_now"] = [pick(c, a, w, 0) for c, a, w in zip(tab["crop"], tab["age"], tab["hw"])]
    tab["prof_index"] = tab["y_roll4"] / np.maximum(tab["prof_now"].fillna(0), 0.05)
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
    rf = lambda: RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=SEED)
    gb = lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)
    return {
        "Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=2.0)),
        "RandomForest": make_pipeline(imp(), rf()),
        "GradientBoosting": make_pipeline(imp(), gb()),
        # average of the two tree models: usually steadier than either one alone
        "Ensemble": make_pipeline(imp(), VotingRegressor([("rf", rf()), ("gb", gb())])),
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
    m = y > MAPE_MIN
    mape = float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan")
    wape = float(np.sum(np.abs(y - p)) / max(np.sum(y), 1e-9) * 100)
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mape": mape,
        "wape": wape,
        "accuracy": max(0.0, 100 - mape),                  # accuracy = 100% - MAPE
        "r2": float(1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
        "n": int(len(y)),
    }


def error_quantiles(y, p):
    """10th and 90th percentile of the forecast error (actual minus forecast), in kg/m2."""
    r = np.asarray(y) - np.asarray(p)
    return float(np.quantile(r, INTERVAL[0])), float(np.quantile(r, INTERVAL[1]))


GROUPS = [  # related features are shuffled together, so credit is not split between near-copies
    ("Recent harvest", lambda f: f.startswith("y_")),
    ("Crop stage and typical yield curve", lambda f: f in ("age", "target_age", "hw", "height", "nodes") or f.startswith("prof_")),
    ("Fruit set, this and earlier weeks", lambda f: f.startswith(("fruits", "set"))),
    ("Planned crop removal", lambda f: f in ("to_end", "last_week")),
    ("Harvested clusters", lambda f: f == "clusters"),
    ("Weather, last 2 weeks", lambda f: f in WEATHER_COLS),
    ("Expected weather", lambda f: f.startswith("fc_")),
    ("Season", lambda f: f.startswith("season_")),
    ("Variety", lambda f: f.startswith(("crop_", "variety_"))),
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
    ap.add_argument("--val-weeks", type=int, default=39)
    ap.add_argument("--val-folds", type=int, default=3)
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
    val_start = cutoff - pd.Timedelta(weeks=a.val_weeks)
    mid_test = cutoff + pd.Timedelta(weeks=a.test_weeks // 2)
    # anything learned from history (weather normals, typical yield curves) only uses weeks before the
    # test period in the backtest; the live forecast uses all of it
    train_hist = df[df["date"] <= cutoff]
    clim_train = climatology(train_hist, present_w) if present_w else {}
    clim_all = climatology(df, present_w) if present_w else {}
    prof_train = (crop_profile(train_hist), harvest_profile(train_hist))
    prof_all = (crop_profile(df), harvest_profile(df))
    print(f"{len(df)} weekly records, {df['polyhouse'].nunique()} polyhouses, {df['date'].min().date()} to {last_date.date()}")
    print(f"Training on target weeks up to {cutoff.date()} (model choice on the last {a.val_weeks} of them, {a.val_folds} rolling folds), "
          f"testing on the {a.test_weeks} weeks after it")

    weather = None
    if a.weather:
        wf = read_weather(a.weather)
        weather = {}
        for _, r in wf.iterrows():
            ph = df["polyhouse"].unique() if str(r.get("polyhouse", "ALL")) == "ALL" else [str(r["polyhouse"])]
            for p in ph:
                weather[(p, r["week_start"])] = {c: r[c] for c in present_w if c in wf.columns}

    metrics = {c: {m: {} for m in MODELS} for c in crops}            # crop -> model -> h -> scores
    overall = {m: {} for m in MODELS + ["Best"]}                     # all crops together
    best, val_mae, skill, coverage, tuned = ({c: {} for c in crops} for _ in range(5))
    importance_by = {c: {} for c in crops}
    fitted, err_q = {c: {} for c in crops}, {c: {} for c in crops}
    store = {}                                                       # (crop, h) -> what the second step needs
    feats_h = {}

    # step 1: for every crop and distance, score every model on the validation weeks and on the test year
    for h in range(1, a.horizon + 1):
        tab = set_profile(build_table(df, h, clim_train, crops, varieties, min_age), prof_train)
        if tab.empty:
            raise SystemExit("No usable rows. Check that weeks are consecutive within each crop cycle.")
        feats = feats_h[h] = [c for c in tab.columns if c not in INFO_COLS]
        full_tab = set_profile(set_expected_weather(tab, clim_all), prof_all)
        for crop in crops:
            tc = tab[tab["crop"] == crop]
            tr = tc[tc["target_date"] <= cutoff].sort_values(["target_date", "polyhouse"])
            te = tc[tc["target_date"] > cutoff].copy()
            fit = tr[tr["target_date"] <= val_start]
            if len(fit) < 60 or len(te) < 10:
                raise SystemExit(f"Too little {crop} data for horizon {h}: {len(fit)} rows before validation, {len(te)} test rows.")
            models = make_models()
            if a.tune:
                models["GradientBoosting"], tuned[crop][h] = tune_gb(fit[feats], fit["y"])
            # rolling-origin validation: each fold is predicted by models trained only on the weeks before it
            err = {name: [] for name in models}
            fold = pd.Timedelta(weeks=a.val_weeks // a.val_folds)
            for k in range(a.val_folds):
                f0 = val_start + k * fold
                f1 = cutoff if k == a.val_folds - 1 else f0 + fold
                ftr, fva = tr[tr["target_date"] <= f0], tr[(tr["target_date"] > f0) & (tr["target_date"] <= f1)]
                if len(fva) == 0:
                    continue
                for name, mdl in models.items():
                    m = clone(mdl).fit(ftr[feats], ftr["y"])
                    err[name].extend(np.abs(fva["y"] - np.clip(m.predict(fva[feats]), 0, None)))
            vm = {name: float(np.mean(e)) for name, e in err.items()}
            val_mae[crop][h] = {k: round(v, 4) for k, v in vm.items()}
            te["pred_Naive"] = te["y_lag0"].values                   # test: train on everything before the test year
            test_models = {}
            for name, mdl in models.items():
                test_models[name] = clone(mdl).fit(tr[feats], tr["y"])
                te[f"pred_{name}"] = np.clip(test_models[name].predict(te[feats]), 0, None)
            for name in MODELS:
                metrics[crop][name][h] = score(te["y"], te[f"pred_{name}"])
            store[(crop, h)] = dict(te=te, models=models, test_models=test_models,
                                    full=full_tab[full_tab["crop"] == crop].sort_values(["target_date", "polyhouse"]))
        print(f"  h={h}: scored " + ", ".join(f"{c} ({len(store[(c, h)]['te'])} test weeks)" for c in crops))

    # step 2: one model per crop, the one with the lowest average validation error over all distances.
    # The test year plays no part in the choice, so its scores are a fair estimate of live accuracy.
    for crop in crops:
        mean_val = {m: np.mean([val_mae[crop][h][m] for h in val_mae[crop]]) for m in MODELS if m != "Naive"}
        choice = min(mean_val, key=mean_val.get)
        for h in range(1, a.horizon + 1):
            st = store[(crop, h)]
            te, feats = st["te"], feats_h[h]
            best[crop][h] = choice
            te["pred_Best"] = te[f"pred_{choice}"]
            skill[crop][h] = round((1 - metrics[crop][choice][h]["mae"] / metrics[crop]["Naive"][h]["mae"]) * 100, 1)
            # 80% range: learned on the first half of the test year, checked on the second half;
            # the live forecast uses the whole test year
            first, second = te[te["target_date"] <= mid_test], te[te["target_date"] > mid_test]
            lo, hi = error_quantiles(first["y"], first["pred_Best"])
            inside = (second["y"] >= second["pred_Best"] + lo) & (second["y"] <= second["pred_Best"] + hi)
            coverage[crop][h] = round(float(inside.mean()) * 100, 1)
            err_q[crop][h] = error_quantiles(te["y"], te["pred_Best"])
            if h in (1, a.horizon):
                importance_by[crop][h] = importance(st["test_models"][choice], te[feats], te["y"], feats)
            fitted[crop][h] = (clone(st["models"][choice]).fit(st["full"][feats], st["full"]["y"]), feats)
        print(f"  {crop}: best model {choice} (mean validation MAE " + ", ".join(f"{m} {v:.3f}" for m, v in mean_val.items()) + ")")
        print("     accuracy by weeks ahead: " + "  ".join(f"{h}:{metrics[crop][choice][h]['accuracy']:.1f}%" for h in range(1, a.horizon + 1)))

    test_rows = {}
    for h in range(1, a.horizon + 1):
        te_all = pd.concat([store[(c, h)]["te"] for c in crops])
        test_rows[h] = te_all
        for name in MODELS + ["Best"]:
            overall[name][h] = score(te_all["y"], te_all[f"pred_{name}"])
    print("  all crops: " + "  ".join(f"{h}:{overall['Best'][h]['accuracy']:.1f}%" for h in range(1, a.horizon + 1)))

    # ---- backtest series per polyhouse, every model and distance ------------
    weeks = pd.date_range(cutoff + pd.Timedelta(weeks=1), last_date, freq="7D")
    widx = {d: i for i, d in enumerate(weeks)}
    backtest = {}
    test_df = df[df["date"] > cutoff]
    for ph, g in test_df.groupby("polyhouse"):
        actual, crop_w = [None] * len(weeks), [None] * len(weeks)
        for d, y, c in zip(g["date"], g["y"], g["crop"]):
            if d in widx:
                actual[widx[d]], crop_w[widx[d]] = round(float(y), 4), c
        preds = {}
        for h, te in test_rows.items():
            t = te[te["polyhouse"] == ph]
            preds[str(h)] = {}
            for name in MODELS:
                arr = [None] * len(weeks)
                for d, p in zip(t["target_date"], t[f"pred_{name}"]):
                    arr[widx[d]] = round(float(p), 3)
                preds[str(h)][name] = arr
        backtest[ph] = dict(area_m2=float(g["area_m2"].iloc[-1]), crop=crop_w, actual=actual, pred=preds)

    # ---- forecast the running crops --------------------------------------
    forecast, idle = {}, {}
    for ph, g in df.groupby("polyhouse"):
        g = g.sort_values("date")
        rec = g.iloc[-1]
        if (last_date - rec["date"]).days > 14:                # no record in the last two weeks: crop not running
            idle[ph] = dict(last_crop=rec["crop"], last_date=rec["date"].date().isoformat(), area_m2=float(rec["area_m2"]))
            continue
        cyc = g[g["cycle_key"] == rec["cycle_key"]].reset_index(drop=True)
        crop = rec["crop"]
        dates, ys, los, his = [], [], [], []
        stop = float(rec["planned_end"]) if "planned_end" in rec and not pd.isna(rec["planned_end"]) else end_age.get(crop, np.inf)
        for h in range(1, a.horizon + 1):
            tdate = rec["date"] + pd.Timedelta(weeks=h)
            tage = rec["crop_age"] + h
            if tage < min_age.get(crop, 1) or tage > stop:
                val = lo = hi = 0.0                            # before first harvest, or after the crop normally ends
            else:
                model, feats = fitted[crop][h]
                row = pd.DataFrame([feature_row(rec, cyc, h, clim_all, crops, varieties, weather)])
                row = set_profile(row.assign(crop=crop), prof_all)
                val = float(max(0.0, model.predict(row[feats])[0]))
                lo, hi = val + err_q[crop][h][0], val + err_q[crop][h][1]
            dates.append(tdate.date().isoformat())
            ys.append(round(val, 4)); los.append(round(max(0.0, lo), 4)); his.append(round(hi, 4))
        hist = cyc.tail(16)
        forecast[ph] = dict(crop=crop, variety=rec["variety"], area_m2=float(rec["area_m2"]), crop_age=float(rec["crop_age"]),
                            expected_end_age=None if np.isinf(stop) else float(stop),
                            last_actual_date=rec["date"].date().isoformat(), last_actual=round(float(rec["y"]), 4),
                            history_dates=[d.date().isoformat() for d in hist["date"]], history=hist["y"].round(4).tolist(),
                            best_model=[best[crop][h] for h in range(1, a.horizon + 1)],
                            dates=dates, y=ys, lo=los, hi=his)

    # ---- write results -----------------------------------------------------
    hk = lambda d: {str(h): v for h, v in d.items()}
    result = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_start": df["date"].min().date().isoformat(), "data_end": last_date.date().isoformat(),
            "n_rows": int(len(df)), "n_polyhouses": int(df["polyhouse"].nunique()), "n_cycles": int(df["cycle_key"].nunique()),
            "crops": crops, "varieties": varieties, "models": MODELS,
            "first_harvest_age": {k: float(v) for k, v in min_age.items()}, "cycle_end_age": {k: float(v) for k, v in end_age.items()},
            "horizon_weeks": a.horizon, "test_start": weeks[0].date().isoformat(), "test_weeks": a.test_weeks,
            "val_start": (val_start + pd.Timedelta(weeks=1)).date().isoformat(), "val_weeks": a.val_weeks, "val_folds": a.val_folds,
            "best_model": {c: hk(v) for c, v in best.items()},
            "best_model_by_crop": {c: v[1] for c, v in best.items()},
            "model_choice": "one model per crop: lowest average validation MAE over all distances",
            "validation_mae": {c: hk(v) for c, v in val_mae.items()},
            "skill_vs_naive_pct": {c: hk(v) for c, v in skill.items()},
            "interval_coverage_pct": {c: hk(v) for c, v in coverage.items()},
            "accuracy": f"100% minus MAPE; weeks under {MAPE_MIN} kg/m2 are left out of MAPE",
            "weather": "weather forecast file" if a.weather else "seasonal normals from history",
            "interval": "80%: 10th to 90th percentile of backtest errors (kg/m2) at each distance ahead, per crop",
            "unit": "kg per m2 per week", "tuned": {c: hk(v) for c, v in tuned.items()} if a.tune else False,
        },
        "metrics": {"by_crop": {c: {m: hk(d) for m, d in v.items()} for c, v in metrics.items()},
                    "overall": {m: hk(d) for m, d in overall.items()}},
        "backtest": {"weeks": [d.date().isoformat() for d in weeks], "polyhouses": backtest},
        "forecast": forecast,
        "idle": idle,
        "importance": {c: hk(v) for c, v in importance_by.items()},
    }
    (out / "forecast_output.json").write_text(json.dumps(result, separators=(",", ":")))
    pd.DataFrame([dict(crop=c, model=m, weeks_ahead=h, best=(best[c][h] == m), **s)
                  for c, v in metrics.items() for m, d in v.items() for h, s in d.items()]).to_csv(out / "model_metrics.csv", index=False)
    pd.DataFrame([dict(polyhouse=ph, crop=f["crop"], variety=f["variety"], week_start=d, weeks_ahead=i + 1, model=f["best_model"][i],
                       yield_kg_m2=y, low_kg_m2=lo, high_kg_m2=hi,
                       yield_kg=round(y * f["area_m2"], 1), low_kg=round(lo * f["area_m2"], 1), high_kg=round(hi * f["area_m2"], 1))
                  for ph, f in forecast.items() for i, (d, y, lo, hi) in enumerate(zip(f["dates"], f["y"], f["lo"], f["hi"]))]).to_csv(out / "forecast.csv", index=False)
    print(f"Done. Results are in {out.resolve()}  (forecast_output.json, forecast.csv, model_metrics.csv)")


if __name__ == "__main__":
    main()
