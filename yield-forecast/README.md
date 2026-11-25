# Crop yield forecasting

Weekly harvest forecasts 1 to 8 weeks ahead for polyhouse crops, with an 80% range.

**Live dashboard:** [crop-yield-forecast.html](https://yasharma9.github.io/YashaSharma_portfolio/crop-yield-forecast.html)

All data here is **sample data** made by `make_sample_data.py` with a fixed seed. It is realistic in shape (crop cycles, seasons, weather effects, pest dips, missing records) but does not come from any farm or company.

## Results on the held-out year

Trained on 2 years of weekly records for 20 polyhouses (tomato and cucumber), tested on the following 52 weeks, which the models never saw.

| Weeks ahead | Best model | MAPE | Naive MAPE | Lower error than naive (MAE) | 80% range held |
|---|---|---|---|---|---|
| 1 | Random forest | 12.1% | 19.0% | 25% | 77% |
| 2 | Random forest | 15.7% | 31.2% | 44% | 72% |
| 4 | Gradient boosting | 17.3% | 45.3% | 62% | 70% |
| 8 | Gradient boosting | 18.0% | 61.2% | 74% | 67% |

The naive forecast repeats this week's harvest. It is hard to beat 1 week ahead and falls apart further out. The models hold their accuracy because they know where each crop is in its cycle and how well fruit has set.

The 80% ranges are a little narrow on the later half of the test year (67% to 77% of weeks inside). That is mostly because of pest and disease dips that no input can see coming.

## How it works

1. **Features** for each polyhouse and week: recent harvest (last 4 weeks and their mean), crop stage (age, plant height, nodes), fruit set (fruits per cluster), harvested clusters, weather over the last 2 weeks, expected weather for the target week (seasonal normals or a weather forecast file), season, crop and variety.
2. **One model per distance** (direct multi-horizon): separate models for 1, 2, … 8 weeks ahead, so errors don't compound the way they do when a 1-week model is fed its own forecasts.
3. **Four candidates** per distance: naive baseline, ridge regression, random forest and gradient boosting (scikit-learn). Each starts with a median imputer, so missing records never stop a run.
4. **Time-based backtest:** train on target weeks up to the cutoff, test on the year after. The lowest-MAE model wins at each distance.
5. **No leakage:**
   - Seasonal weather normals for the backtest come only from weeks before the cutoff.
   - `--tune` uses `TimeSeriesSplit` on rows sorted by date.
   - The tuned settings are kept when the final model is refit.
6. **80% ranges** from backtest errors at each distance, as a ratio of the forecast. They are learned on the first half of the test year and checked on the second half.
7. **Grouped permutation importance:** related inputs (for example crop age, height and nodes) are shuffled together, so credit isn't split between near-copies.
8. **Crop cycles respected:** no harvest is forecast before a crop's first harvest age or after the age at which 90% of past cycles of that crop had ended.
9. **Live forecast:** the best model at each distance is refit on all data and forecasts every polyhouse with a crop in the ground.

## Files

| File | What it is |
|---|---|
| `make_sample_data.py` | Makes `data/weekly_records.csv`: 3 years, 20 polyhouses, 118 crop cycles |
| `yield_forecast.py` | The pipeline: features, models, backtest, ranges, live forecast |
| `embed_results.py` | Puts `forecast_output.json` into the dashboard page |
| `data/weekly_records.csv` | Sample input, one row per polyhouse per week |
| `results/forecast_output.json` | Everything the dashboard shows |
| `results/forecast.csv` | Live forecast per polyhouse and week, in kg/m² and kg, with ranges |
| `results/model_metrics.csv` | MAE, RMSE, MAPE, WAPE and R² for every model and distance |

## Run it

```bash
pip install -r requirements.txt
python make_sample_data.py --out data/weekly_records.csv
python yield_forecast.py --data data/weekly_records.csv --out results
python embed_results.py results/forecast_output.json ../crop-yield-forecast.html
```

The forecast takes about 1 to 2 minutes on a laptop. Options:

- `--horizon 8`: weeks ahead (default 8)
- `--test-weeks 52`: weeks held out for the backtest (default 52)
- `--tune`: grid search for gradient boosting at each distance, with time-ordered folds (slower)
- `--weather file.csv`: use a weather forecast instead of seasonal normals (see below)

### Your own data

Edit `COLUMNS` and `OPTIONAL` at the top of `yield_forecast.py` to match your column names. Only these are required:

- week start date
- polyhouse
- crop
- variety
- harvest in kg
- growing area in m²

Missing optional columns are skipped. If there is no cycle id, a gap of more than 3 weeks or a change of crop starts a new cycle.

### Weather forecast file

A CSV with:

- `polyhouse`: a polyhouse id, or `ALL` for every polyhouse
- `week_start`
- any of `day_temp_c`, `night_temp_c`, `rh_pct`, `solar_radiation_mj_m2`, `vpd_kpa`

Where a week and polyhouse are in the file, its values replace the seasonal normals for that week.
