# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # SPC signal detection — AttainX demo
# MAGIC
# MAGIC This notebook generates fictional operational counts inside AttainX. It reads no external data.
# MAGIC It demonstrates XmR, CUSUM, and EWMA control rules; a classifier that reproduces their
# MAGIC same-window label; a separate **next-business-day count forecast**; chronological evaluation;
# MAGIC analyst review candidates; five-day ARIMA/Holt-Winters forecasts; Isolation Forest;
# MAGIC subgroup analysis; fixture lineage and investigation; historical assessment; lifecycle exercises;
# MAGIC a daily replay dashboard; and optional MLflow and Delta integration.
# MAGIC
# MAGIC **Run all 20 code cells.** Install `requirements-demo-lock.txt` in the notebook environment first.
# MAGIC The lineage, operational events and reviewer actions are fictional fixtures. No notifications are sent.
# MAGIC
# MAGIC **Interpretation:** A signal means a statistical rule fired, not that a real problem was confirmed.
# MAGIC The classifier reproduces a rule-generated label from same-window measurements; it does not predict
# MAGIC a future incident. The forecast is genuinely forward-looking, but its scores describe only this
# MAGIC fictional dataset. Neither model decides whether a process problem occurred.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Set the rules for this run
# MAGIC **Plain English:** Choose the random seed, how much history each rule reads, and whether to save results.
# MAGIC **Technique:** Python libraries handle tables, numerical calculations, charts, and random forest models.
# MAGIC **Check:** `OUTPUT_SCHEMA` and `UC_MODEL_NAME` are empty by default, so Delta writes and model registration are off. MLflow still needs experiment write access when installed.
# MAGIC **Developer note:** Change the settings here, then rerun all cells so labels, splits, charts, and logged metrics agree.

# COMMAND ----------

# DBTITLE 1,Configuration and imports
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, mean_absolute_error,
)
from sklearn.ensemble import RandomForestRegressor

SEED = 42
N_DAYS = 420
WINDOW = 25
BASELINE_DAYS = 90
XMR_MR_MULTIPLIER = 2.66
CUSUM_K_FACTOR = 0.5
CUSUM_H_MULTIPLIER = 4.77
EWMA_LAMBDA = 0.10
EWMA_L = 2.703
DATASET_ID = f"attainx_demo_v2_seed{SEED}"

# Optional: set to an existing Unity Catalog catalog.schema after the first successful run.
# Example: OUTPUT_SCHEMA = "demo_catalog.spc_demo". Empty means charts/MLflow only.
OUTPUT_SCHEMA = ""
# Optional: set to a permitted three-part catalog.schema.model name to register the model.
# Leave empty until the notebook and MLflow experiment work.
UC_MODEL_NAME = ""

plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.2})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Make a safe example dataset
# MAGIC **Plain English:** Create three imaginary daily work queues. We plant a few jumps and sustained changes so there is something for the chart to find.
# MAGIC **Technique:** A seeded NumPy generator makes repeatable fictional time series. Assertions check unique dates and nonnegative counts.
# MAGIC **Output:** `daily_df` has one count per business day and fictional series. `dataset_id` identifies this demo run.
# MAGIC **Developer note:** The planted changes help explain the demo; they are no proof that real agency data behave this way.

# COMMAND ----------

# DBTITLE 1,Generate fictional operational series
rng = np.random.default_rng(SEED)
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dates = pd.date_range("2025-01-06", periods=N_DAYS, freq=BUSINESS_DAY)
series_config = {"Intake A": (110, 11), "Completions A": (94, 10), "Intake B": (75, 9)}
daily_parts = []
for series_id, (center, sigma) in series_config.items():
    t = np.arange(N_DAYS)
    weekly_pattern = 3 * np.sin(2 * np.pi * t / 5)
    shift = np.where((t >= 140) & (t < 155), 19, 0)
    shift += np.where((t >= 260) & (t < 275), -16, 0)
    shift += np.where((t >= 355) & (t < 370), 16, 0)
    spike = np.where(np.isin(t, [105, 220, 319]), 32, 0)
    values = np.maximum(0, np.round(center + weekly_pattern + shift + spike + rng.normal(0, sigma, N_DAYS)))
    daily_parts.append(pd.DataFrame({"series_id": series_id, "run_date": dates, "daily_count": values.astype(int)}))
daily_df = pd.concat(daily_parts, ignore_index=True)
daily_df["source_name"] = "fictional_operational_feed"
daily_df["dataset_id"] = DATASET_ID
assert daily_df.groupby(["series_id", "run_date"]).size().max() == 1
assert daily_df["daily_count"].notna().all() and daily_df["daily_count"].ge(0).all()
print(f"Created {len(daily_df):,} fictitious daily observations across {len(series_config)} series.")
display(daily_df.head()) if "display" in globals() else print(daily_df.head().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3: Define the three SPC checks
# MAGIC **Plain English:** XmR notices unusually large individual values; CUSUM adds small departures from a past average; EWMA smooths recent values to reveal a shift.
# MAGIC **Technique:** XmR uses the average moving range for limits. CUSUM accumulates deviations. EWMA gives newer observations more weight.
# MAGIC **Output:** Three functions return a yes/no rule signal; XmR also returns the center, limits, and moving range for the chart.
# MAGIC **Developer note:** Each function scans the **whole preceding window**. A flag can stay true across several later runs after one event.

# COMMAND ----------

# DBTITLE 1,Define the three SPC rules
def xmr_signal(window):
    """Reference notebook: estimate limits from this window and flag any point beyond them."""
    center = float(np.mean(window))
    mr_bar = float(np.mean(np.abs(np.diff(window))))
    ucl = center + XMR_MR_MULTIPLIER * mr_bar
    lcl = center - XMR_MR_MULTIPLIER * mr_bar
    return bool(np.any((window > ucl) | (window < lcl))), center, ucl, lcl, mr_bar


def cusum_signal(window, baseline_mean, baseline_sigma):
    k = CUSUM_K_FACTOR * baseline_sigma
    h = CUSUM_H_MULTIPLIER * baseline_sigma
    upper, lower = 0.0, 0.0
    fired = False
    for value in window:
        upper = max(0.0, upper + value - baseline_mean - k)
        lower = min(0.0, lower + value - baseline_mean + k)
        fired |= upper > h or lower < -h
    return fired


def ewma_signal(window, baseline_mean, baseline_sigma):
    z = baseline_mean
    fired = False
    for index, value in enumerate(window, start=1):
        z = EWMA_LAMBDA * value + (1 - EWMA_LAMBDA) * z
        band = EWMA_L * baseline_sigma * np.sqrt(
            EWMA_LAMBDA / (2 - EWMA_LAMBDA) * (1 - (1 - EWMA_LAMBDA) ** (2 * index))
        )
        fired |= abs(z - baseline_mean) > band
    return fired

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 4: Turn past counts into signals and features
# MAGIC **Plain English:** For each date, inspect the prior 25 business days. XmR derives limits from that window; CUSUM and EWMA use an earlier 90-day reference period. Save which SPC check fired.
# MAGIC **Technique:** Rolling-window feature engineering produces means, variation, recent trend, and distance from the baseline. `signal_detected` is `xmr_signal OR cusum_signal OR ewma_signal`.
# MAGIC **Output:** `signals_df` contains the rule flags and measurements used later by the models.
# MAGIC **Developer note:** The features and rule label describe the **same past window**. The classifier below imitates these rules; it is not advance warning.

# COMMAND ----------

# DBTITLE 1,Calculate the rule labels and modeling features
rows = []
for series_id, group in daily_df.groupby("series_id", sort=True):
    group = group.sort_values("run_date").reset_index(drop=True)
    values = group["daily_count"].to_numpy(dtype=float)
    # The reference notebook labels a 25-day window ending yesterday for each run_date.
    # Start after 90 baseline days to avoid an unstable early-history fallback.
    for end in range(BASELINE_DAYS + WINDOW, len(group)):
        window = values[end - WINDOW:end]
        baseline = values[end - WINDOW - BASELINE_DAYS:end - WINDOW]
        baseline_mean = float(np.mean(baseline))
        baseline_sigma = max(float(np.std(baseline, ddof=1)), 1e-6)
        xmr, center, ucl, lcl, mr_bar = xmr_signal(window)
        cusum = cusum_signal(window, baseline_mean, baseline_sigma)
        ewma = ewma_signal(window, baseline_mean, baseline_sigma)
        rows.append({
            "series_id": series_id, "run_date": group.loc[end, "run_date"],
            "window_start": group.loc[end - WINDOW, "run_date"],
            "window_end": group.loc[end - 1, "run_date"],
            "window_mean": center, "window_std": float(np.std(window, ddof=1)),
            "window_range": float(np.ptp(window)), "mr_mean": mr_bar,
            "last_value": float(window[-1]), "last_5_mean": float(np.mean(window[-5:])),
            "last_5_std": float(np.std(window[-5:], ddof=1)),
            "trend": float(np.mean(window[-5:]) - np.mean(window[:5])),
            "baseline_mean": baseline_mean, "baseline_sigma": baseline_sigma,
            "ucl": ucl, "lcl": lcl,
            "max_baseline_deviation": float(np.max(np.abs(window - baseline_mean)) / baseline_sigma),
            "max_window_deviation": float(np.max(np.abs(window - center)) / max(mr_bar, 1e-6)),
            "xmr_signal": xmr, "cusum_signal": cusum, "ewma_signal": ewma,
            "signal_detected": bool(xmr or cusum or ewma),
            "source_name": "fictional_operational_feed", "dataset_id": DATASET_ID,
        })

signals_df = pd.DataFrame(rows).sort_values(["run_date", "series_id"]).reset_index(drop=True)
assert len(signals_df) > 100 and signals_df["signal_detected"].nunique() == 2
assert (signals_df["signal_detected"] == signals_df[["xmr_signal", "cusum_signal", "ewma_signal"]].any(axis=1)).all()
print(f"{len(signals_df):,} labeled windows; rule signal rate: {signals_df['signal_detected'].mean():.1%}")
print(signals_df[["xmr_signal", "cusum_signal", "ewma_signal"]].mean().map(lambda v: f"{v:.1%}").to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5: Test whether ML can copy the rule label
# MAGIC **Plain English:** Train on earlier dates and ask a random forest if the SPC rules fired on later dates.
# MAGIC **Technique:** A chronological holdout separates the 25-day train and test windows. The earlier 90-day reference histories can overlap; all inputs still precede their run date. Compare accuracy with constant predictions, and read precision, recall, F1, and ROC AUC.
# MAGIC **Output:** `predictions_df` holds the model calls; `metrics` holds both model and simple-baseline scores.
# MAGIC **Decision:** This classification is useful only if it adds something to directly running the known rules. A strong AUC alone does not establish added mission value.

# COMMAND ----------

# DBTITLE 1,Train chronologically and compare against a simple baseline
# Each feature is known at the end of the 25-day window; no future values are inputs.
# The evaluation starts 25 business days after training ends, so the 25-day
# measurement windows cannot overlap. Earlier 90-day reference histories can overlap.
# Metrics describe rule reproduction on fictional data, not real-world performance.
feature_cols = ["window_mean", "window_std", "window_range", "mr_mean",
                "last_value", "last_5_mean", "last_5_std", "trend", "baseline_mean", "baseline_sigma",
                "max_baseline_deviation", "max_window_deviation"]
unique_dates = sorted(signals_df["run_date"].unique())
cut_index = int(len(unique_dates) * 0.67)
test_index = cut_index + WINDOW
assert test_index < len(unique_dates) - 20
train_df = signals_df[signals_df["run_date"] <= unique_dates[cut_index]].copy()
test_df = signals_df[signals_df["run_date"] >= unique_dates[test_index]].copy()
assert train_df["window_end"].max() < test_df["window_start"].min()
assert train_df["signal_detected"].nunique() == 2 and test_df["signal_detected"].nunique() == 2

X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_train = train_df["signal_detected"].astype(int)
y_test = test_df["signal_detected"].astype(int)
model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3,
                               class_weight="balanced", random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
majority_class = int(y_train.mean() >= 0.5)
baseline_pred = np.full(len(y_test), majority_class)
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, prob),
    "majority_baseline_accuracy": accuracy_score(y_test, baseline_pred),
    "always_signal_accuracy": float(y_test.mean()),
    "always_no_signal_accuracy": float(1 - y_test.mean()),
    "test_signal_rate": float(y_test.mean()),
}
print(f"Training: {len(train_df)} rows through {train_df.run_date.max().date()}")
print(f"Test: {len(test_df)} rows from {test_df.run_date.min().date()} to {test_df.run_date.max().date()}")
print(f"Training signal rate: {y_train.mean():.1%}; test signal rate: {y_test.mean():.1%}")
print(pd.Series(metrics).round(3).to_string())
if metrics["accuracy"] <= max(metrics["majority_baseline_accuracy"], metrics["always_signal_accuracy"]):
    print("The classifier did not beat both simple constant baselines on this test. Use the rules directly.")
else:
    print("The classifier beat constant baselines on this test; this measures rule reproduction, not future warning.")

predictions_df = test_df[["series_id", "run_date", "signal_detected", "xmr_signal", "cusum_signal", "ewma_signal"]].copy()
predictions_df["model_signal"] = pred.astype(bool)
predictions_df["model_probability"] = prob
predictions_df["dataset_id"] = DATASET_ID

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 6: Predict one future business-day count
# MAGIC **Plain English:** Use information available yesterday to estimate today's workload before today's count arrives.
# MAGIC **Technique:** A random forest regressor learns from past counts, recent trend, weekday, and series. A later-date test with a gap compares mean absolute error (MAE) to the simple five-day moving average.
# MAGIC **Output:** `forecast_results_df` holds actual and predicted counts; `forecast_metrics` reports errors in **daily-count units**.
# MAGIC **Developer note:** This is a **one-business-day forecast**. It does not predict a confirmed incident, and the test score comes only from fabricated data.

# COMMAND ----------

# DBTITLE 1,Forecast the next business day's count (separate from rule classification)
# Each run_date's features use only the 25 business days ending the previous day.
# The target is the count observed on run_date, which was unavailable when predicting.
# The last value and trailing mean are transparent baselines; no rule label is used as an input.
forecast_df = signals_df.merge(
    daily_df[["series_id", "run_date", "daily_count"]],
    on=["series_id", "run_date"], validate="one_to_one",
)
forecast_df["weekday"] = forecast_df["run_date"].dt.weekday
forecast_features = ["series_id", "weekday", "last_value", "last_5_mean",
                     "window_mean", "window_std", "trend", "baseline_mean"]
forecast_X = pd.get_dummies(forecast_df[forecast_features], columns=["series_id"], dtype=float)
forecast_train = forecast_df["run_date"].le(unique_dates[cut_index])
forecast_test = forecast_df["run_date"].ge(unique_dates[test_index + 1])
assert forecast_df.loc[forecast_train, "run_date"].max() < forecast_df.loc[forecast_test, "window_start"].min()
forecast_model = RandomForestRegressor(
    n_estimators=120, max_depth=7, min_samples_leaf=4, random_state=SEED, n_jobs=-1
)
forecast_model.fit(forecast_X.loc[forecast_train], forecast_df.loc[forecast_train, "daily_count"])
forecast_values = np.maximum(0, forecast_model.predict(forecast_X.loc[forecast_test]))
forecast_actual = forecast_df.loc[forecast_test, "daily_count"].to_numpy()
forecast_metrics = {
    "forecast_mae": float(mean_absolute_error(forecast_actual, forecast_values)),
    "last_value_baseline_mae": float(mean_absolute_error(
        forecast_actual, forecast_df.loc[forecast_test, "last_value"])),
    "trailing_mean_baseline_mae": float(mean_absolute_error(
        forecast_actual, forecast_df.loc[forecast_test, "last_5_mean"])),
}
forecast_results_df = forecast_df.loc[forecast_test, ["series_id", "run_date", "daily_count", "dataset_id"]].copy()
forecast_results_df["predicted_count"] = forecast_values
forecast_results_df["trailing_mean_baseline"] = forecast_df.loc[forecast_test, "last_5_mean"].to_numpy()
forecast_results_df["absolute_error"] = np.abs(forecast_actual - forecast_values)
print("Next-business-day count forecast: held-out later dates, with a 25-day gap")
print(pd.Series(forecast_metrics).round(2).to_string())
if forecast_metrics["forecast_mae"] >= forecast_metrics["trailing_mean_baseline_mae"]:
    print("The forecast did not beat the trailing-mean baseline here; show the comparison honestly.")
else:
    print("The forecast beat the trailing-mean baseline on this demo test only.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 7: Draw the evidence
# MAGIC **Plain English:** The first chart shows recent counts and XmR limits. The second shows which rule labels the classifier matched or missed. The third compares the forecast with later observed counts.
# MAGIC **Technique:** Matplotlib draws the time series; a confusion matrix displays true and false classifier calls on the held-out dates.
# MAGIC **Read carefully:** A circle means **any** rule fired somewhere in the past 25-day window. The point under the circle does not have to cross the displayed XmR limit.
# MAGIC **Developer note:** These charts support explanation; review actual rows and rule flags before interpreting a cause.

# COMMAND ----------

# DBTITLE 1,Show the control chart and classification results
chosen = "Intake A"
chart_df = signals_df[signals_df.series_id == chosen].tail(90)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(chart_df.run_date, chart_df.last_value, color="#145A7D", label="Window's last observed count")
ax.plot(chart_df.run_date, chart_df.ucl, color="#B8603C", linestyle="--", label="XmR upper limit")
ax.plot(chart_df.run_date, chart_df.lcl, color="#B8603C", linestyle=":", label="XmR lower limit")
marked = chart_df[chart_df.signal_detected]
ax.scatter(marked.run_date, marked.last_value, facecolor="white", edgecolor="#853B32",
           s=30, linewidth=1.2, label="Rule signal for 25-day window")
ax.set(title="Fictional SPC history — Intake A (last 90 runs)", xlabel="Run date", ylabel="Daily count")
ax.legend(loc="upper left", fontsize=8)
fig.text(0.08, 0.01, "Circles mean any rule fired in the prior 25 days; the plotted point need not cross an XmR limit.",
         fontsize=8, color="#444444")
fig.autofmt_xdate()
fig.subplots_adjust(bottom=0.36)
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No signal", "Signal"],
                                        cmap="Blues", colorbar=False, ax=ax)
ax.set_title(f"Demo test: {len(y_test)} labeled windows")
plt.tight_layout()
plt.show()

forecast_chart = forecast_results_df[forecast_results_df.series_id == chosen].tail(55)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(forecast_chart.run_date, forecast_chart.daily_count, label="Observed", color="#145A7D")
ax.plot(forecast_chart.run_date, forecast_chart.predicted_count, label="Predicted before observation", color="#B8603C")
ax.set(title="Next-business-day forecast — held-out demo dates", xlabel="Date", ylabel="Daily count")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 8: Give people a manageable review list
# MAGIC **Plain English:** Consecutive days of rule flags for one series become one review episode instead of a new alert every day.
# MAGIC **Technique:** Group-by and a cumulative count identify each contiguous run of signals. Basic data and model health indicators go into `monitoring_df`.
# MAGIC **Output:** `review_df` shows the 15 episodes with the latest signal dates, their rules, and a pending disposition. The printed total counts **all** episodes.
# MAGIC **Developer note:** This is a proposed review queue. It does not send notifications, record a completed analyst decision, or automate an operational action.

# COMMAND ----------

# DBTITLE 1,Consolidate persistent signals into analyst review episodes
# A 25-day rule window can stay flagged across many runs. One episode is one
# consecutive span of flagged windows within a series, not one email per day.
ordered_signals = signals_df.sort_values(["series_id", "run_date"]).copy()
previous_flag = ordered_signals.groupby("series_id")["signal_detected"].shift(fill_value=False)
ordered_signals["episode_id"] = (ordered_signals["signal_detected"] & ~previous_flag).groupby(
    ordered_signals["series_id"]
).cumsum()
episode_rows = ordered_signals[ordered_signals["signal_detected"]].copy()
review_df = episode_rows.groupby(["series_id", "episode_id"], as_index=False).agg(
    run_date=("run_date", "min"), last_signal_date=("run_date", "max"),
    signal_windows=("run_date", "size"), window_end=("window_end", "max"),
    last_value=("last_value", "last"), xmr_signal=("xmr_signal", "max"),
    cusum_signal=("cusum_signal", "max"), ewma_signal=("ewma_signal", "max"),
    source_name=("source_name", "first"), dataset_id=("dataset_id", "first"),
)
review_df["rule_fired"] = review_df.apply(
    lambda row: ", ".join(name for name in ("xmr", "cusum", "ewma") if row[f"{name}_signal"]), axis=1
)
review_df["disposition"] = "Pending analyst review"
review_df = review_df[["series_id", "episode_id", "run_date", "last_signal_date", "signal_windows",
                       "window_end", "rule_fired", "last_value", "disposition", "source_name", "dataset_id"]]
total_review_episodes = len(review_df)
review_df = review_df.sort_values(["last_signal_date", "series_id"]).tail(15).reset_index(drop=True)
monitoring_df = pd.DataFrame([{
    "dataset_id": DATASET_ID,
    "source_rows": len(daily_df),
    "missing_counts": int(daily_df["daily_count"].isna().sum()),
    "rule_signal_rate": float(signals_df["signal_detected"].mean()),
    "raw_signal_windows": int(signals_df["signal_detected"].sum()),
    "review_episodes": total_review_episodes,
    "test_model_signal_rate": float(predictions_df["model_signal"].mean()),
    "test_rule_disagreement_rate": float((predictions_df["model_signal"] != predictions_df["signal_detected"]).mean()),
    "classifier_accuracy": float(metrics["accuracy"]),
    "majority_baseline_accuracy": float(metrics["majority_baseline_accuracy"]),
    "always_signal_accuracy": float(metrics["always_signal_accuracy"]),
    "forecast_mae": forecast_metrics["forecast_mae"],
    "trailing_mean_baseline_mae": forecast_metrics["trailing_mean_baseline_mae"],
}])
print("Consolidated review episodes (15 latest signal dates; no automatic action):")
display(review_df) if "display" in globals() else print(review_df.tail(5).to_string(index=False))
print("Monitoring snapshot:")
display(monitoring_df) if "display" in globals() else print(monitoring_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 9: Add severity and inspect the data
# MAGIC **Plain English:** A yes/no flag tells us to look. Zone checks add an indication of how unusual the latest pattern is.
# MAGIC **Method:** Moving-range sigma, a latest-point three-sigma check, 2-of-3 beyond two sigma, 4-of-5 beyond one sigma, and eight on one side. These are demo choices, not calibrated agency thresholds.
# MAGIC **Show:** Severity counts, per-series distributions and correlations. A correlation does not establish cause.

# COMMAND ----------
# DBTITLE 1,Zone severity and exploratory analysis
def zone_check(values):
    center = float(np.mean(values))
    sigma = max(float(np.mean(np.abs(np.diff(values)))) / 1.128, 1e-6)
    z = (np.asarray(values) - center) / sigma
    if abs(z[-1]) > 3:
        return "CRITICAL", "latest point beyond 3 sigma"
    if max((z[-3:] > 2).sum(), (z[-3:] < -2).sum()) >= 2:
        return "WARNING", "2 of 3 beyond 2 sigma on one side"
    if np.all(z[-8:] > 0) or np.all(z[-8:] < 0):
        return "WARNING", "8 points on one side"
    if max((z[-5:] > 1).sum(), (z[-5:] < -1).sum()) >= 4:
        return "WATCH", "4 of 5 beyond 1 sigma on one side"
    if abs(z[-1]) > 1:
        return "WATCH", "latest point beyond 1 sigma"
    return "NORMAL", "no zone pattern"

zone_rows = []
for sid, group in daily_df.groupby("series_id"):
    group = group.sort_values("run_date").reset_index(drop=True)
    for end in range(BASELINE_DAYS + WINDOW, len(group)):
        severity, reason = zone_check(group.daily_count.iloc[end-WINDOW:end].to_numpy())
        zone_rows.append({"series_id": sid, "run_date": group.run_date.iloc[end],
                          "zone_severity": severity, "zone_reason": reason})
zone_df = pd.DataFrame(zone_rows)
series_profile_df = daily_df.groupby("series_id").daily_count.agg(["count", "mean", "std", "min", "max"]).reset_index()
print(series_profile_df.round(2).to_string(index=False))
print(zone_df.zone_severity.value_counts().to_string())
pivot_counts = daily_df.pivot(index="run_date", columns="series_id", values="daily_count")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for sid in pivot_counts:
    axes[0].hist(pivot_counts[sid], bins=25, alpha=.45, label=sid)
axes[0].set(title="Fictional daily-count distributions", xlabel="Daily count", ylabel="Days")
axes[0].legend(fontsize=8)
corr = pivot_counts.corr()
im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
axes[1].set_xticks(range(len(corr)), corr.columns, rotation=20)
axes[1].set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
axes[1].set_title("Association between series; not causality")
fig.colorbar(im, ax=axes[1], fraction=.04)
plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 10: Find which subgroup changed
# MAGIC **Plain English:** Split each fictional daily count between two offices, then investigate shifts inside each office.
# MAGIC **Method:** Counts reconcile to the parent series. Past-only rolling z-scores and workload shares support subgroup investigation; cardinality, placeholder and duplicate-dimension checks keep categories useful.
# MAGIC **Boundary:** These are invented office and channel labels, with no agency records or claims about people.

# COMMAND ----------
# DBTITLE 1,Subgroup profiles and anomaly evidence
subgroup_rows = []
for sid, group in daily_df.groupby("series_id"):
    for idx, row in enumerate(group.sort_values("run_date").itertuples()):
        share = .58 + (.18 if 355 <= idx < 370 else 0)
        north = int(round(row.daily_count * share))
        for office, count in [("North", north), ("South", row.daily_count-north)]:
            subgroup_rows.append({"series_id": sid, "run_date": row.run_date, "office": office,
                                  "channel": "Online" if office == "North" else "Assisted",
                                  "daily_count": count, "dataset_id": DATASET_ID})
subgroup_df = pd.DataFrame(subgroup_rows).sort_values(["series_id", "office", "run_date"])
subgroup_df["prior_mean"] = subgroup_df.groupby(["series_id", "office"]).daily_count.transform(lambda x: x.shift(1).rolling(25).mean())
subgroup_df["prior_std"] = subgroup_df.groupby(["series_id", "office"]).daily_count.transform(lambda x: x.shift(1).rolling(25).std()).clip(lower=1)
subgroup_df["zscore"] = (subgroup_df.daily_count-subgroup_df.prior_mean)/subgroup_df.prior_std
subgroup_df["subgroup_anomaly"] = subgroup_df.zscore.abs().gt(3)
subgroup_df["workload_share"] = subgroup_df.daily_count / subgroup_df.groupby(["series_id", "run_date"]).daily_count.transform("sum").clip(lower=1)
reconciled = subgroup_df.groupby(["series_id", "run_date"]).daily_count.sum().sort_index()
assert reconciled.equals(daily_df.set_index(["series_id", "run_date"]).daily_count.sort_index())
dimension_profile_df = pd.DataFrame([{"dimension": col, "distinct_values": subgroup_df[col].nunique(),
                                    "missing": int(subgroup_df[col].isna().sum()),
                                    "placeholder_rows": int(subgroup_df[col].isin(["", "UNKNOWN", "N/A"]).sum())}
                                   for col in ["office", "channel"]])
# Office and channel map one-to-one in this deliberately simple fixture; do not count them as independent evidence.
dimension_profile_df["use_for_modeling"] = [True, False]
print(dimension_profile_df.to_string(index=False))
print("Channel is redundant with office in this fixture; retain office for analysis.")
print(subgroup_df[subgroup_df.subgroup_anomaly].tail(8).to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 4))
for office, group in subgroup_df[subgroup_df.series_id == "Intake A"].groupby("office"):
    ax.plot(group.run_date.tail(90), group.workload_share.tail(90), label=office)
ax.set(title="Fictional workload mix — Intake A", ylabel="Share of parent daily count", xlabel="Date")
ax.legend(); fig.autofmt_xdate(); plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 11: Compare classifiers across five later periods
# MAGIC **Plain English:** Compare random forest and logistic regression over several time periods, not just one favorable split.
# MAGIC **Method:** Five expanding training windows, shared date boundaries across series, and a 25-business-day separation of measurement windows. Scaling fits only on training rows. Compare each fold with a training-majority baseline.
# MAGIC **Decision:** Both models still imitate known rules. Fold scores and feature importance describe this experiment, not mission value or causality.

# COMMAND ----------
# DBTITLE 1,Walk-forward classification and feature importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

fold_specs = []
fold_metrics = []
fold_predictions = []
for fold, start in enumerate(np.linspace(155, len(unique_dates)-30, 5, dtype=int), 1):
    train_dates_end = start-WINDOW-1
    fold_train = signals_df.run_date.le(unique_dates[train_dates_end])
    fold_test = signals_df.run_date.between(unique_dates[start], unique_dates[start+24])
    tr, te = signals_df.loc[fold_train], signals_df.loc[fold_test]
    assert tr.window_end.max() < te.window_start.min()
    fold_specs.append((fold, fold_train, fold_test))
    candidates = {
        "Random forest": RandomForestClassifier(n_estimators=60, max_depth=7, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    }
    for name, candidate in candidates.items():
        candidate.fit(tr[feature_cols], tr.signal_detected.astype(int))
        calls = candidate.predict(te[feature_cols])
        scores = candidate.predict_proba(te[feature_cols])[:,1]
        majority = int(tr.signal_detected.mean() >= .5)
        fold_metrics.append({"fold": fold, "model": name, "train_end": tr.run_date.max(),
                             "test_start": te.run_date.min(), "test_end": te.run_date.max(), "rows": len(te),
                             "accuracy": accuracy_score(te.signal_detected, calls),
                             "f1": f1_score(te.signal_detected, calls, zero_division=0),
                             "auc": roc_auc_score(te.signal_detected, scores) if te.signal_detected.nunique()==2 else np.nan,
                             "majority_accuracy": accuracy_score(te.signal_detected, np.full(len(te), majority))})
        fold_predictions.extend([{"fold":fold,"model":name,"truth":bool(y),"prediction":bool(p)} for y,p in zip(te.signal_detected,calls)])
        if name == "Logistic regression": logistic_model = candidate
fold_metrics_df = pd.DataFrame(fold_metrics)
print(fold_metrics_df.round(3).to_string(index=False))
importance = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=SEED, scoring="accuracy")
importance_df = pd.DataFrame({"feature":feature_cols,"accuracy_drop":importance.importances_mean}).sort_values("accuracy_drop", ascending=False)
fig, axes = plt.subplots(1,2,figsize=(12,4))
for name,g in fold_metrics_df.groupby("model"):
    axes[0].plot(g.fold,g.accuracy,marker="o",label=name)
base=fold_metrics_df[fold_metrics_df.model=="Random forest"]
axes[0].plot(base.fold,base.majority_accuracy,linestyle="--",label="Training-majority baseline")
axes[0].set(title="Five later-period tests",xlabel="Fold",ylabel="Accuracy",ylim=(0,1)); axes[0].legend(fontsize=8)
importance_df.head(6).iloc[::-1].plot.barh(x="feature",y="accuracy_drop",ax=axes[1],legend=False)
axes[1].set(title="Permutation importance on held-out rows",xlabel="Accuracy drop when shuffled",ylabel="")
plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 12: Forecast five business days and test the alternatives
# MAGIC **Plain English:** Compare five-day ARIMA and Holt-Winters forecasts with the simple guess that the next five values repeat the previous five. Then generate actual future-date forecasts, including offices.
# MAGIC **Method:** Six nonoverlapping forecast origins, five steps each, fit only on history at each origin. The last origin is held aside for lifecycle comparison; earlier origins choose the candidate. US federal holidays are excluded from the calendar.
# MAGIC **Boundary:** Five business observations are a compact seasonal approximation; holiday weeks can differ. Show errors and keep the baseline if it wins.

# COMMAND ----------
# DBTITLE 1,Five-day ARIMA Holt-Winters and subgroup forecasts
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def five_day_predictions(values):
    values = np.asarray(values, dtype=float)
    predictions = {"Seasonal naive": np.tile(values[-5:], 1)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = ARIMA(values[-180:], order=(1,0,0), trend="c").fit()
        hw = ExponentialSmoothing(values[-180:], trend="add", damped_trend=True, seasonal="add", seasonal_periods=5,
                                  initialization_method="estimated").fit(optimized=True)
    predictions["ARIMA"] = np.maximum(0, np.asarray(arima.forecast(5)))
    predictions["Holt-Winters"] = np.maximum(0, np.asarray(hw.forecast(5)))
    return predictions

history_rows, future_rows = [], []
for sid, group in daily_df.groupby("series_id"):
    group = group.sort_values("run_date").reset_index(drop=True)
    values = group.daily_count.to_numpy(dtype=float)
    for end in range(len(group)-60, len(group), 10):
        origin = group.run_date.iloc[end-1]
        for method, vals in five_day_predictions(values[:end]).items():
            for h, value in enumerate(vals,1):
                history_rows.append({"dataset_id":DATASET_ID,"series_id":sid,"model":method,"origin_date":origin,
                                     "target_date":group.run_date.iloc[end+h-1],"horizon":h,
                                     "prediction":float(value),"actual":float(values[end+h-1]),
                                     "origin_count":float(values[end-1])})
    future_dates=pd.date_range(group.run_date.iloc[-1]+BUSINESS_DAY,periods=5,freq=BUSINESS_DAY)
    for method,vals in five_day_predictions(values).items():
        future_rows.extend([{"series_id":sid,"office":"All","model":method,"origin_date":group.run_date.iloc[-1],
                             "target_date":d,"horizon":h,"prediction":float(v),"dataset_id":DATASET_ID}
                            for h,(d,v) in enumerate(zip(future_dates,vals),1)])
forecast_history_df=pd.DataFrame(history_rows)
forecast_history_df["absolute_error"]=(forecast_history_df.actual-forecast_history_df.prediction).abs()
selection_cutoff=forecast_history_df.origin_date.max()
selection_scores=forecast_history_df[forecast_history_df.origin_date<selection_cutoff].groupby("model").absolute_error.mean()
selected_forecaster=selection_scores.drop("Seasonal naive").idxmin()
five_day_metrics_df=forecast_history_df.groupby("model").absolute_error.agg(["mean","count"]).rename(columns={"mean":"mae","count":"forecast_rows"}).reset_index()
for (sid,office),group in subgroup_df.groupby(["series_id","office"]):
    group=group.sort_values("run_date")
    vals=five_day_predictions(group.daily_count)[selected_forecaster]
    future_dates=pd.date_range(group.run_date.iloc[-1]+BUSINESS_DAY,periods=5,freq=BUSINESS_DAY)
    future_rows.extend([{"series_id":sid,"office":office,"model":selected_forecaster,"origin_date":group.run_date.iloc[-1],
                         "target_date":d,"horizon":h,"prediction":float(v),"dataset_id":DATASET_ID}
                        for h,(d,v) in enumerate(zip(future_dates,vals),1)])
future_forecasts_df=pd.DataFrame(future_rows)
print(five_day_metrics_df.round(3).to_string(index=False))
print("Candidate selected from first five origins:",selected_forecaster)
print("Subgroup forecasts fit independently and need not sum to the parent forecast.")
print(future_forecasts_df.tail(10).round(2).to_string(index=False))
fig,ax=plt.subplots(figsize=(11,4))
g=daily_df[daily_df.series_id=="Intake A"].tail(25)
ax.plot(g.run_date,g.daily_count,label="Observed fictional counts",color="#145A7D")
for method,group in future_forecasts_df[(future_forecasts_df.series_id=="Intake A")&(future_forecasts_df.office=="All")].groupby("model"):
    ax.plot(group.target_date,group.prediction,marker="o",label=method)
ax.set(title="Five business days beyond the latest observation",xlabel="Date",ylabel="Daily count")
ax.legend(fontsize=8);fig.autofmt_xdate();plt.tight_layout();plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 13: Look for unusual patterns without rule labels
# MAGIC **Plain English:** Isolation Forest finds unusual combinations of measurements. It can disagree with SPC, which gives the analyst another question to investigate.
# MAGIC **Method:** Five chronological folds, a contamination setting estimated only from training data and capped at 10%, then final fitting on the earlier training set. SPC agreement is a proxy check, not incident-detection accuracy.

# COMMAND ----------
# DBTITLE 1,Isolation Forest and disagreement analysis
from sklearn.ensemble import IsolationForest
iso_rows=[]
for fold,trmask,temask in fold_specs:
    tr,te=signals_df.loc[trmask],signals_df.loc[temask]
    contamination=float(np.clip(tr.signal_detected.mean()*.10,.02,.10))
    detector=IsolationForest(n_estimators=80,contamination=contamination,random_state=SEED,n_jobs=-1).fit(tr[feature_cols])
    flags=detector.predict(te[feature_cols])==-1
    iso_rows.append({"fold":fold,"contamination":contamination,"rows":len(te),"anomaly_rate":float(flags.mean()),
                     "rule_agreement":float((flags==te.signal_detected.to_numpy()).mean()),
                     "rule_proxy_f1":f1_score(te.signal_detected,flags,zero_division=0)})
iso_fold_df=pd.DataFrame(iso_rows)
iso_model=IsolationForest(n_estimators=100,contamination=.08,random_state=SEED,n_jobs=-1).fit(X_train)
anomaly_df=test_df[["series_id","run_date","signal_detected","dataset_id"]].copy()
anomaly_df["anomaly_score"]=-iso_model.score_samples(X_test)
anomaly_df["model_anomaly"]=iso_model.predict(X_test)==-1
anomaly_df["disagrees_with_spc"]=anomaly_df.model_anomaly != anomaly_df.signal_detected
print(iso_fold_df.round(3).to_string(index=False))
print(anomaly_df.groupby(["signal_detected","model_anomaly"]).size().rename("windows").to_string())
fig,ax=plt.subplots(figsize=(11,3.5))
g=anomaly_df[anomaly_df.series_id=="Intake A"]
ax.plot(g.run_date,g.anomaly_score,label="Isolation Forest unusualness score")
flagged=g[g.model_anomaly];ax.scatter(flagged.run_date,flagged.anomaly_score,color="#B8603C",label="Model flags")
ax.set(title="Anomaly scores on later fictional observations",xlabel="Run date",ylabel="Higher = more unusual")
ax.legend(fontsize=8);fig.autofmt_xdate();plt.tight_layout();plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 14: Trace fictional inputs upstream
# MAGIC **Plain English:** Follow a count back through the tables that produced it and inspect whether an upstream refresh was late.
# MAGIC **Method:** Breadth-first graph traversal up to five hops, column mappings and timestamp-based freshness checks over fixture metadata.
# MAGIC **Boundary:** The graph is a fictional fixture, not discovered Unity Catalog lineage. The traversal and checks execute locally.

# COMMAND ----------
# DBTITLE 1,Fixture lineage and upstream freshness
lineage_edges_df=pd.DataFrame([
    ("demo.daily_counts","demo.normalized_actions"),("demo.normalized_actions","demo.raw_actions"),
    ("demo.normalized_actions","demo.office_lookup"),("demo.raw_actions","demo.batch_landing"),
    ("demo.batch_landing","demo.transfer_manifest"),("demo.transfer_manifest","demo.source_export"),
],columns=["target","source"])
column_lineage_df=pd.DataFrame([
    ("demo.daily_counts.daily_count","demo.normalized_actions.action_id"),
    ("demo.normalized_actions.office","demo.office_lookup.office_code"),
],columns=["target_column","source_column"])
def trace_upstream(root, edges, max_hops=5):
    pending=[(root,0,root)];seen={root};result=[]
    while pending:
        target,hop,path=pending.pop(0)
        if hop>=max_hops:continue
        for source in edges.loc[edges.target==target,"source"]:
            result.append({"table":source,"hop":hop+1,"path":path+" -> "+source})
            if source not in seen:
                seen.add(source);pending.append((source,hop+1,path+" -> "+source))
    return pd.DataFrame(result)
lineage_df=trace_upstream("demo.daily_counts",lineage_edges_df)
# Use a historical replay date, not wall-clock time, so this fixture is reproducible.
replay_date=pd.Timestamp(daily_df.run_date.max())+pd.Timedelta(hours=9)
upstream_profile_df=lineage_df[["table","hop"]].copy()
upstream_profile_df["last_refresh"]=[replay_date-pd.Timedelta(hours=30 if "raw_actions" in t else 2) for t in upstream_profile_df.table]
upstream_profile_df["age_hours"]=(replay_date-upstream_profile_df.last_refresh).dt.total_seconds()/3600
upstream_profile_df["late_refresh"]=upstream_profile_df.age_hours>24
upstream_profile_df["inferred_role"]=upstream_profile_df.table.map(lambda x:"Reference data" if "lookup" in x else "Data pipeline")
print(lineage_df.to_string(index=False));print(column_lineage_df.to_string(index=False))
print(upstream_profile_df.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 15: Build an investigation hypothesis
# MAGIC **Plain English:** Put the triggered rules, office changes, nearby events and upstream evidence beside each review episode.
# MAGIC **Method:** Evidence joins produce reasons, with temporal checks before linking a late refresh. An event near a signal is an association, not a proven cause.
# MAGIC **Show:** An invented capacity change and a delayed input batch. Evidence labels keep fixture facts distinct from analyst conclusions.

# COMMAND ----------
# DBTITLE 1,Root-cause hypotheses and operational events
change_date=daily_df[daily_df.series_id=="Intake A"].sort_values("run_date").run_date.iloc[355]
events_df=pd.DataFrame([
    {"event_id":"fixture-capacity","event_date":change_date,"series_id":"Intake A","event_type":"Capacity change","description":"Fictional office workload reallocation"},
    {"event_id":"fixture-delay","event_date":replay_date.normalize(),"series_id":"Intake A","event_type":"Input delay","description":"Fictional raw-input batch arrived late"},
])
hypotheses=[]
for row in review_df.itertuples():
    nearby=events_df[(events_df.series_id==row.series_id)&events_df.event_date.between(row.run_date-pd.Timedelta(days=7),row.last_signal_date+pd.Timedelta(days=7))]
    sub=subgroup_df[(subgroup_df.series_id==row.series_id)&subgroup_df.run_date.between(row.run_date,row.last_signal_date)]
    top=sub.loc[sub.zscore.abs().idxmax()] if sub.zscore.notna().any() else None
    late=bool(upstream_profile_df.late_refresh.any() and row.run_date <= replay_date.normalize() <= row.last_signal_date+pd.Timedelta(days=1))
    reason="Rule signal requires investigation"
    if len(nearby):reason="Nearby fixture event; verify timing and mechanism"
    elif top is not None and abs(top.zscore)>3:reason="Localized office deviation; inspect workload mix"
    if late:reason+="; late fixture refresh also needs inspection"
    hypotheses.append({"series_id":row.series_id,"episode_id":row.episode_id,"rule_fired":row.rule_fired,
                       "event_ids":", ".join(nearby.event_id),"top_office":str(top.office) if top is not None else "",
                       "largest_abs_z":float(abs(top.zscore)) if top is not None else 0.,"late_refresh_evidence":late,
                       "hypothesis":reason,"confidence":"Unverified hypothesis","dataset_id":DATASET_ID})
hypotheses_df=pd.DataFrame(hypotheses)
print(events_df.to_string(index=False));print(hypotheses_df.head(6).to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 16: Assess forecasts after actuals arrive
# MAGIC **Plain English:** Keep past forecasts, match them to later observed counts, and decide whether enough evidence exists to reconsider a model.
# MAGIC **Method:** Idempotent upsert keys preserve distinct forecast origins. Assess MAE, RMSE, sMAPE, bias and direction against the origin count. Earlier runs are superseded for current action; fewer than three actuals cannot trigger retraining.
# MAGIC **Boundary:** This is a replay of several historical forecast runs. It demonstrates the feedback calculation, not a live retraining schedule.

# COMMAND ----------
# DBTITLE 1,Historical forecast assessment and safeguards
HISTORY_KEYS=["dataset_id","series_id","model","origin_date","target_date"]
def upsert_history(existing,incoming,keys):
    return pd.concat([existing,incoming],ignore_index=True).drop_duplicates(keys,keep="last").sort_values(keys).reset_index(drop=True)
forecast_ledger_df=upsert_history(pd.DataFrame(),forecast_history_df,HISTORY_KEYS)
assert len(upsert_history(forecast_ledger_df,forecast_history_df,HISTORY_KEYS))==len(forecast_ledger_df)
def assessment_metrics(group):
    err=group.prediction-group.actual
    denominator=(group.prediction.abs()+group.actual.abs())/2
    smape=np.where(denominator>1e-9,err.abs()/denominator.clip(lower=1e-9),0).mean()*100
    return {"n_actuals":len(group),"mae":float(err.abs().mean()),"rmse":float(np.sqrt((err**2).mean())),
            "smape":float(smape),"bias":float(err.mean()),
            "direction_accuracy":float((np.sign(group.prediction-group.origin_count)==np.sign(group.actual-group.origin_count)).mean()) if len(group)>=2 else np.nan}
assessment_rows=[]
for keys,g in forecast_ledger_df.groupby(["series_id","model","origin_date"]):
    assessment_rows.append(dict(zip(["series_id","model","origin_date"],keys))|assessment_metrics(g))
assessment_df=pd.DataFrame(assessment_rows).sort_values(["series_id","model","origin_date"])
assessment_df["previous_smape"]=assessment_df.groupby(["series_id","model"]).smape.shift()
assessment_df["latest_origin"]=assessment_df.groupby(["series_id","model"]).origin_date.transform("max")
def retraining_advice(n, current, previous, latest=True):
    if not latest:return "superseded"
    if n<3:return "insufficient_data"
    if pd.notna(previous) and current>40 and current>previous:return "architecture_review"
    if pd.notna(previous) and current>previous*1.10:return "review_for_retraining"
    if pd.notna(previous) and current<previous*.95:return "improving"
    return "monitor"
assessment_df["recommendation"]=assessment_df.apply(lambda r:retraining_advice(r.n_actuals,r.smape,r.previous_smape,r.origin_date==r.latest_origin),axis=1)
assessment_df["confidence"]=np.where(assessment_df.n_actuals>=10,"higher sample count","limited sample")
# A deliberately partial arrival exercises the insufficient-data safeguard.
partial_assessment=assessment_metrics(forecast_ledger_df.head(1))
partial_assessment["recommendation"]=retraining_advice(1,partial_assessment["smape"],np.nan)
print(assessment_df[assessment_df.origin_date==assessment_df.latest_origin].round(3).to_string(index=False))
print("One-observation safeguard:",partial_assessment["recommendation"])
fig,ax=plt.subplots(figsize=(11,3.5))
for method,g in assessment_df[assessment_df.series_id=="Intake A"].groupby("model"):
    ax.plot(g.origin_date,g.mae,marker="o",label=method)
ax.set(title="Replay: forecast error by origin — Intake A",xlabel="Forecast origin",ylabel="MAE, daily counts")
ax.legend(fontsize=8);fig.autofmt_xdate();plt.tight_layout();plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 17: Challenge the baseline and rehearse withdrawal
# MAGIC **Plain English:** A candidate must beat the current method on a separate period before a person would even consider a pilot. A reviewer can keep the baseline or withdraw a component.
# MAGIC **Method:** Earlier forecast origins choose ARIMA versus Holt-Winters. The last origin compares that candidate with seasonal naive. A 2% improvement is a demo threshold. The small sample supports advisory use only.
# MAGIC **Boundary:** Local lifecycle records and explicit exercise approvals illustrate version selection and rollback. They do not grant production authorization or alter a real registry alias.

# COMMAND ----------
# DBTITLE 1,Champion challenger lifecycle and rollback exercise
last_origin=forecast_ledger_df[forecast_ledger_df.origin_date==selection_cutoff]
last_scores=last_origin.groupby("model").absolute_error.mean()
champion_mae=float(last_scores["Seasonal naive"])
challenger_mae=float(last_scores[selected_forecaster])
improvement=(champion_mae-challenger_mae)/max(champion_mae,1e-9)
lifecycle_df=pd.DataFrame([
    {"component":"Rule-label classifier","version":"rf-v1","recommendation":"withdraw candidate","reason":"Known rules compute the target directly"},
    {"component":"Five-day forecast","version":selected_forecaster,"recommendation":"advisory candidate" if improvement>=.02 else "keep baseline","reason":f"Untouched final-origin relative MAE improvement: {improvement:.1%}; only 15 observations"},
    {"component":"Isolation Forest","version":"if-v1","recommendation":"advisory only","reason":"Rule-label agreement is not verified incident detection"},
])
def rehearse_registry_change(active_version, candidate_version, approved, rollback=False):
    """Pure local exercise: an explicit approval controls a simulated state change."""
    if not approved:
        return active_version, "blocked: exercise approval absent"
    return candidate_version, "rollback" if rollback else "select candidate"

exercise_rows=[{"step":1,"active_version":"baseline-v1","action":"Initial fictional registry state","actor":"Exercise fixture"}]
active_version="baseline-v1"
assert rehearse_registry_change(active_version,"candidate-v1",False)[0]==active_version
for step,target,rollback in [(2,"candidate-v1",False),(3,"baseline-v1",True)]:
    active_version,action=rehearse_registry_change(active_version,target,approved=True,rollback=rollback)
    exercise_rows.append({"step":step,"active_version":active_version,"action":action,"actor":"Simulated reviewer; not a real approval"})
registry_exercise_df=pd.DataFrame(exercise_rows)
assert registry_exercise_df.iloc[-1].active_version==registry_exercise_df.iloc[0].active_version
print(lifecycle_df.to_string(index=False));print(registry_exercise_df.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 18: Replay the daily operating loop
# MAGIC **Plain English:** Replay three observation dates, prepare one review notice per episode, suppress repeats, and record an example analyst disposition separately from the pending queue.
# MAGIC **Method:** Stable episode keys make repeated preparation idempotent. The dashboard summarizes the current replay. No email, chat or external notification is sent.
# MAGIC **Boundary:** Job configuration and SQL queries accompany this notebook. Scheduling, delivery connectors and actual analyst authorization still require the target workspace and named owners.

# COMMAND ----------
# DBTITLE 1,Daily replay notification outbox and human review
outbox_rows=[];seen_episodes=set()
replay_days=sorted(signals_df.run_date.unique())[-3:]
for day in replay_days:
    active=episode_rows[episode_rows.run_date==day]
    for row in active.itertuples():
        key=(DATASET_ID,row.series_id,int(row.episode_id))
        outbox_rows.append({"dataset_id":DATASET_ID,"run_date":pd.Timestamp(day),"series_id":row.series_id,
                           "episode_id":int(row.episode_id),"status":"suppressed_repeat" if key in seen_episodes else "prepared_not_sent",
                           "delivery":"No external connector"})
        seen_episodes.add(key)
outbox_df=pd.DataFrame(outbox_rows)
# Explicitly seeded analyst exercise. The actual proposed review_df stays pending.
operator_disposition_df=review_df.head(1)[["series_id","episode_id"]].copy()
operator_disposition_df["disposition"]="Investigate source refresh"
operator_disposition_df["actor"]="Fictional reviewer exercise"
operator_disposition_df["action_taken"]="No operational action"
dashboard_df=pd.DataFrame([
    {"measure":"Rule-flagged windows","value":int(signals_df.signal_detected.sum())},
    {"measure":"Review episodes","value":total_review_episodes},
    {"measure":"Latest pending queue","value":len(review_df)},
    {"measure":"Prepared notices (not sent)","value":int(outbox_df.status.eq("prepared_not_sent").sum())},
    {"measure":"Repeated notices suppressed","value":int(outbox_df.status.eq("suppressed_repeat").sum())},
])
print(outbox_df.to_string(index=False));print(operator_disposition_df.to_string(index=False))
fig,ax=plt.subplots(figsize=(11,3.5))
bars=ax.barh(dashboard_df.measure.iloc[::-1],dashboard_df.value.iloc[::-1],color="#145A7D")
ax.bar_label(bars,padding=4)
ax.set_xlim(0,float(dashboard_df.value.max())*1.12)
ax.set(title="Local operating dashboard — fictional historical replay",xlabel="Count")
plt.tight_layout();plt.show()
print("Delivery integration is off. Replaying history demonstrates the logic, not a live service.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 19: Record how each model was made
# MAGIC **Plain English:** Save parameters, scores, and four trained models so another person can inspect this run.
# MAGIC **Technique:** MLflow experiment tracking stores run metadata and artifacts. Model registration in Unity Catalog is optional.
# MAGIC **Output:** A run ID in Databricks; if `UC_MODEL_NAME` is set and access is granted, four registered candidate model versions with demo tags and aliases.
# MAGIC **Developer note:** An MLflow run preserves evidence of this execution. Reproducing it also requires the notebook revision, package versions, input dataset ID, and environment to be recorded.

# COMMAND ----------

# DBTITLE 1,Log the result in MLflow when available
try:
    from inspect import signature
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

if mlflow is None:
    print("MLflow is not installed here. In Databricks, this cell logs the model and metrics.")
else:
    # Databricks notebooks automatically use their notebook experiment.
    # MLflow 2 uses artifact_path; MLflow 3 uses name for logged models.
    model_path_arg = "name" if "name" in signature(mlflow.sklearn.log_model).parameters else "artifact_path"
    with mlflow.start_run(run_name="attainx_spc_demo_models") as run:
        mlflow.log_params({"source": "fictional", "seed": SEED, "window": WINDOW,
                           "dataset_id": DATASET_ID, "baseline_days": BASELINE_DAYS, "model": "random_forest",
                           "n_estimators": 100, "max_depth": 8})
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(forecast_metrics)
        classifier_model_info = mlflow.sklearn.log_model(
            model, **{model_path_arg: "model"}, input_example=X_train.head(2)
        )
        forecast_model_info = mlflow.sklearn.log_model(
            forecast_model, **{model_path_arg: "count_forecast"}, input_example=forecast_X.head(2)
        )
        logistic_model_info = mlflow.sklearn.log_model(
            logistic_model, **{model_path_arg: "logistic_classifier"}, input_example=X_train.head(2)
        )
        anomaly_model_info = mlflow.sklearn.log_model(
            iso_model, **{model_path_arg: "isolation_forest"}, input_example=X_train.head(2)
        )
        mlflow.log_metrics({"five_day_" + row.model.lower().replace("-", "_").replace(" ", "_") + "_mae": float(row.mae)
                            for row in five_day_metrics_df.itertuples()})
        print("MLflow run ID:", run.info.run_id)
        print("Logged model URIs:", classifier_model_info.model_uri, forecast_model_info.model_uri, logistic_model_info.model_uri, anomaly_model_info.model_uri)
        print("Five-day statsmodels forecasts and evaluations are in forecast tables; those fitted models are not logged by this cell.")
        if UC_MODEL_NAME:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", UC_MODEL_NAME), (
                "Set UC_MODEL_NAME to a permitted catalog.schema.model using simple identifiers."
            )
            mlflow.set_registry_uri("databricks-uc")
            client = mlflow.MlflowClient()
            for suffix, info in [("", classifier_model_info), ("_count", forecast_model_info),
                                 ("_logistic", logistic_model_info), ("_anomaly", anomaly_model_info)]:
                name = UC_MODEL_NAME + suffix
                registered = mlflow.register_model(info.model_uri, name)
                client.set_model_version_tag(name, registered.version, "demo_only", "true")
                client.set_registered_model_alias(name, "demo_candidate", registered.version)
                print("Registered demo candidate:", name, "version", registered.version)
            print("No production alias changed. Lifecycle and rollback exercises remain local.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 20: Optionally retain queryable demo history
# MAGIC **Plain English:** Save the fictional inputs and results for SQL inspection after confirming a dedicated writable demo schema.
# MAGIC **Method:** Keyed Delta MERGE preserves different forecast origins and replaces an identical replay, instead of deleting all previous runs. Default is off.
# MAGIC **Boundary:** A new dedicated demo schema is recommended. This cell does not migrate incompatible old table schemas. Local tests check the merge contract; real Spark/Delta execution needs workspace verification.

# COMMAND ----------
# DBTITLE 1,Optional managed Delta history
DEMO_TABLES = {
    "demo_daily": (daily_df, ["series_id", "run_date"]),
    "spc_signals": (signals_df, ["series_id", "run_date"]),
    "spc_predictions": (predictions_df, ["series_id", "run_date"]),
    "count_forecasts": (forecast_results_df, ["series_id", "run_date"]),
    "review_queue": (review_df, ["series_id", "episode_id"]),
    "monitoring_snapshot": (monitoring_df, []),
    "zone_checks": (zone_df, ["series_id", "run_date"]),
    "subgroup_counts": (subgroup_df, ["series_id", "office", "run_date"]),
    "five_day_forecasts": (future_forecasts_df, ["series_id", "office", "model", "origin_date", "target_date"]),
    "forecast_history": (forecast_ledger_df, ["series_id", "model", "origin_date", "target_date"]),
    "forecast_assessment": (assessment_df, ["series_id", "model", "origin_date"]),
    "model_anomalies": (anomaly_df, ["series_id", "run_date"]),
    "lineage": (lineage_df, ["table", "path"]),
    "upstream_profiles": (upstream_profile_df, ["table"]),
    "operational_events": (events_df, ["event_id"]),
    "investigation_hypotheses": (hypotheses_df, ["series_id", "episode_id"]),
    "model_lifecycle": (lifecycle_df, ["component"]),
    "notification_outbox": (outbox_df, ["run_date", "series_id", "episode_id"]),
    "review_exercise": (operator_disposition_df, ["series_id", "episode_id"]),
}
if OUTPUT_SCHEMA:
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", OUTPUT_SCHEMA), "Use an existing dedicated catalog.schema."
    assert "spark" in globals(), "Delta outputs require Databricks Spark."
    for table_name, (frame, keys) in DEMO_TABLES.items():
        output_frame = frame.copy()
        output_frame["dataset_id"] = DATASET_ID
        keys = ["dataset_id"] + [key for key in keys if key != "dataset_id"]
        assert not output_frame.duplicated(keys).any(), f"Duplicate keys: {table_name}"
        for col in output_frame:
            if pd.api.types.is_datetime64_any_dtype(output_frame[col]):
                output_frame[col] = output_frame[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        view = "spc_demo_" + table_name
        spark.createDataFrame(output_frame).createOrReplaceTempView(view)
        table = f"{OUTPUT_SCHEMA}.{table_name}"
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table} USING DELTA AS SELECT * FROM {view} WHERE 1=0")
        on = " AND ".join(f"target.`{key}` = source.`{key}`" for key in keys)
        spark.sql(f"MERGE INTO {table} AS target USING {view} AS source ON {on} WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")
        print("Merged", len(output_frame), "rows into", table)
else:
    print(f"Delta writes are off. {len(DEMO_TABLES)} demo tables are ready in memory; set OUTPUT_SCHEMA only after permission checks.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Closing talking points
# MAGIC 1. These are fictional counts and fixture metadata, never USCIS performance measurements.
# MAGIC 2. SPC flags and zones identify review candidates. The classifier imitates the historical rules; Isolation Forest offers a separate unusualness signal.
# MAGIC 3. Compare one-day and five-day forecasts with simple baselines. The five-day forecast includes dates beyond the last observed count.
# MAGIC 4. Office shifts, lineage and events support hypotheses. An analyst must verify cause and record a real decision.
# MAGIC 5. Historical assessment separates insufficient evidence, stale runs and possible retraining needs. Local lifecycle exercises demonstrate withdrawal.
# MAGIC 6. Replay notices are prepared or suppressed, never sent. A live schedule, real registry/Delta writes, external delivery and agency authorization remain separate checks.
