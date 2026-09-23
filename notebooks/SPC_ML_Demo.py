# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # SPC signal detection — AttainX synthetic demo
# MAGIC
# MAGIC This notebook generates fictional operational counts inside AttainX. It reads no external data.
# MAGIC It demonstrates XmR, CUSUM, and EWMA control rules; a classifier that reproduces their
# MAGIC same-window label; a separate **next-business-day count forecast**; chronological evaluation;
# MAGIC analyst review candidates; charts; and an MLflow run when available.
# MAGIC
# MAGIC **Interpretation:** A signal means a statistical rule fired, not that a real problem was confirmed.
# MAGIC The classifier reproduces a rule-generated label from same-window measurements; it does not predict
# MAGIC a future incident. The forecast is genuinely forward-looking, but its scores describe only this
# MAGIC synthetic dataset. Neither model decides whether a process problem occurred.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Set the rules for this run
# MAGIC **Plain English:** Choose the random seed, how much history each rule reads, and whether to save results.
# MAGIC **Technique:** Python libraries handle tables, numerical calculations, charts, and random forest models.
# MAGIC **Check:** `OUTPUT_SCHEMA` and `UC_MODEL_NAME` are empty by default, so the first run needs no write permissions.
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
DATASET_ID = f"attainx_synthetic_v1_seed{SEED}"

# Optional: set to an existing Unity Catalog catalog.schema after the first successful run.
# Example: OUTPUT_SCHEMA = "main.default". Empty means charts/MLflow only.
OUTPUT_SCHEMA = ""
# Optional: set to a permitted three-part catalog.schema.model name to register the model.
# Leave empty until the notebook and MLflow experiment work.
UC_MODEL_NAME = ""

plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.2})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Make a safe example dataset
# MAGIC **Plain English:** Create three imaginary daily work queues. We plant a few jumps and sustained changes so there is something for the chart to find.
# MAGIC **Technique:** A seeded NumPy generator makes repeatable synthetic time series. Assertions check unique dates and nonnegative counts.
# MAGIC **Output:** `daily_df` has one count per business day and fictional series. `dataset_id` identifies this synthetic run.
# MAGIC **Developer note:** The planted changes help explain the demo; they are no proof that real agency data behave this way.

# COMMAND ----------

# DBTITLE 1,Generate synthetic operational series
rng = np.random.default_rng(SEED)
dates = pd.bdate_range("2025-01-06", periods=N_DAYS)
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
# MAGIC **Plain English:** For each date, inspect the prior 25 business days against an earlier 90-day reference period. Save which SPC check fired.
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
# MAGIC **Technique:** A chronological holdout with a 25-business-day gap prevents overlapping train and test windows. Compare accuracy with constant predictions, and read precision, recall, F1, and ROC AUC.
# MAGIC **Output:** `predictions_df` holds the model calls; `metrics` holds both model and simple-baseline scores.
# MAGIC **Decision:** This classification is useful only if it adds something to directly running the known rules. A strong AUC alone does not establish added mission value.

# COMMAND ----------

# DBTITLE 1,Train chronologically and compare against a simple baseline
# Each feature is known at the end of the 25-day window; no future values are inputs.
# The evaluation starts 25 business days after training ends, so train/test windows
# cannot overlap. Metrics describe synthetic rule reproduction, not real-world performance.
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
    print("The forecast beat the trailing-mean baseline on this synthetic test only.")

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
ax.set(title="Synthetic SPC history — Intake A (last 90 runs)", xlabel="Run date", ylabel="Daily count")
ax.legend(loc="upper left", fontsize=8)
ax.text(0.01, -0.33, "Circles mean any rule fired in the prior 25 days; the plotted point need not cross an XmR limit.",
        transform=ax.transAxes, fontsize=8, color="#444444")
fig.autofmt_xdate()
fig.subplots_adjust(bottom=0.34)
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No signal", "Signal"],
                                        cmap="Blues", colorbar=False, ax=ax)
ax.set_title(f"Synthetic test: {len(y_test)} labeled windows")
plt.tight_layout()
plt.show()

forecast_chart = forecast_results_df[forecast_results_df.series_id == chosen].tail(55)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(forecast_chart.run_date, forecast_chart.daily_count, label="Observed", color="#145A7D")
ax.plot(forecast_chart.run_date, forecast_chart.predicted_count, label="Predicted before observation", color="#B8603C")
ax.set(title="Next-business-day forecast — held-out synthetic dates", xlabel="Date", ylabel="Daily count")
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
# MAGIC ### Cell 9: Record how each model was made
# MAGIC **Plain English:** Save parameters, scores, and both trained models so another person can inspect this run.
# MAGIC **Technique:** MLflow experiment tracking stores run metadata and artifacts. Model registration in Unity Catalog is optional.
# MAGIC **Output:** A run ID in Databricks; if `UC_MODEL_NAME` is set and access is granted, a registered model version.
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
    with mlflow.start_run(run_name="attainx_spc_synthetic_models") as run:
        mlflow.log_params({"source": "synthetic", "seed": SEED, "window": WINDOW,
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
        print("MLflow run ID:", run.info.run_id)
        print("Logged model URIs:", classifier_model_info.model_uri, forecast_model_info.model_uri)
        if UC_MODEL_NAME:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", UC_MODEL_NAME), (
                "Set UC_MODEL_NAME to a permitted catalog.schema.model using simple identifiers."
            )
            mlflow.set_registry_uri("databricks-uc")
            registered = mlflow.register_model(classifier_model_info.model_uri, UC_MODEL_NAME)
            print("Registered model:", UC_MODEL_NAME, "version", registered.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 10: Optionally save the demo tables
# MAGIC **Plain English:** After checking permissions, put the fictional input, signals, predictions, review list, and metrics in six queryable tables.
# MAGIC **Technique:** Spark writes managed Delta tables to an existing Unity Catalog `catalog.schema`.
# MAGIC **Output:** Tables for a dashboard or inspection if `OUTPUT_SCHEMA` is set. With the default empty setting, this cell only explains how to enable them.
# MAGIC **Developer note:** The notebook uses `overwrite`. Choose a demo-only schema, and never point this setting at a shared production table.

# COMMAND ----------

# DBTITLE 1,Optional: write synthetic demo tables to Unity Catalog Delta
if OUTPUT_SCHEMA:
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", OUTPUT_SCHEMA), (
        "Set OUTPUT_SCHEMA to an existing catalog.schema using simple identifiers."
    )
    assert "spark" in globals(), "Delta outputs require a Databricks notebook with Spark."
    for table_name, frame in [
        ("synthetic_daily", daily_df),
        ("spc_signals", signals_df),
        ("spc_predictions", predictions_df),
        ("count_forecasts", forecast_results_df),
        ("review_queue", review_df),
        ("monitoring_snapshot", monitoring_df),
    ]:
        # String dates keep schema inference portable across Spark Connect runtimes.
        output_frame = frame.copy()
        for date_col in ["run_date", "last_signal_date", "window_start", "window_end"]:
            if date_col in output_frame:
                output_frame[date_col] = output_frame[date_col].dt.strftime("%Y-%m-%d")
        (spark.createDataFrame(output_frame)
         .write.format("delta").mode("overwrite")
         .saveAsTable(f"{OUTPUT_SCHEMA}.{table_name}"))
        print("Wrote:", f"{OUTPUT_SCHEMA}.{table_name}")
else:
    print("Delta output is off. Set OUTPUT_SCHEMA near the top after confirming your catalog/schema permissions.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Notes
# MAGIC
# MAGIC 1. These are fabricated daily operational counts with planted spikes and shifts.
# MAGIC 2. Three SPC rules examine each preceding 25-day window. Their logical OR is `signal_detected`.
# MAGIC 3. One model approximates the rule label; a separate model predicts the *next* business day's count.
# MAGIC 4. We evaluate on later dates, with a 25-business-day gap, against simple baselines. Read the printed metrics, including cases where ML loses.
# MAGIC 5. The review queue names the triggering rule and leaves disposition pending. No action is automatic.
# MAGIC 6. The source ID, synthetic dataset ID, MLflow run, and optional Delta tables make this run inspectable.
# MAGIC 7. This is a synthetic demonstration. Its metrics do not describe USCIS performance or a deployed capability.
