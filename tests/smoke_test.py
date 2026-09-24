"""Run the entire demo and verify temporal, reconciliation and governance boundaries."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["SPC_DEMO_LOCAL_TEST"] = "1"
# Leave any operator bundle in artifacts/ untouched unless a directory is chosen explicitly.
os.environ.setdefault("SPC_DEMO_MODEL_DIR", tempfile.mkdtemp(prefix="spc-smoke-bundle-"))
root=Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()) as output:
    state=runpy.run_path(str(root/"notebooks"/"SPC_ML_Demo.py"))
s=state
signals=s["signals_df"]
assert len(signals)==915
assert signals.signal_detected.eq(signals[["xmr_signal","cusum_signal","ewma_signal"]].any(axis=1)).all()
assert s["train_df"].window_end.max()<s["test_df"].window_start.min()
assert s["forecast_df"].loc[s["forecast_train"],"run_date"].max()<s["forecast_df"].loc[s["forecast_test"],"window_start"].min()
assert s["review_df"].disposition.eq("Pending analyst review").all()
assert s["fold_metrics_df"].groupby("model").fold.nunique().eq(5).all()
for _,tr,te in s["fold_specs"]:
    assert signals.loc[tr,"window_end"].max()<signals.loc[te,"window_start"].min()
assert s["zone_check"](np.r_[np.zeros(24),100.])[0]=="CRITICAL"
assert s["zone_check"](np.ones(25))[0]=="NORMAL"
assert len(s["zone_df"])==len(signals)
assert s["daily_df"].run_date.dt.dayofweek.lt(5).all()
from pandas.tseries.holiday import USFederalHolidayCalendar
holidays=USFederalHolidayCalendar().holidays(s["daily_df"].run_date.min(),s["future_forecasts_df"].target_date.max())
assert not s["daily_df"].run_date.isin(holidays).any()
f=s["future_forecasts_df"]
assert (f.target_date>f.origin_date).all()
assert f.groupby(["series_id","office","model"]).horizon.apply(list).map(lambda x:x==[1,2,3,4,5]).all()
assert f.prediction.ge(0).all() and not f.target_date.isin(holidays).any()
h=s["forecast_history_df"]
assert (h.origin_date<h.target_date).all()
assert h.groupby(["series_id","origin_date","model"]).size().eq(5).all()
assert h.groupby("model").size().nunique()==1
assert s["selection_cutoff"]==h.origin_date.max()
assert len(s["upsert_history"](h,h,s["HISTORY_KEYS"]))==len(h)
changed=h.head(1).copy();changed["prediction"]=0
merged=s["upsert_history"](h,changed,s["HISTORY_KEYS"])
assert len(merged)==len(h)
assert merged.set_index(s["HISTORY_KEYS"]).loc[tuple(changed[s["HISTORY_KEYS"]].iloc[0]),"prediction"]==0
assert s["retraining_advice"](1,95,20)=="insufficient_data"
assert s["retraining_advice"](10,95,20,False)=="superseded"
assert s["retraining_advice"](10,95,20)=="architecture_review"
assert s["assessment_df"].loc[s["assessment_df"].origin_date<s["assessment_df"].latest_origin,"recommendation"].eq("superseded").all()
assert s["iso_fold_df"].contamination.between(.02,.10).all()
assert len(s["anomaly_df"])==len(s["test_df"])
assert s["lineage_df"].hop.max()==5
cycle=pd.DataFrame({"target":["a","b"],"source":["b","a"]})
assert len(s["trace_upstream"]("a",cycle))==2
assert s["hypotheses_df"].confidence.eq("Unverified hypothesis").all()
out=s["outbox_df"]
assert out[out.status.eq("prepared_not_sent")].groupby(["dataset_id","series_id","episode_id"]).size().eq(1).all()
assert out.status.eq("suppressed_repeat").any()
assert s["registry_exercise_df"].iloc[-1].active_version=="baseline-v1"
assert s["rehearse_registry_change"]("baseline-v1", "candidate-v1", False)[0] == "baseline-v1"
assert s["rehearse_registry_change"]("baseline-v1", "candidate-v1", True)[0] == "candidate-v1"
for name,(frame,keys) in s["DEMO_TABLES"].items():
    assert len(frame)>0,name
    assert not frame.duplicated(keys).any() if keys else len(frame)==1
# Execute portable dashboard SELECTs against the actual in-memory output schemas.
import sqlite3
with sqlite3.connect(":memory:") as connection:
    for name, (frame, _) in (s["DEMO_TABLES"] | s["DRIFT_TABLES"]).items():
        output_frame = frame.copy()
        output_frame["dataset_id"] = s["DATASET_ID"]
        output_frame.to_sql(name, connection, index=False)
    queries = (root/"sql"/"dashboard_queries.sql").read_text().replace("ml_statistical_process_controls.demo_schema.", "").replace("demo_catalog.demo_schema.", "")
    queries = "\n".join(line for line in queries.splitlines() if not line.lstrip().startswith("--"))
    for query in queries.split(";"):
        if query.strip():
            assert len(connection.execute(query).fetchall()) > 0
source=(root/"notebooks"/"SPC_ML_Demo.py").read_text()
cells=source.split("# COMMAND ----------")
code=[(i,c) for i,c in enumerate(cells) if "# DBTITLE 1," in c]
assert len(code)==22
for n,(i,c) in enumerate(code,1):assert f"### Cell {n}:" in cells[i-1]
# Every cell names its method so reviewers can see where statistics ends and ML begins.
for n,(i,c) in enumerate(code,1):assert "**Type:**" in cells[i-1],n
assert "ictional" not in source, "Use 'synthetic' for data and 'simulated' for exercise actions"
assert 'OUTPUT_SCHEMA = "ml_statistical_process_controls.demo_schema"' in source
assert s["OUTPUT_SCHEMA"] == "" and s["LOCAL_TEST"] is True
assert len(s["raw_events_df"]) >= 1_000_000
assert s["raw_events_df"].event_id.is_unique
assert len(s["raw_events_df"]) == len(s["prepared_events_df"]) == s["daily_df"].daily_count.sum()
recomputed=s["prepared_events_df"].groupby(["series_id", "run_date"], observed=True).size().sort_index()
assert np.array_equal(recomputed.to_numpy(),s["daily_df"].set_index(["series_id","run_date"]).daily_count.sort_index().to_numpy())
assert s["model"].get_params()["class_weight"] == "balanced"
# Constant baselines are reported on the same metric as the models (F1 of always predicting a signal).
rate=s["metrics"]["test_signal_rate"]
assert np.isclose(s["metrics"]["always_signal_f1"],2*rate/(1+rate))
for fold,g in s["fold_metrics_df"].groupby("fold"):
    fold_rate=signals.loc[s["fold_specs"][fold-1][2],"signal_detected"].mean()
    assert np.allclose(g.always_signal_f1,2*fold_rate/(1+fold_rate))
assert np.isclose(s["forecast_metrics"]["forecast_bias"],(s["forecast_results_df"].predicted_count-s["forecast_results_df"].daily_count).mean())
assert s["assessment_metrics"](pd.DataFrame({"prediction":[12.],"actual":[10.],"origin_count":[9.]}))["bias"] == 2.
print("Expanded smoke test passed: 22 cells, chronological folds, future forecasts, history, lifecycle and replay checks.")
print(output.getvalue())

assert s["performance_status"](25,25,np.nan,10,False,1)[0] == "Insufficient actuals"
assert s["performance_status"](25,25,10,np.inf,False,1)[0] == "Insufficient actuals"
missing=pd.DataFrame({"prediction":[12.,13.,np.inf],"actual":[10.,np.nan,10.],"origin_count":[9.,9.,9.]})
assert s["assessment_metrics"](missing)["n_actuals"] == 1
assert s["assessment_metrics"](missing)["bias"] == 2.
assert s["selected_forecaster"] == s["selection_scores"].idxmin()

# Compact execution receipt for the deck and review; all values come from this run.
import json
sys.path.insert(0,str(root/"scripts"))
from prepare_deck_evidence import receipt_from_state
receipt=receipt_from_state(s)
assert s["MLFLOW_STATUS"]=="disabled" and s["saved_forecast_manifest"]["model_uri"] is None
assert list(s["forecast_test_dates"])==sorted(s["forecast_results_df"].run_date.unique())
(root/".build").mkdir(exist_ok=True)
(root/".build"/"notebook-metrics.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
print("Metrics receipt: .build/notebook-metrics.json")

assert s["retraining_advice"](25,np.nan,10) == "insufficient_data"
assert np.isnan(s["assessment_metrics"](pd.DataFrame({"prediction":[12.,13.],"actual":[10.,12.],"origin_count":[np.nan,9.]}))["direction_accuracy"])
