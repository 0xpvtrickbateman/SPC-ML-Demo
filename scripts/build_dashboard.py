"""Build the native Databricks dashboard; optionally execute the notebook for a local preview."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "attainx_demo_v2_seed42"

def make_dashboard(schema):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError("Use a dedicated catalog.schema containing letters, numbers and underscores.")
    datasets = []
    def dataset(name, sql):
        datasets.append({"name": name, "displayName": name.replace("_", " ").title(), "queryLines": [sql]})
        return name
    def source(table, scenario=None):
        where = f"dataset_id = '{DATASET_ID}'"
        if scenario:
            where += f" AND scenario = '{scenario}'"
        return f"FROM {schema}.{table} WHERE {where}"
    for prefix, scenario in [("replay", "Recorded replay"), ("exercise", "Performance drift exercise")]:
        dataset(prefix+"_daily", "SELECT series_id, CAST(run_date AS DATE) AS run_date, evaluation_actual, predicted_count, trailing_mean_baseline "+source("demo_drift_daily", scenario))
        dataset(prefix+"_performance", "SELECT series_id, CAST(window_start AS DATE) AS window_start, CAST(window_end AS DATE) AS window_end, window_number, n_actuals, expected_actuals, mae, reference_mae, baseline_mae, bias, review_threshold, status "+source("demo_drift_performance", scenario))
        dataset(prefix+"_latest", "SELECT series_id, n_actuals, mae, reference_mae, baseline_mae, status, window_end "+source("demo_drift_performance", scenario)+" AND window_number = 2")
        parts = []
        for col, label in [("mae", "Model MAE"), ("baseline_mae", "Simple baseline MAE"), ("review_threshold", "Review threshold")]:
            parts.append(f"SELECT series_id, CAST(window_end AS DATE) AS window_end, {col} AS error_count, CONCAT(series_id, ' · {label}') AS measure "+source("demo_drift_performance", scenario))
        dataset(prefix+"_errors", " UNION ALL ".join(parts))
    dataset("input_change", "SELECT series_id, feature, shift_score, threshold, reference_n, current_n, missing_rate, status "+source("demo_drift_inputs", "Recorded replay")+" AND window_number = 2")
    dataset("review_queue", "SELECT series_id, episode_id, CAST(run_date AS DATE) AS run_date, CAST(last_signal_date AS DATE) AS last_signal_date, signal_windows, rule_fired, disposition "+source("review_queue"))
    def place(widget, x,y,w,h):
        return {"widget":widget,"position":{"x":x,"y":y,"width":w,"height":h}}
    def text(name, content, y, h=2):
        return place({"name":name,"multilineTextboxSpec":{"lines":[content]}},0,y,12,h)
    def widget(name, ds, title, kind, columns, encodings, x,y,w,h, description="", aggregate=False):
        fields = [{"name":c,"expression":e} for c,e in columns]
        return place({"name":name,"queries":[{"name":"main_query","query":{"datasetName":ds,"fields":fields,"disaggregated":not aggregate}}],"spec":{"version":2 if kind in ("table","counter") else 3,"widgetType":kind,"encodings":encodings,"frame":{"showTitle":True,"title":title,"showDescription":bool(description),"description":description}}},x,y,w,h)
    def table(name, ds, title, cols,y,h=6):
        return widget(name,ds,title,"table",[(c,f"`{c}`") for c,_ in cols],{"columns":[{"fieldName":c,"displayName":label} for c,label in cols]},0,y,12,h)
    def page(name,title,layout):
        return {"name":name,"displayName":title,"pageType":"PAGE_TYPE_CANVAS","layoutVersion":"GRID_V1","layout":layout}
    def chart(name,ds,title,xcol,ycol,color,y,height=6):
        return widget(name,ds,title,"line",[(c,f"`{c}`") for c in [xcol,ycol,color]],{"x":{"fieldName":xcol,"scale":{"type":"temporal"}},"y":{"fieldName":ycol,"scale":{"type":"quantitative"}},"color":{"fieldName":color,"scale":{"type":"categorical"}}},0,y,12,height)
    def daily_chart(prefix,y):
        columns=[("run_date","`run_date`")]+[(c,f"SUM(`{c}`)") for c in ["evaluation_actual","predicted_count","trailing_mean_baseline"]]
        return widget(prefix+"_trend",prefix+"_daily","Daily counts and one-day forecasts","line",columns,{"x":{"fieldName":"run_date","scale":{"type":"temporal"}},"y":{"fields":[{"fieldName":"evaluation_actual","displayName":"Actual / exercise outcome"},{"fieldName":"predicted_count","displayName":"Frozen model forecast"},{"fieldName":"trailing_mean_baseline","displayName":"Trailing five-day mean"}],"scale":{"type":"quantitative"}}},0,y,12,6,"Counts sum across selected queues. The exercise changes outcomes only.",aggregate=True)
    latest_cols=[("series_id","Queue"),("mae","Latest MAE"),("reference_mae","Reference MAE"),("baseline_mae","Simple baseline MAE"),("n_actuals","Matched actuals"),("status","Review status")]
    performance_cols=[("series_id","Queue"),("window_number","Window: 0 = reference"),("window_end","Through"),("n_actuals","Actuals"),("mae","Model MAE"),("bias","Bias: actual − forecast"),("status","Assessment")]
    overview=page("overview","1 · Overview",[
        text("overview_intro","## SPC and model monitoring\n\nAttainX · Fictional data · Recorded replay. The forecast is frozen; actual counts arrive afterward. Use the queue filter to focus on one process. No live monitoring is connected.",0,3),
        table("replay_latest_table","replay_latest","Latest window · error in count units",latest_cols,3,4),
        daily_chart("replay",7),
        text("workflow","### Counts → historical features → rules and forecasts → review\n\nSPC asks whether the process changed. Input monitoring asks whether model inputs differ from training. Performance monitoring asks whether saved predictions became less accurate after actuals arrived.",13,3),
    ])
    drift=page("drift","2 · Detect drift",[
        text("drift_intro","## Two different questions\n\nInput changes prompt investigation. Worsening prediction error supplies separate evidence. Neither alone establishes the cause or proves concept drift.",0,3),
        widget("input_bar","input_change","Latest input distribution change vs training","bar",[(c,f"`{c}`") for c in ["feature","shift_score","series_id"]],{"x":{"fieldName":"feature","scale":{"type":"categorical"}},"y":{"fieldName":"shift_score","scale":{"type":"quantitative"}},"color":{"fieldName":"series_id","scale":{"type":"categorical"}}},0,3,12,5,"Distance in training standard-deviation units. Above 0.5 prompts investigation; illustrative threshold, three numeric features only."),
        chart("replay_error_chart","replay_errors","Error across three 25-business-day windows","window_end","error_count","measure",8),
        text("drift_rule","**Performance rule:** The first held-out window is the reference. Review after two consecutive windows exceed 125% of reference MAE (one-count floor), with at least 20 matched actuals each. Calibrate thresholds before operational use. Labels must arrive before accuracy can be assessed.",14,3),
        table("replay_performance_table","replay_performance","Window evidence",performance_cols,17,6),
        table("input_detail","input_change","Input coverage and diagnostic evidence",[("series_id","Queue"),("feature","Input"),("reference_n","Training rows"),("current_n","Current rows"),("missing_rate","Missing fraction"),("shift_score","Shift distance"),("status","Assessment")],23,6),
    ])
    exercise=page("exercise","3 · Drift exercise",[
        text("exercise_intro","## Controlled performance drift exercise\n\nAdd 35 counts to outcomes in the last two windows. Inputs, model and saved predictions remain unchanged. These scenario results are separate from the recorded replay; no model is retrained.",0,3),
        table("exercise_latest_table","exercise_latest","Latest exercise window",latest_cols,3,4),
        chart("exercise_error_chart","exercise_errors","Watch one worse window → review after two","window_end","error_count","measure",7),
        table("exercise_performance_table","exercise_performance","Exercise decision evidence",performance_cols,13,6),
        daily_chart("exercise",19),
        text("exercise_lesson","**What this shows:** Input monitoring cannot catch every performance failure. Join actual outcomes to saved predictions as they become available. A warning opens a review; it does not automatically replace a model.",25,3),
    ])
    decisions=page("decisions","4 · Review and retrain",[
        text("decision_intro","## Investigate → train a candidate → validate → approve\n\n1. Check missing or late data, workload mix, seasonality and process changes.\n2. Train a candidate on recent representative history; preserve the current model.\n3. Evaluate on later unseen dates against both the current model and a simple baseline, by queue.\n4. Log data, code, parameters and metrics in MLflow. Obtain approval, promote the version, monitor and retain rollback.",0,6),
        table("pending_reviews","review_queue","SPC review candidates · recorded replay only",[("series_id","Queue"),("episode_id","Episode"),("last_signal_date","Latest signal"),("signal_windows","Flagged windows"),("rule_fired","Rules"),("disposition","Disposition")],6,7),
        text("decision_boundary","**Keep the decisions separate:** SPC candidates are process investigations. Drift warnings concern model usefulness. The exercise never changes the SPC queue. No notification, automatic retraining or promotion occurs here.\n\n**Refresh:** This dashboard reads saved notebook results. Run all retrains the demo; production monitoring would reuse a registered model and join new actuals independently.",13,4),
    ])
    queries=[{"name":"filter_"+d["name"],"query":{"datasetName":d["name"],"fields":[{"name":"series_id","expression":"`series_id`"}],"disaggregated":False}} for d in datasets]
    filters={"name":"filters","displayName":"Queue","pageType":"PAGE_TYPE_GLOBAL_FILTERS","layoutVersion":"GRID_V1","layout":[place({"name":"queue_filter","queries":queries,"spec":{"version":2,"widgetType":"filter-single-select","encodings":{"fields":[{"fieldName":"series_id","displayName":"Work queue","queryName":q["name"]} for q in queries]},"selection":{"defaultSelection":{"values":{"dataType":"STRING","values":[{"value":"Intake A"}]}}},"frame":{"showTitle":True,"title":"Work queue · default Intake A"}}},0,0,4,2)]}
    theme={"canvasBackgroundColor":{"light":"#F2F5F7","dark":"#13232E"},"widgetBackgroundColor":{"light":"#FFFFFF","dark":"#192F3D"},"fontColor":{"light":"#132F42","dark":"#E5EDF1"},"selectionColor":{"light":"#116D88","dark":"#6FCDDF"},"visualizationColors":["#116D88","#D08029","#188B7E","#7C75A2","#587992"],"widgetHeaderAlignment":"LEFT"}
    for canvas in [overview, drift, exercise, decisions]:
        for item in canvas["layout"]:
            spec = item["widget"].get("spec", {})
            if spec.get("widgetType") == "bar":
                spec["mark"] = {"layout": "group"}
    return {"datasets":datasets,"pages":[filters,overview,drift,exercise,decisions],"uiSettings":{"theme":theme}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema",default="demo_catalog.demo_schema")
    parser.add_argument("--preview",action="store_true",help="Execute notebook locally and export its HTML dashboard and reviewed data.")
    args=parser.parse_args()
    out=ROOT/"dashboards";out.mkdir(exist_ok=True)
    (out/"AttainX_SPC_Demo.lvdash.json").write_text(json.dumps(make_dashboard(args.schema),indent=2)+"\n")
    template = (out/"setup-template.html").read_text()
    (out/"setup.html").write_text(template.replace("__TEMPLATE_JSON__", json.dumps(make_dashboard("demo_catalog.demo_schema")).replace("<", "\\u003c")))
    print("Native dashboard built for",args.schema)
    if args.preview:
        os.environ.setdefault("MPLBACKEND","Agg")
        with contextlib.redirect_stdout(io.StringIO()):
            state=runpy.run_path(str(ROOT/"notebooks/SPC_ML_Demo.py"))
        (out/"preview.html").write_text(state["DASHBOARD_HTML"])
        (out/"reviewed-data.json").write_text(json.dumps(state["dashboard_payload"],indent=2,allow_nan=False)+"\n")
        print("Notebook executed; saved preview.html and reviewed-data.json")

if __name__=="__main__":
    main()
