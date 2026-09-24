"""Build deck evidence from one notebook execution receipt, without rerunning models.

`receipt_from_state` captures an execution (the smoke test saves one; the deck test makes a fresh one);
`evidence_from_receipt` turns a receipt into slide values. Running this file converts the saved receipt.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/SPC_ML_Demo.py"
RECEIPT_FRAMES = ["fold_metrics_df", "iso_fold_df", "five_day_metrics_df", "drift_performance_df", "daily_df",
                  "lifecycle_df", "forecast_results_df", "train_df", "test_df"]
PINNED_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "statsmodels"]


def source_identity(dataset_id):
    """Record which notebook revision, dataset and package versions produced a receipt."""
    return {"notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(), "dataset_id": dataset_id,
            "packages": {name: version(name) for name in PINNED_PACKAGES}}


def receipt_from_state(s):
    """Compact execution receipt; every value comes from the same notebook run."""
    signals = s["signals_df"]
    receipt = {"source": source_identity(s["DATASET_ID"]), "dataset": s["dataset_summary"], "dataset_id": s["DATASET_ID"],
               "classifier_metrics": s["metrics"], "forecast_metrics": s["forecast_metrics"],
               "classifier_config": s["CLASSIFIER_CONFIG"],
               "classifier_train_rows": len(s["train_df"]), "classifier_test_rows": len(s["test_df"]),
               "forecast_train_rows": int(s["forecast_train"].sum()), "forecast_test_rows": int(s["forecast_test"].sum()),
               "selected_forecaster": s["selected_forecaster"], "model_manifest": s["saved_forecast_manifest"],
               "signal_rows": len(signals), "review_episodes": len(s["review_df"]),
               "rule_signal_rate": float(signals.signal_detected.mean()), "total_review_episodes": int(s["total_review_episodes"]),
               "isolation_anomaly_rows": int(s["anomaly_df"].model_anomaly.sum()),
               "isolation_disagreement_rows": int(s["anomaly_df"].disagrees_with_spc.sum()),
               "isolation_contamination": s["final_contamination"]}
    for name in RECEIPT_FRAMES:
        receipt[name] = json.loads(s[name].to_json(orient="records", date_format="iso"))
    return receipt


def f1_outcome(model_f1, constant_f1):
    """Compare at the three-decimal precision shown on the slide."""
    model_f1, constant_f1 = round(model_f1, 3), round(constant_f1, 3)
    return "win" if model_f1 > constant_f1 else "tie" if model_f1 == constant_f1 else "loss"


def classification_conclusion(outcomes):
    """Slide conclusion for the holdout and fold outcomes of the best classifier against always-signal."""
    if "win" in outcomes:
        return "A classifier beats always-signal on F1 in at least one experiment, but the rules still compute this label directly."
    if "tie" in outcomes:
        return "No classifier beats always-signal on F1; at best one ties it. Keep the direct rules for this label."
    return "Neither classifier beats always-signal on F1. Keep the direct rules for this label."


def evidence_from_receipt(x):
    d = x['dataset']; m = x['classifier_metrics']; f = x['forecast_metrics']
    dates = sorted({r['run_date'] for r in x['daily_df']})[-30:]
    series = []
    for key, color in [('Intake A', '#007C83'), ('Intake B', '#BC6C25'), ('Completions A', '#315B9B')]:
        values = {r['run_date']: r['daily_count'] for r in x['daily_df'] if r['series_id'] == key}
        series.append({'name': key.replace('Intake', 'Receipts'), 'values': [values[day] for day in dates], 'line': {'fill': color, 'width': 3}})
    fold = {k: mean(r['f1'] for r in x['fold_metrics_df'] if r['model'] == k) for k in ['Random forest', 'Logistic regression']}
    # One always-signal F1 per fold; both model rows of a fold share the same test windows.
    fold_constant = mean(r['always_signal_f1'] for r in x['fold_metrics_df'] if r['model'] == 'Random forest')
    outcomes = {f1_outcome(m['f1'], m['always_signal_f1']), f1_outcome(max(fold.values()), fold_constant)}
    c = x['classifier_config']
    classifier_config = f"{c['n_estimators']} trees, depth {c['max_depth']}, leaf minimum {c['min_samples_leaf']}" + (", balanced classes" if c.get('class_weight') == 'balanced' else "")
    summary = f"{d['event_rows']:,} synthetic event rows aggregate to {d['daily_rows']:,} daily observations across {d['series']} series and {d['business_dates']} business dates."
    return {
        'dataset': x['dataset_id'], 'dataSummary': summary,
        'volume': {'categories': [str(i + 1) for i, _ in enumerate(dates)], 'series': series},
        'trainingSummary': f"Classifier: {x['classifier_train_rows']:,} training windows, {x['classifier_test_rows']:,} later test windows, with a 25-business-day separation.",
        'classificationTable': [['Experiment', 'Model / reference', 'Score'],
                                ['Holdout accuracy', 'Random forest / always signal', f"{m['accuracy']:.1%} / {m['always_signal_accuracy']:.1%}"],
                                ['Holdout F1', 'Random forest / always signal', f"{m['f1']:.3f} / {m['always_signal_f1']:.3f}"],
                                ['Holdout precision / recall', 'Random forest', f"{m['precision']:.1%} / {m['recall']:.1%}"],
                                ['Five-fold mean F1', 'Random forest / logistic', f"{fold['Random forest']:.3f} / {fold['Logistic regression']:.3f}"],
                                ['Five-fold mean F1', 'Always signal', f"{fold_constant:.3f}"]],
        'classificationWidths': [370, 470, 290],
        'classificationConclusion': classification_conclusion(outcomes),
        'classifierConfig': classifier_config,
        'forecastTable': [['Horizon / sample', 'Method', 'MAE (counts)'],
                          ['Next day, 225 rows', 'Random forest', f"{f['forecast_mae']:.2f}"],
                          ['Next day, 225 rows', 'Previous count', f"{f['last_value_baseline_mae']:.2f}"],
                          ['Next day, 225 rows', 'Trailing mean', f"{f['trailing_mean_baseline_mae']:.2f}"]]
                         + [[f"Five days, {r['forecast_rows']} rows", r['model'], f"{r['mae']:.2f}"] for r in x['five_day_metrics_df']],
        'forecastConclusion': 'Lower MAE is better. Five-day values pool six origins and are descriptive.',
        'reuseSummary': 'Fresh processes reload joblib and local MLflow models with matching predictions.',
        'driftSummary': 'Controlled outcome shift: one worse window triggers watch, two trigger review.',
        'validationSummary': f"Rules flag {x['rule_signal_rate']:.1%} of windows ({m['test_signal_rate']:.1%} of the holdout). Isolation Forest flags {x['isolation_anomaly_rows']} of {x['classifier_test_rows']}.",
        'isolationContamination': x['isolation_contamination'],
        'source': 'Final local smoke-test receipt from notebooks/SPC_ML_Demo.py',
        'sourceIdentity': x['source'],
        'rawMetrics': {'classifier': m, 'forecast': f, 'foldMeanF1': fold, 'foldAlwaysSignalF1': fold_constant,
                       'ruleSignalRate': x['rule_signal_rate'], 'reviewEpisodes': x['total_review_episodes'],
                       'fiveDay': x['five_day_metrics_df']},
    }


if __name__ == "__main__":
    receipt = json.loads((ROOT / ".build/notebook-metrics.json").read_text())
    (ROOT / "docs/deck-evidence.json").write_text(json.dumps(evidence_from_receipt(receipt), indent=2) + "\n")
