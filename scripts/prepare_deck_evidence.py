"""Prepare deck evidence from the final smoke-test receipt, without rerunning models."""
import json
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
x=json.loads((ROOT/'.build/notebook-metrics.json').read_text())
d=x['dataset']; m=x['classifier_metrics']; f=x['forecast_metrics']
dates=sorted({r['run_date'] for r in x['daily_df']})[-30:]
series=[]
for key,color in [('Intake A','#007C83'),('Intake B','#BC6C25'),('Completions A','#315B9B')]:
 values={r['run_date']:r['daily_count'] for r in x['daily_df'] if r['series_id']==key}
 series.append({'name':key.replace('Intake','Receipts'),'values':[values[d] for d in dates],'line':{'fill':color,'width':3}})
fold={k:mean(r['f1'] for r in x['fold_metrics_df'] if r['model']==k) for k in ['Random forest','Logistic regression']}
summary=f"{d['event_rows']:,} synthetic event rows aggregate to {d['daily_rows']:,} daily observations across {d['series']} series and {d['business_dates']} business dates."
e={
'dataset':x['dataset_id'],'dataSummary':summary,
'volume':{'categories':[str(i+1) for i,v in enumerate(dates)],'series':series},
 'trainingSummary':f"Classifier: {x['classifier_train_rows']:,} training windows, {x['classifier_test_rows']:,} later test windows, with a 25-business-day separation.",
'classificationTable':[['Experiment','Model / reference','Score'],['Holdout accuracy','Random forest',f"{m['accuracy']:.1%}"],['Holdout accuracy','Always signal',f"{m['always_signal_accuracy']:.1%}"],['Holdout precision / recall','Random forest',f"{m['precision']:.1%} / {m['recall']:.1%}"],['Five-fold mean F1','Random forest',f"{fold['Random forest']:.3f}"],['Five-fold mean F1','Logistic regression',f"{fold['Logistic regression']:.3f}"]],
'classificationWidths':[370,470,290],
'classificationConclusion':'Logistic regression has higher mean fold F1. Keep direct rules for their known target.',
'forecastTable':[['Horizon / sample','Method','MAE (counts)'],['Next day, 225 rows','Random forest',f"{f['forecast_mae']:.2f}"],['Next day, 225 rows','Previous count',f"{f['last_value_baseline_mae']:.2f}"],['Next day, 225 rows','Trailing mean',f"{f['trailing_mean_baseline_mae']:.2f}"]]+[[f"Five days, {r['forecast_rows']} rows",r['model'],f"{r['mae']:.2f}"] for r in x['five_day_metrics_df']],
'forecastConclusion':'Lower MAE is better. Five-day values pool six origins and are descriptive.',
'reuseSummary':'Fresh processes reload joblib and local MLflow models with identical predictions.',
'driftSummary':'Controlled outcome shift: one worse window triggers watch, two trigger review.',
'validationSummary':f"Holdout signal share {m['test_signal_rate']:.1%}. Isolation Forest flags {x['isolation_anomaly_rows']} of {x['classifier_test_rows']} windows.",
'isolationContamination':x['isolation_contamination'],
'source':'Final local smoke-test receipt from notebooks/SPC_ML_Demo.py',
'rawMetrics':{'classifier':m,'forecast':f,'foldMeanF1':fold,'fiveDay':x['five_day_metrics_df']}}
(ROOT/'docs/deck-evidence.json').write_text(json.dumps(e,indent=2)+'\n')
