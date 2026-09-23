"""Execute drift calculations and every native dashboard query against local DuckDB.
Install duckdb==1.4.4 in a test environment. This does not verify Databricks rendering.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import numpy as np
import pandas as pd
import duckdb
os.environ.setdefault('MPLBACKEND','Agg')
os.environ['SPC_DEMO_LOCAL_TEST']='1'
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
from build_dashboard import make_dashboard
with contextlib.redirect_stdout(io.StringIO()):
    s=runpy.run_path(str(root/'notebooks/SPC_ML_Demo.py'))
p=s['drift_performance_df']; inputs=s['drift_inputs_df']; daily=s['drift_daily_df']
assert len(p)==18 and len(inputs)==54 and len(daily)==450
replay=p[p.scenario.eq('Recorded replay')]
exercise=p[p.scenario.eq('Performance drift exercise')]
assert replay.n_actuals.eq(25).all()
assert not replay.status.eq('Review for retraining').any()
assert exercise[exercise.window_number.eq(1)].status.eq('Watch: one worse window').all()
assert exercise[exercise.window_number.eq(2)].status.eq('Review for retraining').all()
a=daily[daily.scenario.eq('Recorded replay')].set_index(['series_id','run_date'])
b=daily[daily.scenario.eq('Performance drift exercise')].set_index(['series_id','run_date'])
np.testing.assert_allclose(a.predicted_count,b.predicted_count)
np.testing.assert_allclose(a.daily_count,b.daily_count)
np.testing.assert_allclose(b.evaluation_actual-a.evaluation_actual,np.where(a.window_number>0,350.,0.))
assert s['forecast_metrics']['forecast_mae']==float(a.model_error.abs().mean()) or np.isclose(s['forecast_metrics']['forecast_mae'],a.model_error.abs().mean())
for key,g in daily.groupby(['scenario','series_id','window_number']):
    row=p.set_index(['scenario','series_id','window_number']).loc[key]
    assert np.isclose(row.mae, np.abs(g.evaluation_actual-g.predicted_count).mean())
    assert np.isclose(row.baseline_mae,np.abs(g.evaluation_actual-g.trailing_mean_baseline).mean())
    assert np.isclose(row.bias,(g.predicted_count-g.evaluation_actual).mean())
for series,g in p.groupby(['series_id','scenario']):
    g=g.sort_values('window_number')
    assert (g.window_start.iloc[1:].reset_index(drop=True)>g.window_end.iloc[:-1].reset_index(drop=True)).all()
    assert g.reference_mae.eq(g.mae.iloc[0]).all()
    assert g.review_threshold.eq(max(g.mae.iloc[0],1)*1.25).all()
cols=['series_id','window_number','feature']
pd.testing.assert_frame_equal(inputs[inputs.scenario.eq('Recorded replay')].set_index(cols).drop(columns='scenario'),inputs[inputs.scenario.eq('Performance drift exercise')].set_index(cols).drop(columns='scenario'))
assert s['input_shift_score'](np.arange(25),np.arange(25))==0
assert s['input_shift_score'](np.arange(25),np.arange(25)+100)>0.5
assert np.isnan(s['input_shift_score']([1]*5,[1]*25))
assert np.isnan(s['input_shift_score']([1]*25,[2]*25))
assert s['performance_status'](19,25,10,50,True,2)==('Insufficient actuals',False)
assert s['performance_status'](25,19,10,50,True,2)==('Insufficient actuals',False)
assert s['performance_status'](25,25,10,12.5,True,2)==('Within demo tolerance',False)
assert s['performance_status'](25,25,10,13,False,1)==('Watch: one worse window',True)
assert s['performance_status'](25,25,10,13,True,2)==('Review for retraining',True)
assert s['performance_status'](25,25,0,.5,True,2)==('Within demo tolerance',False)
for name,(frame,keys) in s['DRIFT_TABLES'].items():
    assert not frame.duplicated(['dataset_id']+keys).any()
# Exercise the optional writer without external side effects.
from types import SimpleNamespace
queries=[];views=[]
class Spark:
    def createDataFrame(self,frame):
        return SimpleNamespace(createOrReplaceTempView=lambda name:views.append(name))
    def sql(self,query):queries.append(query)
cell=(root/'notebooks/SPC_ML_Demo.py').read_text().split('# DBTITLE 1,Interactive drift dashboard and optional Delta outputs\n')[1].split('import json\n')[0]
ns=dict(s,OUTPUT_SCHEMA='ml_statistical_process_controls.demo_schema',spark=Spark())
with contextlib.redirect_stdout(io.StringIO()):exec(cell,ns)
assert len(views)==3 and len(queries)==6
assert all('target.`dataset_id` = source.`dataset_id`' in q and 'target.`scenario` = source.`scenario`' in q for q in queries if q.startswith('MERGE'))
assert not any('DROP ' in q or 'DELETE ' in q or 'OVERWRITE' in q for q in queries)
# Run unmodified generated SQL; native Databricks acceptance remains a separate check.
con=duckdb.connect()
con.execute("ATTACH ':memory:' AS ml_statistical_process_controls")
con.execute('CREATE SCHEMA ml_statistical_process_controls.demo_schema')
for name,(frame,_) in {**s['DEMO_TABLES'],**s['DRIFT_TABLES']}.items():
    copy=frame.copy();copy['dataset_id']=s['DATASET_ID']
    con.register('frame_input',copy)
    con.execute(f'CREATE TABLE ml_statistical_process_controls.demo_schema.{name} AS SELECT * FROM frame_input')
spec=make_dashboard('ml_statistical_process_controls.demo_schema')
saved_spec=json.loads((root/'dashboards/AttainX_SPC_Demo.lvdash.json').read_text())
assert [d['name'] for d in saved_spec['datasets']]==[d['name'] for d in spec['datasets']]
assert saved_spec == spec, 'Committed native dashboard must match its default generator'
assert make_dashboard() == spec
examples='\n'.join(line for line in (root/'sql/dashboard_queries.sql').read_text().splitlines() if not line.lstrip().startswith('--'))
for sql in examples.split(';'):
    if any(line.strip() and not line.lstrip().startswith('--') for line in sql.splitlines()):
        con.execute(sql).fetchall()
schemas={}
for dataset in spec['datasets']:
    data=con.execute(''.join(dataset['queryLines'])).df()
    assert len(data)>0,dataset['name']
    schemas[dataset['name']]=set(data.columns)
    if dataset['name'].endswith('_daily'):assert len(data)==225 and pd.api.types.is_datetime64_any_dtype(data.run_date)
    if dataset['name'].endswith('_performance'):assert len(data)==9
    if dataset['name'].endswith('_latest'):assert len(data)==3
assert len(schemas)==10
filter_widget=spec['pages'][0]['layout'][0]['widget']
assert {q['query']['datasetName'] for q in filter_widget['queries']} == set(schemas)
assert {f['queryName'] for f in filter_widget['spec']['encodings']['fields']} == {q['name'] for q in filter_widget['queries']}
assert filter_widget['spec']['selection']['defaultSelection']['values']['values'] == [{'value':'Intake A'}]
names=[]
for page in spec['pages']:
    for item in page['layout']:
        w=item['widget']
        assert ('spec' in w) != ('multilineTextboxSpec' in w)
        if 'spec' in w:
            assert w['spec']['version'] == (2 if w['spec']['widgetType'] in ('table','counter','filter-single-select') else 3)
            if page['pageType'] == 'PAGE_TYPE_CANVAS':
                fields={f['name'] for q in w['queries'] for f in q['query']['fields']}
                def check_encoding(value):
                    if isinstance(value,dict):
                        if 'fieldName' in value: assert value['fieldName'] in fields
                        for child in value.values(): check_encoding(child)
                    elif isinstance(value,list):
                        for child in value: check_encoding(child)
                check_encoding(w['spec']['encodings'])
        names.append(w['name']);pos=item['position']
        assert pos['x']>=0 and pos['x']+pos['width']<=12
        for q in w.get('queries',[]):
            for field in q['query'].get('fields',[]):
                assert field['name'] in schemas[q['query']['datasetName']],(w['name'],field)
        for other in page['layout']:
            if item is other:continue
            p2=other['position']
            assert pos['x']+pos['width']<=p2['x'] or p2['x']+p2['width']<=pos['x'] or pos['y']+pos['height']<=p2['y'] or p2['y']+p2['height']<=pos['y']
assert len(set(names))==len(names)
try:make_dashboard('bad;DROP TABLE x')
except ValueError:pass
else:raise AssertionError('Invalid schema accepted')
print('PASS: drift math, missing-data/threshold boundaries, separate scenarios, frozen predictions, three simulated Delta MERGEs, all native and example SQL queries in DuckDB, complete filter binding, widget encodings and layout. Local structural checks do not certify Databricks import/rendering.')
