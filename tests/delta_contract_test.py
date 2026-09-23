"""Check the optional Delta cell constructs keyed, idempotent merges; no real Spark writes."""
import contextlib,io,re,runpy
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
root=Path(__file__).resolve().parents[1]
source=(root/'notebooks'/'SPC_ML_Demo.py').read_text()
cell=source.split('# DBTITLE 1,Optional managed Delta history\n',1)[1].split('# COMMAND ----------',1)[0]
# Every table binding is represented by a small fixture with its required key fields.
variables=['daily_df','signals_df','predictions_df','forecast_results_df','review_df','monitoring_df','zone_df','subgroup_df','future_forecasts_df','forecast_ledger_df','assessment_df','anomaly_df','lineage_df','upstream_profile_df','events_df','hypotheses_df','lifecycle_df','outbox_df','operator_disposition_df']
row=dict(series_id='A',run_date=pd.Timestamp('2026-01-06'),episode_id=1,office='North',model='baseline',origin_date=pd.Timestamp('2026-01-05'),target_date=pd.Timestamp('2026-01-06'),table='demo.a',path='demo.b -> demo.a',event_id='event',component='forecast')
ns={name:pd.DataFrame([row]) for name in variables}
queries=[];views=[]
class Spark:
    def createDataFrame(self,frame):return SimpleNamespace(createOrReplaceTempView=lambda name:views.append(name))
    def sql(self,query):queries.append(query)
ns.update(OUTPUT_SCHEMA='demo.spc',DATASET_ID='fixture',re=re,pd=pd,spark=Spark())
with contextlib.redirect_stdout(io.StringIO()):exec(cell,ns)
assert len(views)==19 and len(queries)==38
assert sum(q.startswith('MERGE INTO') for q in queries)==19
assert all('target.`dataset_id` = source.`dataset_id`' in q for q in queries if q.startswith('MERGE'))
assert not any('DROP ' in q or 'DELETE ' in q or 'OVERWRITE' in q for q in queries)
print('Delta contract passed: 19 keyed MERGE statements; simulated Spark only.')
