"""Exercise the optional MLflow cell against the 2.x and 3.x API shapes.

These stand-ins check which API argument and returned model URI the notebook uses;
they do not claim that a Databricks workspace accepted any model.
"""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
notebook = Path(__file__).resolve().parents[1] / "notebooks" / "SPC_ML_Demo.py"
with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(notebook))

mlflow_cell = notebook.read_text().split(
    "# DBTITLE 1,Log the result in MLflow when available\n", 1
)[1].split("# COMMAND ----------", 1)[0]

for api_version in (2, 3):
    mlflow = ModuleType("mlflow")
    mlflow.__path__ = []
    mlflow.sklearn = ModuleType("mlflow.sklearn")
    calls = []

    if api_version == 2:
        def log_model(sk_model, artifact_path, input_example):
            calls.append(artifact_path)
            return SimpleNamespace(model_uri=f"runs:/example-run/{artifact_path}")
    else:
        def log_model(sk_model, *, name, input_example):
            calls.append(name)
            return SimpleNamespace(model_uri=f"models:/example-{name}")

    mlflow.sklearn.log_model = log_model

    @contextlib.contextmanager
    def start_run(*, run_name):
        yield SimpleNamespace(info=SimpleNamespace(run_id="example-run"))

    mlflow.start_run = start_run
    mlflow.log_params = lambda params: None
    mlflow.log_metrics = lambda metrics: None
    mlflow.set_registry_uri = lambda uri: calls.append(uri)
    mlflow.MlflowClient = lambda: SimpleNamespace(
        set_model_version_tag=lambda *args: calls.append(("tag", *args)),
        set_registered_model_alias=lambda *args: calls.append(("alias", *args)),
    )
    mlflow.register_model = lambda uri, name: (calls.append((uri, name)) or SimpleNamespace(version=1))
    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.sklearn"] = mlflow.sklearn
    try:
        namespace = {**state, "UC_MODEL_NAME": "demo.schema.spc_rule_classifier"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(mlflow_cell, namespace)
    finally:
        del sys.modules["mlflow.sklearn"]
        del sys.modules["mlflow"]

    assert calls[:4] == ["model", "count_forecast", "logistic_classifier", "isolation_forest"]
    assert calls[4] == "databricks-uc"
    assert calls[5] == (namespace["classifier_model_info"].model_uri, "demo.schema.spc_rule_classifier")
    aliases = [c for c in calls if isinstance(c, tuple) and c[0] == "alias"]
    assert len(aliases) == 4 and all(c[2] == "demo_candidate" for c in aliases)
    tags = [c for c in calls if isinstance(c, tuple) and c[0] == "tag"]
    assert len(tags) == 4 and all(c[3:] == ("demo_only", "true") for c in tags)

print("MLflow 2.x and 3.x model logging/registration contract checks passed (simulated APIs).")
