"""Exercise the optional MLflow cell against the 2.x and 3.x API shapes and a tracking outage.

These stand-ins check which API argument and returned model URI the notebook uses;
they do not claim that a Databricks workspace accepted any model.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["SPC_DEMO_LOCAL_TEST"] = "1"
os.environ["SPC_DEMO_MODEL_DIR"] = bundle_dir = tempfile.mkdtemp(prefix="spc-mlflow-contract-")
notebook = Path(__file__).resolve().parents[1] / "notebooks" / "SPC_ML_Demo.py"
with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(notebook))

mlflow_cell = notebook.read_text().split(
    "# DBTITLE 1,Log the result in MLflow when available\n", 1
)[1].split("# COMMAND ----------", 1)[0]


def run_cell(mlflow, **settings):
    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.sklearn"] = mlflow.sklearn
    output = io.StringIO()
    try:
        namespace = {**state, "LOCAL_TEST": False, **settings}
        with contextlib.redirect_stdout(output):
            exec(mlflow_cell, namespace)
    finally:
        del sys.modules["mlflow.sklearn"]
        del sys.modules["mlflow"]
    return namespace, output.getvalue()


def stand_in(api_version, calls, start_run=None):
    mlflow = ModuleType("mlflow")
    mlflow.__path__ = []
    mlflow.sklearn = ModuleType("mlflow.sklearn")
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
    def working_run(*, run_name):
        yield SimpleNamespace(info=SimpleNamespace(run_id="example-run"))

    mlflow.start_run = start_run or working_run
    mlflow.log_params = lambda params: None
    mlflow.log_metrics = lambda metrics: None
    mlflow.set_registry_uri = lambda uri: calls.append(uri)
    mlflow.MlflowClient = lambda: SimpleNamespace(
        set_model_version_tag=lambda *args: calls.append(("tag", *args)),
        set_registered_model_alias=lambda *args: calls.append(("alias", *args)),
    )
    mlflow.register_model = lambda uri, name: (calls.append((uri, name)) or SimpleNamespace(version=1))
    return mlflow


def saved_manifest():
    return json.loads((Path(bundle_dir) / "manifest.json").read_text())


for api_version in (2, 3):
    calls = []
    namespace, _ = run_cell(stand_in(api_version, calls), UC_MODEL_NAME="demo.schema.spc_rule_classifier")
    assert calls[:4] == ["model", "count_forecast", "logistic_classifier", "isolation_forest"]
    assert calls[4] == "databricks-uc"
    assert calls[5] == (namespace["classifier_model_info"].model_uri, "demo.schema.spc_rule_classifier")
    aliases = [c for c in calls if isinstance(c, tuple) and c[0] == "alias"]
    assert len(aliases) == 4 and all(c[2] == "demo_candidate" for c in aliases)
    tags = [c for c in calls if isinstance(c, tuple) and c[0] == "tag"]
    assert len(tags) == 4 and all(c[3:] == ("demo_only", "true") for c in tags)
    # A successful run records its forecast URI in the portable bundle for the scorer's MLflow backend.
    assert namespace["MLFLOW_STATUS"] == "logged"
    assert namespace["saved_forecast_manifest"]["model_uri"] == namespace["forecast_model_info"].model_uri
    assert saved_manifest()["model_uri"] == namespace["forecast_model_info"].model_uri


# A later model failure is not reported as successful logging, even when the forecast URI exists.
calls = []
partial = stand_in(3, calls)
working_log_model = partial.sklearn.log_model
def fail_after_forecast(sk_model, *, name, input_example):
    if name == "logistic_classifier":
        raise RuntimeError("later model logging failed")
    return working_log_model(sk_model, name=name, input_example=input_example)
partial.sklearn.log_model = fail_after_forecast
namespace, output = run_cell(partial, UC_MODEL_NAME="")
assert namespace["MLFLOW_STATUS"].startswith("failed: RuntimeError")
assert namespace["forecast_model_info"] is not None and namespace["logistic_model_info"] is None
assert saved_manifest()["model_uri"] == namespace["forecast_model_info"].model_uri
assert "MLflow logging FAILED" in output

# A tracking outage is reported; the bundle is still saved and later cells keep their inputs.
@contextlib.contextmanager
def unavailable_run(*, run_name):
    raise RuntimeError("tracking server unavailable")
    yield


calls = []
namespace, output = run_cell(stand_in(3, calls, unavailable_run), UC_MODEL_NAME="")
assert namespace["MLFLOW_STATUS"].startswith("failed: RuntimeError"), namespace["MLFLOW_STATUS"]
assert "MLflow logging FAILED" in output and namespace["forecast_model_info"] is None
assert namespace["saved_forecast_manifest"] is not None and namespace["saved_forecast_manifest"]["model_uri"] is None
assert saved_manifest()["model_uri"] is None and not calls

# ENABLE_MLFLOW = False skips MLflow without affecting the bundle or Delta settings.
calls = []
namespace, _ = run_cell(stand_in(3, calls), UC_MODEL_NAME="", ENABLE_MLFLOW=False)
assert namespace["MLFLOW_STATUS"] == "disabled" and not calls and namespace["saved_forecast_manifest"] is not None

print("MLflow 2.x/3.x logging and registration contracts, URI recording, partial failures, outage reporting and opt-out passed (simulated APIs).")
