# enterprise-clipboard

```python
import argparse
import importlib
import inspect
import json


def import_function(func_path):
    """
    Import a function dynamically from a module.
    :param func_path: str, in the format "module:function"
    :return: function object
    """
    try:
        module_name, function_name = func_path.split(":")
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not load function '{func_path}': {e}")


def parse_function_args(func, cli_args):
    """
    Map CLI arguments to function arguments dynamically.
    :param func: function object
    :param cli_args: dict of CLI arguments
    :return: dict of valid function arguments
    """
    sig = inspect.signature(func)
    parsed_args = {}

    for param in sig.parameters.values():
        name = param.name
        if name in cli_args:
            arg_value = cli_args[name]
            if param.annotation in [int, float, bool] and arg_value is not None:
                # Convert to the correct type
                parsed_args[name] = param.annotation(arg_value)
            else:
                parsed_args[name] = arg_value

    return parsed_args


def cli():
    parser = argparse.ArgumentParser(description="Dynamically call a function and process its result.")
    
    parser.add_argument(
        "function",
        type=str,
        help="Function to call in the format 'module:function'"
    )
    
    parser.add_argument(
        "--args",
        type=str,
        help="JSON formatted string with function arguments (optional). Example: '{\"name\": \"me\", \"generate_email\": false}'"
    )
    
    args = parser.parse_args()

    # Import and call the function
    func = import_function(args.function)

    # Parse JSON arguments if provided
    func_args = json.loads(args.args) if args.args else {}

    # Validate and execute function
    valid_args = parse_function_args(func, func_args)
    result = func(**valid_args)

    # Print or pass to another function
    print(f"Function Output: {result}")
    return result


if __name__ == "__main__":
    cli()

```


```python
# List of MLflow model flavors
FLAVORS = [
    "catboost",
    "fastai",
    "gluon",
    "h2o",
    "keras",
    "lightgbm",
    "onnx",
    "pyfunc",
    "pytorch",
    "sklearn",
    "spacy",
    "spark",
    "statsmodels",
    "tensorflow",
    "xgboost",
    "paddle",
    "prophet",
    "pmdarima",
]


# Helper function to create a lazy-loadable function
def make_lazy_function(flavor, func_name):
    def lazy_func(*args, **kwargs):
        # Import the module dynamically
        module = importlib.import_module(f"mlflow.{flavor}")
        # Get the function from the module
        func = getattr(module, func_name)
        # Call the function with the provided arguments
        return func(*args, **kwargs)

    return lazy_func


# Create dictionaries with lazy-loadable functions
LOAD_FUNCTIONS = {
    flavor: make_lazy_function(flavor, "load_model") for flavor in FLAVORS
}

LOG_FUNCTIONS = {flavor: make_lazy_function(flavor, "log_model") for flavor in FLAVORS}
```

```python
import os

class FileHandler:
    def __init__(self, file_path, content):
        self.file_path = file_path
        # Create the file when the object is created
        with open(self.file_path, 'w') as f:
            f.write(content)
        print(f"File created: {self.file_path}")
    
    def __del__(self):
        # Delete the file when the object is about to be garbage collected
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"File deleted: {self.file_path}")

# Example usage
file_handler = FileHandler("example.txt", "This is some content.")
# The file will be deleted automatically when the object is garbage collected
```
```
import pickle
import tempfile
import os

class MyModel:
    def __init__(self, data, model):
        self.data = data
        self.model = model  # This model has .save_model() and .load_model()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Use a temporary file to save the model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.model.save_model(tmp.name)
            tmp.seek(0)
            state['model'] = tmp.read()  # Read the binary content of the file
        os.unlink(tmp.name)  # Clean up the temp file
        return state

    def __setstate__(self, state):
        # Write the binary data to a temporary file and load the model from it
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(state['model'])
            tmp.seek(0)
            model = SomeModelClass()  # Instantiate the model class
            model.load_model(tmp.name)
            state['model'] = model
        os.unlink(tmp.name)  # Clean up the temp file
        self.__dict__.update(state)

# Example usage
data = 'some data'
model = SomeModelClass()  # This class would have .save_model() and .load_model()
obj = MyModel(data, model)

# Pickle the object
with open('my_object.pkl', 'wb') as f:
    pickle.dump(obj, f)

# Unpickle the object
with open('my_object.pkl', 'rb') as f:
    loaded_obj = pickle.load(f)

```

```
import pickle
import io

class MyModel:
    def __init__(self, data, model):
        self.data = data
        self.model = model  # Assume this attribute has .save_model() and .load_model()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model to a BytesIO stream
        model_stream = io.BytesIO()
        self.model.save_model(model_stream)
        model_stream.seek(0)  # Important: move to the start of the stream after writing
        state['model'] = model_stream.getvalue()  # Save the byte data
        return state

    def __setstate__(self, state):
        # Load the model from the byte data
        model_stream = io.BytesIO(state['model'])
        model = SomeModelClass()  # Assuming you have a way to instantiate it
        model.load_model(model_stream)
        state['model'] = model
        self.__dict__.update(state)

# Example usage
data = 'some data'
model = SomeModelClass()  # This class would have .save_model() and .load_model()
obj = MyModel(data, model)

# Pickle the object
with open('my_object.pkl', 'wb') as f:
    pickle.dump(obj, f)

# Unpickle the object
with open('my_object.pkl', 'rb') as f:
    loaded_obj = pickle.load(f)

```


```
import polars as pl

def sample_groups(df: pl.DataFrame, group_col: str, n: int) -> pl.DataFrame:
    """
    Take `n` samples from each group in a specified categorical column of a Polars DataFrame.

    Parameters:
    df (pl.DataFrame): The input Polars DataFrame.
    group_col (str): The column name to group by.
    n (int): The number of samples to take from each group.

    Returns:
    pl.DataFrame: A new DataFrame with `n` samples from each group.
    """
    sampled_dfs = []
    
    # Get unique groups in the group_col
    groups = df[group_col].unique()
    
    for group in groups:
        # Filter the DataFrame by the current group
        group_df = df.filter(pl.col(group_col) == group)
        
        # Sample n rows from the current group
        sampled_group_df = group_df.sample(n, with_replacement=True)
        
        # Collect the sampled DataFrame
        sampled_dfs.append(sampled_group_df)
    
    # Concatenate all the sampled DataFrames
    sampled_df = pl.concat(sampled_dfs)
    
    return sampled_df

# Example usage:
df = pl.DataFrame({
    "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "value": [1, 2, 3, 4, 5, 6, 7, 8, 9]
})

# Take 2 samples from each group in the "category" column
sampled_df = sample_groups(df, "category", 2)
print(sampled_df)
```

```python
import pytest
import polars as pl
from google.cloud import bigquery
from unittest.mock import patch, MagicMock

def test_execute_sql_file():
    # Mocking the BigQuery client
    with patch('google.cloud.bigquery.Client', return_value=MagicMock()) as mock_client:
        # Mocking the read_file_contents function
        with patch('your_module.read_file_contents', return_value="SELECT * FROM table WHERE date >= '{start_date}'") as mock_read_file:
            # Mocking the query_polars_df function
            with patch('your_module.query_polars_df', return_value=pl.DataFrame({"column1": [1, 2, 3]})) as mock_query:
                from your_module import execute_sql_file
                
                # Call the function with the SQL file and kwargs
                df = execute_sql_file("query.sql", start_date='2024-01-01')
                
                # Check if the BigQuery client was instantiated
                mock_client.assert_called_once_with(project=PROJECT_ID)
                
                # Check if the SQL file was read
                mock_read_file.assert_called_once_with("query.sql")
                
                # Check if the query was executed
                mock_query.assert_called_once_with("SELECT * FROM table WHERE date >= '2024-01-01'", client=mock_client.return_value)
                
                # Check if the returned DataFrame is correct
                assert df.shape == (3, 1)
                assert df.columns == ["column1"]
                assert df["column1"].to_list() == [1, 2, 3]
```

```markdown
Process Overview
Calculating RFM (Recency, Frequency, Monetary value)

You calculate RFM values for your customer database over a specified period.
Using Models for Prediction

You use the Beta-Geometric/Negative Binomial Distribution (BG/NBD) model to predict customer churn.
You use the Gamma-Gamma model to predict monetary value for the next X amount of time (e.g., 3 months).
Testing Predictions

You test your model predictions by comparing them to actual values for the past 3 months.
You obtain data for two periods:
The actual data for the past 3 months.
Data for the year before that (to serve as a training or reference period).
Updating Models for Production

After validation, you update your models with the latest data and deploy them for production use.
Addressing the Concerns
Validation Process

Historical Validation: It's correct to validate your models using historical data. For example, if you're in January 2024, you could validate your models using data from January 2023 to December 2023 to predict the period from October 2023 to December 2023.
Real-time Validation: Once validated, you update your models with the latest data for production use. However, these models are not immediately validated with real future data since that data hasn't arrived yet.
Model Retraining and Updating

It is standard practice to retrain models with the most recent data to ensure they capture the latest trends and behaviors.
These updated models, although trained on the latest data, will only be validated once the future period has elapsed and you can compare predictions against actual data.
Best Practices for Training and Validation
Backtesting

Perform backtesting where you use historical data to simulate predictions and compare them against actual outcomes. This helps in validating the model's performance.
Example: Use data from January 2022 to December 2022 to predict the period from October 2022 to December 2022, then compare against actual data for those months.
Rolling Window Validation

Use a rolling window approach to continually validate your model. For example, validate the model for multiple 3-month periods in the past, not just the most recent one.
This approach ensures the model's robustness over different periods and reduces the risk of overfitting to a particular period.
Continuous Monitoring

Once the models are in production, continuously monitor their performance and periodically validate them as new data becomes available.
Set up a mechanism to automatically retrain and validate models at regular intervals (e.g., monthly or quarterly).
```

```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="%(levelname)s (%(asctime)s): %(message)s (func: %(funcName)s line: %(lineno)d [%(filename)s])",
    datefmt="%H:%M:%S",
)
# TODO move logging files to a log folder in root

# Print DEBUG level to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

logger.debug(f"Print hi from crypto_labelling {__name__=}")


class ColorFormatter(logging.Formatter):
    # Define the color codes
    COLORS = {
        "DEBUG": "\033[34m",
        "INFO": "\033[0m",  # White
        "FINISHED": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
        "METRICS_LEVEL": "\033[36m",  # Purple,
        "PARAMS_LEVEL": "\033[95m",
        "ARTIFACT_LEVEL": "\033[35m",
    }
    RESET = "\033[0m"  # Reset code

    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        color = self.COLORS.get(levelname, self.RESET)
        return f"{color}{message}{self.RESET}"

formatter = ColorFormatter(
        fmt="%(levelname)s (%(asctime)s): %(message)s (func: %(funcName)s line: %(lineno)d [%(filename)s])",
        datefmt="%H:%M:%S",
    )

```

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "snakeviz",
# ]
# ///
import argparse
import cProfile
import os
import subprocess
import threading
import time

import psutil


def log_resource_usage(interval=1, output_file="resource_usage.log"):
    """Logs the CPU usage per core and memory usage in both percentage and MB every `interval` seconds."""
    with open(output_file, "w") as f:

        core_columns = "\t".join(
            map(lambda i: f"Core {i}[%]", range(psutil.cpu_count()))
        )
        f.write(f"Total CPU[%]\tMemory[%]\tMemory[MB]\t{core_columns}\tTime\n")
        while True:
            # Overall CPU and memory usage
            total_cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_usage_mb = memory.used / (1024**2)  # Convert to MB

            # Per-core CPU usage
            per_core_cpu_usage = psutil.cpu_percent(percpu=True)
            per_core_cpu_str = "\t".join(map(str, per_core_cpu_usage))

            # Current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Log the data to file
            f.write(
                f"{total_cpu_usage}\t{memory_usage_percent}\t{memory_usage_mb:.2f}\t{per_core_cpu_str}\t{timestamp}\n"
            )
            f.flush()
            time.sleep(interval)


def profile_program(script, profiler_file="output.pstats"):
    """Runs the specified program with cProfile."""
    # cProfile.run(f"exec(open('{file}').read())", profiler_file)
    subprocess.run(
        ["python", "-m", "cProfile", "-o", profiler_file, script], check=True
    )


if __name__ == "__main__":
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(
        description="Profile a Python script and log resource usage."
    )
    parser.add_argument("script", help="The Python script to execute and profile.")
    parser.add_argument(
        "--profiler_file",
        default="logs/output.pstats",
        help="File to save profiler results.",
    )
    parser.add_argument(
        "--resource_file",
        default="logs/resource_usage.log",
        help="File to log CPU and memory usage.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval (in seconds) between resource usage logs.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Start resource usage logging in a separate thread
    resource_thread = threading.Thread(
        target=log_resource_usage,
        kwargs=dict(interval=args.interval, output_file=args.resource_file),
    )
    resource_thread.daemon = True  # Daemonize thread to exit with the program
    resource_thread.start()

    # Profile the main program
    profile_program(args.script, args.profiler_file)

    print(
        f"Finalized execution to visualize results in interactive window run:\nuvx snakeviz {args.profiler_file}"
    )
```

```python
# Target for building the Docker image
.PHONY: build
build:
    @echo "Fetching environment variables from Python script..."
    IMAGE_NAME=$$(python3 -c 'from env_vars import IMAGE_NAME; print(IMAGE_NAME)') \
    ENV_VALUE=$$(python3 -c 'from env_vars import ENV_VALUE; print(ENV_VALUE)') \
    DOCKER_BUILDKIT=1 docker build --no-cache --progress=plain -t $$IMAGE_NAME . -f Dockerfile --build-arg VAR_!=$$ENV_VALUE

# Target for cleaning up
.PHONY: clean
clean:
    @echo "Cleaning Docker build artifacts..."
    docker system prune -f
```

```python
import atexit
import json
import logging
import os
import pickle
from posixpath import splitext
from typing import Any, Callable, Optional

import mlflow
import mlflow.tracking.client
import pandas as pd
import plotly.graph_objects as go
import yaml
from mlflow.entities.experiment import Experiment
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.run import Run
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel
from mlflow.tracking.client import MlflowClient

from consts import EXPERIMENT_TAGS
from src.utils.search_file import search_file

logger = logging.getLogger("models")


LOAD_FUNCTIONS = {
    "catboost": mlflow.catboost.load_model,
    "fastai": mlflow.fastai.load_model,
    "gluon": mlflow.gluon.load_model,
    "h2o": mlflow.h2o.load_model,
    "keras": mlflow.keras.load_model,
    "lightgbm": mlflow.lightgbm.load_model,
    "onnx": mlflow.onnx.load_model,
    "pyfunc": mlflow.pyfunc.load_model,
    "pytorch": mlflow.pytorch.load_model,
    "sklearn": mlflow.sklearn.load_model,
    "spacy": mlflow.spacy.load_model,
    "spark": mlflow.spark.load_model,
    "statsmodels": mlflow.statsmodels.load_model,
    "tensorflow": mlflow.tensorflow.load_model,
    "xgboost": mlflow.xgboost.load_model,
    "paddle": mlflow.paddle.load_model,
    "prophet": mlflow.prophet.load_model,
    "pmdarima": mlflow.pmdarima.load_model,
}

LOG_FUNCTIONS = {
    "catboost": mlflow.catboost.log_model,
    "fastai": mlflow.fastai.log_model,
    "gluon": mlflow.gluon.log_model,
    "h2o": mlflow.h2o.log_model,
    "keras": mlflow.keras.log_model,
    "lightgbm": mlflow.lightgbm.log_model,
    "onnx": mlflow.onnx.log_model,
    "pyfunc": mlflow.pyfunc.log_model,
    "pytorch": mlflow.pytorch.log_model,
    "sklearn": mlflow.sklearn.log_model,
    "spacy": mlflow.spacy.log_model,
    "spark": mlflow.spark.log_model,
    "statsmodels": mlflow.statsmodels.log_model,
    "tensorflow": mlflow.tensorflow.log_model,
    "xgboost": mlflow.xgboost.log_model,
    "paddle": mlflow.paddle.log_model,
    "prophet": mlflow.prophet.log_model,
    "pmdarima": mlflow.pmdarima.log_model,
}


@search_file
def save_pickle(
    model: Any,
    local_model_path: str,
):
    with open(local_model_path, "wb") as f:
        pickle.dump(model, f)


class InvalidModelName(Exception):
    """Invalid Model Name Exception"""


class ModelFlavorNotSupported(Exception):
    """MlFlow Model Flavor Not Supported"""


class MlFlowLogger:
    """
    A class used to interact with MLflow's tracking and model registry components.
    """

    load_functions: dict[str, Callable] = LOAD_FUNCTIONS
    log_functions: dict[str, Callable] = LOG_FUNCTIONS

    def __init__(self):
        mlflow.set_tracking_uri()
        self.client = MlflowClient()  # Initialize client

    def start_run(self, experiment_name: str, run_name: str | None = None):
        experiment = self.get_or_create_experiment(experiment_name)
        experiment_id = experiment.experiment_id
        self.run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        self.run_id = self.run.info.run_uuid
        logger.info(f"Starting mlflow run with run_id={self.run_id}")
        atexit.register(self.end_run)
        return self.run_id

    def end_run(self):
        mlflow.end_run()

    def get_experiment_id(self, experiment_name: str) -> str:
        """
        Retrieves the ID of an experiment given its name.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.

        Returns
        -------
        str
            The ID of the experiment.
        """
        retrieved_exp_id = self.get_experiment(experiment_name)

        return retrieved_exp_id.experiment_id

    def get_experiment(self, experiment_name: str) -> Experiment:
        """
        Retrieves an experiment given its name.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.

        Returns
        -------
        Experiment
            The retrieved experiment.
        """

        client = self.client
        return client.get_experiment_by_name(experiment_name)

    def get_experiment_runs(self, experiment_name: str) -> list[Run]:
        """
        Retrieves all runs of an experiment given its name.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.

        Returns
        -------
        list[Run]
            A list of all runs of the experiment.
        """

        client = self.client
        retrieved_exp_id = self.get_experiment_id(experiment_name)
        runs = client.search_runs(retrieved_exp_id)
        return runs

    def get_latest_run(self, experiment_name: str) -> Run:
        """
        Retrieves the latest run of an experiment given its name.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.

        Returns
        -------
        Run
            The latest run of the experiment.
        """

        retrieved_exp_id = self.get_experiment_id(experiment_name)

        latest_run = self.client.search_runs(
            retrieved_exp_id, order_by=["attribute.start_time DESC"], max_results=1
        )

        return latest_run[0]

    def register_latest_model(
        self,
        experiment_name: str,
        model_registry_name: str,
        mlflow_pyfunc_model_path: Optional[str] = None,
    ):
        """
        Registers the latest model of an experiment.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.
        model_registry_name : str
            The name to register the model under in the model registry.
        mlflow_pyfunc_model_path : str
            The path to the model in the MLflow format.

        Returns
        -------
        ModelVersion
            The registered model version.
        """

        latest_run = self.get_latest_run(experiment_name)
        run_id = latest_run.info.run_id

        if mlflow_pyfunc_model_path is None:
            run_data = latest_run.data.tags
            [contents] = json.loads(run_data["mlflow.log-model.history"])
            mlflow_pyfunc_model_path = contents["artifact_path"]

        model_version = mlflow.register_model(
            f"runs:/{run_id}/{mlflow_pyfunc_model_path}", model_registry_name
        )
        return model_version

    def get_latest_model(
        self, model_registry_name: str, stages: str | list[str] | None = None
    ) -> ModelVersion:
        """
        Retrieves the latest model from the model registry.

        Parameters
        ----------
        model_registry_name : str
            The name of the model in the model registry.
        stages : str | list[str] | None, optional
            The stages to consider when retrieving the latest model, by default None.

        Returns
        -------
        ModelVersion
            The latest model version.
        """

        if stages is None:
            stages = ["Staging", "Production"]  # ['None']
        elif isinstance(stages, str):
            stages = [stages]

        latest_models = self.client.get_latest_versions(
            model_registry_name, stages=stages
        )

        latest_models = list(
            filter(lambda model: model.name == model_registry_name, latest_models)
        )

        if len(latest_models) == 0:
            raise Exception(
                f"No model found matching the description. {model_registry_name=}"
            )

        return max(latest_models, key=lambda x: int(x.version))

    def create_new_experiment(
        self,
        experiment_name: str,
        experiment_description: str,
        experiment_tags: dict[str, str],
    ) -> str:
        """Create an experiment.

        Parameters
        ----------
        experiment_name : str
            The experiment name. Must be unique.
        experiment_description : str
            The experiment description to be displayed in the ui tab
        experiment_tags : dict[str, str]
            A dictionary of key-value pairs that are converted into
                                :py:class:`mlflow.entities.ExperimentTag` objects, set as
                                experiment tags upon experiment creation.

        Returns
        -------
        str
            String as an integer ID of the created experiment.

        Examples
        --------

        .. code-block:: python
            :caption: Example

            # Create a new experiment, will fail if it already exists.

            experiment_name = "cltv-lifetime-models-nested-runs"

            experiment_description = (
                "Project SAMS-CLTV Segmentation and Lifetime Model predictions."
                "This project segments the memberships into customer segments and associate_types"
            )

            # Provide searchable tags that define characteristics of the Runs that will be in this Experiment
            experiment_tags = {
                "project_name": "cltv-rfm-lifetime-models",
                "store": "SAMS",
                "model_groups": "associate_types",
                "team": "wmt-mx-dl-iaml",
                "project_quarter": "Q3-2023",
                "mlflow.note.content": experiment_description,
            }
        """

        retrieved_experiment = self.get_experiment(experiment_name)

        if retrieved_experiment is not None:
            logger.warning(
                f"Experiment '{experiment_name}' already exists: {retrieved_experiment}"
            )
            return retrieved_experiment.experiment_id

        experiment_tags["mlflow.note.content"] = experiment_description

        cltv_experiment = self.client.create_experiment(
            name=experiment_name, tags=experiment_tags
        )

        return cltv_experiment

    def get_or_create_experiment(
        self, experiment_name: str, tags: Optional[dict] = None
    ) -> Experiment:
        """
        Retrieves an experiment if it exists, otherwise creates a new one.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.

        Returns
        -------
        Experiment
            The retrieved or created experiment.
        """

        retrieved_experiment = self.get_experiment(experiment_name)

        if retrieved_experiment is not None:
            return retrieved_experiment

        logger.warning(
            f"'{experiment_name}' not found. Creating new experiment. No metadata was provided. Using defaults."
        )

        if tags is None:
            tags = EXPERIMENT_TAGS

        self.client.create_experiment(name=experiment_name, tags=tags)

        return self.get_experiment(experiment_name)

    def log_model(
        self, model: Any, artifact_path: str, flavor: str, **kwargs
    ) -> ModelInfo:

        if flavor not in self.log_functions:
            raise ModelFlavorNotSupported(f"Model flavour {flavor} is not valid.")

        model_info = self.log_functions[flavor](model, artifact_path, **kwargs)
        return model_info

    def log_pyfunc_model(
        self,
        model: Any,
        local_model_path: str,
        mlflow_pyfunc_model_path: str,
        python_model: PythonModel = PythonModel,
        code_path: list[str] | tuple[str, ...] = ("./src", "./config", "consts.py"),
        **kwargs,
    ):
        if "signature" in kwargs:
            signature = kwargs.pop("signature")
        else:
            signature = getattr(python_model, "signature")

        code_path = list(code_path)
        local_model_path = self._assert_local_model_store(
            model, local_model_path, mlflow_pyfunc_model_path
        )
        artifacts = {"model_path": local_model_path}
        model_info = mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=python_model(),
            code_path=code_path,
            artifacts=artifacts,
            signature=signature,
            **kwargs,
        )

        return model_info

    def _assert_local_model_store(
        self, model: Any, local_model_path: str, mlflow_pyfunc_model_path: Optional[str]
    ):
        """
        Asserts that the local model stored is valid.

        Parameters
        ----------
        model : Any
            The model to check.
        local_model_path : str
            The local path to the model.
        mlflow_pyfunc_model_path : Optional[str]
            The path to the model in the MLflow format.

        Returns
        -------
        str
            The validated local model path.
        """

        assert not isinstance(model, str)

        # determine if local_model_path is a directory and that the filename can be constructed
        if not local_model_path.endswith(".pkl"):
            if isinstance(mlflow_pyfunc_model_path, str):
                local_model_path = f"{local_model_path}/{mlflow_pyfunc_model_path}.pkl"
            else:
                logger.error(
                    "When passing local_model_path as a directory, "
                    "mlflow_pyfunc_model_path must be provided too."
                )
                raise InvalidModelName(
                    f"Could not reconstruct the model name from {local_model_path=} & {mlflow_pyfunc_model_path=}"
                )

        # Handle cases when model is not provided, it must be saved already in memory
        if model is None:
            if os.path.isfile(local_model_path) and local_model_path.endswith(".pkl"):
                return local_model_path

            logger.error(
                "If model argument is not provided it must be already saved to memory."
            )
            raise FileNotFoundError(f"Model not found in '{local_model_path}'")

        save_pickle(model=model, local_model_path=local_model_path)

        logger.info(f"Model saved to {local_model_path}. ({model=})")

        return local_model_path

    def register_model(
        self, run_id: str, mlflow_pyfunc_model_path: str, model_registry_name: str
    ):
        """
        Registers a model to the model registry.

        Parameters
        ----------
        run_id : str
            The ID of the run.
        mlflow_pyfunc_model_path : str
            The path to the model in the MLflow format.
        model_registry_name : str
            The name to register the model under in the model registry.

        Returns
        -------
        ModelVersion
            The registered model version.
        """

        model_version = mlflow.register_model(
            f"runs:/{run_id}/{mlflow_pyfunc_model_path}", model_registry_name
        )

        return model_version

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]):
        mlflow.log_params(params)

    def set_tags(self, tags: dict[str, Any]):
        mlflow.set_tags(tags)

    def register_model_to_stage(
        self,
        run_id: str,
        model_registry_name: str,
        mlflow_pyfunc_model_path: str,
        stage: None | str = None,
    ):
        """
        Registers a model to the MLflow Model Registry and optionally sets its stage.

        Parameters
        ----------
        run_id : str
            The ID of the run that produced the model.
        model_registry_name : str
            The name to register the model under in the model registry.
        mlflow_pyfunc_model_path : str
            The path to the model in the MLflow format.
        stage : str, optional
            The stage to set for the model in the model registry. If this is `None`, the function will not set a stage for the model.

        Returns
        -------
        mlflow.entities.model_registry.ModelVersion
            The registered model version.

        Raises
        ------
        MlflowException
            If an error occurs while registering the model or transitioning its stage.
        """

        model_version = self.register_model(
            run_id, mlflow_pyfunc_model_path, model_registry_name
        )
        if stage is not None:
            model_version = self.promote_model_to_stage(
                model_version.name,
                model_version.version,
                stage=stage,
            )
        return model_version

    def promote_model_to_stage(self, model_name: str, model_version: str, stage: str):

        archive_existing_versions = stage in ["Production", "Staging"]
        model_version = self.client.transition_model_version_stage(
            model_name,
            model_version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )
        return model_version

    def load_latest_model(
        self, model_registry_name: str, model_flavour: str = "pyfunc"
    ) -> mlflow.pyfunc.PyFuncModel:

        latest_model = self.get_latest_model(model_registry_name)

        loaded_model = self.load_functions[model_flavour](
            f"models:/{latest_model.name}/{latest_model.version}"
        )

        return loaded_model

    def load_model_version(self, model_registry_name: str, model_version: str):

        model_info = self.client.get_model_version(model_registry_name, model_version)

        loaded_model = mlflow.pyfunc.load_model(
            f"models:/{model_info.name}/{model_info.version}"
        )

        return loaded_model

    def load_model_from_run(self, run_id: str) -> mlflow.pyfunc.PyFuncModel:

        run_obj = self.client.get_run(run_id)

        tags = run_obj.data.tags

        [model_history] = json.loads(tags["mlflow.log-model.history"])
        artifact_path = model_history["artifact_path"]
        logged_model = f"runs:/{run_id}/{artifact_path}"

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        return loaded_model

    def log_table(self, df: pd.DataFrame, table_name: str):

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df.columns),
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[df[col] for col in df],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )

        if not table_name.endswith(".html"):
            root, ext = splitext(table_name)
            table_name = f"{root}.html"

        mlflow.log_figure(fig, table_name)

    def log_parquet(self, df: pd.DataFrame, local_path: str, artifact_path: str):

        df.to_parquet(local_path, index=False)
        mlflow.log_artifact(local_path, artifact_path)

    def log_yaml(self, data: Any, local_path: str, artifact_path: str):

        with open(local_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False)

        mlflow.log_artifact(local_path, artifact_path)


if __name__ == "__main__":
    pass

```
