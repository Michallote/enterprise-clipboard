# enterprise-clipboard

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
