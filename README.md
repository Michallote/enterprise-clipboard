# enterprise-clipboard
This is a glorified clipboard

![image](https://github.com/Michallote/enterprise-clipboard/assets/74160122/cfc0d250-1ce0-4549-849d-3ee074322e43)

![image](https://github.com/Michallote/enterprise-clipboard/assets/74160122/6cead6e6-f487-4b09-9b06-0d8d38fe5d7c)

![image](https://github.com/Michallote/enterprise-clipboard/assets/74160122/060decef-4bdb-42cd-9d7f-b5721e4df272)

![image](https://github.com/Michallote/enterprise-clipboard/assets/74160122/16c239b4-accd-4909-b0d5-a0545df1cf0a)

![image](https://github.com/Michallote/enterprise-clipboard/assets/74160122/39fa8bf1-3aea-4c35-b098-238d685d15a6)

![image](https://github.com/Michallote/enterprise-clipboard/assets/74160122/985b5e61-ffd9-4740-9565-b34cc6e5fbd9)


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
