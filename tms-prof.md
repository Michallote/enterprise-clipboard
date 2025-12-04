Nice, let’s turn the profiling plan into something concrete and code-y. I’ll propose:

1. **What we’re going to compute** (profiling metrics)
2. **A small profiling “API”**: a set of functions using Polars + NumPy that take your daily-aggregated `DataFrame` (`date, sales, price, total_amount`).
3. A couple of **hooks for the raw (non-aggregated) df**.

You can copy-paste these into a `profiling.py` module and adapt.

---

## 1. Profiling: what we want to measure

For the **daily aggregated df** (one row per date):

**A. Basic time-series shape**

* `n_rows`
* `start_date`, `end_date`
* `n_calendar_days` = number of days between start & end (inclusive)
* `n_unique_dates`, presence of duplicates
* `n_missing_dates`, `%_missing_dates`

**B. Sales behavior**

* `sales_mean`, `sales_std`, `sales_min`, `sales_max`, `sales_median`
* `sales_cv` (coefficient of variation = std / mean)
* `zero_sales_days`, `%_zero_sales_over_calendar_days`
* `non_zero_sales_days`, `%_non_zero_sales_over_calendar_days`

**C. Price behavior**

* `price_mean`, `price_std`, `price_min`, `price_max`, `price_median`
* `price_cv`
* optional: `corr_sales_price`

**D. Revenue / total_amount**

* `revenue_mean`, `revenue_std`, `revenue_min`, `revenue_max`, `revenue_median`

**E. Seasonality / autocorrelation (simple version)**

Using NumPy:

* Autocorrelation at lags: 1, 7, 14, 28 (and more if useful)
* Flags like `has_weekly_seasonality` if `acf(lag=7)` > threshold
* Overall “volatility” index (again `sales_cv` is good here)

**F. Sparsity / intermittency**

* `calendar_coverage = n_rows / n_calendar_days`
* If `calendar_coverage < threshold`, you know you have irregular data.
* If `zero_sales_days / n_calendar_days` is high → intermittent demand.

We’ll package all this into one `profile_daily_ts` function.

---

## 2. Code: profiling functions for the aggregated daily DataFrame

Assumptions:

* Polars is installed as `polars` (imported as `pl`)
* `date` is convertible to `pl.Date`
* Data is for a single product (or single product-category combo).
  (Later you can just group-by product and call this function per group.)

```python
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional


def _ensure_sorted_and_cast_date(
    df: pl.DataFrame,
    date_col: str = "date"
) -> pl.DataFrame:
    """
    Ensure date column is pl.Date and df is sorted by date.
    Does not mutate the original df (returns a new one).
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")
    
    out = df.with_columns(
        pl.col(date_col).cast(pl.Date)
    ).sort(date_col)
    return out


def profile_time_coverage(
    df: pl.DataFrame,
    date_col: str = "date"
) -> Dict[str, Any]:
    """
    Basic time coverage, missing dates, and duplicates.
    Assumes one row per date (but checks for duplicates).
    """
    df = _ensure_sorted_and_cast_date(df, date_col=date_col)

    if df.height == 0:
        return {
            "is_empty": True,
            "n_rows": 0
        }

    # basic info
    n_rows = df.height
    min_date = df[0, date_col]
    max_date = df[-1, date_col]

    # unique dates vs. full calendar range
    unique_dates = (
        df.select(pl.col(date_col).unique())
        .to_series()
        .sort()
    )
    n_unique_dates = unique_dates.len()

    full_calendar = pl.Series(
        name=date_col,
        values=pl.date_range(
            start=min_date,
            end=max_date,
            interval="1d",
            eager=True
        )
    )
    n_calendar_days = full_calendar.len()

    # missing dates = calendar dates not in unique_dates
    missing_mask = ~full_calendar.is_in(unique_dates)
    missing_dates = full_calendar.filter(missing_mask)
    n_missing_dates = missing_dates.len()
    pct_missing_dates = (
        float(n_missing_dates) / n_calendar_days * 100.0
        if n_calendar_days > 0 else 0.0
    )

    # duplicates (if n_rows > n_unique_dates)
    n_duplicate_dates = n_rows - n_unique_dates

    coverage = {
        "is_empty": False,
        "n_rows": n_rows,
        "start_date": min_date,
        "end_date": max_date,
        "n_calendar_days": n_calendar_days,
        "n_unique_dates": n_unique_dates,
        "n_missing_dates": n_missing_dates,
        "pct_missing_dates": pct_missing_dates,
        "missing_dates": missing_dates.to_list(),
        "n_duplicate_dates": n_duplicate_dates,
        "calendar_coverage": float(n_unique_dates) / n_calendar_days
        if n_calendar_days > 0 else 0.0,
    }

    return coverage


def profile_sales_price_revenue(
    df: pl.DataFrame,
    sales_col: str = "sales",
    price_col: str = "price",
    total_col: str = "total_amount"
) -> Dict[str, Any]:
    """
    Summary stats for sales, price, and total_amount (revenue).
    """
    required_cols = [sales_col, price_col, total_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")

    # cast numeric columns to float for safety
    numeric_df = df.select(
        [
            pl.col(sales_col).cast(pl.Float64),
            pl.col(price_col).cast(pl.Float64),
            pl.col(total_col).cast(pl.Float64),
        ]
    )

    # Stats in one shot
    stats = numeric_df.select(
        [
            pl.col(sales_col).mean().alias("sales_mean"),
            pl.col(sales_col).std().alias("sales_std"),
            pl.col(sales_col).min().alias("sales_min"),
            pl.col(sales_col).max().alias("sales_max"),
            pl.col(sales_col).median().alias("sales_median"),

            pl.col(price_col).mean().alias("price_mean"),
            pl.col(price_col).std().alias("price_std"),
            pl.col(price_col).min().alias("price_min"),
            pl.col(price_col).max().alias("price_max"),
            pl.col(price_col).median().alias("price_median"),

            pl.col(total_col).mean().alias("revenue_mean"),
            pl.col(total_col).std().alias("revenue_std"),
            pl.col(total_col).min().alias("revenue_min"),
            pl.col(total_col).max().alias("revenue_max"),
            pl.col(total_col).median().alias("revenue_median"),
        ]
    ).to_dicts()[0]

    # Coefficient of variation (handle division by zero)
    sales_mean = stats["sales_mean"]
    price_mean = stats["price_mean"]

    stats["sales_cv"] = (
        float(stats["sales_std"]) / sales_mean
        if sales_mean not in (None, 0.0) else None
    )
    stats["price_cv"] = (
        float(stats["price_std"]) / price_mean
        if price_mean not in (None, 0.0) else None
    )

    # correlation sales vs price (if variance > 0)
    # Using numpy for convenience
    sales_arr = numeric_df[sales_col].to_numpy()
    price_arr = numeric_df[price_col].to_numpy()

    if np.all(np.isfinite(sales_arr)) and np.all(np.isfinite(price_arr)):
        if np.std(sales_arr) > 0 and np.std(price_arr) > 0:
            stats["corr_sales_price"] = float(
                np.corrcoef(sales_arr, price_arr)[0, 1]
            )
        else:
            stats["corr_sales_price"] = None
    else:
        stats["corr_sales_price"] = None

    return stats


def profile_sparsity(
    df: pl.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales"
) -> Dict[str, Any]:
    """
    Measures sparsity / intermittency: zeros vs. non-zeros,
    relative to both observed rows and full calendar days.
    """
    df = _ensure_sorted_and_cast_date(df, date_col=date_col)
    coverage = profile_time_coverage(df, date_col=date_col)

    if coverage["is_empty"]:
        return {
            "zero_sales_days": 0,
            "pct_zero_sales_over_rows": 0.0,
            "pct_zero_sales_over_calendar": 0.0,
            "non_zero_sales_days": 0,
        }

    n_rows = coverage["n_rows"]
    n_calendar_days = coverage["n_calendar_days"]

    zero_sales_days = df.filter(pl.col(sales_col) == 0).height
    non_zero_sales_days = df.filter(pl.col(sales_col) > 0).height

    sparsity = {
        "zero_sales_days": zero_sales_days,
        "non_zero_sales_days": non_zero_sales_days,
        "pct_zero_sales_over_rows": (
            float(zero_sales_days) / n_rows * 100.0
            if n_rows > 0 else 0.0
        ),
        "pct_zero_sales_over_calendar": (
            float(zero_sales_days) / n_calendar_days * 100.0
            if n_calendar_days > 0 else 0.0
        ),
        "calendar_coverage": coverage["calendar_coverage"],
    }
    return sparsity


def compute_acf(
    x: np.ndarray,
    max_lag: int
) -> np.ndarray:
    """
    Simple autocorrelation function up to max_lag.
    x is a 1D numpy array.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([])

    x = x - x.mean()
    n = x.size
    # full autocorrelation via convolution
    autocorr_full = np.correlate(x, x, mode="full")
    autocorr = autocorr_full[n - 1 :]  # keep non-negative lags
    autocorr = autocorr / autocorr[0]  # normalize

    if max_lag + 1 < autocorr.size:
        return autocorr[: max_lag + 1]
    return autocorr


def profile_seasonality(
    df: pl.DataFrame,
    sales_col: str = "sales",
    max_lag: int = 60,
    lags_of_interest: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Basic seasonality diagnostics using autocorrelation.

    - Computes ACF up to max_lag
    - Reports ACF at selected lags (e.g., 1, 7, 14, 28)
    - Flags if weekly seasonality is strong
    """
    if lags_of_interest is None:
        lags_of_interest = [1, 7, 14, 28]

    if sales_col not in df.columns:
        raise ValueError(f"Column '{sales_col}' not found in DataFrame.")

    sales_arr = (
        df.select(pl.col(sales_col).cast(pl.Float64))
        .to_series()
        .to_numpy()
    )

    if sales_arr.size < 2:
        return {
            "acf": np.array([]),
            "acf_at_lags": {},
            "strong_weekly_seasonality": False,
        }

    acf_vals = compute_acf(sales_arr, max_lag=max_lag)
    acf_at_lags = {
        lag: float(acf_vals[lag]) for lag in lags_of_interest
        if lag < acf_vals.size
    }

    # crude rule: strong weekly seasonality if acf at lag=7 > 0.5
    weekly_acf = acf_at_lags.get(7, 0.0)
    strong_weekly = bool(weekly_acf > 0.5)

    return {
        "acf": acf_vals,
        "acf_at_lags": acf_at_lags,
        "strong_weekly_seasonality": strong_weekly,
    }


def profile_daily_time_series(
    df: pl.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales",
    price_col: str = "price",
    total_col: str = "total_amount",
    max_lag: int = 60
) -> Dict[str, Any]:
    """
    High-level function that returns a consolidated profiling dict
    for the daily aggregated time series.
    """
    df = _ensure_sorted_and_cast_date(df, date_col=date_col)

    time_cov = profile_time_coverage(df, date_col=date_col)
    stats = profile_sales_price_revenue(
        df,
        sales_col=sales_col,
        price_col=price_col,
        total_col=total_col,
    )
    sparsity = profile_sparsity(
        df,
        date_col=date_col,
        sales_col=sales_col,
    )
    seasonality = profile_seasonality(
        df,
        sales_col=sales_col,
        max_lag=max_lag,
    )

    profile = {
        "time_coverage": time_cov,
        "stats": stats,
        "sparsity": sparsity,
        "seasonality": seasonality,
    }
    return profile
```

### Example usage

```python
import polars as pl

# df_daily: your aggregated daily DataFrame with columns:
#   ["date", "sales", "price", "total_amount", ...]
profile = profile_daily_time_series(df_daily)

# Access pieces:
profile["time_coverage"]
profile["stats"]
profile["sparsity"]
profile["seasonality"]["acf_at_lags"]
```

---

## 3. Hooks for the raw (non-aggregated) dataframe

For the **raw df** (multiple stores/zones, same product), useful extra profiles:

* Distribution of sales by store / zone
* Number of stores active per day
* Data completeness by store (missing dates per store)
* Whether some stores systematically have missing data (data quality issue).

You can build a function like:

```python
def profile_raw_by_group(
    df_raw: pl.DataFrame,
    group_cols: list,
    date_col: str = "date",
    sales_col: str = "sales"
) -> pl.DataFrame:
    """
    Returns a per-group summary with basic metrics:
    - n_rows
    - n_unique_dates
    - total_sales
    - mean_sales
    """
    return (
        df_raw
        .with_columns(pl.col(date_col).cast(pl.Date))
        .groupby(group_cols)
        .agg(
            [
                pl.count().alias("n_rows"),
                pl.col(date_col).n_unique().alias("n_unique_dates"),
                pl.col(sales_col).sum().alias("total_sales"),
                pl.col(sales_col).mean().alias("mean_sales"),
            ]
        )
    )
```

Then you can spot problematic stores/zones quickly and decide if you trust their data when aggregating.

---

If you like this structure, next we can:

* Add **“classification flags”** (stable vs seasonal vs intermittent) based on these metrics.
* Or wire this profiling module into a loop that profiles *each product* (using `groupby("product_id")` or similar) and returns a big summary table.
