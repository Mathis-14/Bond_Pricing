"""
Utilities to download and cache US Treasury constant-maturity yields from FRED.

Key points:
----------
- We use `fredapi.Fred` with an API key stored in `.env` (FRED_API_KEY).
- We currently download the following series:
    * DGS3MO  - 3-month
    * DGS6MO  - 6-month
    * DGS1    - 1-year
    * DGS2    - 2-year
    * DGS3    - 3-year
    * DGS5    - 5-year
- The data are stored in `data/treasury_yields_3M_6M_1Y.csv` (name kept
  for backwards compatibility, even though it now holds more columns).

`load_or_fetch_treasury_yields()`:
    - Loads from the CSV cache if present and complete.
    - Otherwise calls `fetch_treasury_yields()` to download from FRED and
      overwrite the cache.

The function returns:
    df          : full time series DataFrame for all configured series
    curve_date  : latest date with data for all series
    yields_today: the cross-section of yields at `curve_date` (still in percent)
"""


import os
from pathlib import Path

from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd


DATA_DIR = Path("data")
CACHE_PATH = DATA_DIR / "treasury_yields_3M_6M_1Y.csv"

# Treasury series used throughout the project.
# Short end (bills) + 1Y–5Y notes for curve shape.
SERIES_IDS = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5"]


def load_fred_client() -> Fred:
    """Instantiate a FRED client using the API key stored in .env."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY missing in environment or .env file.")
    return Fred(api_key=api_key)


def fetch_treasury_yields() -> tuple[pd.DataFrame, pd.Timestamp, pd.Series]:
    """
    Download configured Treasury constant-maturity yields from FRED.

    Returns
    -------
    df : DataFrame
        Full time series with columns equal to SERIES_IDS.
    curve_date : Timestamp
        Latest date where all series are available.
    yields_today : Series
        The yields at curve_date, ready for bootstrapping.
    """
    fred = load_fred_client()
    data = {sid: fred.get_series(sid) for sid in SERIES_IDS}
    df = pd.DataFrame(data).dropna()

    curve_date = df.index.max()
    yields_today = df.loc[curve_date]
    return df, curve_date, yields_today


def load_or_fetch_treasury_yields(
    use_cache: bool = True, cache_path: Path = CACHE_PATH
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Series]:
    """
    Robust loader that reuses cached CSV if available and complete.

    If `use_cache` is True and `cache_path` exists and contains all
    required SERIES_IDS, data is loaded from disk. Otherwise, data is
    fetched from FRED and saved to `cache_path`.
    """
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # If cache is missing some series, ignore and refetch
        missing = set(SERIES_IDS) - set(df.columns)
        if not missing:
            curve_date = df.index.max()
            yields_today = df.loc[curve_date]
            return df, curve_date, yields_today

    # Fetch from FRED and cache (either no cache or incomplete cache)
    df, curve_date, yields_today = fetch_treasury_yields()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df, curve_date, yields_today


if __name__ == "__main__":
    df, curve_date, yields_today = load_or_fetch_treasury_yields()
    print("Latest curve date:", curve_date)
    print("Yields on that date:")
    print(yields_today)
    print("\nFull dataframe:")
    print(df)