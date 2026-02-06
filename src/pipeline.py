import pandas as pd

from .cleaning import normalize_columns
from .transforms import combine_retro_date, normalize_time, to_numeric_cols


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full data cleaning and transformation pipeline."""
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    df = normalize_columns(df)
    df = combine_retro_date(df)
    df = normalize_time(df)
    df = to_numeric_cols(df)
    return df
