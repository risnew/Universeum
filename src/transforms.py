import pandas as pd


def _clean_timestamp(series: pd.Series) -> pd.Series:
    """Strip Swedish am/pm and timezone from timestamps before parsing."""
    return (
        series
        .str.replace(r"\s*CET\s*", " ", regex=True)
        .str.replace(r"\s*fm\b", " AM", regex=True)
        .str.replace(r"\s*em\b", " PM", regex=True)
        .str.strip()
    )


def combine_retro_date(df: pd.DataFrame) -> pd.DataFrame:
    """Merge retrospective date field with timestamp for activities logged on different dates."""
    df = df.copy()

    df["date"] = pd.to_datetime(_clean_timestamp(df["date"]), format="mixed", errors="coerce")
    df["retro_date"] = pd.to_datetime(_clean_timestamp(df["retro_date"]), format="mixed", errors="coerce")

    mask = df["retro_date"].notna()
    has_time = mask & df["date"].notna()

    df.loc[has_time, "date"] = pd.to_datetime(
        df.loc[has_time, "retro_date"].dt.date.astype(str)
        + " "
        + df.loc[has_time, "date"].dt.time.astype(str)
    )
    df.loc[mask & ~has_time, "date"] = df.loc[mask & ~has_time, "retro_date"]

    return df.drop(columns="retro_date")


def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Swedish time notation (fm/em) to standard format."""
    df = df.copy()

    df["time"] = (
        df["time"]
        .str.lower()
        .str.replace("fm", "am")
        .str.replace("em", "pm")
        .str.strip()
    )

    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce").dt.time

    return df


def to_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert grade to numeric type."""
    df = df.copy()
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    return df
