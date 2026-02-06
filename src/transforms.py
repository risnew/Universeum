import pandas as pd


def combine_retro_date(df: pd.DataFrame) -> pd.DataFrame:
    """Merge retrospective date field with timestamp for activities logged on different dates."""
    df = df.copy()

    df[["date", "retro_date"]] = df[["date", "retro_date"]].apply(
        pd.to_datetime, errors="coerce"
    )

    mask = df["retro_date"].notna()

    df.loc[mask, "date"] = pd.to_datetime(
        df.loc[mask, "retro_date"].dt.date.astype(str)
        + " "
        + df.loc[mask, "date"].dt.time.astype(str)
    )

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
