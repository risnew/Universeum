import pandas as pd

def avg_range(value: str | float) -> float | None:
    if pd.isna(value):
        return None
    if value == "0":
        return 0
    if isinstance(value, str) and value.endswith("+"):
        return float(value[:-1])
    if isinstance(value, str) and "-" in value:
        low, high = map(float, value.split("-"))
        return (low + high) / 2
    return float(value)
