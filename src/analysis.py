import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PeriodComparison:
    """Comparison metrics between two time periods."""
    period1_label: str
    period2_label: str
    activities_change: float
    participants_change: float
    grade_change: float
    period1_activities: int
    period2_activities: int
    period1_avg_participants: float
    period2_avg_participants: float
    period1_avg_grade: float
    period2_avg_grade: float


def compare_periods(
    df: pd.DataFrame,
    period1_start: pd.Timestamp,
    period1_end: pd.Timestamp,
    period2_start: pd.Timestamp,
    period2_end: pd.Timestamp,
) -> PeriodComparison:
    """Compare metrics between two time periods."""
    p1 = df[(df["date"] >= period1_start) & (df["date"] <= period1_end)]
    p2 = df[(df["date"] >= period2_start) & (df["date"] <= period2_end)]

    p1_activities = len(p1)
    p2_activities = len(p2)
    p1_participants = p1["participants_avg"].mean() if len(p1) > 0 else 0
    p2_participants = p2["participants_avg"].mean() if len(p2) > 0 else 0
    p1_grade = p1["grade"].mean() if len(p1) > 0 else 0
    p2_grade = p2["grade"].mean() if len(p2) > 0 else 0

    def pct_change(old, new):
        if old == 0:
            return 0 if new == 0 else 100
        return ((new - old) / old) * 100

    return PeriodComparison(
        period1_label=f"{period1_start.strftime('%Y-%m-%d')} to {period1_end.strftime('%Y-%m-%d')}",
        period2_label=f"{period2_start.strftime('%Y-%m-%d')} to {period2_end.strftime('%Y-%m-%d')}",
        activities_change=pct_change(p1_activities, p2_activities),
        participants_change=pct_change(p1_participants, p2_participants),
        grade_change=pct_change(p1_grade, p2_grade),
        period1_activities=p1_activities,
        period2_activities=p2_activities,
        period1_avg_participants=round(p1_participants, 1),
        period2_avg_participants=round(p2_participants, 1),
        period1_avg_grade=round(p1_grade, 2),
        period2_avg_grade=round(p2_grade, 2),
    )


def calculate_trend(df: pd.DataFrame, column: str, freq: str = "M") -> pd.DataFrame:
    """Calculate trend over time with slope indicator."""
    grouped = df.groupby(df["date"].dt.to_period(freq))[column].mean().reset_index()
    grouped["date"] = grouped["date"].dt.to_timestamp()
    grouped = grouped.dropna()

    if len(grouped) < 2:
        grouped["trend"] = 0
        return grouped

    x = np.arange(len(grouped))
    y = grouped[column].values
    slope, _ = np.polyfit(x, y, 1)
    grouped["trend"] = slope

    return grouped
