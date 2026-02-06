import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

# Consistent color palette
COLORS = {
    "primary": "#0077B6",
    "secondary": "#00B4D8",
    "accent": "#90E0EF",
    "warm": "#F77F00",
    "success": "#06D6A0",
    "neutral": "#577590",
}

COLOR_SEQUENCE = ["#0077B6", "#00B4D8", "#90E0EF", "#F77F00", "#06D6A0", "#577590", "#CAF0F8", "#FCBF49"]

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=30, b=20),
)


def grade_chart(activity_stats: pd.DataFrame) -> Figure:
    """Create horizontal bar chart showing average grade per activity."""
    max_height = 800
    bar_height = 35
    fig_height = min(max_height, len(activity_stats) * bar_height)

    fig = px.bar(
        activity_stats,
        x="avg_grade",
        y="activity",
        orientation="h",
        text="avg_grade",
        height=fig_height,
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        textfont=dict(color="#E0E0E0"),
        marker=dict(cornerradius=4),
    )

    max_grade = activity_stats["avg_grade"].max()
    fig.update_xaxes(
        range=[0, max_grade * 1.15],
        showticklabels=False,
        ticks="",
        title="",
        showgrid=False,
    )
    fig.update_yaxes(showgrid=False)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        yaxis={"categoryorder": "total ascending"},
        yaxis_title="",
        xaxis_title="",
        width=900,
    )

    return fig


def participants_chart(activity_participants: pd.DataFrame) -> Figure:
    """Create horizontal bar chart showing average participants per activity."""
    bar_height = 35
    fig_height = max(400, len(activity_participants) * bar_height)

    fig = px.bar(
        activity_participants,
        x="participants_avg",
        y="activity",
        orientation="h",
        text="participants_avg",
        height=fig_height,
        labels={"participants_avg": "Average participants", "activity": ""},
        color_discrete_sequence=[COLORS["secondary"]],
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont=dict(color="#E0E0E0"),
        marker=dict(cornerradius=4),
    )

    max_participants = activity_participants["participants_avg"].max()
    fig.update_xaxes(
        range=[0, max_participants * 1.15],
        showticklabels=False,
        ticks="",
        title="",
        showgrid=False,
    )
    fig.update_yaxes(showgrid=False)

    fig.update_layout(
        **{**LAYOUT_DEFAULTS, "margin": dict(l=20, r=40, t=30, b=20)},
        yaxis=dict(automargin=True),
    )

    return fig


def interest_chart(df: pd.DataFrame) -> Figure:
    """Create pie chart showing interest level distribution."""
    counts = df["interest_level"].value_counts().reset_index()
    counts.columns = ["interest_level", "count"]

    fig = px.pie(
        counts,
        values="count",
        names="interest_level",
        hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE,
    )

    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont=dict(color="#E0E0E0"),
        pull=[0.02] * len(counts),
    )

    fig.update_layout(**LAYOUT_DEFAULTS)

    return fig


def activities_over_time_chart(df: pd.DataFrame) -> Figure:
    """Create line chart showing activity count by month."""
    monthly = df.groupby(df["date"].dt.to_period("M")).size().reset_index(name="count")
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig = px.line(
        monthly,
        x="date",
        y="count",
        markers=True,
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Sessions",
    )

    return fig


def grade_distribution_chart(df: pd.DataFrame) -> Figure:
    """Create histogram showing grade distribution."""
    fig = px.histogram(
        df,
        x="grade",
        nbins=6,
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_traces(marker=dict(cornerradius=4))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="Grade",
        yaxis_title="Count",
        bargap=0.15,
    )

    return fig


def activity_heatmap(df: pd.DataFrame) -> Figure:
    """Create heatmap showing activity count by day of week and hour."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_name()
    df["hour"] = df["time"].apply(lambda t: t.hour if pd.notna(t) else None)

    pivot = df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="activity",
        aggfunc="count",
        fill_value=0,
    )

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[[0, "#CAF0F8"], [0.5, "#00B4D8"], [1, "#0077B6"]],
        labels={"color": "Sessions"},
    )

    fig.update_traces(
        hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Sessions: %{z}<extra></extra>"
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="Hour",
        yaxis_title="",
    )

    return fig


def trend_chart(trend_data: pd.DataFrame, column: str, title: str = "") -> Figure:
    """Create line chart with trend indicator."""
    fig = px.line(
        trend_data,
        x="date",
        y=column,
        markers=True,
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
    )

    if len(trend_data) >= 2:
        x_num = list(range(len(trend_data)))
        slope, intercept = np.polyfit(x_num, trend_data[column].values, 1)
        trend_line = [slope * x + intercept for x in x_num]

        fig.add_trace(go.Scatter(
            x=trend_data["date"],
            y=trend_line,
            mode="lines",
            name="Trend",
            line=dict(dash="dash", color=COLORS["warm"], width=2),
        ))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def line_chart(data: pd.DataFrame, x: str, y: str, y_title: str = "") -> Figure:
    """Create simple line chart."""
    fig = px.line(data, x=x, y=y, markers=True, color_discrete_sequence=[COLORS["primary"]])

    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title=y_title,
    )
    return fig


def pie_chart(data: pd.DataFrame, values: str, names: str) -> Figure:
    """Create donut chart."""
    fig = px.pie(data, values=values, names=names, hole=0.45, color_discrete_sequence=COLOR_SEQUENCE)

    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont=dict(color="#E0E0E0"),
        pull=[0.02] * len(data),
    )

    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def bar_chart(data: pd.DataFrame, x: str, y: str, x_title: str = "", y_title: str = "") -> Figure:
    """Create vertical bar chart."""
    fig = px.bar(data, x=x, y=y, text=y, color_discrete_sequence=[COLORS["primary"]])

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont=dict(color="#E0E0E0"), marker=dict(cornerradius=4))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def horizontal_bar_chart(data: pd.DataFrame, x: str, y: str, x_title: str = "", text_format: str = ".2f") -> Figure:
    """Create horizontal bar chart."""
    fig = px.bar(data, x=x, y=y, orientation="h", text=x, color_discrete_sequence=[COLORS["secondary"]])

    fig.update_traces(texttemplate=f"%{{text:{text_format}}}", textposition="outside", textfont=dict(color="#E0E0E0"), marker=dict(cornerradius=4))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title=x_title,
        yaxis_title="",
    )
    return fig


def scatter_with_trendline(data: pd.DataFrame, x: str, y: str, x_title: str = "", y_title: str = "") -> Figure:
    """Create scatter plot with OLS trendline."""
    fig = px.scatter(data, x=x, y=y, opacity=0.6, trendline="ols", color_discrete_sequence=[COLORS["primary"]])

    fig.update_traces(marker=dict(size=10))
    fig.data[1].line.color = COLORS["warm"]
    fig.data[1].line.width = 2

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def hour_distribution_chart(data: pd.DataFrame) -> Figure:
    """Create bar chart showing distribution by hour."""
    hour_counts = data["hour"].value_counts().sort_index().reset_index()
    hour_counts.columns = ["hour", "count"]

    fig = px.bar(hour_counts, x="hour", y="count", color_discrete_sequence=[COLORS["neutral"]])

    fig.update_traces(marker=dict(cornerradius=4))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="Hour",
        yaxis_title="Sessions",
    )
    return fig


def participants_by_hour_chart(data: pd.DataFrame) -> Figure:
    """Create bar chart showing average participants by hour."""
    hourly_avg = data.groupby("hour")["participants_avg"].mean().reset_index()
    hourly_avg.columns = ["hour", "avg_participants"]
    hourly_avg = hourly_avg.sort_values("hour")

    fig = px.bar(hourly_avg, x="hour", y="avg_participants", text="avg_participants", color_discrete_sequence=[COLORS["success"]])

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", textfont=dict(color="#E0E0E0"), marker=dict(cornerradius=4))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="Hour",
        yaxis_title="Avg Participants",
    )
    return fig
