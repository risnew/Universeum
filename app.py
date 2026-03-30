import streamlit as st
import pandas as pd

from src import process_data
from src.metrics import avg_range
from src.charts import (
    grade_chart,
    participants_chart,
    interest_chart,
    activities_over_time_chart,
    grade_distribution_chart,
    activity_heatmap,
    trend_chart,
    line_chart,
    bar_chart_time,
    pie_chart,
    bar_chart,
    horizontal_bar_chart,
    interest_bar_chart,
    scatter_with_trendline,
    hour_distribution_chart,
    participants_by_hour_chart,
    INTEREST_LEVEL_COLORS,
)
from src.analysis import (
    compare_periods,
    calculate_trend,
)
from src.classifier import (
    AdjustmentClassifier,
    load_training_data,
    get_category_distribution,
    CATEGORY_NAMES,
)

st.set_page_config(page_title="Universeum Analytics Dashboard", layout="wide", page_icon=":bar_chart:")

# Custom CSS for styling
st.markdown("""
<style>
    /* Main title styling */
    h1 {
        color: #0077B6;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #00B4D8;
        margin-bottom: 1.5rem;
    }

    /* Metric styling - clean, no boxes */
    [data-testid="stMetricLabel"] {
        color: #B0B0B0;
        font-size: 0.9rem;
        font-weight: 500;
    }

    [data-testid="stMetricValue"] {
        color: #E0E0E0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1A1A2E;
    }

    /* Subheader styling */
    h2, h3 {
        color: #1A1A2E;
        font-weight: 600;
    }

    /* Caption styling */
    .stCaption {
        color: #577590;
        font-weight: 500;
    }

    /* Divider styling */
    hr {
        border-color: #e9ecef;
        margin: 1.5rem 0;
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Universeum Analytics Dashboard")

# --- Data Loading ---
uploaded_file = st.file_uploader("Upload Google Forms CSV", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
    st.stop()

df = process_data(pd.read_csv(uploaded_file))

# Compute average participants from range (e.g., "3-6" -> 4.5)
df["participants_avg"] = df["participants"].apply(avg_range)


def extract_activity_type(name: str) -> str:
    """Extract activity type from activity name (text before ':'), or 'Övrigt'."""
    if isinstance(name, str) and ":" in name:
        return name.split(":")[0].strip()
    return "Övrigt"


MIN_SESSIONS = 10

# --- Classifier (loaded once, used in tab6 and raw data export) ---
training_file = "data/additional_data.csv"
try:
    @st.cache_resource
    def get_trained_classifier():
        texts, labels = load_training_data(training_file)
        clf = AdjustmentClassifier()
        metrics = clf.train(texts, labels)
        return clf, metrics

    classifier, classifier_metrics = get_trained_classifier()
except FileNotFoundError:
    classifier, classifier_metrics = None, None

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total sessions", len(df))
col2.metric("Average participants per session", round(df["participants_avg"].mean(), 1))
col3.metric("Average grade per session", round(df["grade"].mean(), 2))

st.divider()

# --- Tabs for organization ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Activity Deep Dive",
    "Correlations",
    "Trends",
    "Period Comparison",
    "Adjustment Predictor (Beta)",
])

# =============================================================================
# TAB 1: Overview
# =============================================================================
with tab1:
    # Row 1: Distributions
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Interest Level Distribution")
        st.plotly_chart(interest_chart(df), use_container_width=True)
    with col2:
        st.caption("Performance Distribution")
        st.plotly_chart(grade_distribution_chart(df), use_container_width=True)

    # Row 2: Time-based charts
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Activities Over Time")
        st.plotly_chart(activities_over_time_chart(df), use_container_width=True)
    with col2:
        st.caption("Activity Heatmap (Weekday & Hour)")
        st.plotly_chart(activity_heatmap(df), use_container_width=True)

    # Row 3: Participation by hour
    df_with_hour = df.copy()
    df_with_hour["hour"] = df_with_hour["time"].apply(lambda t: t.hour if pd.notna(t) else None)
    df_with_hour = df_with_hour.dropna(subset=["hour"])
    if len(df_with_hour) > 0:
        st.caption("Average Participation by Hour")
        st.caption(f"Average total participants per session at each hour of the day.")
        st.plotly_chart(participants_by_hour_chart(df_with_hour), use_container_width=True)

    # Row 4: Activity details (expandable)
    with st.expander("Average Performance per Activity"):
        activity_stats = (
            df.groupby("activity")
            .agg(avg_grade=("grade", "mean"), grade_std=("grade", "std"), count=("grade", "size"))
            .reset_index()
            .fillna({"avg_grade": 0, "grade_std": 0})
        )
        filtered_stats = activity_stats[activity_stats["count"] >= MIN_SESSIONS]
        st.caption(f"Only showing activities with at least {MIN_SESSIONS} sessions. ({len(filtered_stats)} of {len(activity_stats)} activities)")
        st.plotly_chart(grade_chart(filtered_stats), use_container_width=True)

    with st.expander("Average Participants per Activity"):
        activity_participants = (
            df.groupby("activity")
            .agg(participants_avg=("participants_avg", "mean"), count=("participants_avg", "size"))
            .reset_index()
            .sort_values("participants_avg")
        )
        filtered_participants = activity_participants[activity_participants["count"] >= MIN_SESSIONS]
        st.caption(f"Only showing activities with at least {MIN_SESSIONS} sessions. ({len(filtered_participants)} of {len(activity_participants)} activities)")
        st.plotly_chart(participants_chart(filtered_participants), use_container_width=True)

    st.divider()

    # Row 5: Activity type aggregations
    st.subheader("Average Performance & Participants per Activity Type")
    df_typed = df.copy()
    df_typed["activity_type"] = df_typed["activity"].apply(extract_activity_type)

    type_stats = (
        df_typed.groupby("activity_type")
        .agg(avg_grade=("grade", "mean"), participants_avg=("participants_avg", "mean"), count=("grade", "size"))
        .reset_index()
    )
    filtered_type_stats = type_stats[type_stats["count"] >= MIN_SESSIONS]
    st.caption(f"Only showing activity types with at least {MIN_SESSIONS} sessions.")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Average Performance per Activity Type")
        st.plotly_chart(
            grade_chart(filtered_type_stats.rename(columns={"activity_type": "activity"})),
            use_container_width=True,
        )
    with col2:
        st.caption("Average Participants per Activity Type")
        st.plotly_chart(
            participants_chart(filtered_type_stats.rename(columns={"activity_type": "activity"})),
            use_container_width=True,
        )

# =============================================================================
# TAB 2: Activity Deep Dive
# =============================================================================
with tab2:
    st.subheader("Activity Deep Dive")

    activities = sorted(df["activity"].unique())
    selected_activity = st.selectbox("Select an activity", activities, key="activity_select")

    activity_data = df[df["activity"] == selected_activity]

    # Key metrics for this activity
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total number of sessions", len(activity_data))
    col2.metric("Average grade", round(activity_data["grade"].mean(), 2))
    col3.metric("Average participants", round(activity_data["participants_avg"].mean(), 1))

    interest_mode = activity_data["interest_level"].mode()
    col4.metric("Most Common Interest", interest_mode.iloc[0] if len(interest_mode) > 0 else "N/A")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Grade Over Time (Monthly Average)")
        if len(activity_data) > 1:
            activity_by_month = activity_data.groupby(activity_data["date"].dt.to_period("M"))["grade"].mean().reset_index()
            activity_by_month["date"] = activity_by_month["date"].dt.to_timestamp()
            st.plotly_chart(bar_chart_time(activity_by_month, "date", "grade", "Grade"), use_container_width=True)
        else:
            st.info("Not enough data for trend.")

    with col2:
        st.caption("Participants Over Time (Monthly Average)")
        if len(activity_data) > 1:
            participants_by_month = activity_data.groupby(activity_data["date"].dt.to_period("M"))["participants_avg"].mean().reset_index()
            participants_by_month["date"] = participants_by_month["date"].dt.to_timestamp()
            st.plotly_chart(bar_chart_time(participants_by_month, "date", "participants_avg", "Participants"), use_container_width=True)
        else:
            st.info("Not enough data for trend.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Interest Level Distribution")
        interest_counts = activity_data["interest_level"].value_counts().reset_index()
        interest_counts.columns = ["interest_level", "count"]
        if len(interest_counts) > 0:
            st.plotly_chart(
                interest_bar_chart(interest_counts, "count", "Sessions", ".0f"),
                use_container_width=True,
            )

    with col2:
        st.caption("When Does This Activity Run?")
        if activity_data["time"].notna().any():
            activity_data_copy = activity_data.copy()
            activity_data_copy["hour"] = activity_data_copy["time"].apply(lambda t: t.hour if pd.notna(t) else None)
            st.plotly_chart(hour_distribution_chart(activity_data_copy), use_container_width=True)
        else:
            st.info("No time data available.")

    # Participation by hour for this activity
    if activity_data["time"].notna().any():
        activity_with_hour = activity_data.copy()
        activity_with_hour["hour"] = activity_with_hour["time"].apply(lambda t: t.hour if pd.notna(t) else None)
        activity_with_hour = activity_with_hour.dropna(subset=["hour"])
        if len(activity_with_hour) > 0:
            st.caption("Average Participation by Hour")
            st.caption("Average total participants per session at each hour of the day.")
            st.plotly_chart(participants_by_hour_chart(activity_with_hour), use_container_width=True)

    with st.expander("Recent Sessions"):
        recent = activity_data.sort_values("date", ascending=False).head(10)[["date", "participants_avg", "grade", "interest_level", "adjustments"]]
        st.dataframe(recent, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: Correlations
# =============================================================================
with tab3:
    st.subheader("Correlation Insights")

    # Activity type filter
    df_typed_corr = df.copy()
    df_typed_corr["activity_type"] = df_typed_corr["activity"].apply(extract_activity_type)
    all_types = sorted(df_typed_corr["activity_type"].unique())
    selected_types = st.multiselect(
        "Filter by activity type (leave empty for all)",
        all_types,
        default=all_types,
        key="corr_type_filter",
    )
    df_corr = df_typed_corr[df_typed_corr["activity_type"].isin(selected_types)] if selected_types else df_typed_corr

    if len(selected_types) == len(all_types) or not selected_types:
        st.caption("Based on all activity types.")
    else:
        st.caption(f"Based on activity types: {', '.join(selected_types)}.")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Participants vs Performance")
        scatter_fig = scatter_with_trendline(df_corr, "participants_avg", "grade", "Participants", "Performance")
        scatter_fig.update_yaxes(tickmode="array", tickvals=[1, 2, 3, 4, 5, 6, 7])
        st.plotly_chart(scatter_fig, use_container_width=True)

        corr = df_corr[["participants_avg", "grade"]].corr().iloc[0, 1]
        st.caption(f"Each dot = one session. Y-axis = grade given (1–7). X-axis = number of participants. The line shows the overall trend. Correlation: **{corr:.2f}**.")
        if corr > 0.1:
            st.caption("Larger groups tend to give slightly higher grades.")
        elif corr < -0.1:
            st.caption("Larger groups tend to give slightly lower grades.")
        else:
            st.caption("No significant relationship found between group size and performance.")

    with col2:
        st.caption("Average Performance by Interest Level")
        interest_grade = df_corr.groupby("interest_level")["grade"].mean().reset_index()
        st.plotly_chart(interest_bar_chart(interest_grade, "grade", "Average Performance"), use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Average Participants by Interest Level")
        interest_participants = df_corr.groupby("interest_level")["participants_avg"].mean().reset_index()
        st.plotly_chart(interest_bar_chart(interest_participants, "participants_avg", "Average Participants", ".1f"), use_container_width=True)

    with col2:
        st.caption("Average Performance by Day of Week")
        df_copy = df_corr.copy()
        df_copy["day_of_week"] = df_copy["date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_grade = df_copy.groupby("day_of_week")["grade"].mean().reset_index()
        day_grade["day_of_week"] = pd.Categorical(day_grade["day_of_week"], categories=day_order, ordered=True)
        day_grade = day_grade.sort_values("day_of_week")
        st.plotly_chart(bar_chart(day_grade, "day_of_week", "grade", "", "Average Performance"), use_container_width=True)

# =============================================================================
# TAB 4: Trends
# =============================================================================
MIN_MONTHS = 5
MIN_SESSIONS_PER_MONTH = 3

with tab4:
    # Filter out months with too few sessions to avoid misleading outlier months
    monthly_counts = df.groupby(df["date"].dt.to_period("M")).size()
    valid_months = monthly_counts[monthly_counts >= MIN_SESSIONS_PER_MONTH].index
    df_trend = df[df["date"].dt.to_period("M").isin(valid_months)]

    excluded = len(monthly_counts) - len(valid_months)
    note = f" ({excluded} month(s) with fewer than {MIN_SESSIONS_PER_MONTH} sessions excluded)" if excluded > 0 else ""

    st.subheader("Performance Trend Over Time")
    grade_trend = calculate_trend(df_trend, "grade")
    if len(grade_trend) >= MIN_MONTHS:
        trend_direction = "improving" if grade_trend["trend"].iloc[0] > 0 else "declining"
        st.caption(f"Monthly average performance across all activities. Overall trend: **{trend_direction}**{note}.")
        st.plotly_chart(trend_chart(grade_trend, "grade", "Average Performance"), use_container_width=True)
    else:
        st.warning(f"Not enough data for trend analysis. At least {MIN_MONTHS} months of data required (currently {len(grade_trend)}).")

    st.divider()

    st.subheader("Participants Trend Over Time")
    participants_trend = calculate_trend(df_trend, "participants_avg")
    if len(participants_trend) >= MIN_MONTHS:
        trend_direction = "increasing" if participants_trend["trend"].iloc[0] > 0 else "decreasing"
        st.caption(f"Monthly average total participants per session, across all activities. Overall trend: **{trend_direction}**{note}.")
        st.plotly_chart(trend_chart(participants_trend, "participants_avg", "Average Participants"), use_container_width=True)
    else:
        st.warning(f"Not enough data for trend analysis. At least {MIN_MONTHS} months of data required (currently {len(participants_trend)}).")

# =============================================================================
# TAB 5: Period Comparison
# =============================================================================
with tab5:
    st.subheader("Compare Two Periods")

    # Build list of available months as display strings
    months = sorted(df["date"].dt.to_period("M").unique())
    month_labels = [m.strftime("%b %Y") for m in months]
    month_starts = {m.strftime("%b %Y"): m.to_timestamp(how="start") for m in months}
    month_ends   = {m.strftime("%b %Y"): m.to_timestamp(how="end")   for m in months}

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Period 1")
        p1_range = st.select_slider(
            "Period 1",
            options=month_labels,
            value=(month_labels[0], month_labels[len(month_labels) // 2 - 1]),
            key="p1_range",
            label_visibility="collapsed",
        )

    with col2:
        st.caption("Period 2")
        p2_range = st.select_slider(
            "Period 2",
            options=month_labels,
            value=(month_labels[len(month_labels) // 2], month_labels[-1]),
            key="p2_range",
            label_visibility="collapsed",
        )

    p1_start = month_starts[p1_range[0]]
    p1_end   = month_ends[p1_range[1]]
    p2_start = month_starts[p2_range[0]]
    p2_end   = month_ends[p2_range[1]]

    if st.button("Compare Periods"):
        comparison = compare_periods(
            df,
            p1_start,
            p1_end,
            p2_start,
            p2_end,
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sessions",
            comparison.period2_activities,
            delta=f"{comparison.activities_change:+.1f}%",
        )
        col2.metric(
            "Avg Participants",
            comparison.period2_avg_participants,
            delta=f"{comparison.participants_change:+.1f}%",
        )
        col3.metric(
            "Avg Grade",
            comparison.period2_avg_grade,
            delta=f"{comparison.grade_change:+.1f}%",
        )

        st.divider()

        comparison_df = pd.DataFrame({
            "Metric": ["Sessions", "Avg Participants", "Avg Grade"],
            "Period 1": [comparison.period1_activities, comparison.period1_avg_participants, comparison.period1_avg_grade],
            "Period 2": [comparison.period2_activities, comparison.period2_avg_participants, comparison.period2_avg_grade],
            "Change %": [f"{comparison.activities_change:+.1f}%", f"{comparison.participants_change:+.1f}%", f"{comparison.grade_change:+.1f}%"],
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 6: Adjustment Predictor (Beta)
# =============================================================================
with tab6:
    if classifier is not None:
        metrics = classifier_metrics

        # --- Adjustments in Your Data ---
        st.subheader("Adjustments in Your Data")

        adjustments_with_text = df[df["adjustments"].notna() & (df["adjustments"].str.strip() != "")]

        if len(adjustments_with_text) > 0:
            predictions = []
            for text in adjustments_with_text["adjustments"]:
                result = classifier.predict(text)
                predictions.append(result.predicted_category)

            pred_counts = pd.Series(predictions).value_counts().reset_index()
            pred_counts.columns = ["category", "count"]
            pred_counts["percentage"] = (pred_counts["count"] / pred_counts["count"].sum() * 100).round(1)

            col1, col2 = st.columns(2)

            with col1:
                st.caption(f"Distribution of {len(adjustments_with_text)} adjustments")
                st.plotly_chart(pie_chart(pred_counts, "count", "category"), use_container_width=True)

            with col2:
                st.caption("Breakdown")
                st.dataframe(
                    pred_counts.rename(columns={"category": "Category", "count": "Count", "percentage": "%"}),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No adjustment texts found in the uploaded data.")

        st.divider()

        # --- Adjustment Category Predictor ---
        st.subheader("Adjustment Category Predictor")
        st.caption("Predict adjustment category from free-text descriptions using machine learning.")

        distribution = get_category_distribution(training_file)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Training Data Distribution")
            st.plotly_chart(horizontal_bar_chart(distribution, "count", "category", "Samples", ".0f"), use_container_width=True)

        with col2:
            st.caption("Category Descriptions")
            category_desc = pd.DataFrame([
                {"Code": k, "Category": v}
                for k, v in CATEGORY_NAMES.items()
            ])
            st.dataframe(category_desc, use_container_width=True, hide_index=True)

        st.caption(f"Model trained on {metrics['n_samples']} samples | Accuracy: {metrics['accuracy']:.1%} (+/- {metrics['std']:.1%})")

        with st.expander("View Full Data with Predictions"):
            export_df = df.copy()
            export_df["predicted_category"] = export_df["adjustments"].apply(
                lambda t: classifier.predict(t).predicted_category
                if pd.notna(t) and t.strip() else ""
            )
            export_df = export_df[export_df["predicted_category"] != ""]
            st.dataframe(export_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download as CSV",
                export_df.to_csv(index=False),
                file_name="universeum_data_with_predictions.csv",
                mime="text/csv",
            )

        st.divider()

        # --- Test Prediction ---
        st.subheader("Test Prediction")

        input_text = st.text_area(
            "Enter adjustment description (Swedish):",
            placeholder="E.g., 'Körde på engelska eftersom gruppen var internationell'",
            height=100,
        )

        if st.button("Predict Category", type="primary"):
            if input_text.strip():
                result = classifier.predict(input_text)

                if result.confidence > 0:
                    st.success(f"**Predicted Category:** {result.predicted_category}")
                    st.caption(f"Confidence: {result.confidence:.1%}")

                    st.divider()

                    st.caption("Top Predictions:")
                    for category, prob in result.top_predictions:
                        st.progress(prob, text=f"{category}: {prob:.1%}")
                else:
                    st.warning("Could not make a prediction. Try a longer description.")
            else:
                st.warning("Please enter some text to predict.")

    else:
        st.warning("Training data not found. Please ensure 'data/additional_data.csv' exists.")

# =============================================================================
# Raw Data
# =============================================================================
st.divider()

with st.expander("View Raw Data"):
    raw_columns = [col for col in df.columns if col != "participants_avg"]
    st.dataframe(df[raw_columns], use_container_width=True)

with st.expander("Data for Lena"):
    export_df = df.copy()
    export_df["activity_type"] = export_df["activity"].apply(extract_activity_type)
    export_df["hour"] = export_df["time"].apply(lambda t: t.hour if pd.notna(t) else None)
    export_df["day_of_week"] = export_df["date"].dt.day_name()
    if classifier is not None:
        export_df["predicted_category"] = export_df["adjustments"].apply(
            lambda t: classifier.predict(t).predicted_category if pd.notna(t) and str(t).strip() else ""
        )
    st.dataframe(export_df, use_container_width=True)
    st.download_button(
        "Download as CSV",
        export_df.to_csv(index=False),
        file_name="universeum_data_enriched.csv",
        mime="text/csv",
    )
