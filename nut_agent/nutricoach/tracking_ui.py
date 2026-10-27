"""Journal dashboard and dated meal and weight tracking."""

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shared.config import SECRETS_DIR
from shared.memory import MemoryManager
from shared.schemas import DailyLog


def display_nutrition_dashboard():
    """Display a real dashboard from the user's daily logs."""
    username = st.session_state.get("username")
    if not username:
        st.info("Log in to see your nutrition dashboard.")
        return

    memory = MemoryManager(username, SECRETS_DIR)
    recent_logs = memory.load_recent_daily_logs(n=7)
    targets = memory.load_nutrition_targets()

    if not recent_logs:
        st.info("No daily logs yet. Start tracking your meals to see trends here!")
        return

    rows = []
    for log in reversed(recent_logs):
        rows.append({
            "Date": log.date,
            "Calories": log.total_calories or 0,
            "Protein (g)": log.total_protein_g or 0,
            "Carbs (g)": log.total_carbs_g or 0,
            "Fat (g)": log.total_fat_g or 0,
            "Weight (kg)": log.weight_kg,
            "Compliance": (log.compliance_score or 0) * 100,
        })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily Calories**")
        fig = px.line(df, x="Date", y="Calories", markers=True)
        if targets:
            fig.add_hline(y=targets.target_calories, line_dash="dash", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if any(df["Weight (kg)"] > 0):
            st.markdown("**Weight Trend**")
            weight_df = df[df["Weight (kg)"] > 0]
            fig = px.line(weight_df, x="Date", y="Weight (kg)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            latest = rows[-1] if rows else {}
            if latest.get("Protein (g)") or latest.get("Carbs (g)") or latest.get("Fat (g)"):
                st.markdown("**Today's Macros**")
                fig = go.Figure(data=[go.Pie(
                    labels=["Protein", "Carbs", "Fat"],
                    values=[latest["Protein (g)"], latest["Carbs (g)"], latest["Fat (g)"]],
                )])
                st.plotly_chart(fig, use_container_width=True)


def display_daily_results(send_message):
    with st.expander("Daily Results Tracking", expanded=False):
        date_tracked = st.date_input("Meals date:", max_value=date.today())
        record_weight = st.checkbox("I measured my weight")
        current_weight = None
        weight_date = None
        if record_weight:
            st.caption("Use the morning measurement, before breakfast. Date it on the morning it was measured, even when reviewing yesterday's meals.")
            col1, col2 = st.columns(2)
            with col1:
                current_weight = st.number_input("Measured weight (kg):", min_value=30.0, max_value=200.0, value=None, step=0.1)
            with col2:
                weight_date = st.date_input("Weight measurement date:", max_value=date.today())

        st.subheader("Meal Compliance")
        meals = ["Breakfast", "Lunch", "Dinner", "Snacks"]
        meal_data = {}
        for meal in meals:
            st.write(f"**{meal}:**")
            col1, col2 = st.columns([1, 2])
            with col1:
                followed = st.checkbox(f"Followed {meal.lower()} plan exactly", key=f"exact_{meal.lower()}")
            with col2:
                if not followed:
                    diff = st.text_area(f"Differences for {meal.lower()}:", key=f"diff_{meal.lower()}", height=60)
                else:
                    diff = "Followed plan exactly"
            meal_data[meal] = {"followed_exactly": followed, "differences": diff if not followed else "Followed plan exactly"}

        additional_info = st.text_area("Additional information:", placeholder="Energy levels, cravings, exercise, water intake, sleep...", height=80)

        if st.button("Submit Daily Results"):
            if st.session_state["agent_graph"]:
                if record_weight and current_weight is None:
                    st.error("Enter the measured weight or uncheck 'I measured my weight'.")
                    return
                if record_weight:
                    memory = MemoryManager(st.session_state["username"], SECRETS_DIR)
                    weight_day = weight_date.isoformat()
                    log = memory.load_daily_log(weight_day) or DailyLog(date=weight_day)
                    log.weight_kg = current_weight
                    memory.save_daily_log(log)
                meal_lines = []
                for meal, data in meal_data.items():
                    if data["followed_exactly"]:
                        meal_lines.append(f"- {meal}: Followed plan exactly")
                    else:
                        meal_lines.append(f"- {meal}: {data['differences']}")

                message = f"""Please analyze my daily results:

**Daily Tracking Results for {date_tracked}**
**Morning weight:** {f'{current_weight} kg on {weight_date}' if record_weight else 'Not measured'}
**Meal Compliance:**
{chr(10).join(meal_lines)}
**Additional Information:** {additional_info or "None provided"}

Please provide analysis of my progress and any recommendations."""

                send_message(message)

