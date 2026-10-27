"""Account registration and initial profile persistence."""

from datetime import datetime

import streamlit as st

from shared.auth import register_user
from shared.config import SECRETS_DIR
from shared.memory import MemoryManager


def registration_form(users_file):
    st.title("Register New Account")
    username = st.session_state.get("username", "")
    password = st.session_state.get("password", "")
    st.write(f"Username: **{username}**")

    st.markdown("---")
    st.subheader("Basic Information")

    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        age = st.number_input("Age", min_value=5, max_value=120, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        activity_level = st.selectbox("Activity Level", [
            "Sedentary (desk job, no exercise)",
            "Lightly active (light exercise 1-3 days/week)",
            "Moderately active (moderate exercise 3-5 days/week)",
            "Very active (hard exercise 6-7 days/week)",
            "Extremely active (very hard exercise, physical job)",
        ])

    st.markdown("---")
    st.subheader("Goals & Preferences")

    col3, col4 = st.columns(2)
    with col3:
        primary_goal = st.selectbox("Primary Goal", [
            "Lose weight", "Gain weight", "Maintain weight", "Build muscle",
            "Improve health", "Increase energy", "Better sleep",
        ])
        dietary_preferences = st.multiselect("Dietary Preferences/Restrictions", [
            "None", "Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo",
            "Low-carb", "Low-fat", "Gluten-free", "Dairy-free", "Halal",
            "Kosher", "Nut allergies", "Shellfish allergies",
        ])
    with col4:
        cooking_experience = st.selectbox("Cooking Experience", [
            "Beginner (prefer simple recipes)",
            "Intermediate (comfortable with most recipes)",
            "Advanced (enjoy complex cooking)",
        ])
        budget_range = st.selectbox("Weekly Food Budget", [
            "$50-100", "$100-150", "$150-200", "$200-300", "$300+", "No specific budget",
        ])

    st.markdown("---")
    st.subheader("Meal Schedule")

    col5a, col5b = st.columns(2)
    with col5a:
        st.markdown("**Breakfast**")
        breakfast_enabled = st.checkbox("I eat breakfast", value=True)
        breakfast_location = st.selectbox("Breakfast location", ["At home", "On the go", "At work/office", "Varies"], key="breakfast_location")
        breakfast_time = st.selectbox("Breakfast cooking time", ["0 minutes", "5-10 minutes", "15-20 minutes", "30+ minutes"], key="breakfast_time")

        st.markdown("**Lunch**")
        lunch_enabled = st.checkbox("I eat lunch", value=True)
        lunch_location = st.selectbox("Lunch location", ["At work/office", "At home", "Restaurant/cafeteria", "Varies"], key="lunch_location")
        lunch_time = st.selectbox("Lunch cooking time", ["0 minutes", "5-10 minutes", "15-20 minutes", "30+ minutes", "Meal prep"], key="lunch_time")

    with col5b:
        st.markdown("**Dinner**")
        dinner_enabled = st.checkbox("I eat dinner", value=True)
        dinner_location = st.selectbox("Dinner location", ["At home", "Restaurant", "At work", "Varies"], key="dinner_location")
        dinner_time = st.selectbox("Dinner cooking time", ["0 minutes", "15-30 minutes", "30-60 minutes", "1+ hours", "Varies"], key="dinner_time")

        st.markdown("**Snacks**")
        snacks_enabled = st.checkbox("I eat snacks", value=True)
        snacks_frequency = st.selectbox("Snack frequency", ["No snacks", "1/day", "2/day", "3+/day", "As needed"], key="snacks_frequency")
        snacks_type = st.selectbox("Preferred snack type", ["Quick/ready-made", "Light prep", "Homemade", "Mixed", "Healthy only"], key="snacks_type")

    st.markdown("---")
    st.subheader("Health & Food Preferences")

    health_conditions = st.multiselect("Health Conditions (optional)", [
        "None", "Diabetes", "High blood pressure", "High cholesterol",
        "Heart disease", "PCOS", "Thyroid issues", "IBS/Digestive issues",
        "Food allergies", "Other",
    ])
    water_intake_goal = st.selectbox("Daily Water Intake Goal", ["1-2 liters", "2-3 liters", "3-4 liters", "4+ liters", "Not sure"])
    favorite_cuisines = st.multiselect("Favorite Cuisines", [
        "American", "Italian", "Mexican", "Asian", "Indian",
        "Mediterranean", "Middle Eastern", "French", "African", "Latin American", "No preference",
    ])
    foods_to_avoid = st.text_area("Foods to avoid (optional)", height=60)
    additional_notes = st.text_area("Additional notes (optional)", height=80)

    if st.button("Complete Registration", type="primary"):
        if not username or not password:
            st.error("Username and password are required.")
            return

        profile = {
            "weight": weight, "height": height, "age": age, "gender": gender,
            "activity_level": activity_level, "primary_goal": primary_goal,
            "dietary_preferences": dietary_preferences, "cooking_experience": cooking_experience,
            "budget_range": budget_range,
            "meal_schedule": {
                "breakfast": {"enabled": breakfast_enabled, "location": breakfast_location, "cooking_time": breakfast_time},
                "lunch": {"enabled": lunch_enabled, "location": lunch_location, "cooking_time": lunch_time},
                "dinner": {"enabled": dinner_enabled, "location": dinner_location, "cooking_time": dinner_time},
                "snacks": {"enabled": snacks_enabled, "frequency": snacks_frequency, "type": snacks_type},
            },
            "health_conditions": health_conditions, "water_intake_goal": water_intake_goal,
            "favorite_cuisines": favorite_cuisines, "foods_to_avoid": foods_to_avoid,
            "additional_notes": additional_notes, "registration_date": datetime.now().isoformat(),
        }

        success = register_user(username, password, profile, users_file)
        if success:
            st.session_state["is_new_user"] = False
            st.session_state["initialized"] = True
            st.session_state["show_register"] = False

            memory = MemoryManager(username, SECRETS_DIR)
            memory.migrate_from_legacy(profile)

            st.success("Registration complete! Welcome to NutriCoach!")
            st.rerun()
        else:
            st.error("Username already exists. Please choose another.")

