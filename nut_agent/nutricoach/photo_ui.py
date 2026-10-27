"""Food photo analysis and explicit meal review before journal writes."""

import hashlib
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from nutricoach.tools import save_photo_analysis
from nutricoach.food_vision.nutrition_db import NutritionDB
from nutricoach.food_vision.review import NUTRIENT_FIELDS, photo_edit_rows, apply_photo_edits


def _save_upload(uploaded, secrets_dir) -> Path:
    secrets_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile("wb", dir=secrets_dir, suffix=suffix, delete=False) as f:
        f.write(uploaded.getvalue())
        return Path(f.name)


def display_food_analysis(thread_config, analyzer, secrets_dir):
    """Tab for analyzing food photos."""
    st.subheader("Analyze Food Photo")
    st.markdown("Upload a photo of your meal to estimate its total mass, calories, and macros.")
    method = st.selectbox("Analysis method", ["vlm_single", "foodlmm_4bit", "dpf_reproduction"],
                          format_func=lambda value: {"vlm_single": "One-shot VLM",
                                                     "foodlmm_4bit": "FoodLMM (4-bit)",
                                                     "dpf_reproduction": "DPF (reproduction)"}[value],
                          key="food_vision_method")
    guided_prompt = st.toggle("Use guided nutrition prompt", value=True, key="food_vision_guided_prompt") \
        if method == "vlm_single" else False
    if method == "vlm_single":
        st.caption("Uses the provider and model selected in the sidebar, always with high reasoning.")
    elif method == "foodlmm_4bit":
        st.caption("Runs locally, with no API key. Model files download on first use.")
        st.caption("FoodLMM (4-bit) requires its separate environment. A prepared 4-bit checkpoint reduces loading memory.")
    else:
        st.caption("Runs locally, with no API key. Model files download on first use.")
        st.caption("DPF estimates whole-dish nutrition using RGB and predicted depth.")

    uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png", "webp"])
    active_method = "vlm_single_guided" if method == "vlm_single" and guided_prompt else method
    context = None
    if uploaded:
        context = (st.session_state.get("username"), active_method,
                   hashlib.sha256(uploaded.getvalue()).hexdigest())
        if method == "vlm_single":
            context += (st.session_state.get("llm_provider"), st.session_state.get("chat_model_id"))
    if st.session_state.get("photo_context") != context:
        st.session_state["photo_context"] = context
        st.session_state.pop("photo_result", None)
        st.session_state["photo_logged"] = False
    if not uploaded:
        return
    st.image(uploaded, caption="Your meal", use_container_width=True)
    if st.button("Analyze photo"):
        st.session_state.pop("photo_result", None)
        tmp_path = _save_upload(uploaded, secrets_dir)
        try:
            with st.spinner("Analyzing your meal..."):
                result = analyzer.invoke({"image_path": str(tmp_path), "method": method,
                                                    "guided_prompt": guided_prompt}, config=thread_config())
            if result.get("error"):
                st.error(result["error"])
            elif result.get("method") != active_method:
                st.error("The analysis returned a different method than selected. Please restart the app and analyze again.")
            else:
                st.session_state["photo_result"] = result
                st.session_state["photo_review_id"] = uuid.uuid4().hex[:8]
                st.session_state["photo_logged"] = False
        finally:
            tmp_path.unlink(missing_ok=True)
    if st.session_state.get("photo_result"):
        display_photo_review(thread_config)


def display_photo_review(thread_config):
    original = st.session_state["photo_result"]
    review_id = st.session_state["photo_review_id"]
    logged = st.session_state.get("photo_logged", False)
    st.markdown("**Review your meal**")
    st.caption("Analysis method: " + {"vlm_single": "One-shot VLM", "vlm_single_guided": "One-shot VLM (guided)",
                                      "foodlmm_4bit": "FoodLMM (4-bit)", "dpf_reproduction": "DPF (reproduction)"}
               .get(original["method"], original["method"]))
    meal_name = st.text_input("Meal name", value="Photo meal", key=f"photo_name_{review_id}", disabled=logged)
    st.caption("Edit grams, add rows, or select rows to delete ingredients. When changing a food, choose its nutrition source. "
               "Manual nutrient columns are per 100g; selecting a database food overrides those columns.")
    try:
        rows = st.session_state["photo_saved_rows"] if logged else photo_edit_rows(original)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
        return
    sources = ["Photo estimate", "Manual", *[f"Database: {name}" for name in sorted(NutritionDB().db)]]
    columns = {
        "original_index": None,
        "name": st.column_config.TextColumn("Food composition", required=True),
        "quantity_grams": st.column_config.NumberColumn("Grams", min_value=0.01, required=True),
        "nutrition_source": st.column_config.SelectboxColumn("Nutrition source", options=sources, required=True),
        **{field: st.column_config.NumberColumn(f"{field.replace('_g', '')} / 100g", min_value=0.0)
           for field in NUTRIENT_FIELDS},
    }
    edited = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", hide_index=True,
                            column_config=columns, disabled=True if logged else ["original_index"],
                            key=f"photo_editor_{review_id}", use_container_width=True)
    try:
        corrected = apply_photo_edits(original, edited.to_dict("records"))
    except (ValueError, TypeError, KeyError) as exc:
        st.info(f"Complete your corrections: {exc}")
        return
    st.write(f"**{corrected.total_calories:.0f} kcal** · Protein: {corrected.total_protein_g:.1f}g · "
             f"Carbs: {corrected.total_carbs_g:.1f}g · Fat: {corrected.total_fat_g:.1f}g")
    if logged:
        st.success("This meal has been saved to today's journal.")
    elif st.button("Save reviewed meal to today's journal", key=f"photo_save_{review_id}"):
        try:
            save_photo_analysis(corrected, thread_config(), meal_name=meal_name)
            st.session_state["photo_saved_rows"] = edited.to_dict("records")
            st.session_state["photo_logged"] = True
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(f"Could not save meal: {exc}")

