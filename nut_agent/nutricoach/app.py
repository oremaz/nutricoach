"""NutriCoach session lifecycle, provider settings and Streamlit page routing."""

import sys
import os
import logging
import uuid
from pathlib import Path

# On macOS, LightGBM must load before the agent's PyTorch dependencies.
if sys.platform == "darwin":
    try:
        import lightgbm
    except (ImportError, OSError):
        pass

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

from shared.config import OPENROUTER_REASONING_EFFORT, SECRETS_DIR
from shared.auth import authenticate_user, load_user_info
from shared.memory import MemoryManager
from nutricoach import photo_ui, registration_ui, tracking_ui
from nutricoach.tools import analyze_food_image
from nutricoach.agent import build_nutricoach_graph
from shared.llm_provider import PROVIDERS, REASONING_EFFORTS, default_model

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="NutriCoach",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

USERS_FILE = SECRETS_DIR / "users.json"


# --- Session State ---


def initialize_session_state():
    defaults = {
        "username": None,
        "password": None,
        "initialized": False,
        "is_new_user": True,
        "show_register": False,
        "agent_graph": None,
        "thread_id": None,
        "chat_history": [],
        "llm_provider": "openrouter",
        "chat_model_id": default_model("openrouter"),
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "local_quantization": "none",
        "local_device": os.getenv("LOCAL_LLM_DEVICE") or None,
        "reasoning_effort": OPENROUTER_REASONING_EFFORT,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


initialize_session_state()


# --- Agent Invocation ---


def _thread_config() -> dict:
    """LangGraph config with thread_id for checkpointer."""
    return {"configurable": {
        "thread_id": st.session_state.get("thread_id", "default"),
        "username": st.session_state.get("username"),
        "openrouter_api_key": st.session_state.get("api_key"),
        "groq_api_key": st.session_state.get("groq_api_key"),
        "chat_model_id": st.session_state.get("chat_model_id"),
        "llm_provider": st.session_state.get("llm_provider", "openrouter"),
        "local_device": st.session_state.get("local_device"),
        "local_quantization": st.session_state.get("local_quantization", "none"),
        "reasoning_effort": st.session_state.get("reasoning_effort", OPENROUTER_REASONING_EFFORT),
    }}


def invoke_agent(user_message: str) -> str:
    """
    Send a message to the agent and return the response.
    The checkpointer handles state persistence automatically.
    """
    agent_graph = st.session_state["agent_graph"]
    if agent_graph is None:
        return "Agent not initialized. Please enter your API key."

    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={
                **_thread_config(),
                "recursion_limit": 20,
            },
        )

        # Extract last AI message
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I processed your message. How can I help?"
    except Exception as e:
        logger.error("Agent invocation error: %s", e)
        return f"I encountered an error: {str(e)}. Please try again."


def send_message(user_message: str):
    """Send a message through the agent and update chat history."""
    st.session_state["chat_history"].append(("user", user_message))
    with st.spinner("NutriCoach is thinking..."):
        response = invoke_agent(user_message)
    st.session_state["chat_history"].append(("assistant", response))
    st.rerun()


# --- Session Management (simplified with checkpointer) ---


def start_new_chat_session():
    """Start a new chat thread."""
    st.session_state["thread_id"] = str(uuid.uuid4())[:8]
    st.session_state["chat_history"] = []
    username = st.session_state.get("username")
    if username:
        _ensure_memory_migrated(username)


def _ensure_memory_migrated(username):
    """Ensure the user's profile is migrated to structured memory."""
    memory = MemoryManager(username, SECRETS_DIR)
    if memory.load_user_profile() is None:
        user_info = load_user_info(username, USERS_FILE)
        if user_info:
            memory.migrate_from_legacy(user_info)


# --- Init Agent ---


def _change_llm_provider():
    provider = st.session_state["llm_provider"]
    st.session_state["chat_model_id"] = default_model(provider)
    st.session_state["reasoning_effort"] = "none" if provider == "local" else "high"
    st.session_state["agent_graph"] = None


def display_llm_settings():
    provider = st.selectbox("Model provider", PROVIDERS, key="llm_provider",
                            format_func=lambda value: {"openrouter": "OpenRouter", "groq": "Groq", "local": "Local (Qwen)"}[value],
                            on_change=_change_llm_provider)
    model_id = st.session_state.get("chat_model_id", "")
    if (not model_id or not model_id.strip()
            or (provider == "openrouter" and model_id == "dots-studio/dots-3-note-preview:free")):
        st.session_state["chat_model_id"] = default_model(provider)
    if provider != "local":
        label, key = ("OpenRouter", "api_key") if provider == "openrouter" else ("Groq", "groq_api_key")
        if provider == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            st.session_state[key] = os.environ["OPENROUTER_API_KEY"]
            st.text_input(f"{label} API Key", value="***", disabled=True)
            st.caption("Key configured in the environment.")
        else:
            st.text_input(f"{label} API Key", type="password", key=key)
    else:
        st.selectbox("Local precision", ("none", "4bit"), key="local_quantization",
                     format_func=lambda value: "BF16" if value == "none" else "4-bit (CUDA)")
        st.caption("Local inference needs downloaded weights and sufficient memory. 4-bit mode reduces GPU memory use.")
    st.text_input("Chat model ID", key="chat_model_id")
    st.selectbox("Reasoning effort", REASONING_EFFORTS[provider], key="reasoning_effort",
                 help="Applies to NutriCoach chat.")


def init_agent():
    provider = st.session_state.get("llm_provider", "openrouter")
    reasoning_effort = st.session_state.get("reasoning_effort", OPENROUTER_REASONING_EFFORT)
    api_key = ((os.getenv("OPENROUTER_API_KEY") or st.session_state.get("api_key", "")) if provider == "openrouter"
               else st.session_state.get("groq_api_key", ""))
    model_id = st.session_state.get("chat_model_id") or default_model(provider)
    signature = (provider, api_key if provider != "local" else None, model_id, reasoning_effort,
                 st.session_state.get("local_device"), st.session_state.get("local_quantization", "none"))
    if (st.session_state["agent_graph"] is not None
            and st.session_state.get("agent_settings") == signature):
        return True
    if api_key or provider == "local":
        username = st.session_state.get("username", "anonymous")
        try:
            st.session_state["agent_graph"] = build_nutricoach_graph(
                api_key, username, model_id=model_id.strip() or default_model(provider),
                reasoning_effort=reasoning_effort, provider=provider,
                local_device=st.session_state.get("local_device"),
                local_quantization=st.session_state.get("local_quantization", "none"))
            st.session_state["agent_settings"] = signature
            st.success("Agent initialized!")
            return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return False
    else:
        st.warning(f"Please enter your {provider.title()} API key.")
        return False


# --- Dashboard ---


def display_nutrition_dashboard():
    tracking_ui.display_nutrition_dashboard()


# --- Chat Interface ---


def display_chat_interface():
    st.subheader("Chat with NutriCoach")

    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state["chat_history"]:
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(message)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)

    user_input = st.chat_input("Type your message here...")
    if user_input and st.session_state["agent_graph"]:
        send_message(user_input)


def _save_upload(uploaded) -> Path:
    return photo_ui._save_upload(uploaded, SECRETS_DIR)


def display_food_analysis():
    photo_ui.display_food_analysis(_thread_config, analyze_food_image, SECRETS_DIR)


def display_photo_review():
    photo_ui.display_photo_review(_thread_config)


# --- Quick Actions ---


def display_quick_actions():
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Plan Today's Meals", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Please create a meal plan for today based on my profile and any constraints I mentioned.")
        if st.button("What Can I Still Eat Today?", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("What do I have left in my nutrition budget today? Suggest a meal or snack that fits.")

    with col2:
        if st.button("Calculate Nutrition Targets", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Please calculate my daily nutrition targets based on my profile.")
        if st.button("Grocery List From My Plan", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Build a grocery list from my meal plan for this week, grouped by store section.")

    with col3:
        if st.button("Check My Progress", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("How have I been doing this week? Please analyze my recent progress.")
        if st.button("Weekly Summary", use_container_width=True):
            if st.session_state["agent_graph"]:
                send_message("Generate my weekly summary and tell me what to improve next week.")


def display_daily_results():
    tracking_ui.display_daily_results(send_message)


# --- User Profile Display ---


def display_user_profile():
    username = st.session_state.get("username")
    if not username:
        return
    user_info = load_user_info(username, USERS_FILE)
    if not user_info:
        return
    with st.expander("Your Profile", expanded=False):
        st.write(f"**Age:** {user_info.get('age')}")
        st.write(f"**Weight:** {user_info.get('weight')} kg")
        st.write(f"**Height:** {user_info.get('height')} cm")
        st.write(f"**Goal:** {user_info.get('primary_goal')}")
        st.write(f"**Activity:** {user_info.get('activity_level')}")
        prefs = user_info.get("dietary_preferences", [])
        if prefs:
            st.write(f"**Diet:** {', '.join(prefs)}")


def registration_form():
    registration_ui.registration_form(USERS_FILE)


# --- Landing Page ---


def landing_page():
    st.title("Welcome to NutriCoach")
    username = st.text_input("Username", key="landing_username")
    password = st.text_input("Password", type="password", key="landing_password")

    col1, col2 = st.columns(2)
    with col1:
        login_btn = st.button("Login")
    with col2:
        register_btn = st.button("Register")

    if login_btn and username and password:
        user_info = authenticate_user(username, password, USERS_FILE)
        if user_info is not None:
            st.session_state["username"] = username
            st.session_state["is_new_user"] = False
            st.session_state["initialized"] = True
            start_new_chat_session()
            st.success(f"Login successful! Thread: {st.session_state['thread_id']}")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again or register.")

    elif register_btn and username and password:
        st.session_state["username"] = username
        st.session_state["password"] = password
        st.session_state["show_register"] = True
        st.rerun()


# --- Main App ---


def main_app():
    if not st.session_state.get("thread_id"):
        start_new_chat_session()

    st.title("NutriCoach")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.title("Control Panel")
        display_llm_settings()
        if not init_agent():
            st.stop()
        st.markdown("---")
        display_user_profile()

        thread_id = st.session_state.get("thread_id")
        if thread_id:
            st.info(f"Thread: {thread_id}")

        st.markdown("---")
        if st.button("New Conversation"):
            start_new_chat_session()
            st.rerun()

    # Main content tabs
    tab_chat, tab_food, tab_recipe, tab_dashboard, tab_actions, tab_track = st.tabs([
        "Chat", "Food Analysis", "Recipe Lab", "Dashboard", "Quick Actions", "Daily Tracking",
    ])

    with tab_chat:
        display_chat_interface()

    with tab_food:
        display_food_analysis()

    with tab_recipe:
        from recipe_lab.app import main as display_recipe_lab
        display_recipe_lab()

    with tab_dashboard:
        display_nutrition_dashboard()

    with tab_actions:
        display_quick_actions()

    with tab_track:
        display_daily_results()

    st.markdown("---")
    st.caption("NutriCoach v2: LangGraph, SqliteSaver, and Food Vision")


# --- Entry Point ---


def run():
    initialize_session_state()
    if not st.session_state.get("initialized", False):
        if st.session_state["show_register"]:
            registration_form()
        else:
            landing_page()
    else:
        main_app()


if __name__ == "__main__":
    run()
