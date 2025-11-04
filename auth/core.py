import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from typing import Tuple, Optional


def load_auth_config(yaml_path: str = "credentials.yaml") -> Optional[dict]:
    """Load authentication configuration from YAML"""
    try:
        with open(yaml_path) as file:
            return yaml.load(file, Loader=SafeLoader)
    except Exception as e:
        st.error(f"Failed to load credentials: {e}")
        st.stop()  # stop immediately if config can't be loaded


def create_authenticator(config: dict) -> stauth.Authenticate:
    """Create authenticator instance"""
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config.get("preauthorized"),
    )


def authenticate(yaml_path: str = "credentials.yaml") -> Tuple[bool, Optional[stauth.Authenticate]]:
    """
    Force login page until authenticated.
    Entire app execution stops if not logged in.
    """
    # Apply dark theme CSS for login page
    st.markdown("""
    <style>
        /* Dark theme for login page */
        .stApp {
            background-color: #0f0f0f;
        }

        .main {
            background-color: #0f0f0f;
            color: #e5e5e5;
        }

        /* Toolbar - Make it black */
        [data-testid="stToolbar"],
        .stAppToolbar,
        .st-emotion-cache-14vh5up {
            background-color: #0f0f0f !important;
            border-bottom: 1px solid #2a2a2a !important;
        }

        /* Toolbar inner containers */
        [data-testid="stToolbar"] > div,
        .stAppToolbar > div,
        .st-emotion-cache-1j22a0y,
        .st-emotion-cache-70qvj9,
        .st-emotion-cache-scp8yw {
            background-color: transparent !important;
        }

        /* Toolbar buttons */
        [data-testid="stToolbar"] button,
        .stAppToolbar button {
            color: #e5e5e5 !important;
            background-color: transparent !important;
        }

        /* Main menu button */
        [data-testid="stMainMenu"],
        .stMainMenu {
            color: #e5e5e5 !important;
        }

        [data-testid="stMainMenu"] button {
            background-color: transparent !important;
        }

        /* Main menu icon */
        [data-testid="stMainMenu"] svg {
            color: #e5e5e5 !important;
            fill: #e5e5e5 !important;
        }

        /* Deploy button */
        [data-testid="stAppDeployButton"] button {
            background-color: #3b82f6 !important;
            color: white !important;
        }

        /* Login form container */
        [data-testid="stForm"] {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 2rem;
        }

        /* Input fields */
        .stTextInput > div > div > input {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            color: #e5e5e5;
        }

        .stTextInput > div > div > input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 1px #3b82f6;
        }

        /* Labels */
        label {
            color: #e5e5e5 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            width: 100%;
        }

        .stButton > button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }

        /* Error/Warning messages */
        .stAlert {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            color: #e5e5e5;
        }

        /* Success message */
        .stSuccess {
            background-color: #1a3a1a;
            border: 1px solid #10b981;
            color: #10b981;
        }

        /* Error message */
        .stError {
            background-color: #3a1a1a;
            border: 1px solid #ef4444;
            color: #ef4444;
        }

        /* Warning message */
        .stWarning {
            background-color: #3a2f1a;
            border: 1px solid #f59e0b;
            color: #f59e0b;
        }

        /* Markdown text */
        .stMarkdown {
            color: #e5e5e5;
        }

        /* Sidebar for logout */
        [data-testid="stSidebar"] {
            background-color: #171717;
            border-right: 1px solid #2a2a2a;
        }

        /* Dividers */
        hr {
            border-color: #2a2a2a;
        }
    </style>
    """, unsafe_allow_html=True)

    config = load_auth_config(yaml_path)
    authenticator = create_authenticator(config)

    name, auth_status, username = authenticator.login("Login", "main")

    # Save session state
    st.session_state["authentication_status"] = auth_status
    st.session_state["name"] = name
    st.session_state["username"] = username

    # Gatekeeping logic
    if auth_status is False:
        st.error("❌ Incorrect username or password")
        st.stop()
    elif auth_status is None:
        st.warning("🔐 Please log in to continue")
        st.stop()

    return True, authenticator


def show_user_info(authenticator: stauth.Authenticate) -> None:
    """Sidebar user info + logout button"""
    if st.session_state.get("authentication_status"):
        with st.sidebar:
            st.success(f"👋 Welcome *{st.session_state.get('name', 'User')}*")
            authenticator.logout("🚪 Logout", "sidebar")
            st.divider()