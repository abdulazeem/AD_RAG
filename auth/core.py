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