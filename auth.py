import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


def _cfg(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


@st.cache_resource
def get_supabase() -> Client:
    """Single cached client — persists across rerenders so PKCE verifier survives the OAuth redirect."""
    return create_client(_cfg("SUPABASE_URL"), _cfg("SUPABASE_ANON_KEY"))


def app_url() -> str:
    return _cfg("APP_URL") or "http://localhost:8501"


# ── OAuth flow ────────────────────────────────────────────────────────────────

def get_google_oauth_url(supabase: Client) -> str:
    """Return the Google sign-in URL. Supabase stores the PKCE verifier inside the cached client."""
    resp = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": app_url(),
            "skip_browser_redirect": True,
        },
    })
    return resp.url


def handle_oauth_callback(supabase: Client) -> bool:
    """Exchange ?code= param for a session. Returns True if a new session was established."""
    code = st.query_params.get("code")
    if not code:
        return False
    try:
        resp = supabase.auth.exchange_code_for_session({"auth_code": code})
        _store_session(resp.user, resp.session.access_token, resp.session.refresh_token)
        st.query_params.clear()
        return True
    except Exception as e:
        st.query_params.clear()
        st.error(f"Login failed — please try again. ({e})")
        return False


# ── Session helpers ───────────────────────────────────────────────────────────

def restore_session(supabase: Client) -> bool:
    """Try to restore a session from st.session_state tokens. Returns True if valid."""
    access = st.session_state.get("sb_access_token")
    refresh = st.session_state.get("sb_refresh_token", "")
    if not access:
        return False
    try:
        resp = supabase.auth.set_session(access, refresh)
        st.session_state["user"] = resp.user
        return True
    except Exception:
        _clear_session()
        return False


def logout(supabase: Client) -> None:
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    _clear_session()


def current_user():
    return st.session_state.get("user")


def is_logged_in() -> bool:
    return "user" in st.session_state


# ── Internals ─────────────────────────────────────────────────────────────────

def _store_session(user, access_token: str, refresh_token: str) -> None:
    st.session_state["user"] = user
    st.session_state["sb_access_token"] = access_token
    st.session_state["sb_refresh_token"] = refresh_token


def _clear_session() -> None:
    for k in ("user", "sb_access_token", "sb_refresh_token"):
        st.session_state.pop(k, None)
