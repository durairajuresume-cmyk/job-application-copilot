import os
import secrets as _secrets
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
    """Single cached Supabase client shared across rerenders."""
    return create_client(_cfg("SUPABASE_URL"), _cfg("SUPABASE_ANON_KEY"))


def app_url() -> str:
    return _cfg("APP_URL") or "http://localhost:8501"


# ── OAuth flow ────────────────────────────────────────────────────────────────

def get_google_oauth_url(supabase: Client) -> str:
    """Return the Google sign-in URL.

    Persists the PKCE code verifier to Supabase so it survives process
    restarts and multi-worker Streamlit Cloud deployments. A random token
    is embedded in the redirect URL so the callback can look it up.
    """
    token = _secrets.token_urlsafe(16)
    redirect = f"{app_url()}?pkce={token}"

    resp = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": redirect,
            "skip_browser_redirect": True,
        },
    })

    # supabase-py stores the verifier on the GoTrueClient instance after
    # sign_in_with_oauth. Persist it so any worker can retrieve it on callback.
    verifier = getattr(supabase.auth, "_code_verifier", None)
    if verifier:
        try:
            supabase.table("pkce_verifiers").insert(
                {"token": token, "verifier": verifier}
            ).execute()
        except Exception:
            pass  # fallback to in-memory verifier if table not ready

    return resp.url


def handle_oauth_callback(supabase: Client) -> bool:
    """Exchange the ?code= param for a session. Returns True on success."""
    code = st.query_params.get("code")
    if not code:
        return False
    try:
        # Restore the PKCE verifier from DB using the token embedded in the URL
        verifier = None
        pkce_token = st.query_params.get("pkce")
        if pkce_token:
            try:
                r = (
                    supabase.table("pkce_verifiers")
                    .select("verifier")
                    .eq("token", pkce_token)
                    .limit(1)
                    .execute()
                )
                if r.data:
                    verifier = r.data[0]["verifier"]
                    # Single-use — delete immediately
                    supabase.table("pkce_verifiers").delete().eq("token", pkce_token).execute()
            except Exception:
                pass

        # If we have the verifier, pass it explicitly; otherwise rely on
        # whatever the current process has in memory (same-process fallback)
        params: dict = {"auth_code": code}
        if verifier:
            params["code_verifier"] = verifier

        resp = supabase.auth.exchange_code_for_session(params)
        _store_session(resp.user, resp.session.access_token, resp.session.refresh_token)
        st.query_params.clear()
        return True
    except Exception as e:
        st.query_params.clear()
        st.error(f"Login failed — please try again. ({e})")
        return False


# ── Session helpers ───────────────────────────────────────────────────────────

def restore_session(supabase: Client) -> bool:
    """Try to restore a session from st.session_state tokens."""
    access = st.session_state.get("sb_access_token")
    refresh = st.session_state.get("sb_refresh_token", "")
    if not access:
        return False
    try:
        resp = supabase.auth.set_session(access, refresh)
        st.session_state["user"] = resp.user
        if resp.session:
            st.session_state["sb_access_token"] = resp.session.access_token
            st.session_state["sb_refresh_token"] = resp.session.refresh_token
        return True
    except Exception:
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
