# auth.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from storage.gsheets import NeuroGSheets, GSheetsConfig
from storage.repo import NeuroRepo, RepoUser


# ----------------------------
# Repo factory
# ----------------------------

def make_gspread_client_from_secrets() -> gspread.Client:
    """
    Expects Streamlit secrets:
      st.secrets["gcp_service_account"] = { ... service account json ... }
    """
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Missing st.secrets['gcp_service_account'] for Google service account credentials.")

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def get_repo(spreadsheet_name: str = "NeuroFrame_DB") -> NeuroRepo:
    """
    Cached repo instance.
    """
    @st.cache_resource
    def _build_repo(name: str) -> NeuroRepo:
        gc = make_gspread_client_from_secrets()
        db = NeuroGSheets(gc, GSheetsConfig(spreadsheet_name=name))
        return NeuroRepo(db)

    return _build_repo(spreadsheet_name)


# ----------------------------
# Session helpers
# ----------------------------

def is_logged_in() -> bool:
    return bool(st.session_state.get("username"))


def logout():
    for k in ["username"]:
        if k in st.session_state:
            del st.session_state[k]


# ----------------------------
# UI
# ----------------------------

@dataclass
class AuthResult:
    ok: bool
    user: Optional[RepoUser] = None
    message: str = ""


def login_guard(repo: NeuroRepo, title: str = "로그인") -> AuthResult:
    """
    Call this at the top of your app.
    - If logged in: returns ok=True, user loaded
    - Else: renders sidebar login/signup UI and stops the app flow
    """
    if is_logged_in():
        u = repo.get_user(st.session_state["username"])
        if not u:
            # user disappeared?
            logout()
            st.warning("세션이 만료되었을 수 있습니다. 다시 로그인해주세요.")
            st.stop()
        return AuthResult(ok=True, user=u)

    st.sidebar.header(title)
    tab_login, tab_signup = st.sidebar.tabs(["로그인", "회원가입"])

    # --- Login tab ---
    with tab_login:
        username = st.text_input("아이디", key="auth_login_username")
        password = st.text_input("비밀번호", type="password", key="auth_login_password")
        if st.button("로그인", type="primary", use_container_width=True, key="auth_login_btn"):
            if not username or not password:
                st.sidebar.error("아이디/비밀번호를 입력해주세요.")
            elif repo.verify_login(username, password):
                st.session_state["username"] = username
                repo.touch_login(username)
                st.sidebar.success("로그인되었습니다.")
                st.rerun()
            else:
                st.sidebar.error("로그인 정보가 올바르지 않습니다.")

    # --- Signup tab ---
    with tab_signup:
        new_username = st.text_input("아이디", key="auth_signup_username")
        new_password = st.text_input("비밀번호", type="password", key="auth_signup_password")
        new_password2 = st.text_input("비밀번호 확인", type="password", key="auth_signup_password2")

        if st.button("회원가입", use_container_width=True, key="auth_signup_btn"):
            if not new_username or not new_password:
                st.sidebar.error("아이디/비밀번호를 입력해주세요.")
            elif new_password != new_password2:
                st.sidebar.error("비밀번호가 일치하지 않습니다.")
            else:
                ok = repo.create_user(new_username, new_password)
                if ok:
                    st.sidebar.success("회원가입이 완료되었습니다. 로그인해주세요.")
                else:
                    st.sidebar.error("이미 존재하는 아이디입니다.")

    st.sidebar.divider()
    st.sidebar.caption("NeuroFrame은 치료 앱이 아니라, 에너지 곡선을 기반으로 하루를 설계하는 엔진입니다.")
    st.stop()


def render_user_badge(user: RepoUser):
    """
    Optional: show current user + logout button in sidebar.
    """
    st.sidebar.divider()
    st.sidebar.markdown(f"**로그인됨:** `{user.username}`")
    if st.sidebar.button("로그아웃", use_container_width=True):
        logout()
        st.rerun()
