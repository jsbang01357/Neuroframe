# admin.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import datetime as dt

import streamlit as st

from auth import get_repo, login_guard, render_user_badge
from storage.gsheets import NeuroGSheets  # repo.db 접근용 (records/raw)
from storage.repo import NeuroRepo


# ----------------------------
# Admin gate
# ----------------------------

def _parse_allowlist(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _load_admin_policy() -> tuple[str, List[str]]:
    secret = None
    if "admin_pass" in st.secrets:
        secret = st.secrets["admin_pass"]
    elif "admin" in st.secrets and "admin_pass" in st.secrets["admin"]:
        secret = st.secrets["admin"]["admin_pass"]
    if secret is None:
        raise RuntimeError("admin_pass가 secrets에 없습니다. (admin_pass 또는 [admin].admin_pass)")

    allow_raw = None
    if "admin_allowlist" in st.secrets:
        allow_raw = st.secrets["admin_allowlist"]
    elif "admin" in st.secrets and "allowlist" in st.secrets["admin"]:
        allow_raw = st.secrets["admin"]["allowlist"]
    allowlist = _parse_allowlist(allow_raw)
    return str(secret).strip(), allowlist


def _admin_gate(repo: NeuroRepo):
    st.title("NeuroFrame Admin")
    auth = login_guard(repo, title="관리자 로그인")
    user = auth.user
    assert user is not None
    render_user_badge(user)

    try:
        secret, allowlist = _load_admin_policy()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if allowlist and user.username not in allowlist:
        st.error("관리자 권한이 없는 계정입니다.")
        st.stop()

    st.sidebar.header("관리자 인증")
    pw = st.sidebar.text_input("Admin Pass", type="password")
    pw = str(pw or "").strip()

    if pw != secret:
        st.info("관리자 비밀번호를 입력해주세요.")
        st.stop()
    return user



# ----------------------------
# Helpers
# ----------------------------

def _to_bool_str(v: bool) -> str:
    return "true" if v else "false"

def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _sheet_time_str(hhmm: str, fallback: str) -> str:
    s = (hhmm or "").strip()
    if not s:
        return fallback
    # minimal validation
    if ":" not in s:
        return fallback
    hh, mm = s.split(":")
    hh = int(hh); mm = int(mm)
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    return f"{hh:02d}:{mm:02d}"

def _fetch_all_users(db: NeuroGSheets) -> List[Dict[str, Any]]:
    # users sheet with header -> get_all_records
    return db.users.get_all_records()

def _fetch_logs_for_user(db: NeuroGSheets, username: str) -> List[Dict[str, Any]]:
    logs = db.logs.get_all_records()
    out = [r for r in logs if str(r.get("username", "")).strip() == username]
    # sort by date desc (YYYY-MM-DD)
    out.sort(key=lambda r: str(r.get("date", "")), reverse=True)
    return out


@st.cache_data(ttl=30, show_spinner=False)
def _cached_fetch_all_users(_db: NeuroGSheets) -> List[Dict[str, Any]]:
    return _fetch_all_users(_db)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_fetch_logs_for_user(_db: NeuroGSheets, username: str) -> List[Dict[str, Any]]:
    return _fetch_logs_for_user(_db, username)


def _invalidate_admin_caches() -> None:
    _cached_fetch_all_users.clear()
    _cached_fetch_logs_for_user.clear()


# ----------------------------
# UI
# ----------------------------

def render_admin_page(repo: NeuroRepo, spreadsheet_name: str = "NeuroFrame_DB") -> None:
    """
    Admin dashboard:
      - list users
      - edit baseline/weights
      - view recent daily logs
    """
    admin_user = _admin_gate(repo)

    # We want direct access to sheets for bulk listing
    # repo.db is NeuroGSheets instance
    db: NeuroGSheets = repo.db  # type: ignore

    st.caption(f"관리자 세션: `{admin_user.username}`")
    st.caption("운영 전용 페이지입니다. 의료 조언을 제공하지 않습니다.")

    st.divider()
    st.subheader("보안 상태")
    pw_stat = repo.count_password_rows()
    csec1, csec2, csec3, csec4 = st.columns(4)
    csec1.metric("users 총계", str(pw_stat.get("total", 0)))
    csec2.metric("해시 저장", str(pw_stat.get("hashed", 0)))
    csec3.metric("평문 잔존", str(pw_stat.get("plaintext", 0)))
    csec4.metric("비어있음", str(pw_stat.get("empty", 0)))

    if st.button("평문 비밀번호 일괄 해시 마이그레이션", use_container_width=True):
        migrated = repo.migrate_plaintext_passwords()
        _invalidate_admin_caches()
        st.success(f"{migrated}개 계정을 해시로 변환했습니다.")
        st.rerun()

    st.subheader("유저 목록")
    users = _cached_fetch_all_users(db)
    if not users:
        st.warning("users 시트에 유저가 없습니다.")
        return

    # Build selector options
    usernames = [u.get("username", "") for u in users if u.get("username", "")]
    usernames = sorted(set(usernames))

    col1, col2 = st.columns([1, 2])
    with col1:
        selected = st.selectbox("유저 선택", usernames)

    # Pull the selected user row (via repo.get_user gives parsed baseline)
    uobj = repo.get_user(selected)
    if not uobj:
        st.error("유저 정보를 불러오지 못했습니다.")
        return

    with col2:
        st.markdown(f"**선택된 유저:** `{uobj.username}`")
        st.markdown(f"**Onboarded:** `{uobj.onboarded}`")

    st.divider()
    st.subheader("Baseline / Weights 편집")

    # Read raw user row too (for password/created_at etc if you want)
    raw_user = db.get_user(selected) or {}

    # Editable fields (strings for time)
    c1, c2, c3 = st.columns(3)
    with c1:
        sleep_start = st.text_input(
            "baseline_sleep_start (HH:MM)",
            value=_sheet_time_str(raw_user.get("baseline_sleep_start", ""), "23:30"),
        )
        wake = st.text_input(
            "baseline_wake (HH:MM)",
            value=_sheet_time_str(raw_user.get("baseline_wake", ""), "07:30"),
        )
        chronoshift = st.number_input(
            "chronotype_shift_hours",
            value=_safe_float(raw_user.get("chronotype_shift_hours", 0.0), 0.0),
            step=0.5
        )
    with c2:
        half_life = st.number_input(
            "caffeine_half_life_hours",
            value=_safe_float(raw_user.get("caffeine_half_life_hours", 5.0), 5.0),
            step=0.5
        )
        sensitivity = st.number_input(
            "caffeine_sensitivity",
            value=_safe_float(raw_user.get("caffeine_sensitivity", 1.0), 1.0),
            step=0.1
        )
        offset = st.number_input(
            "baseline_offset",
            value=_safe_float(raw_user.get("baseline_offset", 0.0), 0.0),
            step=0.05
        )
    with c3:
        w_c = st.number_input(
            "circadian_weight",
            value=_safe_float(raw_user.get("circadian_weight", 1.0), 1.0),
            step=0.1
        )
        w_s = st.number_input(
            "sleep_pressure_weight",
            value=_safe_float(raw_user.get("sleep_pressure_weight", 1.2), 1.2),
            step=0.1
        )
        w_d = st.number_input(
            "drug_weight",
            value=_safe_float(raw_user.get("drug_weight", 0.004), 0.004),
            step=0.001,
            format="%.6f"
        )
        w_l = st.number_input(
            "load_weight",
            value=_safe_float(raw_user.get("load_weight", 0.2), 0.2),
            step=0.05
        )

    onboarded = st.toggle("onboarded", value=bool(uobj.onboarded))

    save = st.button("저장", type="primary")
    if save:
        patch = {
            "baseline_sleep_start": _sheet_time_str(sleep_start, "23:30"),
            "baseline_wake": _sheet_time_str(wake, "07:30"),
            "chronotype_shift_hours": str(float(chronoshift)),
            "caffeine_half_life_hours": str(float(half_life)),
            "caffeine_sensitivity": str(float(sensitivity)),
            "baseline_offset": str(float(offset)),
            "circadian_weight": str(float(w_c)),
            "sleep_pressure_weight": str(float(w_s)),
            "drug_weight": str(float(w_d)),
            "load_weight": str(float(w_l)),
            "onboarded": _to_bool_str(onboarded),
        }
        ok = repo.update_user_baseline(selected, patch)
        if ok:
            _invalidate_admin_caches()
            st.success("저장되었습니다.")
        else:
            st.error("저장에 실패했습니다.")
        st.rerun()

    st.divider()
    st.subheader("모델 개인화 (자동 가중치 튜닝)")
    st.caption("최근 7일간의 주관적 명료도 로그를 분석하여 카페인 민감도와 일주기 가중치를 최적화합니다.")
    if st.button("가중치 자동 튜닝 실행", use_container_width=True):
        from neuroframe.engine import tune_user_weights
        recent_logs = _fetch_logs_for_user(db, selected)[:7]
        
        if not recent_logs:
            st.warning("분석할 최근 7일간의 로그 데이터가 없습니다.")
        else:
            updated_weights = tune_user_weights(uobj.baseline, recent_logs)
            if not updated_weights:
                st.info("유의미한 주관적 명료도 데이터가 부족하여 튜닝이 보류되었습니다 (최소 1일치 필요).")
            else:
                ok = repo.update_user_baseline(selected, {
                    "caffeine_sensitivity": str(updated_weights.get("caffeine_sensitivity", sensitivity)),
                    "circadian_weight": str(updated_weights.get("circadian_weight", w_c))
                })
                if ok:
                    st.success(f"가중치 자동 튜닝 완료! 변경 사항: {updated_weights}")
                    st.rerun()
                else:
                    st.error("가중치 업데이트에 실패했습니다.")

    st.divider()
    st.subheader("Daily Logs 조회")

    logs = _cached_fetch_logs_for_user(db, selected)
    n = st.slider("표시 개수", 5, 100, 20)
    logs = logs[:n]

    if not logs:
        st.info("해당 유저의 daily_logs가 없습니다.")
    else:
        # Display as table
        # Keep only key columns for readability
        cols = [
            "date",
            "sleep_override_on",
            "sleep_cross_midnight",
            "sleep_start",
            "wake_time",
            "workload_level",
            "subjective_clarity",
            "updated_at",
        ]
        table = [{k: r.get(k, "") for k in cols} for r in logs]
        st.dataframe(table, use_container_width=True)

        # Optional: show doses_json on demand
        with st.expander("doses_json 보기 (최근 로그)"):
            for r in logs[:10]:
                st.markdown(f"**{r.get('date','')}**  · doses_json:")
                st.code(r.get("doses_json", "[]") or "[]", language="json")

    st.divider()
    st.subheader("최근 오류 로그")
    err_rows = repo.get_recent_admin_logs(limit=30)
    err_rows = [r for r in err_rows if str(r.get("level", "")).lower() in ("error", "warn", "warning")]
    if not err_rows:
        st.caption("최근 오류 로그가 없습니다.")
    else:
        st.dataframe(err_rows[:20], use_container_width=True)


# ----------------------------
# Entry point for streamlit
# ----------------------------

def main():
    repo = get_repo("NeuroFrame_DB")
    render_admin_page(repo)


if __name__ == "__main__":
    main()
