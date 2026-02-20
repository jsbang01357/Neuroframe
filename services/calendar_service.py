from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import streamlit as st

from shared.calendar_integration import CalendarEvent, fetch_google_calendar_events_for_day


def _read_calendar_config() -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Returns (service_account_info, calendar_id, delegated_user).
    calendar_id can be configured as:
      - st.secrets["google_calendar_id"]
      - st.secrets["calendar"]["id"]
    delegated user can be configured as:
      - st.secrets["google_calendar_delegated_user"]
      - st.secrets["calendar"]["delegated_user"]
    """
    sa = None
    if "gcp_service_account" in st.secrets:
        sa = dict(st.secrets["gcp_service_account"])

    calendar_id = ""
    delegated = ""
    if "google_calendar_id" in st.secrets:
        calendar_id = str(st.secrets["google_calendar_id"] or "").strip()
    elif "calendar" in st.secrets and "id" in st.secrets["calendar"]:
        calendar_id = str(st.secrets["calendar"]["id"] or "").strip()

    if "google_calendar_delegated_user" in st.secrets:
        delegated = str(st.secrets["google_calendar_delegated_user"] or "").strip()
    elif "calendar" in st.secrets and "delegated_user" in st.secrets["calendar"]:
        delegated = str(st.secrets["calendar"]["delegated_user"] or "").strip()
    return sa, calendar_id, delegated


@st.cache_data(ttl=180, show_spinner=False)
def _cached_google_calendar_events(day_iso: str, tz_name: str) -> List[CalendarEvent]:
    sa, cal_id, delegated = _read_calendar_config()
    if not sa or not cal_id:
        return []
    day = dt.date.fromisoformat(day_iso)
    tz = ZoneInfo(tz_name)
    try:
        return fetch_google_calendar_events_for_day(
            service_account_info=sa,
            calendar_id=cal_id,
            day=day,
            tz=tz,
            delegated_user=delegated,
        )
    except Exception:
        return []


def _calendar_readiness_message() -> Tuple[bool, str]:
    sa, cal_id, delegated = _read_calendar_config()
    if not sa:
        return (
            False,
            "Calendar 연동 비활성화: `st.secrets['gcp_service_account']`가 없습니다. 서비스 계정 키를 추가하세요.",
        )
    if not cal_id:
        return (
            False,
            "Calendar 연동 비활성화: `google_calendar_id`(또는 `[calendar].id`)가 없습니다.",
        )
    deleg_txt = delegated if delegated else "(미설정)"
    return (
        True,
        f"calendar_id=`{cal_id}`, delegated_user=`{deleg_txt}`",
    )


def _read_google_calendar_events_with_error(day: dt.date, tz: dt.tzinfo) -> Tuple[List[CalendarEvent], Optional[str]]:
    sa, cal_id, delegated = _read_calendar_config()
    try:
        events = fetch_google_calendar_events_for_day(
            service_account_info=sa or {},
            calendar_id=cal_id,
            day=day,
            tz=tz,
            delegated_user=delegated,
        )
        return events, None
    except Exception as e:
        return [], str(e)


def _invalidate_calendar_cache() -> None:
    _cached_google_calendar_events.clear()
