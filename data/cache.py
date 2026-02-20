from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import streamlit as st


def _parse_date_iso(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(s))
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_daily_logs_for_user(_repo, username: str, limit: int = 120) -> List[Dict[str, Any]]:
    return _repo.get_daily_logs_for_user(username, limit=limit)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_daily_log(_repo, username: str, date_iso: str) -> Optional[Dict[str, Any]]:
    d = dt.date.fromisoformat(date_iso)
    return _repo.get_daily_log(username, d)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_checkin(_repo, username: str, date_iso: str) -> Optional[Dict[str, Any]]:
    d = dt.date.fromisoformat(date_iso)
    return _repo.get_checkin(username, d)


def _invalidate_repo_read_caches() -> None:
    _cached_daily_logs_for_user.clear()
    _cached_daily_log.clear()
    _cached_checkin.clear()


def _recent_logs_window(repo, username: str, end_date: dt.date, days: int = 7) -> List[Dict[str, Any]]:
    start_date = end_date - dt.timedelta(days=days - 1)
    logs = _cached_daily_logs_for_user(repo, username, 120)
    out: List[Dict[str, Any]] = []
    for r in logs:
        d = _parse_date_iso(r.get("date", ""))
        if not d:
            continue
        if start_date <= d <= end_date:
            out.append(r)
    out.sort(key=lambda r: str(r.get("date", "")), reverse=True)
    return out
