from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import streamlit as st

from domain.shift import _parse_shift_blocks_json, _shift_blocks_to_spans
from neuroframe.engine import DayInputs
from shared.today_input import doses_from_json, drafts_to_engine_doses, parse_sleep_override

from .cache import _cached_daily_log, _cached_daily_logs_for_user, _invalidate_repo_read_caches

TZ = ZoneInfo("Asia/Seoul")


def _parse_date_iso(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(s))
    except Exception:
        return None


def _parse_time_hhmm(s: str, fallback: dt.time) -> dt.time:
    try:
        hh, mm = str(s).strip().split(":")
        return dt.time(int(hh), int(mm))
    except Exception:
        return fallback


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _task_done_map_for_date(date: dt.date) -> Dict[str, bool]:
    prefix = f"{date.isoformat()}::"
    raw = st.session_state.get("task_done_map", {})
    out: Dict[str, bool] = {}
    for k, v in raw.items():
        if str(k).startswith(prefix):
            out[str(k)] = bool(v)
    return out


def _task_done_json_for_date(date: dt.date) -> str:
    return json.dumps(_task_done_map_for_date(date), ensure_ascii=False)


def _load_task_done_json_for_date(date: dt.date, s: str) -> None:
    prefix = f"{date.isoformat()}::"
    cur = dict(st.session_state.get("task_done_map", {}))
    for k in list(cur.keys()):
        if str(k).startswith(prefix):
            del cur[k]

    if s:
        try:
            obj = json.loads(s)
        except Exception:
            obj = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).startswith(prefix):
                    cur[str(k)] = bool(v)
    st.session_state["task_done_map"] = cur


def _build_inputs_from_log_row(row: Dict[str, Any], date: dt.date) -> DayInputs:
    sleep_on = str(row.get("sleep_override_on", "false")).lower() == "true"
    cross_mid = str(row.get("sleep_cross_midnight", "true")).lower() == "true"
    sleep_start = _parse_time_hhmm(row.get("sleep_start", "23:30"), dt.time(23, 30))
    wake_time = _parse_time_hhmm(row.get("wake_time", "07:30"), dt.time(7, 30))

    sleep_override: Optional[Tuple[dt.datetime, dt.datetime]] = None
    if sleep_on:
        sleep_override = parse_sleep_override(
            date=date,
            tz=TZ,
            sleep_start=sleep_start,
            wake_time=wake_time,
            crosses_midnight=cross_mid,
        )

    doses = drafts_to_engine_doses(date, TZ, doses_from_json(row.get("doses_json", "[]")))
    shift_spans = _shift_blocks_to_spans(date, _parse_shift_blocks_json(str(row.get("shift_blocks_json", "[]") or "[]")))
    return DayInputs(
        date=date,
        timezone=TZ,
        sleep_override=sleep_override,
        doses=doses,
        workload_level=_safe_float(row.get("workload_level", 1.0), 1.0),
        shift_blocks=shift_spans,
    )


def _apply_recent_pattern(repo, username: str, target_date: dt.date) -> bool:
    logs = _cached_daily_logs_for_user(repo, username, 120)
    candidates: List[Dict[str, Any]] = []
    for r in logs:
        d = _parse_date_iso(r.get("date", ""))
        if d is None or d >= target_date:
            continue
        candidates.append(r)
        if len(candidates) >= 7:
            break
    if not candidates:
        return False

    sleep_on_votes = 0
    cross_mid_votes = 0
    workloads: List[float] = []
    clarities: List[float] = []
    shifts: List[float] = []
    sleep_starts: List[dt.time] = []
    wakes: List[dt.time] = []
    latest_doses_json = "[]"
    latest_shift_blocks_json = "[]"

    for r in candidates:
        sleep_on = str(r.get("sleep_override_on", "false")).lower() == "true"
        cross_mid = str(r.get("sleep_cross_midnight", "true")).lower() == "true"
        sleep_on_votes += 1 if sleep_on else 0
        cross_mid_votes += 1 if cross_mid else 0

        sleep_starts.append(_parse_time_hhmm(r.get("sleep_start", "23:30"), dt.time(23, 30)))
        wakes.append(_parse_time_hhmm(r.get("wake_time", "07:30"), dt.time(7, 30)))
        workloads.append(_safe_float(r.get("workload_level", 1.0), 1.0))
        clarities.append(_safe_float(r.get("subjective_clarity", 5.0), 5.0))
        shifts.append(_safe_float(r.get("day_shift_hours", 0.0), 0.0))

        djson = str(r.get("doses_json", "[]") or "[]")
        if latest_doses_json == "[]" and djson != "[]":
            latest_doses_json = djson

        sjson = str(r.get("shift_blocks_json", "[]") or "[]")
        if latest_shift_blocks_json == "[]" and sjson != "[]":
            latest_shift_blocks_json = sjson

    n = max(len(candidates), 1)
    st.session_state["today_sleep_override_on"] = sleep_on_votes >= (n / 2)
    st.session_state["today_sleep_cross_midnight"] = cross_mid_votes >= (n / 2)

    sleep_starts_sorted = sorted(sleep_starts, key=lambda t: (t.hour, t.minute))
    wakes_sorted = sorted(wakes, key=lambda t: (t.hour, t.minute))
    st.session_state["today_sleep_start"] = sleep_starts_sorted[len(sleep_starts_sorted) // 2]
    st.session_state["today_wake_time"] = wakes_sorted[len(wakes_sorted) // 2]

    st.session_state["today_doses_json"] = latest_doses_json
    st.session_state["today_shift_blocks_json"] = latest_shift_blocks_json
    st.session_state["today_shift_hours"] = round(sum(shifts) / len(shifts), 1)
    st.session_state["today_workload"] = round(sum(workloads) / len(workloads), 1)
    st.session_state["today_clarity"] = round(sum(clarities) / len(clarities), 1)
    return True


def load_today_state(repo, username: str):
    date = st.session_state["today_date"]
    row = _cached_daily_log(repo, username, date.isoformat())
    if not row:
        _load_task_done_json_for_date(date, "{}")
        return

    st.session_state["today_sleep_override_on"] = str(row.get("sleep_override_on", "false")).lower() == "true"
    st.session_state["today_sleep_cross_midnight"] = str(row.get("sleep_cross_midnight", "true")).lower() == "true"
    st.session_state["today_sleep_start"] = _parse_time_hhmm(row.get("sleep_start", "23:30"), dt.time(23, 30))
    st.session_state["today_wake_time"] = _parse_time_hhmm(row.get("wake_time", "07:30"), dt.time(7, 30))
    st.session_state["today_doses_json"] = row.get("doses_json", "[]") or "[]"
    st.session_state["today_shift_blocks_json"] = row.get("shift_blocks_json", "[]") or "[]"
    st.session_state["today_shift_hours"] = _safe_float(row.get("day_shift_hours", 0.0), 0.0)
    st.session_state["today_workload"] = _safe_float(row.get("workload_level", 1.0), 1.0)
    st.session_state["today_clarity"] = _safe_float(row.get("subjective_clarity", 5.0), 5.0)
    _load_task_done_json_for_date(date, str(row.get("task_done_json", "{}") or "{}"))


def save_today_state(repo, username: str):
    date = st.session_state["today_date"]
    payload = {
        "sleep_override_on": str(bool(st.session_state["today_sleep_override_on"])).lower(),
        "sleep_cross_midnight": str(bool(st.session_state["today_sleep_cross_midnight"])).lower(),
        "sleep_start": st.session_state["today_sleep_start"].strftime("%H:%M"),
        "wake_time": st.session_state["today_wake_time"].strftime("%H:%M"),
        "doses_json": st.session_state["today_doses_json"],
        "shift_blocks_json": st.session_state["today_shift_blocks_json"],
        "task_done_json": _task_done_json_for_date(date),
        "day_shift_hours": str(float(st.session_state["today_shift_hours"])),
        "workload_level": str(st.session_state["today_workload"]),
        "subjective_clarity": str(st.session_state["today_clarity"]),
    }
    repo.upsert_daily_log(username, date, payload)
    _invalidate_repo_read_caches()
