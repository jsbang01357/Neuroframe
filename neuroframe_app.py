# neuroframe_app.py
from __future__ import annotations

import datetime as dt
import io
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import streamlit as st

from auth import get_repo, login_guard, render_user_badge
from neuroframe.coach import design_schedule, interpret_zones
from neuroframe.engine import DayInputs, UserBaseline, calibrate_baseline_offset, predict_day
from neuroframe.plots import plot_net_energy
from shared.today_input import (
    DoseDraft,
    doses_from_json,
    doses_to_json,
    drafts_to_engine_doses,
    parse_sleep_override,
)
from shared.calendar_integration import CalendarEvent, fetch_google_calendar_events_for_day

TZ = ZoneInfo("Asia/Seoul")


# ----------------------------
# Session init
# ----------------------------

def init_session_defaults():
    st.session_state.setdefault("edit_today_open", False)

    today = dt.datetime.now(TZ).date()
    st.session_state.setdefault("today_date", today)

    st.session_state.setdefault("today_sleep_override_on", False)
    st.session_state.setdefault("today_sleep_cross_midnight", True)
    st.session_state.setdefault("today_sleep_start", dt.time(23, 30))
    st.session_state.setdefault("today_wake_time", dt.time(7, 30))

    st.session_state.setdefault("today_doses_json", "[]")
    st.session_state.setdefault("today_shift_hours", 0.0)
    st.session_state.setdefault("today_shift_blocks_json", "[]")
    st.session_state.setdefault("today_workload", 1.0)
    st.session_state.setdefault("today_clarity", 5.0)
    st.session_state.setdefault("task_done_map", {})
    st.session_state.setdefault("profile_is_shift_worker", False)
    st.session_state.setdefault("profile_uses_adhd_medication", False)

    st.session_state.setdefault("morning_checkin_done_date", "")


def toggle_edit_panel():
    st.session_state["edit_today_open"] = not st.session_state.get("edit_today_open", False)


# ----------------------------
# Utility
# ----------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


def _mask_to_spans(t: List[dt.datetime], mask: List[bool]) -> List[Tuple[dt.datetime, dt.datetime]]:
    spans: List[Tuple[dt.datetime, dt.datetime]] = []
    if not t or not mask or len(t) != len(mask):
        return spans

    in_span = False
    start: Optional[dt.datetime] = None
    step = (t[1] - t[0]) if len(t) >= 2 else dt.timedelta(minutes=10)
    for i, m in enumerate(mask):
        if m and not in_span:
            in_span = True
            start = t[i]
        elif not m and in_span and start is not None:
            in_span = False
            spans.append((start, t[i]))
            start = None
    if in_span and start is not None:
        spans.append((start, t[-1] + step))
    return spans


def _span_minutes(span: Tuple[dt.datetime, dt.datetime]) -> float:
    return (span[1] - span[0]).total_seconds() / 60.0


def _primary_span_text(spans: List[Tuple[dt.datetime, dt.datetime]], sleep_gate: bool = False) -> str:
    if not spans:
        return "없음"
    best = sorted(spans, key=_span_minutes, reverse=True)[0]
    if sleep_gate:
        mid = best[0] + (best[1] - best[0]) / 2
        return f"{mid.strftime('%H:%M')}±"
    return f"{best[0].strftime('%H:%M')}–{best[1].strftime('%H:%M')}"


def _effective_baseline(baseline: UserBaseline, day_shift_hours: float) -> UserBaseline:
    return UserBaseline(
        baseline_sleep_start=baseline.baseline_sleep_start,
        baseline_wake=baseline.baseline_wake,
        chronotype_shift_hours=float(baseline.chronotype_shift_hours) + float(day_shift_hours),
        caffeine_half_life_hours=baseline.caffeine_half_life_hours,
        caffeine_sensitivity=baseline.caffeine_sensitivity,
        baseline_offset=baseline.baseline_offset,
        circadian_weight=baseline.circadian_weight,
        sleep_pressure_weight=baseline.sleep_pressure_weight,
        drug_weight=baseline.drug_weight,
        load_weight=baseline.load_weight,
    )


def _sleep_duration_hours(sleep_start: dt.time, wake_time: dt.time, crosses_midnight: bool) -> float:
    d = dt.date(2026, 1, 1)
    wake_dt = dt.datetime(d.year, d.month, d.day, wake_time.hour, wake_time.minute, tzinfo=TZ)
    sleep_dt = dt.datetime(d.year, d.month, d.day, sleep_start.hour, sleep_start.minute, tzinfo=TZ)
    if crosses_midnight:
        sleep_dt -= dt.timedelta(days=1)
    elif sleep_dt > wake_dt:
        sleep_dt -= dt.timedelta(days=1)
    return max(0.0, (wake_dt - sleep_dt).total_seconds() / 3600.0)


def _baseline_sleep_hours(baseline: UserBaseline) -> float:
    crosses_mid = baseline.baseline_sleep_start > baseline.baseline_wake
    return _sleep_duration_hours(baseline.baseline_sleep_start, baseline.baseline_wake, crosses_mid)


def _sleep_midpoint_minutes(sleep_start: dt.time, wake_time: dt.time, crosses_midnight: bool) -> float:
    d = dt.date(2026, 1, 1)
    wake_dt = dt.datetime(d.year, d.month, d.day, wake_time.hour, wake_time.minute, tzinfo=TZ)
    sleep_dt = dt.datetime(d.year, d.month, d.day, sleep_start.hour, sleep_start.minute, tzinfo=TZ)
    if crosses_midnight:
        sleep_dt -= dt.timedelta(days=1)
    elif sleep_dt > wake_dt:
        sleep_dt -= dt.timedelta(days=1)
    mid = sleep_dt + (wake_dt - sleep_dt) / 2
    return mid.hour * 60 + mid.minute


def _minutes_to_hhmm(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _badge_html(label: str, value: str, bg: str, fg: str = "#0b1220") -> str:
    return (
        f"<span style='display:inline-block;padding:6px 10px;margin:3px 6px 3px 0;"
        f"border-radius:999px;background:{bg};color:{fg};font-size:0.88rem;font-weight:600;'>"
        f"{label} · {value}</span>"
    )


def _sum_caffeine_mg(doses_json: str) -> float:
    total = 0.0
    for d in doses_from_json(doses_json):
        if str(getattr(d, "dose_type", "caffeine") or "caffeine").strip().lower() == "caffeine":
            total += max(0.0, float(d.amount_mg))
    return total


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
def _cached_google_calendar_events(day_iso: str) -> List[CalendarEvent]:
    sa, cal_id, delegated = _read_calendar_config()
    if not sa or not cal_id:
        return []
    day = dt.date.fromisoformat(day_iso)
    try:
        return fetch_google_calendar_events_for_day(
            service_account_info=sa,
            calendar_id=cal_id,
            day=day,
            tz=TZ,
            delegated_user=delegated,
        )
    except Exception:
        return []


def _invalidate_calendar_cache() -> None:
    _cached_google_calendar_events.clear()


def _ics_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def _blocks_to_ics_bytes(blocks) -> bytes:
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//NeuroFrame//Schedule Export//EN",
        "CALSCALE:GREGORIAN",
    ]
    for b in blocks:
        start_utc = b.start.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end_utc = b.end.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        uid = f"{uuid.uuid4()}@neuroframe"
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{now_utc}",
                f"DTSTART:{start_utc}",
                f"DTEND:{end_utc}",
                f"SUMMARY:{_ics_escape('NeuroFrame ' + b.label)}",
                f"DESCRIPTION:{_ics_escape(b.rationale)}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def _weekly_report_pdf_bytes(
    end_date: dt.date,
    avg_prime_start_text: str,
    avg_crash_len: float,
    sleep_48h: float,
    sleep_7d_avg: float,
    sleep_debt_7d: float,
    caffeine_total_mg: float,
    recommendation: str,
) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            "NeuroFrame Weekly Report",
            f"기준일: {end_date.isoformat()}",
            "",
            f"Prime 평균 시작: {avg_prime_start_text}",
            f"Crash 평균 길이: {avg_crash_len:.0f}분",
            f"최근48h 수면: {sleep_48h:.1f}h",
            f"7일 평균 수면: {sleep_7d_avg:.1f}h",
            f"수면부채(7d): {sleep_debt_7d:.1f}h",
            f"카페인 총량(7d): {caffeine_total_mg:.0f}mg",
            "",
            "추천",
            recommendation,
        ]
        y = 0.95
        for line in lines:
            ax.text(0.05, y, line, fontsize=11, va="top", family="DejaVu Sans")
            y -= 0.05
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buf.getvalue()


def _estimate_confidence(day_inputs: DayInputs, recent_logs_count: int) -> Tuple[str, int, List[str]]:
    score = 90
    reasons: List[str] = []
    if not day_inputs.sleep_override:
        score -= 20
        reasons.append("수면 오버라이드 미입력")
    if not (day_inputs.doses or []):
        score -= 12
        reasons.append("카페인/약물 입력 없음")
    if day_inputs.workload_level <= 0.0:
        score -= 8
        reasons.append("업무 부하가 0")
    if recent_logs_count < 3:
        score -= 20
        reasons.append("최근 7일 로그 부족")

    score = int(_clip(float(score), 5.0, 99.0))
    if score >= 75:
        return "높음", score, reasons
    if score >= 50:
        return "중간", score, reasons
    return "낮음", score, reasons


def _dose_type_label(v: str) -> str:
    key = str(v or "caffeine").strip().lower()
    if key == "mph_ir":
        return "MPH IR"
    if key == "mph_xr":
        return "MPH XR"
    return "Caffeine"


def _render_stimulant_caffeine_notice(day_inputs: DayInputs):
    doses = day_inputs.doses or []
    has_stimulant = False
    late_caffeine = False
    for d in doses:
        typ = str(getattr(d, "dose_type", "caffeine") or "caffeine").strip().lower()
        if typ in ("mph_ir", "mph_xr", "stimulant_ir", "stimulant_xr"):
            has_stimulant = True
        if typ == "caffeine" and (d.time.hour >= 15):
            late_caffeine = True

    if has_stimulant and late_caffeine:
        st.warning("오늘은 자극제 입력이 있어, 늦은 카페인은 Sleep Gate를 뒤로 미룰 수 있습니다. 가능한 이른 시간대로 조정하는 편이 유리할 수 있습니다.")
    elif has_stimulant:
        st.info("오늘은 자극제 입력이 있습니다. 추가 카페인은 필요 최소량을 이른 시간대에 배치하면 저녁 반동(rebound)과 수면 지연 가능성을 줄이는 데 도움이 될 수 있습니다.")


def _task_key(date: dt.date, label: str, start: dt.datetime, end: dt.datetime) -> str:
    return f"{date.isoformat()}::{label}::{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def _is_task_done(task_key: str) -> bool:
    done_map = st.session_state.get("task_done_map", {})
    return bool(done_map.get(task_key, False))


def _set_task_done(task_key: str, done: bool) -> None:
    done_map = dict(st.session_state.get("task_done_map", {}))
    done_map[task_key] = bool(done)
    st.session_state["task_done_map"] = done_map


def _shift_template_blocks(template: str) -> List[Dict[str, str]]:
    if template == "Day":
        return [{"start": "08:00", "end": "18:00", "type": "day"}]
    if template == "Evening":
        return [{"start": "14:00", "end": "23:00", "type": "evening"}]
    if template == "Night":
        return [{"start": "22:00", "end": "08:00", "type": "night"}]
    if template == "24h-call":
        return [{"start": "08:00", "end": "08:00", "type": "call24"}]
    return []


def _template_shift_hours(template: str) -> float:
    if template == "Day":
        return 0.0
    if template == "Evening":
        return 1.0
    if template == "Night":
        return 2.5
    if template == "24h-call":
        return 1.5
    return 0.0


def _parse_shift_blocks_json(s: str) -> List[Dict[str, str]]:
    if not s:
        return []
    try:
        arr = json.loads(s)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    if not isinstance(arr, list):
        return out
    for item in arr:
        if not isinstance(item, dict):
            continue
        stt = str(item.get("start", "")).strip()
        end = str(item.get("end", "")).strip()
        typ = str(item.get("type", "shift")).strip() or "shift"
        if len(stt) == 5 and len(end) == 5 and ":" in stt and ":" in end:
            out.append({"start": stt, "end": end, "type": typ})
    return out


def _shift_blocks_to_json(blocks: List[Dict[str, str]]) -> str:
    return json.dumps(blocks, ensure_ascii=False)


def _shift_blocks_to_spans(date: dt.date, blocks: List[Dict[str, str]]) -> List[Tuple[dt.datetime, dt.datetime, str]]:
    spans: List[Tuple[dt.datetime, dt.datetime, str]] = []
    for b in blocks:
        stt = _parse_time_hhmm(b.get("start", "08:00"), dt.time(8, 0))
        end = _parse_time_hhmm(b.get("end", "18:00"), dt.time(18, 0))
        typ = str(b.get("type", "shift"))
        start_dt = dt.datetime(date.year, date.month, date.day, stt.hour, stt.minute, tzinfo=TZ)
        end_dt = dt.datetime(date.year, date.month, date.day, end.hour, end.minute, tzinfo=TZ)
        if end_dt <= start_dt:
            end_dt += dt.timedelta(days=1)
        spans.append((start_dt, end_dt, typ))
    return spans


def _overlap_minutes(a: Tuple[dt.datetime, dt.datetime], b: Tuple[dt.datetime, dt.datetime]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e <= s:
        return 0.0
    return (e - s).total_seconds() / 60.0


def render_calendar_conflicts(out):
    st.subheader("Calendar 충돌")
    events = _cached_google_calendar_events(st.session_state["today_date"].isoformat())
    if not events:
        st.caption("Google Calendar 연동이 없거나 오늘 일정이 없습니다.")
        return

    prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
    collisions: List[Tuple[CalendarEvent, float]] = []
    for ev in events:
        overlap = 0.0
        for ps in prime_spans:
            overlap += _overlap_minutes((ev.start, ev.end), ps)
        if overlap > 0:
            collisions.append((ev, overlap))

    if not collisions:
        st.success("오늘 Prime Zone과 겹치는 회의가 크지 않습니다.")
        return

    st.warning(f"Prime Zone과 겹치는 일정 {len(collisions)}건이 있습니다.")
    for ev, ov in collisions[:5]:
        tr = f"{ev.start.strftime('%H:%M')}–{ev.end.strftime('%H:%M')}"
        st.write(f"- **{ev.summary}** ({tr}) · Prime 겹침 {ov:.0f}분")
    st.info("이 구간은 Deep Work가 유리할 수 있는데, 이미 회의가 있습니다. 가능하면 회의를 이동하거나 Deep Work를 인접 구간으로 이동해보세요.")


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
    return DayInputs(
        date=date,
        timezone=TZ,
        sleep_override=sleep_override,
        doses=doses,
        workload_level=_safe_float(row.get("workload_level", 1.0), 1.0),
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


def _add_preset_dose(drafts: List[DoseDraft], preset: str) -> List[DoseDraft]:
    new_drafts = list(drafts)
    if preset == "아메리카노(150mg)":
        new_drafts.append(DoseDraft(9, 0, 150.0, "caffeine"))
    elif preset == "샷 추가(75mg)":
        new_drafts.append(DoseDraft(13, 30, 75.0, "caffeine"))
    elif preset == "에너지드링크(120mg)":
        new_drafts.append(DoseDraft(15, 0, 120.0, "caffeine"))
    elif preset == "디카페인(20mg)":
        new_drafts.append(DoseDraft(16, 0, 20.0, "caffeine"))
    return new_drafts[:6]


def _plan_to_drafts(plan: str) -> List[DoseDraft]:
    if plan == "없음":
        return []
    if plan == "라이트":
        return [DoseDraft(9, 0, 100.0, "caffeine")]
    if plan == "보통":
        return [DoseDraft(9, 0, 150.0, "caffeine"), DoseDraft(14, 0, 70.0, "caffeine")]
    return [DoseDraft(8, 30, 180.0, "caffeine"), DoseDraft(13, 30, 120.0, "caffeine")]


# ----------------------------
# Repo load/save hooks
# ----------------------------

def load_today_state(repo, username: str):
    date = st.session_state["today_date"]
    row = _cached_daily_log(repo, username, date.isoformat())
    if not row:
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


def save_today_state(repo, username: str):
    date = st.session_state["today_date"]
    payload = {
        "sleep_override_on": str(bool(st.session_state["today_sleep_override_on"])).lower(),
        "sleep_cross_midnight": str(bool(st.session_state["today_sleep_cross_midnight"])).lower(),
        "sleep_start": st.session_state["today_sleep_start"].strftime("%H:%M"),
        "wake_time": st.session_state["today_wake_time"].strftime("%H:%M"),
        "doses_json": st.session_state["today_doses_json"],
        "shift_blocks_json": st.session_state["today_shift_blocks_json"],
        "day_shift_hours": str(float(st.session_state["today_shift_hours"])),
        "workload_level": str(st.session_state["today_workload"]),
        "subjective_clarity": str(st.session_state["today_clarity"]),
    }
    repo.upsert_daily_log(username, date, payload)
    _invalidate_repo_read_caches()


# ----------------------------
# UI
# ----------------------------

def render_topbar():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("NeuroFrame")
        st.caption("Design Your Day Around Your Brain.")
    with c2:
        st.write("")
        st.write("")
        st.button("오늘 입력 수정", on_click=toggle_edit_panel, use_container_width=True)


def render_setup_wizard(repo, username: str, baseline: UserBaseline):
    st.header("Setup Wizard")
    st.caption("최초 1회 설정입니다. 나중에 관리자/설정에서 언제든 수정할 수 있습니다.")
    st.info("안내: NeuroFrame은 의료 조언/진단/치료를 제공하지 않는 일정 설계 도구입니다.")

    st.subheader("Step 0 — 프로필")
    p1, p2 = st.columns(2)
    with p1:
        is_shift_worker = st.toggle("교대근무를 하나요?", value=bool(st.session_state.get("profile_is_shift_worker", False)))
    with p2:
        uses_adhd_med = st.toggle("ADHD 약물을 사용하나요?", value=bool(st.session_state.get("profile_uses_adhd_medication", False)))
    st.session_state["profile_is_shift_worker"] = bool(is_shift_worker)
    st.session_state["profile_uses_adhd_medication"] = bool(uses_adhd_med)

    st.subheader("Step 1 — Baseline")
    c1, c2, c3 = st.columns(3)
    with c1:
        default_sleep = baseline.baseline_sleep_start if not is_shift_worker else dt.time(0, 30)
        default_wake = baseline.baseline_wake if not is_shift_worker else dt.time(8, 0)
        baseline_sleep_start = st.time_input("평균 취침", value=default_sleep)
        baseline_wake = st.time_input("평균 기상", value=default_wake)
    with c2:
        chronoshift = st.number_input("크로노타입 시프트(시간)", value=float(baseline.chronotype_shift_hours), step=0.5)
    with c3:
        baseline_offset = st.number_input("baseline_offset", value=float(baseline.baseline_offset), step=0.05)

    st.subheader("Step 2 — 약물/진정 영향")
    c4, c5 = st.columns(2)
    with c4:
        caffeine_half_life = st.number_input("카페인 반감기(시간)", value=float(baseline.caffeine_half_life_hours), step=0.5)
        caffeine_sensitivity = st.number_input("카페인 민감도", value=float(baseline.caffeine_sensitivity), step=0.1)
    with c5:
        if uses_adhd_med:
            st.caption("약물 입력 폼에서 MPH IR/XR를 함께 설정할 수 있습니다.")
        else:
            st.caption("기본 입력은 카페인 중심으로 단순화됩니다.")

    st.subheader("Step 3 — 엔진 가중치")
    c6, c7, c8, c9 = st.columns(4)
    with c6:
        w_c = st.number_input("circadian_weight", value=float(baseline.circadian_weight), step=0.1)
    with c7:
        w_s = st.number_input("sleep_pressure_weight", value=float(baseline.sleep_pressure_weight), step=0.1)
    with c8:
        w_d = st.number_input("drug_weight", value=float(baseline.drug_weight), step=0.001, format="%.6f")
    with c9:
        w_l = st.number_input("load_weight", value=float(baseline.load_weight), step=0.05)

    if st.button("저장하고 시작하기", type="primary", use_container_width=True):
        patch = {
            "baseline_sleep_start": baseline_sleep_start.strftime("%H:%M"),
            "baseline_wake": baseline_wake.strftime("%H:%M"),
            "chronotype_shift_hours": str(float(chronoshift)),
            "caffeine_half_life_hours": str(float(caffeine_half_life)),
            "caffeine_sensitivity": str(float(caffeine_sensitivity)),
            "baseline_offset": str(float(baseline_offset)),
            "circadian_weight": str(float(w_c)),
            "sleep_pressure_weight": str(float(w_s)),
            "drug_weight": str(float(w_d)),
            "load_weight": str(float(w_l)),
            "is_shift_worker": "true" if is_shift_worker else "false",
            "uses_adhd_medication": "true" if uses_adhd_med else "false",
            "onboarded": "true",
        }
        ok = repo.update_user_baseline(username, patch)
        if ok:
            st.success("온보딩이 완료되었습니다.")
            st.rerun()
        else:
            st.error("저장에 실패했습니다. users 시트/권한/헤더를 확인해주세요.")


def render_morning_checkin(repo, username: str, is_shift_worker: bool):
    now = dt.datetime.now(TZ)
    today = now.date()
    if st.session_state["today_date"] != today:
        return

    done_date = str(st.session_state.get("morning_checkin_done_date", ""))
    if not (now.hour < 12 and done_date != today.isoformat()):
        return

    st.subheader("아침 체크인 (30초)")
    st.caption("수면/Shift/카페인/부하를 빠르게 선택하고 오늘 요약을 생성합니다.")

    c1, c2 = st.columns(2)
    with c1:
        sleep_on = st.toggle("실제 수면 반영", value=True, key="morning_sleep_on")
        cross_mid = st.checkbox("취침이 전날부터 이어짐", value=True, key="morning_cross_mid")
        sleep_start = st.time_input("실제 취침", value=st.session_state["today_sleep_start"], key="morning_sleep_start")
        wake_time = st.time_input("실제 기상", value=st.session_state["today_wake_time"], key="morning_wake_time")
    with c2:
        shift_template = "Off"
        if is_shift_worker:
            shift_template = st.selectbox("Shift 템플릿", ["Off", "Day", "Evening", "Night", "24h-call"], index=0)
        plan = st.selectbox("카페인 계획(대략)", ["없음", "라이트", "보통", "높음"], index=2)
        workload = st.slider("오늘 workload", 0.0, 3.0, value=float(st.session_state["today_workload"]), step=0.5)

    if st.button("아침 체크인 완료", type="primary", use_container_width=True):
        st.session_state["today_sleep_override_on"] = bool(sleep_on)
        st.session_state["today_sleep_cross_midnight"] = bool(cross_mid)
        st.session_state["today_sleep_start"] = sleep_start
        st.session_state["today_wake_time"] = wake_time
        st.session_state["today_doses_json"] = doses_to_json(today, TZ, _plan_to_drafts(plan))
        st.session_state["today_shift_blocks_json"] = _shift_blocks_to_json(_shift_template_blocks(shift_template))
        st.session_state["today_shift_hours"] = _template_shift_hours(shift_template)
        st.session_state["today_workload"] = float(workload)

        try:
            save_today_state(repo, username)
            st.session_state["morning_checkin_done_date"] = today.isoformat()
            st.success("체크인 완료. 오늘 요약을 업데이트합니다.")
            st.rerun()
        except Exception as e:
            st.error(f"아침 체크인 저장 실패: {e}")


def render_edit_panel(repo, username: str, uses_adhd_medication: bool):
    st.sidebar.header("오늘 입력")

    date = st.sidebar.date_input("날짜", value=st.session_state["today_date"])
    if date != st.session_state["today_date"]:
        st.session_state["today_date"] = date
        load_today_state(repo, username)

    if st.sidebar.button("최근 7일 패턴으로 자동 채우기", use_container_width=True):
        ok = _apply_recent_pattern(repo, username, st.session_state["today_date"])
        if ok:
            st.sidebar.success("최근 패턴으로 기본값을 채웠습니다.")
            st.rerun()
        st.sidebar.warning("참조할 최근 로그가 부족합니다.")

    if st.sidebar.button("어제 카페인 입력 복사", use_container_width=True):
        yesterday = st.session_state["today_date"] - dt.timedelta(days=1)
        prev = _cached_daily_log(repo, username, yesterday.isoformat()) or {}
        prev_doses = str(prev.get("doses_json", "[]") or "[]")
        if prev_doses != "[]":
            drafts = [d for d in doses_from_json(prev_doses) if str(getattr(d, "dose_type", "caffeine")).strip().lower() == "caffeine"]
            st.session_state["today_doses_json"] = doses_to_json(st.session_state["today_date"], TZ, drafts)
            st.sidebar.success("어제 카페인 패턴을 복사했습니다.")
            st.rerun()
        st.sidebar.info("어제 복사할 카페인 입력이 없습니다.")

    st.sidebar.subheader("Shift 템플릿")
    tcol1, tcol2, tcol3 = st.sidebar.columns(3)
    if tcol1.button("Day", use_container_width=True):
        st.session_state["today_shift_blocks_json"] = _shift_blocks_to_json(_shift_template_blocks("Day"))
        st.session_state["today_shift_hours"] = _template_shift_hours("Day")
        st.rerun()
    if tcol2.button("Evening", use_container_width=True):
        st.session_state["today_shift_blocks_json"] = _shift_blocks_to_json(_shift_template_blocks("Evening"))
        st.session_state["today_shift_hours"] = _template_shift_hours("Evening")
        st.rerun()
    if tcol3.button("Night", use_container_width=True):
        st.session_state["today_shift_blocks_json"] = _shift_blocks_to_json(_shift_template_blocks("Night"))
        st.session_state["today_shift_hours"] = _template_shift_hours("Night")
        st.rerun()
    tcol4, tcol5 = st.sidebar.columns(2)
    if tcol4.button("24h-call", use_container_width=True):
        st.session_state["today_shift_blocks_json"] = _shift_blocks_to_json(_shift_template_blocks("24h-call"))
        st.session_state["today_shift_hours"] = _template_shift_hours("24h-call")
        st.rerun()
    if tcol5.button("Off", use_container_width=True):
        st.session_state["today_shift_blocks_json"] = _shift_blocks_to_json(_shift_template_blocks("Off"))
        st.session_state["today_shift_hours"] = _template_shift_hours("Off")
        st.rerun()

    blocks = _parse_shift_blocks_json(st.session_state.get("today_shift_blocks_json", "[]"))
    if blocks:
        st.sidebar.caption("현재 Shift 블록")
        for b in blocks:
            st.sidebar.write(f"- {b.get('type','shift')}: {b.get('start','')}–{b.get('end','')}")
    else:
        st.sidebar.caption("현재 Shift: Off")

    st.sidebar.subheader("수면")
    sleep_on = st.sidebar.toggle("수면 시간 오버라이드", value=st.session_state["today_sleep_override_on"])
    st.session_state["today_sleep_override_on"] = sleep_on

    if sleep_on:
        cross_mid = st.sidebar.checkbox("취침이 자정 이전(전날)부터 이어짐", value=st.session_state["today_sleep_cross_midnight"])
        st.session_state["today_sleep_cross_midnight"] = cross_mid
        st.session_state["today_sleep_start"] = st.sidebar.time_input("취침 시각", value=st.session_state["today_sleep_start"])
        st.session_state["today_wake_time"] = st.sidebar.time_input("기상 시각", value=st.session_state["today_wake_time"])
    else:
        st.sidebar.caption("기본값(온보딩 baseline)을 사용합니다.")

    st.sidebar.subheader("자극제/카페인")
    drafts = doses_from_json(st.session_state["today_doses_json"])
    if not drafts:
        drafts = [DoseDraft(9, 0, 150.0, "caffeine")]

    preset = st.sidebar.selectbox(
        "빠른 프리셋",
        ["선택 안 함", "아메리카노(150mg)", "샷 추가(75mg)", "에너지드링크(120mg)", "디카페인(20mg)"],
        index=0,
    )
    if st.sidebar.button("프리셋 추가", use_container_width=True) and preset != "선택 안 함":
        drafts = _add_preset_dose(drafts, preset)
        st.session_state["today_doses_json"] = doses_to_json(date, TZ, drafts)
        st.rerun()

    n = st.sidebar.slider("도즈 개수", 0, 6, min(len(drafts), 6))
    drafts = drafts[:n]

    new_drafts: List[DoseDraft] = []
    for i in range(n):
        st.sidebar.markdown(f"**Dose {i + 1}**")
        hour = st.number_input("시", 0, 23, value=int(drafts[i].hour), key=f"dose_h_{i}")
        minute = st.number_input("분", 0, 59, value=int(drafts[i].minute), key=f"dose_m_{i}")
        mg = st.number_input("mg", 0.0, 400.0, value=float(drafts[i].amount_mg), step=10.0, key=f"dose_mg_{i}")
        if uses_adhd_medication:
            dose_type = st.selectbox(
                "타입",
                ["caffeine", "mph_ir", "mph_xr"],
                index=["caffeine", "mph_ir", "mph_xr"].index(
                    str(getattr(drafts[i], "dose_type", "caffeine") or "caffeine").strip().lower()
                )
                if str(getattr(drafts[i], "dose_type", "caffeine") or "caffeine").strip().lower() in ("caffeine", "mph_ir", "mph_xr")
                else 0,
                key=f"dose_type_{i}",
                format_func=_dose_type_label,
            )
        else:
            dose_type = "caffeine"
        new_drafts.append(DoseDraft(int(hour), int(minute), float(mg), str(dose_type)))

    st.sidebar.subheader("오늘 shift 보정")
    st.session_state["today_shift_hours"] = st.sidebar.slider(
        "day_shift_hours", -3.0, 3.0, value=float(st.session_state["today_shift_hours"]), step=0.5
    )

    st.sidebar.subheader("업무 부하")
    st.session_state["today_workload"] = st.sidebar.slider("workload_level", 0.0, 3.0, value=float(st.session_state["today_workload"]), step=0.5)

    st.sidebar.subheader("자가 선명도(선택)")
    st.session_state["today_clarity"] = st.sidebar.slider(
        "subjective_clarity (0–10)", 0.0, 10.0, value=float(st.session_state["today_clarity"]), step=0.5
    )

    apply = st.sidebar.button("저장하고 적용", type="primary", use_container_width=True)
    if apply:
        try:
            st.session_state["today_doses_json"] = doses_to_json(date, TZ, new_drafts)
            save_today_state(repo, username)
            st.session_state["edit_today_open"] = False
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"오늘 입력 저장 실패: {e}")


def build_day_inputs() -> DayInputs:
    date = st.session_state["today_date"]
    doses = drafts_to_engine_doses(date, TZ, doses_from_json(st.session_state["today_doses_json"]))

    sleep_override = None
    if st.session_state["today_sleep_override_on"]:
        sleep_dt, wake_dt = parse_sleep_override(
            date=date,
            tz=TZ,
            sleep_start=st.session_state["today_sleep_start"],
            wake_time=st.session_state["today_wake_time"],
            crosses_midnight=st.session_state["today_sleep_cross_midnight"],
        )
        sleep_override = (sleep_dt, wake_dt)

    return DayInputs(
        date=date,
        timezone=TZ,
        sleep_override=sleep_override,
        doses=doses,
        workload_level=float(st.session_state["today_workload"]),
    )


def render_summary_cards(out):
    prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
    crash_spans = _mask_to_spans(out.t, out.zones.get("crash", []))
    gate_spans = _mask_to_spans(out.t, out.zones.get("sleep_gate", []))
    rebound_spans = _mask_to_spans(out.t, out.zones.get("rebound_candidate", []))

    c1, c2 = st.columns(2)
    c1.metric("Prime", _primary_span_text(prime_spans))
    c2.metric("Crash", _primary_span_text(crash_spans))
    c3, c4 = st.columns(2)
    c3.metric("Sleep Gate", _primary_span_text(gate_spans, sleep_gate=True))
    c4.metric("Rebound 후보", _primary_span_text(rebound_spans))

    badges = [
        _badge_html("Prime", _primary_span_text(prime_spans), "#cdeccd"),
        _badge_html("Crash", _primary_span_text(crash_spans), "#ffe1c4"),
        _badge_html("Sleep Gate", _primary_span_text(gate_spans, sleep_gate=True), "#dbeafe"),
    ]
    st.markdown("".join(badges), unsafe_allow_html=True)


def render_today_recommendations(out):
    st.subheader("오늘의 추천 3개")
    blocks = design_schedule(out, deep_work_target_minutes=120, max_blocks=6)
    if not blocks:
        st.caption("오늘은 추천 블록을 만들기 어렵습니다.")
        return

    today = st.session_state["today_date"]
    done = 0
    for i, b in enumerate(blocks[:5], start=1):
        time_range = f"{b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}"
        k = _task_key(today, b.label, b.start, b.end)
        checked = st.checkbox(
            f"{i}. {b.label} · {time_range}",
            value=_is_task_done(k),
            key=f"task_chk_{k}",
        )
        _set_task_done(k, checked)
        if checked:
            done += 1
        st.caption(f"근거: {b.rationale}")
    st.caption(f"완료 {done}/{min(len(blocks), 5)}")

    ics_bytes = _blocks_to_ics_bytes(blocks[:5])
    st.download_button(
        "추천 블록 iCal(.ics) 다운로드",
        data=ics_bytes,
        file_name=f"neuroframe_{st.session_state['today_date'].isoformat()}_blocks.ics",
        mime="text/calendar",
        use_container_width=True,
    )


def render_shift_aware_insights(out, shift_spans: List[Tuple[dt.datetime, dt.datetime, str]]):
    st.subheader("Shift-aware 해석")
    if not shift_spans:
        st.caption("오늘 Shift 블록이 없어서 일반 해석만 적용됩니다.")
        return

    prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
    crash_spans = _mask_to_spans(out.t, out.zones.get("crash", []))

    prime_overlap = 0.0
    crash_overlap = 0.0
    night_like = False

    for ss, se, typ in shift_spans:
        if typ in ("night", "call24"):
            night_like = True
        for ps in prime_spans:
            prime_overlap += _overlap_minutes((ss, se), ps)
        for cs in crash_spans:
            crash_overlap += _overlap_minutes((ss, se), cs)

    if prime_overlap > 0:
        st.warning(f"Prime Zone이 근무 블록과 **{prime_overlap:.0f}분** 겹칩니다. 개인 집중 블록 확보가 어려울 수 있습니다.")
    else:
        st.success("Prime Zone이 근무 블록과 크게 겹치지 않습니다.")

    if night_like and crash_overlap > 0:
        st.warning(
            f"Night/24h-call 구간에서 Crash가 **{crash_overlap:.0f}분** 겹칩니다. 해당 시간대는 의사결정 전 체크리스트를 더 천천히 확인하는 운영이 안전할 수 있습니다."
        )
    elif crash_overlap > 0:
        st.info(f"근무 중 Crash 겹침이 **{crash_overlap:.0f}분** 있습니다. 저부하 업무를 우선 배치해 리스크를 줄이세요.")
    else:
        st.success("근무 중 Crash 겹침이 크지 않습니다.")


def render_nap_recommendations(out, shift_spans: List[Tuple[dt.datetime, dt.datetime, str]]):
    st.subheader("Nap 추천")

    crash_spans = sorted(_mask_to_spans(out.t, out.zones.get("crash", [])), key=_span_minutes, reverse=True)
    suggestions: List[str] = []

    # Crash-based nap
    if crash_spans:
        s, e = crash_spans[0]
        dur = _span_minutes((s, e))
        if dur >= 90:
            suggestions.append(f"- Full cycle nap 90분: **{s.strftime('%H:%M')}–{(s + dt.timedelta(minutes=90)).strftime('%H:%M')}**")
        else:
            mid = s + (e - s) / 2
            p_start = mid - dt.timedelta(minutes=12)
            p_end = p_start + dt.timedelta(minutes=25)
            suggestions.append(f"- Power nap 25분: **{p_start.strftime('%H:%M')}–{p_end.strftime('%H:%M')}**")

    # Shift boundary nap
    if shift_spans:
        first = sorted(shift_spans, key=lambda x: x[0])[0]
        pre_start = first[0] - dt.timedelta(minutes=35)
        pre_end = pre_start + dt.timedelta(minutes=25)
        suggestions.append(f"- Shift 전 Power nap 25분: **{pre_start.strftime('%H:%M')}–{pre_end.strftime('%H:%M')}**")

        last = sorted(shift_spans, key=lambda x: x[1])[-1]
        if last[2] in ("night", "call24"):
            post_start = last[1] + dt.timedelta(minutes=15)
            post_end = post_start + dt.timedelta(minutes=90)
            suggestions.append(f"- Shift 후 Full cycle nap 90분: **{post_start.strftime('%H:%M')}–{post_end.strftime('%H:%M')}**")

    if not suggestions:
        st.caption("현재 조건에서는 뚜렷한 nap window를 계산하기 어렵습니다.")
        return

    for s in suggestions[:3]:
        st.write(s)


def render_tomorrow_plan(repo, username: str, baseline: UserBaseline):
    st.divider()
    st.subheader("내일 자동 계획")

    tomorrow = st.session_state["today_date"] + dt.timedelta(days=1)
    today_drafts = doses_from_json(st.session_state["today_doses_json"])
    tomorrow_doses = drafts_to_engine_doses(tomorrow, TZ, today_drafts)

    sleep_override = None
    if st.session_state["today_sleep_override_on"]:
        sleep_override = parse_sleep_override(
            date=tomorrow,
            tz=TZ,
            sleep_start=st.session_state["today_sleep_start"],
            wake_time=st.session_state["today_wake_time"],
            crosses_midnight=st.session_state["today_sleep_cross_midnight"],
        )

    tomorrow_inputs = DayInputs(
        date=tomorrow,
        timezone=TZ,
        sleep_override=sleep_override,
        doses=tomorrow_doses,
        workload_level=float(st.session_state["today_workload"]),
    )
    baseline_tomorrow = _effective_baseline(baseline, st.session_state["today_shift_hours"])
    out = predict_day(baseline_tomorrow, tomorrow_inputs, step_minutes=10)
    blocks = design_schedule(out, deep_work_target_minutes=150)

    st.caption(f"기준일: {tomorrow.isoformat()} (오늘 입력 템플릿 사용)")
    if not blocks:
        st.info("내일 추천 블록을 만들기 어렵습니다. 수면/카페인 입력을 보완해보세요.")
    for b in blocks:
        st.write(f"- **{b.label}** · {b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}")

    if st.button("내일 계획 초안 저장", use_container_width=True):
        payload = {
            "sleep_override_on": str(bool(st.session_state["today_sleep_override_on"])).lower(),
            "sleep_cross_midnight": str(bool(st.session_state["today_sleep_cross_midnight"])).lower(),
            "sleep_start": st.session_state["today_sleep_start"].strftime("%H:%M"),
            "wake_time": st.session_state["today_wake_time"].strftime("%H:%M"),
            "doses_json": doses_to_json(tomorrow, TZ, today_drafts),
            "shift_blocks_json": st.session_state["today_shift_blocks_json"],
            "day_shift_hours": str(float(st.session_state["today_shift_hours"])),
            "workload_level": str(st.session_state["today_workload"]),
            "subjective_clarity": "",
        }
        try:
            repo.upsert_daily_log(username, tomorrow, payload)
            _invalidate_repo_read_caches()
            st.success("내일 daily_log 초안을 저장했습니다.")
        except Exception as e:
            st.error(f"내일 계획 저장 실패: {e}")


def render_daily_checkin(repo, username: str, baseline: UserBaseline, out):
    st.divider()
    st.subheader("저녁 마감 체크인")

    date = st.session_state["today_date"]
    existing = _cached_checkin(repo, username, date.isoformat()) or {}

    clarity_default = _safe_float(
        existing.get("subjective_clarity", existing.get("energy_satisfaction", st.session_state["today_clarity"])),
        st.session_state["today_clarity"],
    )
    focus_success_default = str(existing.get("focus_success", "")).lower() == "true"
    notes_default = str(existing.get("notes", ""))

    c1, c2 = st.columns(2)
    with c1:
        clarity = st.slider("subjective_clarity (0–10)", 0.0, 10.0, value=float(clarity_default), step=0.5)
        focus_success = st.toggle("오늘 집중 목표 달성", value=focus_success_default)
    with c2:
        notes = st.text_area("메모", value=notes_default, height=95)

    def _save_checkin() -> None:
        repo.upsert_checkin(
            username,
            date,
            {
                "subjective_clarity": str(float(clarity)),
                "focus_success": "true" if focus_success else "false",
                # legacy compatibility columns
                "actual_focus_minutes": "120" if focus_success else "60",
                "energy_satisfaction": str(float(clarity)),
                "notes": notes,
            },
        )
        _invalidate_repo_read_caches()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("체크인 저장", use_container_width=True):
            try:
                _save_checkin()
                st.success("체크인을 저장했습니다.")
            except Exception as e:
                st.error(f"체크인 저장 실패: {e}")

    with c4:
        if st.button("체크인 저장 + 적응 업데이트", use_container_width=True):
            try:
                _save_checkin()
            except Exception as e:
                st.error(f"체크인 저장 실패: {e}")
                return

            adjusted_clarity = float(clarity) + (0.5 if focus_success else -0.5)
            adjusted_clarity = _clip(adjusted_clarity, 0.0, 10.0)
            predicted_mean = sum(out.net) / max(len(out.net), 1)
            new_offset = calibrate_baseline_offset(
                current_offset=baseline.baseline_offset,
                subjective_clarity_0_10=float(adjusted_clarity),
                predicted_net_mean_0_1=float(predicted_mean),
                eta=0.10,
                k=0.50,
            )

            base_cross = baseline.baseline_sleep_start > baseline.baseline_wake
            base_mid = _sleep_midpoint_minutes(baseline.baseline_sleep_start, baseline.baseline_wake, base_cross)
            actual_mid = _sleep_midpoint_minutes(
                st.session_state["today_sleep_start"],
                st.session_state["today_wake_time"],
                bool(st.session_state["today_sleep_cross_midnight"]),
            )
            delta_m = actual_mid - base_mid
            if delta_m > 720:
                delta_m -= 1440
            elif delta_m < -720:
                delta_m += 1440
            delta_h = _clip(delta_m / 60.0, -3.0, 3.0)
            new_chrono = _clip(float(baseline.chronotype_shift_hours) + 0.08 * delta_h, -5.0, 5.0)

            baseline.baseline_offset = float(new_offset)
            baseline.chronotype_shift_hours = float(new_chrono)
            ok = repo.update_user_baseline(
                username,
                {
                    "baseline_offset": str(baseline.baseline_offset),
                    "chronotype_shift_hours": str(baseline.chronotype_shift_hours),
                },
            )
            if ok:
                st.success(f"적응 업데이트 완료: offset={baseline.baseline_offset:.3f}, phase={baseline.chronotype_shift_hours:.2f}h")
                st.session_state["today_clarity"] = float(clarity)
                st.rerun()
            else:
                st.error("users 시트 업데이트 실패")


def render_weekly_report(repo, username: str, baseline: UserBaseline):
    st.divider()
    st.subheader("주간 리포트 (최근 7일)")

    end_date = st.session_state["today_date"]
    start_date = end_date - dt.timedelta(days=6)
    logs = _cached_daily_logs_for_user(repo, username, 120)

    log_by_date: Dict[dt.date, Dict[str, Any]] = {}
    for r in logs:
        d = _parse_date_iso(r.get("date", ""))
        if not d:
            continue
        log_by_date[d] = r

    week_days = [start_date + dt.timedelta(days=i) for i in range(7)]
    target_sleep_h = _baseline_sleep_hours(baseline)

    prime_start_mins: List[float] = []
    crash_lengths: List[float] = []
    sleep_7d_total = 0.0
    caffeine_total_mg = 0.0

    for d in week_days:
        row = log_by_date.get(d)
        if row:
            day_shift = _safe_float(row.get("day_shift_hours", 0.0), 0.0)
            base_for_day = _effective_baseline(baseline, day_shift)
            day_inputs = _build_inputs_from_log_row(row, d)
            out = predict_day(base_for_day, day_inputs, step_minutes=10)

            prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
            if prime_spans:
                first_prime = sorted(prime_spans, key=lambda x: x[0])[0][0]
                prime_start_mins.append(first_prime.hour * 60 + first_prime.minute)

            crash_spans = _mask_to_spans(out.t, out.zones.get("crash", []))
            crash_lengths.append(sum(_span_minutes(s) for s in crash_spans))

            sleep_on = str(row.get("sleep_override_on", "false")).lower() == "true"
            if sleep_on:
                actual_sleep_h = _sleep_duration_hours(
                    _parse_time_hhmm(row.get("sleep_start", "23:30"), dt.time(23, 30)),
                    _parse_time_hhmm(row.get("wake_time", "07:30"), dt.time(7, 30)),
                    str(row.get("sleep_cross_midnight", "true")).lower() == "true",
                )
            else:
                actual_sleep_h = target_sleep_h

            sleep_7d_total += actual_sleep_h
            caffeine_total_mg += _sum_caffeine_mg(str(row.get("doses_json", "[]") or "[]"))
        else:
            # no record -> fallback to baseline estimate
            sleep_7d_total += target_sleep_h

    # 48h sleep (today + yesterday)
    sleep_48h = 0.0
    for d in [end_date - dt.timedelta(days=1), end_date]:
        row = log_by_date.get(d)
        if row and str(row.get("sleep_override_on", "false")).lower() == "true":
            sleep_48h += _sleep_duration_hours(
                _parse_time_hhmm(row.get("sleep_start", "23:30"), dt.time(23, 30)),
                _parse_time_hhmm(row.get("wake_time", "07:30"), dt.time(7, 30)),
                str(row.get("sleep_cross_midnight", "true")).lower() == "true",
            )
        else:
            sleep_48h += target_sleep_h

    avg_prime_start = sum(prime_start_mins) / len(prime_start_mins) if prime_start_mins else None
    avg_crash_len = sum(crash_lengths) / len(crash_lengths) if crash_lengths else 0.0
    sleep_7d_avg = sleep_7d_total / 7.0
    sleep_debt_7d = max(0.0, target_sleep_h * 7.0 - sleep_7d_total)
    avg_prime_start_text = _minutes_to_hhmm(avg_prime_start) if avg_prime_start is not None else "-"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prime 평균 시작", avg_prime_start_text)
    c2.metric("Crash 평균 길이", f"{avg_crash_len:.0f}분")
    c3.metric("최근48h 수면", f"{sleep_48h:.1f}h")
    c4.metric("7일 평균 수면", f"{sleep_7d_avg:.1f}h")

    c5, c6 = st.columns(2)
    c5.metric("수면부채(7d)", f"{sleep_debt_7d:.1f}h")
    c6.metric("카페인 총량(7d)", f"{caffeine_total_mg:.0f}mg")

    if sleep_debt_7d >= 4.0:
        rec = "이번 주는 수면부채가 큰 편입니다. Off 또는 Day 다음날 우선 회복 수면을 확보하세요."
    elif avg_crash_len >= 150.0:
        rec = "Crash가 길게 나타납니다. 업무 강도를 분산하고 체크리스트 기반으로 리듬을 낮추세요."
    elif caffeine_total_mg >= 1400.0:
        rec = "카페인 총량이 높습니다. 야간 전후 도즈 타이밍을 앞당기거나 양을 줄이는 편이 유리합니다."
    else:
        rec = "패턴이 비교적 안정적입니다. Prime 시작 30분 전 준비 루틴을 고정하면 효율이 더 오릅니다."
    st.write(f"- 이번 주 추천: {rec}")

    pdf_bytes = _weekly_report_pdf_bytes(
        end_date=end_date,
        avg_prime_start_text=avg_prime_start_text,
        avg_crash_len=avg_crash_len,
        sleep_48h=sleep_48h,
        sleep_7d_avg=sleep_7d_avg,
        sleep_debt_7d=sleep_debt_7d,
        caffeine_total_mg=caffeine_total_mg,
        recommendation=rec,
    )
    st.download_button(
        "주간 리포트 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"neuroframe_weekly_{end_date.isoformat()}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_dashboard(repo, username: str, baseline: UserBaseline):
    st.caption("면책: 본 예측은 입력값 기반 참고 정보이며 의료 조언이 아닙니다.")

    day_inputs = build_day_inputs()
    effective_baseline = _effective_baseline(baseline, st.session_state["today_shift_hours"])
    out = predict_day(effective_baseline, day_inputs, step_minutes=10)

    recent_logs = _recent_logs_window(repo, username, st.session_state["today_date"], days=7)
    conf_label, conf_score, conf_reasons = _estimate_confidence(day_inputs, len(recent_logs))
    st.info(f"예측 신뢰도: **{conf_label}** ({conf_score}/100)")
    if conf_reasons:
        st.caption("신뢰도 하락 요인: " + ", ".join(conf_reasons))
    _render_stimulant_caffeine_notice(day_inputs)

    # 1.0 Summary-first dashboard
    render_summary_cards(out)
    render_today_recommendations(out)
    render_calendar_conflicts(out)

    # 2.x Shift mode
    blocks = _parse_shift_blocks_json(st.session_state.get("today_shift_blocks_json", "[]"))
    shift_spans = _shift_blocks_to_spans(st.session_state["today_date"], blocks)
    render_shift_aware_insights(out, shift_spans)
    render_nap_recommendations(out, shift_spans)

    with st.expander("자세히 보기: 에너지 그래프", expanded=False):
        fig = plot_net_energy(out, show_components=False)
        st.pyplot(fig, clear_figure=True)
        st.subheader("Zone 해석")
        for line in interpret_zones(out):
            st.write("- " + line)

    st.divider()
    st.subheader("개인화(빠른 보정)")
    st.caption("주관 선명도 입력으로 baseline_offset을 소폭 조정합니다.")

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("오늘 자가평가로 보정 적용", use_container_width=True):
            predicted_mean = sum(out.net) / max(len(out.net), 1)
            new_offset = calibrate_baseline_offset(
                current_offset=baseline.baseline_offset,
                subjective_clarity_0_10=float(st.session_state["today_clarity"]),
                predicted_net_mean_0_1=float(predicted_mean),
                eta=0.10,
                k=0.50,
            )
            baseline.baseline_offset = float(new_offset)
            ok = repo.update_user_baseline(username, {"baseline_offset": str(baseline.baseline_offset)})
            if ok:
                st.success(f"baseline_offset이 {baseline.baseline_offset:.3f}로 업데이트되었습니다.")
                st.rerun()
            else:
                st.error("users 시트 업데이트에 실패했습니다.")

    with colB:
        st.write(f"- 현재 baseline_offset: **{baseline.baseline_offset:.3f}**")
        st.write(f"- 오늘 평균 예측(net mean): **{(sum(out.net)/max(len(out.net),1)):.3f}**")
        st.write(f"- 오늘 자가 선명도: **{float(st.session_state['today_clarity']):.1f}/10**")
        st.write(f"- 오늘 shift 보정: **{float(st.session_state['today_shift_hours']):+.1f}h**")

    render_tomorrow_plan(repo, username, baseline)
    render_daily_checkin(repo, username, baseline, out)
    render_weekly_report(repo, username, baseline)


# ----------------------------
# Main
# ----------------------------

def main():
    init_session_defaults()

    repo = get_repo("NeuroFrame_DB")
    auth = login_guard(repo)
    user = auth.user
    assert user is not None
    username = user.username

    render_user_badge(user)
    baseline = user.baseline
    st.session_state["profile_is_shift_worker"] = bool(getattr(user, "is_shift_worker", False))
    st.session_state["profile_uses_adhd_medication"] = bool(getattr(user, "uses_adhd_medication", False))

    if not user.onboarded:
        st.sidebar.info("최초 1회 설정이 필요합니다.")
        render_setup_wizard(repo, username, baseline)
        return

    if "today_loaded_once" not in st.session_state:
        load_today_state(repo, username)
        st.session_state["today_loaded_once"] = True

    render_topbar()
    render_morning_checkin(repo, username, bool(getattr(user, "is_shift_worker", False)))

    if st.session_state["edit_today_open"]:
        render_edit_panel(repo, username, bool(getattr(user, "uses_adhd_medication", False)))
    else:
        st.sidebar.info("상단의 **오늘 입력 수정** 버튼으로 수면/카페인/근무블록/부하를 조정할 수 있습니다.")
        if st.sidebar.button("오늘 로그 다시 불러오기", use_container_width=True):
            _invalidate_repo_read_caches()
            load_today_state(repo, username)
            st.rerun()

    render_dashboard(repo, username, baseline)


if __name__ == "__main__":
    main()
