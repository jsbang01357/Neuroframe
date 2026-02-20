# neuroframe_app.py
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import streamlit as st

from auth import get_repo, login_guard, render_user_badge

from neuroframe.components.wizard import render_setup_wizard
from neuroframe.components.checkin import render_morning_checkin
from neuroframe.components.edit_panel import render_edit_panel
from neuroframe.components.dashboard import (
    render_summary_cards,
    render_today_recommendations,
    render_shift_aware_insights,
    render_nap_recommendations,
    render_tomorrow_plan,
)

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


def _recent_logs_window(repo, username: str, end_date: dt.date, days: int = 7) -> List[Dict[str, Any]]:
    start_date = end_date - dt.timedelta(days=days - 1)
    logs = repo.get_daily_logs_for_user(username, limit=120)
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


def _sum_caffeine_mg(doses_json: str) -> float:
    total = 0.0
    for d in doses_from_json(doses_json):
        total += max(0.0, float(d.amount_mg))
    return total


def _estimate_confidence(day_inputs: DayInputs, recent_logs_count: int) -> Tuple[str, int, List[str]]:
    score = 90
    reasons: List[str] = []
    if not day_inputs.sleep_override:
        score -= 20
        reasons.append("수면 오버라이드 미입력")
    if not (day_inputs.doses or []):
        score -= 12
        reasons.append("카페인 입력 없음")
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
    logs = repo.get_daily_logs_for_user(username, limit=120)
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
        new_drafts.append(DoseDraft(9, 0, 150.0))
    elif preset == "샷 추가(75mg)":
        new_drafts.append(DoseDraft(13, 30, 75.0))
    elif preset == "에너지드링크(120mg)":
        new_drafts.append(DoseDraft(15, 0, 120.0))
    elif preset == "디카페인(20mg)":
        new_drafts.append(DoseDraft(16, 0, 20.0))
    return new_drafts[:6]


def _plan_to_drafts(plan: str) -> List[DoseDraft]:
    if plan == "없음":
        return []
    if plan == "라이트":
        return [DoseDraft(9, 0, 100.0)]
    if plan == "보통":
        return [DoseDraft(9, 0, 150.0), DoseDraft(14, 0, 70.0)]
    return [DoseDraft(8, 30, 180.0), DoseDraft(13, 30, 120.0)]


# ----------------------------
# Repo load/save hooks
# ----------------------------

def load_today_state(repo, username: str):
    date = st.session_state["today_date"]
    row = repo.get_daily_log(username, date)
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






def render_daily_checkin(repo, username: str, baseline: UserBaseline, out):
    st.divider()
    st.subheader("저녁 마감 체크인")

    date = st.session_state["today_date"]
    existing = repo.get_checkin(username, date) or {}

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

    c3, c4 = st.columns(2)
    with c3:
        if st.button("체크인 저장", use_container_width=True):
            _save_checkin()
            st.success("체크인을 저장했습니다.")

    with c4:
        if st.button("체크인 저장 + 적응 업데이트", use_container_width=True):
            _save_checkin()

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
    logs = repo.get_daily_logs_for_user(username, limit=120)

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prime 평균 시작", _minutes_to_hhmm(avg_prime_start) if avg_prime_start is not None else "-")
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

    # 1.0 Summary-first dashboard
    render_summary_cards(out)
    render_today_recommendations(out)

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

    render_tomorrow_plan(repo, username, baseline, TZ)
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

    if not user.onboarded:
        st.sidebar.info("최초 1회 설정이 필요합니다.")
        render_setup_wizard(repo, username, baseline)
        return

    if "today_loaded_once" not in st.session_state:
        load_today_state(repo, username)
        st.session_state["today_loaded_once"] = True

    render_topbar()
    render_morning_checkin(repo, username, save_today_state)

    if st.session_state["edit_today_open"]:
        render_edit_panel(repo, username, load_today_state, save_today_state)
    else:
        st.sidebar.info("상단의 **오늘 입력 수정** 버튼으로 수면/카페인/근무블록/부하를 조정할 수 있습니다.")
        if st.sidebar.button("오늘 로그 다시 불러오기", use_container_width=True):
            load_today_state(repo, username)
            st.rerun()

    render_dashboard(repo, username, baseline)


if __name__ == "__main__":
    main()
