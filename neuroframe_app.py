# neuroframe_app.py
from __future__ import annotations

import datetime as dt
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
    st.session_state.setdefault("today_workload", 1.0)
    st.session_state.setdefault("today_clarity", 5.0)


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
    logs = repo.get_daily_logs_for_user(username, limit=90)
    out: List[Dict[str, Any]] = []
    for r in logs:
        d = _parse_date_iso(r.get("date", ""))
        if not d:
            continue
        if start_date <= d <= end_date:
            out.append(r)
    out.sort(key=lambda r: str(r.get("date", "")), reverse=True)
    return out


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


def _estimate_confidence(out, day_inputs: DayInputs, recent_logs_count: int) -> Tuple[str, int, List[str]]:
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
        reasons.append("업무 부하가 0으로 설정됨")
    if recent_logs_count < 3:
        score -= 20
        reasons.append("최근 7일 로그가 부족함")

    score = int(_clip(float(score), 5.0, 99.0))
    if score >= 75:
        return "높음", score, reasons
    if score >= 50:
        return "중간", score, reasons
    return "낮음", score, reasons


def _apply_recent_pattern(repo, username: str, target_date: dt.date) -> bool:
    logs = repo.get_daily_logs_for_user(username, limit=90)
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
    sleep_starts: List[dt.time] = []
    wakes: List[dt.time] = []
    latest_doses_json = "[]"

    for r in candidates:
        sleep_on = str(r.get("sleep_override_on", "false")).lower() == "true"
        cross_mid = str(r.get("sleep_cross_midnight", "true")).lower() == "true"
        sleep_on_votes += 1 if sleep_on else 0
        cross_mid_votes += 1 if cross_mid else 0

        sleep_starts.append(_parse_time_hhmm(r.get("sleep_start", "23:30"), dt.time(23, 30)))
        wakes.append(_parse_time_hhmm(r.get("wake_time", "07:30"), dt.time(7, 30)))

        workloads.append(_safe_float(r.get("workload_level", 1.0), 1.0))
        clarities.append(_safe_float(r.get("subjective_clarity", 5.0), 5.0))

        djson = str(r.get("doses_json", "[]") or "[]")
        if latest_doses_json == "[]" and djson != "[]":
            latest_doses_json = djson

    n = max(len(candidates), 1)
    st.session_state["today_sleep_override_on"] = sleep_on_votes >= (n / 2)
    st.session_state["today_sleep_cross_midnight"] = cross_mid_votes >= (n / 2)

    # Use median-by-sort time values for stability.
    sleep_starts_sorted = sorted(sleep_starts, key=lambda t: (t.hour, t.minute))
    wakes_sorted = sorted(wakes, key=lambda t: (t.hour, t.minute))
    st.session_state["today_sleep_start"] = sleep_starts_sorted[len(sleep_starts_sorted) // 2]
    st.session_state["today_wake_time"] = wakes_sorted[len(wakes_sorted) // 2]

    st.session_state["today_doses_json"] = latest_doses_json
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


def render_setup_wizard(repo, username: str, baseline: UserBaseline):
    st.header("Setup Wizard")
    st.caption("최초 1회 설정입니다. 나중에 관리자/설정에서 언제든 수정할 수 있습니다.")

    st.subheader("Step 1 — Baseline")
    c1, c2, c3 = st.columns(3)
    with c1:
        baseline_sleep_start = st.time_input("평균 취침", value=baseline.baseline_sleep_start)
        baseline_wake = st.time_input("평균 기상", value=baseline.baseline_wake)
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
        st.caption("MVP는 카페인 중심입니다.")

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
            "onboarded": "true",
        }
        ok = repo.update_user_baseline(username, patch)
        if ok:
            st.success("온보딩이 완료되었습니다.")
            st.rerun()
        else:
            st.error("저장에 실패했습니다. users 시트/권한/헤더를 확인해주세요.")


def render_edit_panel(repo, username: str):
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

    st.sidebar.subheader("수면")
    sleep_on = st.sidebar.toggle("수면 시간 오버라이드", value=st.session_state["today_sleep_override_on"])
    st.session_state["today_sleep_override_on"] = sleep_on

    if sleep_on:
        cross_mid = st.sidebar.checkbox(
            "취침이 자정 이전(전날)부터 이어짐",
            value=st.session_state["today_sleep_cross_midnight"],
        )
        st.session_state["today_sleep_cross_midnight"] = cross_mid

        st.session_state["today_sleep_start"] = st.sidebar.time_input("취침 시각", value=st.session_state["today_sleep_start"])
        st.session_state["today_wake_time"] = st.sidebar.time_input("기상 시각", value=st.session_state["today_wake_time"])
    else:
        st.sidebar.caption("기본값(온보딩 baseline)을 사용합니다.")

    st.sidebar.subheader("카페인")
    drafts = doses_from_json(st.session_state["today_doses_json"])
    if not drafts:
        drafts = [DoseDraft(9, 0, 150.0)]

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
        cc1, cc2 = st.sidebar.columns(2)
        with cc1:
            hour = st.number_input("시", 0, 23, value=int(drafts[i].hour), key=f"dose_h_{i}")
            minute = st.number_input("분", 0, 59, value=int(drafts[i].minute), key=f"dose_m_{i}")
        with cc2:
            mg = st.number_input("mg", 0.0, 400.0, value=float(drafts[i].amount_mg), step=10.0, key=f"dose_mg_{i}")
        new_drafts.append(DoseDraft(int(hour), int(minute), float(mg)))

    st.sidebar.subheader("업무 부하")
    st.session_state["today_workload"] = st.sidebar.slider(
        "workload_level", 0.0, 3.0, value=float(st.session_state["today_workload"]), step=0.5
    )

    st.sidebar.subheader("자가 선명도(선택)")
    st.session_state["today_clarity"] = st.sidebar.slider(
        "subjective_clarity (0–10)", 0.0, 10.0, value=float(st.session_state["today_clarity"]), step=0.5
    )

    apply = st.sidebar.button("저장하고 적용", type="primary", use_container_width=True)
    if apply:
        st.session_state["today_doses_json"] = doses_to_json(date, TZ, new_drafts)
        save_today_state(repo, username)
        st.session_state["edit_today_open"] = False
        st.rerun()


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

    out = predict_day(baseline, tomorrow_inputs, step_minutes=10)
    blocks = design_schedule(out, deep_work_target_minutes=150)

    st.caption(f"기준일: {tomorrow.isoformat()} (오늘 입력을 템플릿으로 사용)")
    if not blocks:
        st.info("내일 추천 블록을 만들기 어렵습니다. 수면/카페인 입력을 보완해보세요.")
    for b in blocks:
        st.write(f"- **{b.label}** · {b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("내일 계획 초안 저장", use_container_width=True):
            payload = {
                "sleep_override_on": str(bool(st.session_state["today_sleep_override_on"])).lower(),
                "sleep_cross_midnight": str(bool(st.session_state["today_sleep_cross_midnight"])).lower(),
                "sleep_start": st.session_state["today_sleep_start"].strftime("%H:%M"),
                "wake_time": st.session_state["today_wake_time"].strftime("%H:%M"),
                "doses_json": doses_to_json(tomorrow, TZ, today_drafts),
                "workload_level": str(st.session_state["today_workload"]),
                "subjective_clarity": "",
            }
            repo.upsert_daily_log(username, tomorrow, payload)
            st.success("내일 daily_log 초안을 저장했습니다.")
    with c2:
        st.caption("내일 아침 실제 상태에 맞게 미세 조정하세요.")


def render_daily_checkin(repo, username: str, baseline: UserBaseline, out):
    st.divider()
    st.subheader("하루 마감 체크인 (10초)")

    date = st.session_state["today_date"]
    existing = repo.get_checkin(username, date) or {}

    focus_default = int(_safe_float(existing.get("actual_focus_minutes", 0), 0.0))
    sat_default = _safe_float(existing.get("energy_satisfaction", st.session_state["today_clarity"]), st.session_state["today_clarity"])
    notes_default = str(existing.get("notes", ""))

    c1, c2 = st.columns(2)
    with c1:
        focus = st.number_input("실제 집중 시간(분)", min_value=0, max_value=720, value=focus_default, step=10)
        sat = st.slider("오늘 에너지 만족도 (0–10)", 0.0, 10.0, value=float(sat_default), step=0.5)
    with c2:
        notes = st.text_area("메모(선택)", value=notes_default, height=95)

    c3, c4 = st.columns(2)
    with c3:
        if st.button("체크인 저장", use_container_width=True):
            repo.upsert_checkin(
                username,
                date,
                {
                    "actual_focus_minutes": str(int(focus)),
                    "energy_satisfaction": str(float(sat)),
                    "notes": notes,
                },
            )
            st.success("체크인을 저장했습니다.")

    with c4:
        if st.button("체크인 기준 보정 적용", use_container_width=True):
            predicted_mean = sum(out.net) / max(len(out.net), 1)
            focus_bonus = _clip((float(focus) - 120.0) / 300.0, -0.2, 0.2)
            synthetic_clarity = _clip(float(sat) + focus_bonus * 10.0, 0.0, 10.0)
            new_offset = calibrate_baseline_offset(
                current_offset=baseline.baseline_offset,
                subjective_clarity_0_10=float(synthetic_clarity),
                predicted_net_mean_0_1=float(predicted_mean),
                eta=0.10,
                k=0.50,
            )
            baseline.baseline_offset = float(new_offset)
            ok = repo.update_user_baseline(username, {"baseline_offset": str(baseline.baseline_offset)})
            if ok:
                st.success(f"baseline_offset 업데이트: {baseline.baseline_offset:.3f}")
                st.rerun()
            else:
                st.error("users 시트 업데이트 실패")


def render_weekly_report(repo, username: str, baseline: UserBaseline):
    st.divider()
    st.subheader("주간 리포트 (최근 7일)")

    end_date = st.session_state["today_date"]
    start_date = end_date - dt.timedelta(days=6)

    logs = repo.get_daily_logs_for_user(username, limit=120)
    week_logs: List[Tuple[dt.date, Dict[str, Any]]] = []
    for r in logs:
        d = _parse_date_iso(r.get("date", ""))
        if not d:
            continue
        if start_date <= d <= end_date:
            week_logs.append((d, r))

    if not week_logs:
        st.info("최근 7일 로그가 아직 없습니다.")
        return

    # Aggregate from predictions over each day.
    prime_hour_counts: Dict[int, int] = {h: 0 for h in range(24)}
    crash_minutes_total = 0.0
    pred_means: List[float] = []
    clarity_values: List[float] = []

    for d, row in week_logs:
        day_inputs = _build_inputs_from_log_row(row, d)
        out = predict_day(baseline, day_inputs, step_minutes=10)

        pred_means.append(sum(out.net) / max(len(out.net), 1))
        clarity_values.append(_safe_float(row.get("subjective_clarity", 0.0), 0.0))

        for i, t in enumerate(out.t):
            if out.zones.get("prime", [False] * len(out.t))[i]:
                prime_hour_counts[t.hour] += 1
            if out.zones.get("crash", [False] * len(out.t))[i]:
                crash_minutes_total += out.meta.get("step_minutes", 10)

    peak_hour = max(prime_hour_counts, key=lambda h: prime_hour_counts[h])
    avg_pred = sum(pred_means) / len(pred_means)
    avg_clarity = sum(clarity_values) / len(clarity_values) if clarity_values else 0.0
    avg_crash_minutes = crash_minutes_total / max(len(week_logs), 1)

    checkins = repo.get_checkins_for_user(username, limit=120)
    week_checkins = []
    for r in checkins:
        d = _parse_date_iso(r.get("date", ""))
        if not d:
            continue
        if start_date <= d <= end_date:
            week_checkins.append(r)

    avg_focus = 0.0
    avg_sat = 0.0
    focus_goal_hit = 0
    if week_checkins:
        focus_vals = [_safe_float(r.get("actual_focus_minutes", 0), 0.0) for r in week_checkins]
        sat_vals = [_safe_float(r.get("energy_satisfaction", 0), 0.0) for r in week_checkins]
        avg_focus = sum(focus_vals) / len(focus_vals)
        avg_sat = sum(sat_vals) / len(sat_vals)
        focus_goal_hit = sum(1 for x in focus_vals if x >= 120)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prime 피크 시간", f"{peak_hour:02d}:00")
    c2.metric("평균 Crash(분/일)", f"{avg_crash_minutes:.0f}")
    c3.metric("평균 자가 선명도", f"{avg_clarity:.1f}/10")
    c4.metric("평균 예측 에너지", f"{avg_pred:.2f}")

    if week_checkins:
        st.write(
            f"- 체크인 평균: 집중 **{avg_focus:.0f}분**, 만족도 **{avg_sat:.1f}/10**, 120분 목표 달성 **{focus_goal_hit}/{len(week_checkins)}일**"
        )
    st.write(
        f"- 해석: 최근 7일 기준 고집중 배치는 **{peak_hour:02d}:00 전후**가 상대적으로 유리할 가능성이 높습니다."
    )


def render_dashboard(repo, username: str, baseline: UserBaseline):
    day_inputs = build_day_inputs()
    out = predict_day(baseline, day_inputs, step_minutes=10)

    recent_logs = _recent_logs_window(repo, username, st.session_state["today_date"], days=7)
    conf_label, conf_score, conf_reasons = _estimate_confidence(out, day_inputs, len(recent_logs))

    st.info(f"예측 신뢰도: **{conf_label}** ({conf_score}/100)")
    if conf_reasons:
        st.caption("신뢰도 하락 요인: " + ", ".join(conf_reasons))

    fig = plot_net_energy(out, show_components=False)
    st.pyplot(fig, clear_figure=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Zone 해석")
        for line in interpret_zones(out):
            st.write("- " + line)

    with c2:
        st.subheader("오늘의 설계 제안")
        blocks = design_schedule(out, deep_work_target_minutes=120)
        if not blocks:
            st.caption("추천 블록이 충분히 생성되지 않았을 수 있습니다. 입력을 조정해보세요.")
        for b in blocks:
            st.write(f"**{b.label}** · {b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}")
            st.caption(b.rationale)

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

    if not user.onboarded:
        st.sidebar.info("최초 1회 설정이 필요합니다.")
        render_setup_wizard(repo, username, baseline)
        return

    if "today_loaded_once" not in st.session_state:
        load_today_state(repo, username)
        st.session_state["today_loaded_once"] = True

    render_topbar()

    if st.session_state["edit_today_open"]:
        render_edit_panel(repo, username)
    else:
        st.sidebar.info("상단의 **오늘 입력 수정** 버튼으로 수면/카페인/부하를 조정할 수 있습니다.")
        if st.sidebar.button("오늘 로그 다시 불러오기", use_container_width=True):
            load_today_state(repo, username)
            st.rerun()

    render_dashboard(repo, username, baseline)


if __name__ == "__main__":
    main()
