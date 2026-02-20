# neuroframe/components/edit_panel.py
import datetime as dt
from typing import List, Dict, Any
import streamlit as st
from zoneinfo import ZoneInfo

from shared.today_input import doses_from_json, doses_to_json, DoseDraft

TZ = ZoneInfo("Asia/Seoul")

def _parse_time_hhmm(s: str, fallback: dt.time) -> dt.time:
    try:
        hh, mm = str(s).strip().split(":")
        return dt.time(int(hh), int(mm))
    except Exception:
        return fallback

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

def _shift_blocks_to_json(blocks: List[Dict[str, str]]) -> str:
    import json
    return json.dumps(blocks, ensure_ascii=False)

def _parse_shift_blocks_json(s: str) -> List[Dict[str, str]]:
    import json
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

def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _parse_date_iso(s: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(s))
    except Exception:
        return None

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

def render_edit_panel(repo, username: str, load_func, save_func, uses_adhd_medication: bool):
    st.sidebar.header("오늘 입력")

    date = st.sidebar.date_input("날짜", value=st.session_state["today_date"])
    if date != st.session_state["today_date"]:
        st.session_state["today_date"] = date
        load_func(repo, username)

    if st.sidebar.button("최근 7일 패턴으로 자동 채우기", use_container_width=True):
        ok = _apply_recent_pattern(repo, username, st.session_state["today_date"])
        if ok:
            st.sidebar.success("최근 패턴으로 기본값을 채웠습니다.")
            st.rerun()
        st.sidebar.warning("참조할 최근 로그가 부족합니다.")

    if st.sidebar.button("어제 카페인 입력 복사", use_container_width=True):
        from data.cache import _cached_daily_log
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

    def _dose_type_label(v: str) -> str:
        key = str(v or "caffeine").strip().lower()
        if key == "mph_ir":
            return "MPH IR"
        if key == "mph_xr":
            return "MPH XR"
        return "Caffeine"

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
            save_func(repo, username)
            st.session_state["edit_today_open"] = False
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"오늘 입력 저장 실패: {e}")
