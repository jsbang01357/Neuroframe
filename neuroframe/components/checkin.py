# neuroframe/components/checkin.py
import datetime as dt
from zoneinfo import ZoneInfo
import streamlit as st

from shared.today_input import doses_to_json, DoseDraft

TZ = ZoneInfo("Asia/Seoul")


def _shift_template_blocks(template: str) -> list[dict[str, str]]:
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


def _shift_blocks_to_json(blocks: list[dict[str, str]]) -> str:
    import json
    return json.dumps(blocks, ensure_ascii=False)


def _plan_to_drafts(plan: str) -> list[DoseDraft]:
    if plan == "없음":
        return []
    if plan == "라이트":
        return [DoseDraft(9, 0, 100.0)]
    if plan == "보통":
        return [DoseDraft(9, 0, 150.0), DoseDraft(14, 0, 70.0)]
    return [DoseDraft(8, 30, 180.0), DoseDraft(13, 30, 120.0)]


def render_morning_checkin(repo, username: str, save_today_state_func):
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

        save_today_state_func(repo, username)
        st.session_state["morning_checkin_done_date"] = today.isoformat()
        st.success("체크인 완료. 오늘 요약을 업데이트합니다.")
        st.rerun()
