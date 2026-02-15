# neuroframe_app.py
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo
import streamlit as st

from auth import get_repo, login_guard, render_user_badge
from neuroframe.engine import UserBaseline, DayInputs, predict_day, calibrate_baseline_offset
from neuroframe.plots import plot_net_energy
from neuroframe.coach import interpret_zones, design_schedule
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

    # 오늘 입력 기본값 (로그 저장/로드로 덮어씌워짐)
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
# Repo load/save hooks (gsheets)
# ----------------------------

def load_today_state(repo, username: str):
    """
    Load daily log for (username, date) into session_state if exists.
    """
    date = st.session_state["today_date"]
    row = repo.get_daily_log(username, date)
    if not row:
        return

    st.session_state["today_sleep_override_on"] = str(row.get("sleep_override_on", "false")).lower() == "true"
    st.session_state["today_sleep_cross_midnight"] = str(row.get("sleep_cross_midnight", "true")).lower() == "true"

    def tparse(s: str | None, fallback: str) -> dt.time:
        s = (s or fallback).strip()
        hh, mm = s.split(":")
        return dt.time(int(hh), int(mm))

    st.session_state["today_sleep_start"] = tparse(row.get("sleep_start"), "23:30")
    st.session_state["today_wake_time"] = tparse(row.get("wake_time"), "07:30")

    st.session_state["today_doses_json"] = row.get("doses_json", "[]") or "[]"
    st.session_state["today_workload"] = float(row.get("workload_level") or 1.0)
    st.session_state["today_clarity"] = float(row.get("subjective_clarity") or 5.0)


def save_today_state(repo, username: str):
    """
    Upsert daily log from session_state.
    """
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
# UI: top bar
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


# ----------------------------
# Onboarding (Setup Wizard)
# ----------------------------

def render_setup_wizard(repo, username: str, baseline: UserBaseline):
    st.header("Setup Wizard")
    st.caption("최초 1회 설정입니다. 나중에 관리자/설정에서 언제든 수정할 수 있습니다.")

    st.subheader("Step 1 — Baseline")
    c1, c2, c3 = st.columns(3)
    with c1:
        baseline_sleep_start = st.time_input("평균 취침", value=baseline.baseline_sleep_start)
        baseline_wake = st.time_input("평균 기상", value=baseline.baseline_wake)
    with c2:
        chronoshift = st.number_input(
            "크로노타입 시프트(시간)",
            value=float(baseline.chronotype_shift_hours),
            step=0.5,
            help="음수: 아침형 / 양수: 저녁형 (peak을 좌/우로 이동)"
        )
    with c3:
        baseline_offset = st.number_input(
            "baseline_offset",
            value=float(baseline.baseline_offset),
            step=0.05,
            help="전반적 컨디션 보정(오늘/나의 평균)"
        )

    st.subheader("Step 2 — 약물/진정 영향")
    c4, c5 = st.columns(2)
    with c4:
        caffeine_half_life = st.number_input("카페인 반감기(시간)", value=float(baseline.caffeine_half_life_hours), step=0.5)
        caffeine_sensitivity = st.number_input("카페인 민감도", value=float(baseline.caffeine_sensitivity), step=0.1)
    with c5:
        st.caption("MVP는 카페인만 우선 지원합니다. MPH 등은 dose 타입 확장으로 붙이기 쉬운 구조입니다.")

    st.subheader("Step 3 — 엔진 가중치(초기값 권장)")
    c6, c7, c8, c9 = st.columns(4)
    with c6:
        w_c = st.number_input("circadian_weight", value=float(baseline.circadian_weight), step=0.1)
    with c7:
        w_s = st.number_input("sleep_pressure_weight", value=float(baseline.sleep_pressure_weight), step=0.1)
    with c8:
        w_d = st.number_input("drug_weight", value=float(baseline.drug_weight), step=0.001, format="%.6f")
    with c9:
        w_l = st.number_input("load_weight", value=float(baseline.load_weight), step=0.05)

    st.divider()
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
            st.error("저장에 실패했습니다. (users 시트/권한/헤더를 확인해주세요.)")


# ----------------------------
# Today edit panel (sidebar)
# ----------------------------

def render_edit_panel(repo, username: str):
    st.sidebar.header("오늘 입력")

    date = st.sidebar.date_input("날짜", value=st.session_state["today_date"])
    # 날짜가 바뀌면 해당 날짜 로그를 다시 로드
    if date != st.session_state["today_date"]:
        st.session_state["today_date"] = date
        load_today_state(repo, username)

    st.sidebar.subheader("수면")
    sleep_on = st.sidebar.toggle("수면 시간 오버라이드", value=st.session_state["today_sleep_override_on"])
    st.session_state["today_sleep_override_on"] = sleep_on

    if sleep_on:
        cross_mid = st.sidebar.checkbox(
            "취침이 자정 이전(전날)부터 이어짐",
            value=st.session_state["today_sleep_cross_midnight"]
        )
        st.session_state["today_sleep_cross_midnight"] = cross_mid

        ss = st.sidebar.time_input("취침 시각", value=st.session_state["today_sleep_start"])
        ww = st.sidebar.time_input("기상 시각", value=st.session_state["today_wake_time"])
        st.session_state["today_sleep_start"] = ss
        st.session_state["today_wake_time"] = ww
    else:
        st.sidebar.caption("기본값(온보딩 baseline)을 사용합니다.")

    st.sidebar.subheader("카페인")
    drafts = doses_from_json(st.session_state["today_doses_json"])
    if not drafts:
        drafts = [DoseDraft(9, 0, 150.0)]

    n = st.sidebar.slider("도즈 개수", 0, 6, min(len(drafts), 6))
    drafts = drafts[:n]

    new_drafts = []
    for i in range(n):
        st.sidebar.markdown(f"**Dose {i+1}**")
        cc1, cc2 = st.sidebar.columns(2)
        with cc1:
            hour = st.number_input("시", 0, 23, value=int(drafts[i].hour), key=f"dose_h_{i}")
            minute = st.number_input("분", 0, 59, value=int(drafts[i].minute), key=f"dose_m_{i}")
        with cc2:
            mg = st.number_input("mg", 0.0, 400.0, value=float(drafts[i].amount_mg), step=10.0, key=f"dose_mg_{i}")
        new_drafts.append(DoseDraft(int(hour), int(minute), float(mg)))

    st.sidebar.subheader("업무 부하")
    workload = st.sidebar.slider("workload_level", 0.0, 3.0, value=float(st.session_state["today_workload"]), step=0.5)
    st.session_state["today_workload"] = workload

    st.sidebar.subheader("자가 선명도(선택)")
    clarity = st.sidebar.slider("subjective_clarity (0–10)", 0.0, 10.0, value=float(st.session_state["today_clarity"]), step=0.5)
    st.session_state["today_clarity"] = clarity

    st.sidebar.divider()
    apply = st.sidebar.button("저장하고 적용", type="primary", use_container_width=True)
    if apply:
        st.session_state["today_doses_json"] = doses_to_json(date, TZ, new_drafts)
        save_today_state(repo, username)
        st.session_state["edit_today_open"] = False
        st.rerun()


# ----------------------------
# Build inputs + dashboard
# ----------------------------

def build_day_inputs(baseline: UserBaseline) -> DayInputs:
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
    # 예측
    day_inputs = build_day_inputs(baseline)
    out = predict_day(baseline, day_inputs, step_minutes=10)

    # 그래프
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
            st.caption("추천 블록이 충분히 생성되지 않았을 수 있습니다. 입력(수면/카페인/부하)을 조정해보세요.")
        for b in blocks:
            st.write(f"**{b.label}** · {b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}")
            st.caption(b.rationale)

    # 개인화(Level 1): baseline_offset 업데이트 → users 시트 반영
    st.divider()
    st.subheader("개인화(권장)")
    st.caption("자가 선명도 입력을 기반으로 baseline_offset을 소폭 조정합니다. (머신러닝 없이도 적응형 느낌)")

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


# ----------------------------
# Main
# ----------------------------

def main():
    # 0) Session init
    init_session_defaults()

    # 1) Repo + Auth
    repo = get_repo("NeuroFrame_DB")
    auth = login_guard(repo)
    user = auth.user
    assert user is not None
    username = user.username

    # 2) Sidebar user badge + edit panel
    render_user_badge(user)

    # 3) Load baseline (fresh each run; source of truth = users sheet)
    # (RepoUser.baseline already parsed from sheet)
    baseline = user.baseline

    # 4) If not onboarded -> wizard only
    if not user.onboarded:
        st.sidebar.info("최초 1회 설정이 필요합니다.")
        render_setup_wizard(repo, username, baseline)
        return

    # 5) On first entry of the session, load today's state from daily_logs
    # Prevent re-loading after user edits unless date changes or manual reload.
    if "today_loaded_once" not in st.session_state:
        load_today_state(repo, username)
        st.session_state["today_loaded_once"] = True

    # 6) Topbar + edit toggle
    render_topbar()

    if st.session_state["edit_today_open"]:
        render_edit_panel(repo, username)
    else:
        st.sidebar.info("상단의 **오늘 입력 수정** 버튼으로 수면/카페인/부하를 조정할 수 있습니다.")
        if st.sidebar.button("오늘 로그 다시 불러오기", use_container_width=True):
            load_today_state(repo, username)
            st.rerun()

    # 7) Dashboard
    render_dashboard(repo, username, baseline)


if __name__ == "__main__":
    main()
