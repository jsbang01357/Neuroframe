# neuroframe_app.py
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import streamlit as st

from auth import get_repo, login_guard, render_user_badge

from neuroframe.components.wizard import render_setup_wizard
from neuroframe.components.checkin import render_morning_checkin, render_daily_checkin
from neuroframe.components.edit_panel import render_edit_panel
from neuroframe.components.dashboard import (
    render_tomorrow_plan,
    render_weekly_report,
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
# from shared.weekly_metrics import compute_caffeine_total_mg_from_doses_json, compute_sleep_debt_hours
# from services.export_service import weekly_report_pdf_bytes
from data.cache import (
    _invalidate_repo_read_caches,
    _recent_logs_window,
)
from data.state_io import load_today_state, save_today_state
from domain.shift import (
    _parse_shift_blocks_json,
    _shift_blocks_to_spans,
)
from ui.dashboard_views import (
    render_calendar_conflicts as render_calendar_conflicts_view,
    render_nap_recommendations as render_nap_recommendations_view,
    render_privacy_controls as render_privacy_controls_view,
    render_rebound_explanation as render_rebound_explanation_view,
    render_shift_action_plan as render_shift_action_plan_view,
    render_shift_aware_insights as render_shift_aware_insights_view,
    render_summary_cards as render_summary_cards_view,
    render_today_recommendations as render_today_recommendations_view,
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
    st.session_state.setdefault("task_done_map", {})
    st.session_state.setdefault("profile_is_shift_worker", False)
    st.session_state.setdefault("profile_uses_adhd_medication", False)
    st.session_state.setdefault(
        "profile_medication_tags",
        {
            "atomoxetine": False,
            "ssri": False,
            "aripiprazole": False,
            "beta_blocker": False,
        },
    )

    st.session_state.setdefault("morning_checkin_done_date", "")


def toggle_edit_panel():
    st.session_state["edit_today_open"] = not st.session_state.get("edit_today_open", False)


# ----------------------------
# Utility
# ----------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x






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


def _baseline_with_profile_tags_hint(
    baseline: UserBaseline,
    tags: Dict[str, bool],
) -> Tuple[UserBaseline, List[str]]:
    # Keep adjustments very small and non-diagnostic.
    adj = UserBaseline(
        baseline_sleep_start=baseline.baseline_sleep_start,
        baseline_wake=baseline.baseline_wake,
        chronotype_shift_hours=baseline.chronotype_shift_hours,
        caffeine_half_life_hours=baseline.caffeine_half_life_hours,
        caffeine_sensitivity=baseline.caffeine_sensitivity,
        baseline_offset=baseline.baseline_offset,
        circadian_weight=baseline.circadian_weight,
        sleep_pressure_weight=baseline.sleep_pressure_weight,
        drug_weight=baseline.drug_weight,
        load_weight=baseline.load_weight,
    )
    hints: List[str] = []

    if bool(tags.get("beta_blocker", False)):
        adj.caffeine_sensitivity = float(_clip(adj.caffeine_sensitivity * 0.96, 0.5, 2.5))
        hints.append("beta blocker 태그가 있어 카페인 민감도를 소폭 낮춰 추정할 수 있습니다.")
    if bool(tags.get("ssri", False)):
        adj.baseline_offset = float(_clip(adj.baseline_offset - 0.01, -1.0, 1.0))
        hints.append("SSRI 태그가 있어 baseline offset을 미세 조정해 볼 수 있습니다.")
    if bool(tags.get("aripiprazole", False)):
        adj.drug_weight = float(_clip(adj.drug_weight * 0.97, 0.0, 1.0))
        hints.append("aripiprazole 태그가 있어 stimulant 가중치를 소폭 보수적으로 반영할 수 있습니다.")
    if bool(tags.get("atomoxetine", False)):
        hints.append("atomoxetine 태그는 자극제 rebound 모델에 직접 주입하지 않고 해석 힌트로만 반영될 수 있습니다.")

    return adj, hints



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
        if st.button("늦은 카페인 12:30으로 자동 당기기", use_container_width=True):
            date = st.session_state["today_date"]
            drafts = doses_from_json(st.session_state.get("today_doses_json", "[]"))
            shifted: List[DoseDraft] = []
            for d in drafts:
                typ = str(getattr(d, "dose_type", "caffeine") or "caffeine").strip().lower()
                if typ == "caffeine" and int(d.hour) >= 15:
                    shifted.append(DoseDraft(12, 30, float(d.amount_mg), "caffeine"))
                else:
                    shifted.append(d)
            st.session_state["today_doses_json"] = doses_to_json(date, TZ, shifted)
            st.success("늦은 카페인을 12:30으로 조정했습니다. 저장 후 적용하면 내일 템플릿에도 반영됩니다.")
            st.rerun()
    elif has_stimulant:
        st.info("오늘은 자극제 입력이 있습니다. 추가 카페인은 필요 최소량을 이른 시간대에 배치하면 저녁 반동(rebound)과 수면 지연 가능성을 줄이는 데 도움이 될 수 있습니다.")



def _session_shift_spans_for_date(date: dt.date) -> List[Tuple[dt.datetime, dt.datetime, str]]:
    blocks = _parse_shift_blocks_json(st.session_state.get("today_shift_blocks_json", "[]"))
    return _shift_blocks_to_spans(date, blocks)



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
        shift_blocks=_session_shift_spans_for_date(date),
    )








def render_dashboard(repo, username: str, baseline: UserBaseline):
    st.caption("면책: 본 예측은 입력값 기반 참고 정보이며 의료 조언이 아닙니다.")

    day_inputs = build_day_inputs()
    tags = dict(
        st.session_state.get(
            "profile_medication_tags",
            {"atomoxetine": False, "ssri": False, "aripiprazole": False, "beta_blocker": False},
        )
    )
    base_with_tags, tag_hints = _baseline_with_profile_tags_hint(baseline, tags)
    effective_baseline = _effective_baseline(base_with_tags, st.session_state["today_shift_hours"])
    out = predict_day(effective_baseline, day_inputs, step_minutes=10)
    if tag_hints:
        st.caption("참고: SSRI/aripiprazole/atomoxetine/beta blocker는 당일 곡선이 아니라 장기 상태 태그로 취급될 수 있습니다.")
        for hint in tag_hints:
            st.caption(f"- {hint}")

    recent_logs = _recent_logs_window(repo, username, st.session_state["today_date"], days=7)
    conf_label, conf_score, conf_reasons = _estimate_confidence(day_inputs, len(recent_logs))
    st.info(f"예측 신뢰도: **{conf_label}** ({conf_score}/100)")
    if conf_reasons:
        st.caption("신뢰도 하락 요인: " + ", ".join(conf_reasons))
    _render_stimulant_caffeine_notice(day_inputs)

    # 1.0 Summary-first dashboard
    render_summary_cards_view(out)
    render_rebound_explanation_view(out)
    render_today_recommendations_view(repo, username, out, design_schedule, save_today_state)
    render_calendar_conflicts_view(out, TZ)

    # 2.x Shift mode
    shift_spans = _session_shift_spans_for_date(st.session_state["today_date"])
    render_shift_action_plan_view(out, shift_spans)
    render_shift_aware_insights_view(out, shift_spans)
    render_nap_recommendations_view(out, shift_spans)

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
    render_weekly_report(repo, username, baseline, TZ)
    render_privacy_controls_view(repo, username)


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
    st.session_state["profile_medication_tags"] = dict(
        getattr(
            user,
            "medication_tags",
            {"atomoxetine": False, "ssri": False, "aripiprazole": False, "beta_blocker": False},
        )
    )

    if not user.onboarded:
        st.sidebar.info("최초 1회 설정이 필요합니다.")
        render_setup_wizard(repo, username, baseline)
        return

    if "today_loaded_once" not in st.session_state:
        load_today_state(repo, username)
        st.session_state["today_loaded_once"] = True

    render_topbar()
    # merged from main: boolean flags and injected helpers
    render_morning_checkin(repo, username, save_today_state, bool(getattr(user, "is_shift_worker", False)))

    if st.session_state["edit_today_open"]:
        render_edit_panel(repo, username, load_today_state, save_today_state, bool(getattr(user, "uses_adhd_medication", False)))
    else:
        st.sidebar.info("상단의 **오늘 입력 수정** 버튼으로 수면/카페인/근무블록/부하를 조정할 수 있습니다.")
        if st.sidebar.button("오늘 로그 다시 불러오기", use_container_width=True):
            _invalidate_repo_read_caches()
            load_today_state(repo, username)
            st.rerun()

    render_dashboard(repo, username, baseline)


if __name__ == "__main__":
    main()
