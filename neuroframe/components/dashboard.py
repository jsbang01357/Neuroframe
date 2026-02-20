# neuroframe/components/dashboard.py
import datetime as dt
from typing import List, Tuple, Dict, Any
import streamlit as st

from neuroframe.engine import UserBaseline, predict_day, DayInputs
from neuroframe.coach import design_schedule
from shared.today_input import doses_from_json, drafts_to_engine_doses, parse_sleep_override

def _span_minutes(span: Tuple[dt.datetime, dt.datetime]) -> float:
    return (span[1] - span[0]).total_seconds() / 60.0

def _mask_to_spans(t: List[dt.datetime], mask: List[bool]) -> List[Tuple[dt.datetime, dt.datetime]]:
    spans: List[Tuple[dt.datetime, dt.datetime]] = []
    if not t or not mask or len(t) != len(mask):
        return spans

    in_span = False
    start: dt.datetime | None = None
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

def _primary_span_text(spans: List[Tuple[dt.datetime, dt.datetime]], sleep_gate: bool = False) -> str:
    if not spans:
        return "없음"
    best = sorted(spans, key=_span_minutes, reverse=True)[0]
    if sleep_gate:
        mid = best[0] + (best[1] - best[0]) / 2
        return f"{mid.strftime('%H:%M')}±"
    return f"{best[0].strftime('%H:%M')}–{best[1].strftime('%H:%M')}"

def _overlap_minutes(a: Tuple[dt.datetime, dt.datetime], b: Tuple[dt.datetime, dt.datetime]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e <= s:
        return 0.0
    return (e - s).total_seconds() / 60.0

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

def render_summary_cards(out):
    prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
    crash_spans = _mask_to_spans(out.t, out.zones.get("crash", []))
    gate_spans = _mask_to_spans(out.t, out.zones.get("sleep_gate", []))

    c1, c2, c3 = st.columns(3)
    c1.metric("Prime", _primary_span_text(prime_spans))
    c2.metric("Crash", _primary_span_text(crash_spans))
    c3.metric("Sleep Gate", _primary_span_text(gate_spans, sleep_gate=True))


def render_today_recommendations(out):
    st.subheader("오늘의 추천 3개")
    blocks = design_schedule(out, deep_work_target_minutes=120, max_blocks=6)

    by_label: Dict[str, str | None] = {"Deep Work": None, "Low Load": None, "Wind-down": None}
    for b in blocks:
        if b.label in by_label and by_label[b.label] is None:
            by_label[b.label] = f"{b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}"

    st.write(f"- **Deep Work**: {by_label['Deep Work'] or '추천 시간 부족'}")
    st.write(f"- **Low Load**: {by_label['Low Load'] or '추천 시간 부족'}")
    st.write(f"- **Wind-down**: {by_label['Wind-down'] or '추천 시간 부족'}")


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


def render_tomorrow_plan(repo, username: str, baseline: UserBaseline, TZ):
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

    by_label: Dict[str, str | None] = {"Deep Work": None, "Low Load": None, "Wind-down": None}
    for b in blocks:
        if b.label in by_label and by_label[b.label] is None:
            by_label[b.label] = f"{b.start.strftime('%H:%M')}–{b.end.strftime('%H:%M')}"

    st.write(f"- **Deep Work**: {by_label['Deep Work'] or '추천 시간 부족'}")
    st.write(f"- **Low Load**: {by_label['Low Load'] or '추천 시간 부족'}")
    st.write(f"- **Wind-down**: {by_label['Wind-down'] or '추천 시간 부족'}")
    
    if st.button("내일 초안으로 복사", use_container_width=True):
        st.info("초안 생성 기능은 향후 지원 예정입니다.")
