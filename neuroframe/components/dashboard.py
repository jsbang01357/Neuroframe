# neuroframe/components/dashboard.py
import datetime as dt
from typing import List, Tuple, Dict, Any
import streamlit as st

from neuroframe.engine import UserBaseline, predict_day, DayInputs
from neuroframe.coach import design_schedule, interpret_zones
from shared.today_input import doses_from_json, drafts_to_engine_doses, parse_sleep_override, doses_to_json
from shared.weekly_metrics import compute_caffeine_total_mg_from_doses_json, compute_sleep_debt_hours
from services.export_service import weekly_report_pdf_bytes
from data.cache import _cached_daily_logs_for_user
import json
from zoneinfo import ZoneInfo

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _baseline_sleep_hours(baseline: UserBaseline) -> float:
    s = baseline.baseline_sleep_start
    w = baseline.baseline_wake
    cross = s > w
    dm = (w.hour * 60 + w.minute) - (s.hour * 60 + s.minute)
    if cross:
        dm += 1440
    return dm / 60.0

def _parse_date_iso(s: str) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None

def _parse_time_hhmm(s: str, default_t: dt.time) -> dt.time:
    if not s or ":" not in s:
        return default_t
    try:
        h_str, m_str = s.split(":")
        return dt.time(int(h_str), int(m_str))
    except (ValueError, TypeError):
        return default_t

def _sleep_duration_hours(start: dt.time, end: dt.time, cross: bool) -> float:
    m1 = start.hour * 60 + start.minute
    m2 = end.hour * 60 + end.minute
    dm = m2 - m1
    if cross:
        dm += 1440
    return max(0.0, dm / 60.0)

def _minutes_to_hhmm(mins: float) -> str:
    m = int(mins) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"

def _task_done_counts_from_json(js: str) -> Tuple[int, int]:
    try:
        data = json.loads(js)
        done = sum(1 for v in data.values() if bool(v))
        return done, len(data)
    except:
        return 0, 0

def _build_inputs_from_log_row(row: Dict[str, Any], date_obj: dt.date, TZ: ZoneInfo) -> DayInputs:
    doses = drafts_to_engine_doses(date_obj, TZ, doses_from_json(row.get("doses_json", "[]")))
    sleep_override = None
    if str(row.get("sleep_override_on", "false")).lower() == "true":
        sleep_dt, wake_dt = parse_sleep_override(
            date=date_obj,
            tz=TZ,
            sleep_start=_parse_time_hhmm(row.get("sleep_start", "23:30"), dt.time(23, 30)),
            wake_time=_parse_time_hhmm(row.get("wake_time", "07:30"), dt.time(7, 30)),
            crosses_midnight=str(row.get("sleep_cross_midnight", "true")).lower() == "true",
        )
        sleep_override = (sleep_dt, wake_dt)
    return DayInputs(
        date=date_obj,
        timezone=TZ,
        sleep_override=sleep_override,
        doses=doses,
        workload_level=_safe_float(row.get("workload_level", 0.0), 0.0),
    )

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


def render_tomorrow_plan(repo, username: str, baseline: UserBaseline, TZ: ZoneInfo):
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
            "task_done_json": "{}",
            "day_shift_hours": str(float(st.session_state["today_shift_hours"])),
            "workload_level": str(st.session_state["today_workload"]),
            "subjective_clarity": "",
        }
        try:
            repo.upsert_daily_log(username, tomorrow, payload)
            # st.session_state["today_loaded_once"] = False # optional rerun trigger
            st.success("내일 daily_log 초안을 저장했습니다.")
        except Exception as e:
            st.error(f"내일 계획 저장 실패: {e}")

def render_weekly_report(repo, username: str, baseline: UserBaseline, TZ: ZoneInfo):
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
    weekly_sleep_hours: List[float] = []
    weekly_doses_json: List[str] = []
    plan_total_7d = 0
    plan_done_7d = 0

    for d in week_days:
        row = log_by_date.get(d)
        if row:
            day_shift = _safe_float(row.get("day_shift_hours", 0.0), 0.0)
            base_for_day = _effective_baseline(baseline, day_shift)
            day_inputs = _build_inputs_from_log_row(row, d, TZ)
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

            weekly_sleep_hours.append(actual_sleep_h)
            weekly_doses_json.append(str(row.get("doses_json", "[]") or "[]"))
            d_done, d_total = _task_done_counts_from_json(str(row.get("task_done_json", "{}") or "{}"))
            plan_done_7d += d_done
            plan_total_7d += d_total
        else:
            weekly_sleep_hours.append(target_sleep_h)
            weekly_doses_json.append("[]")

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
    sleep_7d_total = sum(weekly_sleep_hours)
    sleep_7d_avg = sleep_7d_total / 7.0
    sleep_debt_7d = compute_sleep_debt_hours(target_sleep_h, weekly_sleep_hours)
    caffeine_total_mg = compute_caffeine_total_mg_from_doses_json(weekly_doses_json)
    avg_prime_start_text = _minutes_to_hhmm(avg_prime_start) if avg_prime_start is not None else "-"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prime 평균 시작", avg_prime_start_text)
    c2.metric("Crash 평균 길이", f"{avg_crash_len:.0f}분")
    c3.metric("최근48h 수면", f"{sleep_48h:.1f}h")
    c4.metric("7일 평균 수면", f"{sleep_7d_avg:.1f}h")

    c5, c6 = st.columns(2)
    c5.metric("수면부채(7d)", f"{sleep_debt_7d:.1f}h")
    c6.metric("카페인 총량(7d)", f"{caffeine_total_mg:.0f}mg")

    exec_rate = (100.0 * plan_done_7d / plan_total_7d) if plan_total_7d > 0 else None
    c7, c8 = st.columns(2)
    c7.metric("계획-실행 일치율(7d)", f"{exec_rate:.0f}%" if exec_rate is not None else "-")
    c8.metric("완료/계획 블록(7d)", f"{plan_done_7d}/{plan_total_7d}")

    if sleep_debt_7d >= 4.0:
        rec = "이번 주는 수면부채가 큰 편입니다. Off 또는 Day 다음날 우선 회복 수면을 확보하세요."
    elif avg_crash_len >= 150.0:
        rec = "Crash가 길게 나타납니다. 업무 강도를 분산하고 체크리스트 기반으로 리듬을 낮추세요."
    elif caffeine_total_mg >= 1400.0:
        rec = "카페인 총량이 높습니다. 야간 전후 도즈 타이밍을 앞당기거나 양을 줄이는 편이 유리합니다."
    else:
        rec = "패턴이 비교적 안정적입니다. Prime 시작 30분 전 준비 루틴을 고정하면 효율이 더 오릅니다."
    st.write(f"- 이번 주 추천: {rec}")

    pdf_bytes = weekly_report_pdf_bytes(
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
