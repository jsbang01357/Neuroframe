from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from auth import logout
from domain.shift import _overlap_minutes
from services import calendar_service
from services.export_service import blocks_to_ics_bytes


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


def _badge_html(label: str, value: str, bg: str, fg: str = "#0b1220") -> str:
    return (
        f"<span style='display:inline-block;padding:6px 10px;margin:3px 6px 3px 0;"
        f"border-radius:999px;background:{bg};color:{fg};font-size:0.88rem;font-weight:600;'>"
        f"{label} · {value}</span>"
    )


def _task_key(date: dt.date, label: str, start: dt.datetime, end: dt.datetime) -> str:
    return f"{date.isoformat()}::{label}::{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def _is_task_done(task_key: str) -> bool:
    done_map = st.session_state.get("task_done_map", {})
    return bool(done_map.get(task_key, False))


def _set_task_done(task_key: str, done: bool) -> None:
    done_map = dict(st.session_state.get("task_done_map", {}))
    done_map[task_key] = bool(done)
    st.session_state["task_done_map"] = done_map


def _task_done_counts_from_json(s: str) -> Tuple[int, int]:
    if not s:
        return 0, 0
    try:
        obj = json.loads(s)
    except Exception:
        return 0, 0
    if not isinstance(obj, dict):
        return 0, 0
    total = len(obj)
    done = sum(1 for v in obj.values() if bool(v))
    return done, total


def render_calendar_conflicts(out, tz: dt.tzinfo):
    st.subheader("Calendar 충돌")
    day = st.session_state["today_date"]
    ok, msg = calendar_service._calendar_readiness_message()
    if not ok:
        st.info(msg)
        st.caption("필요 시 도메인 위임 환경에서는 `google_calendar_delegated_user`도 설정하세요.")
        return

    events, err = calendar_service._read_google_calendar_events_with_error(day, tz)
    if err is not None:
        st.error("Google Calendar 읽기 실패")
        st.caption(msg)
        st.caption(
            "확인 항목: 1) calendar_id 접근 권한 공유 2) 서비스 계정 Calendar API 권한 "
            "3) Google Workspace 도메인 위임이 필요한 경우 delegated_user 설정"
        )
        st.code(err)
        return

    if not events:
        st.caption("오늘 일정이 없습니다.")
        return

    prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
    collisions: List[Tuple[Any, float]] = []
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


def render_rebound_explanation(out):
    rebound_spans = _mask_to_spans(out.t, out.zones.get("rebound_candidate", []))
    if not rebound_spans:
        return
    primary = sorted(rebound_spans, key=_span_minutes, reverse=True)[0]
    tr = f"{primary[0].strftime('%H:%M')}–{primary[1].strftime('%H:%M')}"
    st.info(f"Rebound 설명: 약효 감소 추정 구간이 **{tr}**에 있어 Crash 후보에 합산되었습니다. 이 구간은 의사결정 난이도를 낮춘 체크리스트 모드가 유리할 수 있습니다.")


def render_today_recommendations(repo, username: str, out, design_schedule_fn, save_today_state_fn):
    st.subheader("오늘의 추천 3개")
    blocks = design_schedule_fn(out, deep_work_target_minutes=120, max_blocks=6)
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

    if st.button("추천 체크 상태 저장", use_container_width=True):
        try:
            save_today_state_fn(repo, username)
            st.success("추천 체크 상태를 저장했습니다.")
        except Exception as e:
            st.error(f"추천 체크 저장 실패: {e}")

    ics_bytes = blocks_to_ics_bytes(blocks[:5])
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


def render_shift_action_plan(out, shift_spans: List[Tuple[dt.datetime, dt.datetime, str]]):
    st.subheader("Shift 자동 배치 계획")
    if not shift_spans:
        st.caption("오늘은 Shift 블록이 없어 일반 추천만 적용됩니다.")
        return

    crash_spans = _mask_to_spans(out.t, out.zones.get("crash", []))
    ordered = sorted(shift_spans, key=lambda x: x[0])
    first = ordered[0]
    last = ordered[-1]

    pre_start = first[0] - dt.timedelta(minutes=45)
    pre_end = pre_start + dt.timedelta(minutes=25)
    st.write(f"- **Shift 전 준비**: {pre_start.strftime('%H:%M')}–{pre_end.strftime('%H:%M')} Power nap + 인수인계 체크리스트 확인")

    crash_overlap = 0.0
    for ss, se, _ in shift_spans:
        for cs in crash_spans:
            crash_overlap += _overlap_minutes((ss, se), cs)
    if crash_overlap > 0:
        st.write(f"- **Shift 중 운영**: Crash 겹침 {crash_overlap:.0f}분 → 고위험 판단/처치는 체크리스트 모드로 전환")
    else:
        st.write("- **Shift 중 운영**: Crash 겹침이 작아 기본 운영 유지 가능")

    post_start = last[1] + dt.timedelta(minutes=15)
    post_end = post_start + dt.timedelta(minutes=90)
    st.write(f"- **Shift 후 회복**: {post_start.strftime('%H:%M')}–{post_end.strftime('%H:%M')} 90분 회복 nap + 수면부채 회복 우선")


def render_privacy_controls(repo, username: str):
    st.divider()
    with st.expander("개인정보 / 데이터 관리", expanded=False):
        if hasattr(repo, "export_user_data"):
            export_payload = repo.export_user_data(username)
        elif hasattr(getattr(repo, "db", None), "export_user_data"):
            export_payload = repo.db.export_user_data(username)  # type: ignore[attr-defined]
        else:
            export_payload = {
                "exported_at": "",
                "username": username,
                "error": "export_user_data API unavailable on current repo object; clear Streamlit cache and retry.",
                "user": {},
                "daily_logs": [],
                "checkins": [],
            }
        export_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "내 데이터 JSON 다운로드",
            data=export_bytes,
            file_name=f"neuroframe_export_{username}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.caption("계정 삭제 시 users/daily_logs/checkins 데이터가 삭제됩니다.")
        confirm = st.text_input("삭제 확인: 사용자명을 입력", value="", key="delete_confirm_username")
        anonymize = st.checkbox("익명화(삭제 대신 username 익명화)", value=False, key="delete_anonymize")
        if st.button("계정 삭제 요청", type="secondary", use_container_width=True):
            if confirm.strip() != username:
                st.error("입력한 사용자명이 현재 계정과 다릅니다.")
            else:
                try:
                    if hasattr(repo, "delete_user_data"):
                        result = repo.delete_user_data(username, anonymize=bool(anonymize))
                    elif hasattr(getattr(repo, "db", None), "delete_user_data"):
                        result = repo.db.delete_user_data(username, anonymize=bool(anonymize))  # type: ignore[attr-defined]
                    else:
                        raise RuntimeError("delete_user_data API unavailable on current repo object; clear Streamlit cache and retry.")
                    st.success(
                        f"처리 완료: users={result.get('users',0)}, "
                        f"daily_logs={result.get('daily_logs',0)}, checkins={result.get('checkins',0)}"
                    )
                    logout()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
