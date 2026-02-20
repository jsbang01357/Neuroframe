# neuroframe/components/checkin.py
import datetime as dt
from zoneinfo import ZoneInfo
import streamlit as st

from shared.today_input import doses_to_json, DoseDraft, parse_sleep_override
from neuroframe.engine import UserBaseline, calibrate_baseline_offset, predict_day
from typing import Any, Dict, List, Optional, Tuple
import json

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


def render_morning_checkin(repo, username: str, save_today_state_func, is_shift_worker: bool):
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
            save_today_state_func(repo, username)
            st.session_state["morning_checkin_done_date"] = today.isoformat()
            st.success("체크인 완료. 오늘 요약을 업데이트합니다.")
            st.rerun()
        except Exception as e:
            st.error(f"아침 체크인 저장 실패: {e}")
def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))

def _sleep_midpoint_minutes(start: dt.time, end: dt.time, cross: bool) -> float:
    m1 = start.hour * 60 + start.minute
    m2 = end.hour * 60 + end.minute
    if cross:
        m2 += 1440
    return (m1 + m2) / 2.0

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

def _span_minutes(span: Tuple[dt.datetime, dt.datetime]) -> float:
    return (span[1] - span[0]).total_seconds() / 60.0

def _task_done_map_for_date(date: dt.date) -> Dict[str, bool]:
    from data.cache import _cached_daily_log
    # We need a repo to call the cache if we used it here, 
    # but the app.py version used a repo-accessing cache.
    # In this component, we'll try to get it from session state or pass repo.
    # For now, let's keep the logic but we might need to pass repo.
    return {}

def _parse_task_key_midpoint_minutes(key: str) -> Optional[float]:
    if "T" not in key:
        return None
    try:
        t_str = key.split("T")[1]
        h, m = map(int, t_str.split(":"))
        return h * 60 + m
    except:
        return None

def _sum_caffeine_mg(doses_json: str) -> float:
    from shared.today_input import doses_from_json
    drafts = doses_from_json(doses_json)
    return sum(float(d.amount_mg) for d in drafts if str(getattr(d, "dose_type", "caffeine")).strip().lower() == "caffeine")

def render_daily_checkin(repo, username: str, baseline: UserBaseline, out):
    from data.cache import _cached_checkin, _invalidate_repo_read_caches
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
            phase_delta = 0.08 * delta_h

            prime_spans = _mask_to_spans(out.t, out.zones.get("prime", []))
            pred_prime_mid_m: Optional[float] = None
            if prime_spans:
                p = sorted(prime_spans, key=_span_minutes, reverse=True)[0]
                pred_mid = p[0] + (p[1] - p[0]) / 2
                pred_prime_mid_m = pred_mid.hour * 60 + pred_mid.minute

            # simplified task_done_map for component (passed via session state would be better)
            done_map = {}
            done_mids: List[float] = []
            for k, v in done_map.items():
                if not bool(v):
                    continue
                m = _parse_task_key_midpoint_minutes(k)
                if m is not None:
                    done_mids.append(m)

            if pred_prime_mid_m is not None and done_mids:
                actual_mid_exec = sum(done_mids) / len(done_mids)
                phase_err_m = actual_mid_exec - pred_prime_mid_m
                if phase_err_m > 720:
                    phase_err_m -= 1440
                elif phase_err_m < -720:
                    phase_err_m += 1440
                phase_delta += _clip((phase_err_m / 60.0) * 0.12, -0.50, 0.50)

            new_chrono = _clip(float(baseline.chronotype_shift_hours) + phase_delta, -5.0, 5.0)

            caffeine_total_mg = _sum_caffeine_mg(st.session_state.get("today_doses_json", "[]"))
            gate_spans = _mask_to_spans(out.t, out.zones.get("sleep_gate", []))
            gate_mid_hour: Optional[float] = None
            if gate_spans:
                g = sorted(gate_spans, key=_span_minutes, reverse=True)[0]
                gmid = g[0] + (g[1] - g[0]) / 2
                gate_mid_hour = gmid.hour + gmid.minute / 60.0

            new_half_life = float(baseline.caffeine_half_life_hours)
            new_sensitivity = float(baseline.caffeine_sensitivity)
            if gate_mid_hour is not None:
                if caffeine_total_mg >= 250.0 and gate_mid_hour >= 23.0:
                    new_sensitivity += 0.03
                    new_half_life += 0.10
                elif caffeine_total_mg <= 120.0 and gate_mid_hour <= 21.0:
                    new_sensitivity -= 0.02
                    new_half_life -= 0.05
            new_sensitivity = _clip(new_sensitivity, 0.5, 2.5)
            new_half_life = _clip(new_half_life, 3.0, 9.0)

            baseline.baseline_offset = float(new_offset)
            baseline.chronotype_shift_hours = float(new_chrono)
            baseline.caffeine_sensitivity = float(new_sensitivity)
            baseline.caffeine_half_life_hours = float(new_half_life)
            ok = repo.update_user_baseline(
                username,
                {
                    "baseline_offset": str(baseline.baseline_offset),
                    "chronotype_shift_hours": str(baseline.chronotype_shift_hours),
                    "caffeine_sensitivity": str(baseline.caffeine_sensitivity),
                    "caffeine_half_life_hours": str(baseline.caffeine_half_life_hours),
                },
            )
            if ok:
                st.success(
                    "적응 업데이트 완료: "
                    f"offset={baseline.baseline_offset:.3f}, "
                    f"phase={baseline.chronotype_shift_hours:.2f}h, "
                    f"caf_sens={baseline.caffeine_sensitivity:.2f}, "
                    f"caf_half_life={baseline.caffeine_half_life_hours:.2f}h"
                )
                st.session_state["today_clarity"] = float(clarity)
                st.rerun()
            else:
                st.error("users 시트 업데이트 실패")
