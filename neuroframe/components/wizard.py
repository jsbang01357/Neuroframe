# neuroframe/components/wizard.py
import streamlit as st
from neuroframe.engine import UserBaseline

def render_setup_wizard(repo, username: str, baseline: UserBaseline):
    st.header("Setup Wizard")
    st.caption("최초 1회 설정입니다. 나중에 관리자/설정에서 언제든 수정할 수 있습니다.")
    st.info("안내: NeuroFrame은 의료 조언/진단/치료를 제공하지 않는 일정 설계 도구입니다.")

    st.subheader("Step 0 — 프로필")
    p1, p2 = st.columns(2)
    with p1:
        is_shift_worker = st.toggle("교대근무를 하나요?", value=bool(st.session_state.get("profile_is_shift_worker", False)))
    with p2:
        uses_adhd_med = st.toggle("ADHD 약물을 사용하나요?", value=bool(st.session_state.get("profile_uses_adhd_medication", False)))
    st.session_state["profile_is_shift_worker"] = bool(is_shift_worker)
    st.session_state["profile_uses_adhd_medication"] = bool(uses_adhd_med)

    st.caption("복용/동반 약물 태그 (선택): 당일 곡선 입력이 아니라 장기 상태 태그로 취급되며, PK 직접 입력 대신 미세 보정/해석 힌트에만 사용될 수 있습니다.")
    tag_defaults = dict(
        st.session_state.get(
            "profile_medication_tags",
            {"atomoxetine": False, "ssri": False, "aripiprazole": False, "beta_blocker": False},
        )
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        tag_atomoxetine = st.checkbox("atomoxetine", value=bool(tag_defaults.get("atomoxetine", False)))
    with m2:
        tag_ssri = st.checkbox("ssri", value=bool(tag_defaults.get("ssri", False)))
    with m3:
        tag_aripiprazole = st.checkbox("aripiprazole", value=bool(tag_defaults.get("aripiprazole", False)))
    with m4:
        tag_beta_blocker = st.checkbox("beta_blocker", value=bool(tag_defaults.get("beta_blocker", False)))
    med_tags = {
        "atomoxetine": bool(tag_atomoxetine),
        "ssri": bool(tag_ssri),
        "aripiprazole": bool(tag_aripiprazole),
        "beta_blocker": bool(tag_beta_blocker),
    }
    st.session_state["profile_medication_tags"] = med_tags

    st.subheader("Step 1 — Baseline")
    c1, c2, c3 = st.columns(3)
    with c1:
        default_sleep = baseline.baseline_sleep_start if not is_shift_worker else __import__("datetime").time(0, 30)
        default_wake = baseline.baseline_wake if not is_shift_worker else __import__("datetime").time(8, 0)
        baseline_sleep_start = st.time_input("평균 취침", value=default_sleep)
        baseline_wake = st.time_input("평균 기상", value=default_wake)
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
        if uses_adhd_med:
            st.caption("약물 입력 폼에서 MPH IR/XR를 함께 설정할 수 있습니다.")
        else:
            st.caption("기본 입력은 카페인 중심으로 단순화됩니다.")

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
        import json
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
            "is_shift_worker": "true" if is_shift_worker else "false",
            "uses_adhd_medication": "true" if uses_adhd_med else "false",
            "profile_json": json.dumps(med_tags, ensure_ascii=False),
            "onboarded": "true",
        }
        ok = repo.update_user_baseline(username, patch)
        if ok:
            st.success("온보딩이 완료되었습니다.")
            st.rerun()
        else:
            st.error("저장에 실패했습니다. users 시트/권한/헤더를 확인해주세요.")
