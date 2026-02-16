# neuroframe/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional, Tuple
import math
import datetime as dt


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Dose:
    """Single dose event."""
    time: dt.datetime
    amount_mg: float
    dose_type: str = "caffeine"  # caffeine | mph_ir | mph_xr


@dataclass
class UserBaseline:
    """
    MVP baseline parameters.
    - baseline_sleep_start / baseline_wake are "typical" times used only to set phase.
    - chronotype_shift_hours: negative = earlier type, positive = later type.
    """
    baseline_sleep_start: dt.time
    baseline_wake: dt.time
    chronotype_shift_hours: float = 0.0

    caffeine_half_life_hours: float = 5.0
    caffeine_sensitivity: float = 1.0  # multiplier on caffeine effect

    baseline_offset: float = 0.0  # global bias
    circadian_weight: float = 1.0
    sleep_pressure_weight: float = 1.0
    drug_weight: float = 1.0
    load_weight: float = 1.0


@dataclass
class DayInputs:
    """
    Daily overrides and knobs.
    - sleep_override: (sleep_start_dt, wake_dt) if user slept off-schedule.
    - workload_level: 0..3 (or 0..10). MVP treats it as constant load for the day.
    """
    date: dt.date
    timezone: dt.tzinfo

    sleep_override: Optional[Tuple[dt.datetime, dt.datetime]] = None
    doses: Optional[List[Dose]] = None
    workload_level: float = 0.0  # constant daily load
    shift_blocks: Optional[List[Tuple[dt.datetime, dt.datetime, str]]] = None


@dataclass
class CurveOutput:
    t: List[dt.datetime]
    net: List[float]                 # normalized 0..1
    raw: List[float]                 # raw (pre-normalization)
    circadian: List[float]
    sleep_pressure: List[float]
    drug: List[float]
    load: List[float]
    zones: Dict[str, List[bool]]     # prime/crash/sleep_gate masks
    meta: Dict[str, Any]             # thresholds, params summary, etc.


# ----------------------------
# Helpers
# ----------------------------

def _datetime_on_date(d: dt.date, t: dt.time, tz: dt.tzinfo) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, t.hour, t.minute, t.second, tzinfo=tz)

def _linspace_datetimes(start: dt.datetime, end: dt.datetime, step_minutes: int) -> List[dt.datetime]:
    out = []
    cur = start
    step = dt.timedelta(minutes=step_minutes)
    while cur < end:
        out.append(cur)
        cur += step
    return out

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _normalize_minmax(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        return [0.5 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

def _quantile(xs: List[float], q: float) -> float:
    """Simple quantile; q in [0,1]."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * _clip(q, 0.0, 1.0)))
    return ys[idx]


# ----------------------------
# Core components
# ----------------------------

def circadian_signal(t: dt.datetime, wake_dt: dt.datetime, chronotype_shift_hours: float = 0.0) -> float:
    """
    Simple 24h sine wave.
    Peak is set ~10 hours after wake as a rough heuristic, shifted by chronotype.
    Returns roughly in [-1, +1].
    """
    peak_dt = wake_dt + dt.timedelta(hours=10 + chronotype_shift_hours)
    # Convert t relative to peak in hours
    delta_h = (t - peak_dt).total_seconds() / 3600.0
    # sine with period 24h; peak at delta=0 => sin(pi/2)=1
    angle = 2 * math.pi * (delta_h / 24.0) + (math.pi / 2)
    return math.sin(angle)

def sleep_pressure_signal(t: dt.datetime, wake_dt: dt.datetime, sleep_dt: dt.datetime, H: float = 16.0) -> float:
    """
    MVP: pressure increases linearly while awake.
    - 0 at wake
    - ~1 after H hours awake
    During sleep window, returns low value (0), since 'awake_hours' is not accumulating.
    """
    if sleep_dt <= t < wake_dt:
        # in the sleep interval (if your sleep window crosses midnight, you'll supply correct dt)
        return 0.0
    awake_h = (t - wake_dt).total_seconds() / 3600.0
    if awake_h < 0:
        # before wake time (early morning) treat as still low pressure
        return 0.0
    return _clip(awake_h / H, 0.0, 1.2)  # allow slight >1 for late nights

def _dose_type(d: Dose) -> str:
    raw = str(getattr(d, "dose_type", "caffeine") or "caffeine").strip().lower()
    if raw == "stimulant_ir":
        return "mph_ir"
    if raw == "stimulant_xr":
        return "mph_xr"
    if raw in ("caffeine", "mph_ir", "mph_xr"):
        return raw
    return "caffeine"


def _caffeine_effect_single(
    t: dt.datetime,
    d: Dose,
    half_life_hours: float,
    sensitivity: float = 1.0,
    onset_minutes: int = 20,
) -> float:
    start = d.time + dt.timedelta(minutes=onset_minutes)
    if t < start:
        return 0.0
    lam = math.log(2) / max(half_life_hours, 1e-6)
    dh = (t - start).total_seconds() / 3600.0
    return sensitivity * d.amount_mg * math.exp(-lam * dh)


def _mph_ir_effect_single(t: dt.datetime, d: Dose) -> float:
    # Quick rise + short decay.
    start = d.time + dt.timedelta(minutes=25)
    if t < start:
        return 0.0
    dh = (t - start).total_seconds() / 3600.0
    rise = 1.0 - math.exp(-dh / 0.6)
    decay = math.exp(-dh / 3.2)
    return d.amount_mg * rise * decay * 3.0


def _mph_xr_effect_single(t: dt.datetime, d: Dose) -> float:
    # Two-phase release heuristic:
    # 30% immediate + 70% delayed (~4h).
    d_immediate = Dose(time=d.time, amount_mg=d.amount_mg * 0.30, dose_type="mph_ir")
    d_delayed = Dose(time=d.time + dt.timedelta(hours=4), amount_mg=d.amount_mg * 0.70, dose_type="mph_ir")
    return _mph_ir_effect_single(t, d_immediate) + _mph_ir_effect_single(t, d_delayed)


DrugEffectFn = Callable[[dt.datetime, Dose, UserBaseline], float]
ReboundWindowFn = Callable[[Dose], Optional[Tuple[dt.datetime, dt.datetime]]]


def _caffeine_effect_model(t: dt.datetime, d: Dose, baseline: UserBaseline) -> float:
    return _caffeine_effect_single(
        t,
        d,
        half_life_hours=baseline.caffeine_half_life_hours,
        sensitivity=baseline.caffeine_sensitivity,
    )


def _mph_ir_effect_model(t: dt.datetime, d: Dose, baseline: UserBaseline) -> float:
    _ = baseline
    return _mph_ir_effect_single(t, d)


def _mph_xr_effect_model(t: dt.datetime, d: Dose, baseline: UserBaseline) -> float:
    _ = baseline
    return _mph_xr_effect_single(t, d)


def _no_rebound_window(d: Dose) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    _ = d
    return None


def _mph_ir_rebound_window(d: Dose) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    return (d.time + dt.timedelta(hours=4.0), d.time + dt.timedelta(hours=7.0))


def _mph_xr_rebound_window(d: Dose) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    return (d.time + dt.timedelta(hours=8.0), d.time + dt.timedelta(hours=12.0))


DRUG_MODEL_REGISTRY: Dict[str, Tuple[DrugEffectFn, ReboundWindowFn]] = {
    "caffeine": (_caffeine_effect_model, _no_rebound_window),
    "mph_ir": (_mph_ir_effect_model, _mph_ir_rebound_window),
    "mph_xr": (_mph_xr_effect_model, _mph_xr_rebound_window),
}


def _dose_effect_at_time(t: dt.datetime, d: Dose, baseline: UserBaseline) -> float:
    typ = _dose_type(d)
    effect_fn = DRUG_MODEL_REGISTRY.get(typ, DRUG_MODEL_REGISTRY["caffeine"])[0]
    return effect_fn(t, d, baseline)


def _total_drug_effect_at_time(t: dt.datetime, doses: List[Dose], baseline: UserBaseline) -> float:
    if not doses:
        return 0.0
    return sum(_dose_effect_at_time(t, d, baseline) for d in doses)


def stimulant_effect(
    t: dt.datetime,
    doses: List[Dose],
    caffeine_half_life_hours: float,
    caffeine_sensitivity: float = 1.0,
) -> float:
    """
    Backward-compatible wrapper. Internal path uses the drug registry.
    """
    baseline = UserBaseline(
        baseline_sleep_start=dt.time(23, 30),
        baseline_wake=dt.time(7, 30),
        caffeine_half_life_hours=caffeine_half_life_hours,
        caffeine_sensitivity=caffeine_sensitivity,
    )
    return _total_drug_effect_at_time(t, doses, baseline)


def _rebound_windows(doses: List[Dose]) -> List[Tuple[dt.datetime, dt.datetime]]:
    windows: List[Tuple[dt.datetime, dt.datetime]] = []
    for d in doses:
        typ = _dose_type(d)
        rebound_fn = DRUG_MODEL_REGISTRY.get(typ, DRUG_MODEL_REGISTRY["caffeine"])[1]
        window = rebound_fn(d)
        if window is not None:
            windows.append(window)
    return windows


def _rebound_mask(t_grid: List[dt.datetime], doses: List[Dose]) -> List[bool]:
    windows = _rebound_windows(doses)
    if not windows:
        return [False for _ in t_grid]
    mask: List[bool] = []
    for t in t_grid:
        mask.append(any(s <= t < e for (s, e) in windows))
    return mask

def _shift_load_boost(shift_type: str) -> float:
    typ = str(shift_type or "shift").strip().lower()
    if typ == "day":
        return 0.60
    if typ == "evening":
        return 0.70
    if typ == "night":
        return 1.00
    if typ == "call24":
        return 1.20
    return 0.70


def _post_shift_boost(shift_type: str) -> float:
    typ = str(shift_type or "shift").strip().lower()
    if typ in ("night", "call24"):
        return 0.40
    return 0.25


def load_signal_with_shifts(
    t: dt.datetime,
    workload_level: float,
    shift_blocks: Optional[List[Tuple[dt.datetime, dt.datetime, str]]] = None,
) -> float:
    """
    Shift-aware load signal:
    - no shift blocks: slightly lower than base (off-day profile)
    - pre-shift (2h): prep load bump
    - in-shift: elevated load by shift type
    - post-shift (2h): residual fatigue bump
    """
    base = max(0.0, workload_level)
    blocks = shift_blocks or []
    if not blocks:
        return _clip(base * 0.90, 0.0, 4.0)

    in_shift_boost = 0.0
    prep_boost = 0.0
    recovery_boost = 0.0
    for s, e, typ in blocks:
        if s <= t < e:
            in_shift_boost = max(in_shift_boost, _shift_load_boost(typ))
        if (s - dt.timedelta(hours=2)) <= t < s:
            prep_boost = max(prep_boost, 0.20)
        if e <= t < (e + dt.timedelta(hours=2)):
            recovery_boost = max(recovery_boost, _post_shift_boost(typ))

    load = base * 0.85
    if in_shift_boost > 0:
        load = base + in_shift_boost
    load += prep_boost + recovery_boost
    return _clip(load, 0.0, 4.0)


# ----------------------------
# Public API
# ----------------------------

def predict_day(
    baseline: UserBaseline,
    day: DayInputs,
    step_minutes: int = 10,
    sleep_pressure_H: float = 16.0
) -> CurveOutput:
    """
    Generates 24h curve for the given date in day.timezone.
    Returns raw components + normalized net + zones.
    """

    tz = day.timezone
    start = dt.datetime(day.date.year, day.date.month, day.date.day, 0, 0, tzinfo=tz)
    end = start + dt.timedelta(days=1)
    t_grid = _linspace_datetimes(start, end, step_minutes)

    # Determine wake/sleep anchors for this date
    if day.sleep_override is not None:
        sleep_dt, wake_dt = day.sleep_override
        # Ensure tz-aware
        if sleep_dt.tzinfo is None:
            sleep_dt = sleep_dt.replace(tzinfo=tz)
        if wake_dt.tzinfo is None:
            wake_dt = wake_dt.replace(tzinfo=tz)
    else:
        # Baseline times: interpret as typical for that day
        wake_dt = _datetime_on_date(day.date, baseline.baseline_wake, tz)
        sleep_dt = _datetime_on_date(day.date, baseline.baseline_sleep_start, tz)
        # If baseline sleep is "before wake" by clock (e.g., 23:30) it belongs to previous day.
        if sleep_dt > wake_dt:
            sleep_dt = sleep_dt - dt.timedelta(days=1)

    doses = day.doses or []

    circ = []
    sp = []
    drug = []
    load = []
    raw = []

    shift_blocks = day.shift_blocks or []

    for t in t_grid:
        c = circadian_signal(t, wake_dt, baseline.chronotype_shift_hours)
        s = sleep_pressure_signal(t, wake_dt, sleep_dt, H=sleep_pressure_H)
        d = _total_drug_effect_at_time(t, doses=doses, baseline=baseline)
        l = load_signal_with_shifts(t, day.workload_level, shift_blocks=shift_blocks)

        # Weighted raw net
        net_raw = (
            baseline.circadian_weight * c
            - baseline.sleep_pressure_weight * s
            + baseline.drug_weight * d
            - baseline.load_weight * l
            + baseline.baseline_offset
        )

        circ.append(c)
        sp.append(s)
        drug.append(d)
        load.append(l)
        raw.append(net_raw)

    # Normalize net to 0..1 for UI/zone logic
    net = _normalize_minmax(raw)

    # Zones (MVP thresholds: quantiles)
    prime_th = _quantile(net, 0.80)
    crash_th = _quantile(net, 0.20)

    prime = [x >= prime_th for x in net]
    crash = [x <= crash_th for x in net]

    rebound_candidate = _rebound_mask(t_grid, doses)
    crash_with_rebound = [crash[i] or rebound_candidate[i] for i in range(len(crash))]

    # Sleep gate: net low-ish AND sleep pressure high-ish (heuristic)
    # Using normalized net + raw sleep pressure.
    sleep_gate = [(net[i] <= 0.35 and sp[i] >= 0.70) for i in range(len(net))]

    zones = {
        "prime": prime,
        "crash": crash_with_rebound,
        "sleep_gate": sleep_gate,
        "rebound_candidate": rebound_candidate,
    }

    dose_types = [_dose_type(d) for d in doses]
    stimulant_types = [t for t in dose_types if t in ("mph_ir", "mph_xr", "stimulant_ir", "stimulant_xr")]
    rebound_windows = _rebound_windows(doses)

    meta = {
        "step_minutes": step_minutes,
        "wake_dt": wake_dt.isoformat(),
        "sleep_dt": sleep_dt.isoformat(),
        "prime_threshold": prime_th,
        "crash_threshold": crash_th,
        "sleep_gate_net_le": 0.35,
        "sleep_gate_sp_ge": 0.70,
        "sleep_pressure_H": sleep_pressure_H,
        "weights": {
            "circadian_weight": baseline.circadian_weight,
            "sleep_pressure_weight": baseline.sleep_pressure_weight,
            "drug_weight": baseline.drug_weight,
            "load_weight": baseline.load_weight,
        },
        "dose_summary": {
            "total_doses": len(doses),
            "caffeine_doses": sum(1 for t in dose_types if t == "caffeine"),
            "stimulant_doses": len(stimulant_types),
            "stimulant_types": sorted(set(stimulant_types)),
        },
        "rebound_windows": [(s.isoformat(), e.isoformat()) for (s, e) in rebound_windows],
        "shift_blocks_count": len(shift_blocks),
    }

    return CurveOutput(
        t=t_grid,
        net=net,
        raw=raw,
        circadian=circ,
        sleep_pressure=sp,
        drug=drug,
        load=load,
        zones=zones,
        meta=meta,
    )


def calibrate_baseline_offset(
    current_offset: float,
    subjective_clarity_0_10: float,
    predicted_net_mean_0_1: float,
    eta: float = 0.10,
    k: float = 0.50
) -> float:
    """
    Simple Level-1 adaptation.
    - subjective_clarity_0_10: user rating
    - predicted_net_mean_0_1: mean(net) for the day (0..1)
    Idea: if user felt better than predicted, offset slightly increases.
    """
    # map clarity 0..10 -> 0..1
    clarity = _clip(subjective_clarity_0_10 / 10.0, 0.0, 1.0)
    err = clarity - _clip(predicted_net_mean_0_1, 0.0, 1.0)
    return (1 - eta) * current_offset + eta * (current_offset + k * err)
