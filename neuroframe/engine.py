# neuroframe/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import datetime as dt


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Dose:
    """Single dose event."""
    time: dt.datetime
    amount_mg: float  # e.g., caffeine mg


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

def caffeine_effect(
    t: dt.datetime,
    doses: List[Dose],
    half_life_hours: float,
    sensitivity: float = 1.0,
    onset_minutes: int = 20
) -> float:
    """
    Caffeine: exponential decay from dose time (with simple onset delay).
    Returns in arbitrary units; later normalized by weights.
    """
    if not doses:
        return 0.0
    lam = math.log(2) / max(half_life_hours, 1e-6)
    total = 0.0
    for d in doses:
        start = d.time + dt.timedelta(minutes=onset_minutes)
        if t < start:
            continue
        dh = (t - start).total_seconds() / 3600.0
        total += d.amount_mg * math.exp(-lam * dh)
    return sensitivity * total

def load_signal_constant(workload_level: float) -> float:
    """
    Constant daily load. Keep it dead simple for MVP.
    """
    return max(0.0, workload_level)


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

    # Precompute constant load for speed
    Lc = load_signal_constant(day.workload_level)

    for t in t_grid:
        c = circadian_signal(t, wake_dt, baseline.chronotype_shift_hours)
        s = sleep_pressure_signal(t, wake_dt, sleep_dt, H=sleep_pressure_H)
        d = caffeine_effect(
            t,
            doses=doses,
            half_life_hours=baseline.caffeine_half_life_hours,
            sensitivity=baseline.caffeine_sensitivity,
        )
        l = Lc

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

    # Sleep gate: net low-ish AND sleep pressure high-ish (heuristic)
    # Using normalized net + raw sleep pressure.
    sleep_gate = [(net[i] <= 0.35 and sp[i] >= 0.70) for i in range(len(net))]

    zones = {"prime": prime, "crash": crash, "sleep_gate": sleep_gate}

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
