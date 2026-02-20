# shared/today_input.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import datetime as dt
import json


@dataclass
class DoseDraft:
    hour: int
    minute: int
    amount_mg: float
    dose_type: str = "caffeine"


def _normalize_dose_type(v: Any) -> str:
    raw = str(v or "caffeine").strip().lower()
    if raw in ("mph_ir", "stimulant_ir"):
        return "mph_ir"
    if raw in ("mph_xr", "stimulant_xr"):
        return "mph_xr"
    return "caffeine"


def doses_to_json(date: dt.date, tz, drafts: List[DoseDraft]) -> str:
    """
    Store minimal dose info as JSON (local time).
    Example:
      [{"hh":9,"mm":0,"mg":150.0,"type":"caffeine"}, ...]
    """
    data = []
    for d in drafts:
        if d.amount_mg <= 0:
            continue
        data.append(
            {
                "hh": int(d.hour),
                "mm": int(d.minute),
                "mg": float(d.amount_mg),
                "type": _normalize_dose_type(getattr(d, "dose_type", "caffeine")),
            }
        )
    return json.dumps(data, ensure_ascii=False)


def doses_from_json(s: Optional[str]) -> List[DoseDraft]:
    if not s:
        return []
    try:
        arr = json.loads(s)
        if not isinstance(arr, list):
            return []
        out: List[DoseDraft] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            # Legacy compatibility:
            # - old key names: hour/minute/amount_mg/dose_type
            # - missing type -> caffeine fallback
            hh = item.get("hh", item.get("hour", 0))
            mm = item.get("mm", item.get("minute", 0))
            mg = item.get("mg", item.get("amount_mg", 0.0))
            typ = item.get("type", item.get("dose_type", "caffeine"))
            out.append(DoseDraft(
                hour=int(hh),
                minute=int(mm),
                amount_mg=float(mg),
                dose_type=_normalize_dose_type(typ),
            ))
        return out
    except Exception:
        return []


def drafts_to_engine_doses(date: dt.date, tz, drafts: List[DoseDraft]):
    """
    Convert DoseDraft -> engine.Dose list with tz-aware datetimes.
    Import engine lazily to avoid circular imports.
    """
    from neuroframe.engine import Dose
    out = []
    for d in drafts:
        if d.amount_mg <= 0:
            continue
        ts = dt.datetime(date.year, date.month, date.day, d.hour, d.minute, tzinfo=tz)
        out.append(Dose(time=ts, amount_mg=float(d.amount_mg), dose_type=_normalize_dose_type(d.dose_type)))
    return out


def parse_sleep_override(
    date: dt.date,
    tz,
    sleep_start: dt.time,
    wake_time: dt.time,
    crosses_midnight: bool
) -> Tuple[dt.datetime, dt.datetime]:
    """
    returns (sleep_dt, wake_dt) tz-aware.
    If crosses_midnight=True: sleep starts previous day, wake on date.
    """
    wake_dt = dt.datetime(date.year, date.month, date.day, wake_time.hour, wake_time.minute, tzinfo=tz)
    sleep_dt = dt.datetime(date.year, date.month, date.day, sleep_start.hour, sleep_start.minute, tzinfo=tz)
    if crosses_midnight:
        sleep_dt = sleep_dt - dt.timedelta(days=1)
    else:
        # sleep and wake within same calendar day (e.g., nap-style) — not typical, but allow
        if sleep_dt > wake_dt:
            sleep_dt = sleep_dt - dt.timedelta(days=1)
    return sleep_dt, wake_dt
