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


def doses_to_json(date: dt.date, tz, drafts: List[DoseDraft]) -> str:
    """
    Store minimal dose info as JSON (local time).
    Example:
      [{"hh":9,"mm":0,"mg":150.0}, ...]
    """
    data = [
        {"hh": d.hour, "mm": d.minute, "mg": float(d.amount_mg), "type": str(d.dose_type or "caffeine")}
        for d in drafts
        if d.amount_mg > 0
    ]
    return json.dumps(data, ensure_ascii=False)


def doses_from_json(s: Optional[str]) -> List[DoseDraft]:
    if not s:
        return []
    try:
        arr = json.loads(s)
        out: List[DoseDraft] = []
        for item in arr:
            out.append(DoseDraft(
                hour=int(item.get("hh", 0)),
                minute=int(item.get("mm", 0)),
                amount_mg=float(item.get("mg", 0.0)),
                dose_type=str(item.get("type", "caffeine") or "caffeine"),
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
        out.append(Dose(time=ts, amount_mg=float(d.amount_mg), dose_type=str(d.dose_type or "caffeine")))
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
