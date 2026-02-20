from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Seoul")


def _parse_time_hhmm(s: str, fallback: dt.time) -> dt.time:
    try:
        hh, mm = str(s).strip().split(":")
        return dt.time(int(hh), int(mm))
    except Exception:
        return fallback


def _shift_template_blocks(template: str) -> List[Dict[str, str]]:
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


def _parse_shift_blocks_json(s: str) -> List[Dict[str, str]]:
    if not s:
        return []
    try:
        arr = json.loads(s)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    if not isinstance(arr, list):
        return out
    for item in arr:
        if not isinstance(item, dict):
            continue
        stt = str(item.get("start", "")).strip()
        end = str(item.get("end", "")).strip()
        typ = str(item.get("type", "shift")).strip() or "shift"
        if len(stt) == 5 and len(end) == 5 and ":" in stt and ":" in end:
            out.append({"start": stt, "end": end, "type": typ})
    return out


def _shift_blocks_to_json(blocks: List[Dict[str, str]]) -> str:
    return json.dumps(blocks, ensure_ascii=False)


def _shift_blocks_to_spans(date: dt.date, blocks: List[Dict[str, str]]) -> List[Tuple[dt.datetime, dt.datetime, str]]:
    spans: List[Tuple[dt.datetime, dt.datetime, str]] = []
    for b in blocks:
        stt = _parse_time_hhmm(b.get("start", "08:00"), dt.time(8, 0))
        end = _parse_time_hhmm(b.get("end", "18:00"), dt.time(18, 0))
        typ = str(b.get("type", "shift"))
        start_dt = dt.datetime(date.year, date.month, date.day, stt.hour, stt.minute, tzinfo=TZ)
        end_dt = dt.datetime(date.year, date.month, date.day, end.hour, end.minute, tzinfo=TZ)
        if end_dt <= start_dt:
            end_dt += dt.timedelta(days=1)
        spans.append((start_dt, end_dt, typ))
    return spans


def _overlap_minutes(a: Tuple[dt.datetime, dt.datetime], b: Tuple[dt.datetime, dt.datetime]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e <= s:
        return 0.0
    return (e - s).total_seconds() / 60.0
