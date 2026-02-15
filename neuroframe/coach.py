# neuroframe/coach.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import datetime as dt

from .engine import CurveOutput


@dataclass
class BlockSuggestion:
    start: dt.datetime
    end: dt.datetime
    label: str               # "Deep Work" / "Low Load" / "Wind-down"
    rationale: str           # 제안형 문장


def _mask_to_spans(t: List[dt.datetime], mask: List[bool]) -> List[Tuple[dt.datetime, dt.datetime]]:
    spans = []
    if not t or not mask or len(t) != len(mask):
        return spans

    in_span = False
    start = None

    for i in range(len(mask)):
        if mask[i] and not in_span:
            in_span = True
            start = t[i]
        elif not mask[i] and in_span:
            in_span = False
            end = t[i]
            spans.append((start, end))
            start = None

    if in_span and start is not None:
        spans.append((start, t[-1] + (t[-1] - t[-2] if len(t) >= 2 else dt.timedelta(minutes=10))))
    return spans


def _span_duration_minutes(span: Tuple[dt.datetime, dt.datetime]) -> float:
    return (span[1] - span[0]).total_seconds() / 60.0


def _top_spans_by_duration(spans: List[Tuple[dt.datetime, dt.datetime]], max_n: int = 3) -> List[Tuple[dt.datetime, dt.datetime]]:
    spans_sorted = sorted(spans, key=_span_duration_minutes, reverse=True)
    return spans_sorted[:max_n]


def _format_time_range(s: dt.datetime, e: dt.datetime) -> str:
    return f"{s.strftime('%H:%M')}–{e.strftime('%H:%M')}"


def interpret_zones(out: CurveOutput) -> List[str]:
    """
    Returns short suggestion-style insights about the day.
    Tone: "~일 수 있습니다", "~가 유리할 수 있습니다"
    """
    t = out.t
    zones = out.zones or {}

    prime_spans = _mask_to_spans(t, zones.get("prime", []))
    crash_spans = _mask_to_spans(t, zones.get("crash", []))
    gate_spans = _mask_to_spans(t, zones.get("sleep_gate", []))

    insights: List[str] = []

    if prime_spans:
        top = _top_spans_by_duration(prime_spans, max_n=2)
        ranges = ", ".join(_format_time_range(s, e) for s, e in top)
        insights.append(f"오늘은 **Prime Zone**이 {ranges}에 형성될 수 있습니다. 이 시간대에 고집중 작업을 배치하는 것이 유리할 수 있습니다.")
    else:
        insights.append("오늘은 뚜렷한 **Prime Zone**이 길게 형성되지 않을 수 있습니다. 짧은 단위(25–45분)로 작업을 쪼개는 방식이 더 유리할 수 있습니다.")

    if crash_spans:
        top = _top_spans_by_duration(crash_spans, max_n=2)
        ranges = ", ".join(_format_time_range(s, e) for s, e in top)
        insights.append(f"**Crash Zone**은 {ranges}에서 두드러질 수 있습니다. 이 구간에는 회의/행정/정리 같은 저부하 과제를 배치하는 편이 안전할 수 있습니다.")
    else:
        insights.append("오늘은 큰 **Crash Zone**이 두드러지지 않을 수 있습니다. 다만 오후 후반에는 체감 피로가 누적될 수 있어, 짧은 회복 시간을 끼워 넣는 것이 유리할 수 있습니다.")

    if gate_spans:
        top = _top_spans_by_duration(gate_spans, max_n=1)
        s, e = top[0]
        insights.append(f"**Sleep Gate**는 {_format_time_range(s, e)} 부근에서 열릴 수 있습니다. 이 시간대엔 자극을 줄이고 정리 루틴으로 전환하는 것이 도움이 될 수 있습니다.")
    else:
        insights.append("오늘은 뚜렷한 **Sleep Gate**가 약하게 나타날 수 있습니다. 취침 목표가 있다면, 취침 60–90분 전부터 스크린/카페인을 줄이는 것이 유리할 수 있습니다.")

    return insights


def design_schedule(
    out: CurveOutput,
    deep_work_target_minutes: int = 120,
    max_blocks: int = 5,
    min_block_minutes: int = 25
) -> List[BlockSuggestion]:
    """
    Suggest schedule blocks based on zones.
    - Prime -> Deep Work
    - Crash -> Low Load
    - Sleep Gate -> Wind-down
    Greedy selection of longer spans first.
    """
    t = out.t
    zones = out.zones or {}

    prime_spans = _mask_to_spans(t, zones.get("prime", []))
    crash_spans = _mask_to_spans(t, zones.get("crash", []))
    gate_spans = _mask_to_spans(t, zones.get("sleep_gate", []))

    suggestions: List[BlockSuggestion] = []

    # 1) Deep Work blocks from Prime Zone
    remaining = deep_work_target_minutes
    for (s, e) in _top_spans_by_duration(prime_spans, max_n=3):
        if remaining <= 0:
            break
        dur = _span_duration_minutes((s, e))
        if dur < min_block_minutes:
            continue

        block_minutes = min(dur, remaining)
        block_end = s + dt.timedelta(minutes=block_minutes)

        suggestions.append(
            BlockSuggestion(
                start=s,
                end=block_end,
                label="Deep Work",
                rationale="Prime Zone일 가능성이 있어, 고집중 과제를 배치하는 편이 유리할 수 있습니다."
            )
        )
        remaining -= int(block_minutes)

    # 2) Low Load blocks from Crash Zone
    for (s, e) in _top_spans_by_duration(crash_spans, max_n=2):
        dur = _span_duration_minutes((s, e))
        if dur < min_block_minutes:
            continue
        suggestions.append(
            BlockSuggestion(
                start=s,
                end=e,
                label="Low Load",
                rationale="Crash Zone일 수 있어, 행정/정리/루틴 작업처럼 부담이 낮은 과제를 권합니다."
            )
        )

    # 3) Wind-down around Sleep Gate (take first gate span and suggest a 45–90 min wind-down)
    if gate_spans:
        s, e = _top_spans_by_duration(gate_spans, max_n=1)[0]
        wind = min(_span_duration_minutes((s, e)), 90.0)
        wind = max(45.0, wind)  # ensure meaningful suggestion
        suggestions.append(
            BlockSuggestion(
                start=s,
                end=s + dt.timedelta(minutes=wind),
                label="Wind-down",
                rationale="Sleep Gate가 열릴 수 있어, 자극을 낮추고 수면 루틴으로 전환하는 것이 도움이 될 수 있습니다."
            )
        )

    # Sort and cap
    suggestions = sorted(suggestions, key=lambda b: b.start)[:max_blocks]
    return suggestions
