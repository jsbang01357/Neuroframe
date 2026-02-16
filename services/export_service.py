from __future__ import annotations

import datetime as dt
import io
import uuid
from pathlib import Path
from typing import Any


def ics_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def blocks_to_ics_bytes(blocks) -> bytes:
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//NeuroFrame//Schedule Export//EN",
        "CALSCALE:GREGORIAN",
    ]
    for b in blocks:
        start_utc = b.start.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end_utc = b.end.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        uid = f"{uuid.uuid4()}@neuroframe"
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{now_utc}",
                f"DTSTART:{start_utc}",
                f"DTEND:{end_utc}",
                f"SUMMARY:{ics_escape('NeuroFrame ' + b.label)}",
                f"DESCRIPTION:{ics_escape(b.rationale)}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def weekly_report_pdf_bytes(
    end_date: dt.date,
    avg_prime_start_text: str,
    avg_crash_len: float,
    sleep_48h: float,
    sleep_7d_avg: float,
    sleep_debt_7d: float,
    caffeine_total_mg: float,
    recommendation: str,
) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties

    buf = io.BytesIO()
    font_path = Path(__file__).resolve().parent.parent / "NanumGothic.ttf"
    font_prop = FontProperties(fname=str(font_path)) if font_path.exists() else None

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            "NeuroFrame Weekly Report",
            f"기준일: {end_date.isoformat()}",
            "",
            f"Prime 평균 시작: {avg_prime_start_text}",
            f"Crash 평균 길이: {avg_crash_len:.0f}분",
            f"최근48h 수면: {sleep_48h:.1f}h",
            f"7일 평균 수면: {sleep_7d_avg:.1f}h",
            f"수면부채(7d): {sleep_debt_7d:.1f}h",
            f"카페인 총량(7d): {caffeine_total_mg:.0f}mg",
            "",
            "추천",
            recommendation,
        ]
        y = 0.95
        for line in lines:
            if font_prop is not None:
                ax.text(0.05, y, line, fontsize=11, va="top", fontproperties=font_prop)
            else:
                ax.text(0.05, y, line, fontsize=11, va="top", family="DejaVu Sans")
            y -= 0.05
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buf.getvalue()
