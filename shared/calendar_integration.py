from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import datetime as dt
from urllib.parse import quote

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials


@dataclass
class CalendarEvent:
    start: dt.datetime
    end: dt.datetime
    summary: str


def _parse_google_event_time(value: Dict[str, Any], tz: dt.tzinfo) -> Optional[dt.datetime]:
    if not isinstance(value, dict):
        return None
    if value.get("dateTime"):
        try:
            parsed = dt.datetime.fromisoformat(str(value["dateTime"]).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=tz)
            return parsed.astimezone(tz)
        except Exception:
            return None
    if value.get("date"):
        try:
            d = dt.date.fromisoformat(str(value["date"]))
            return dt.datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
        except Exception:
            return None
    return None


def fetch_google_calendar_events_for_day(
    service_account_info: Dict[str, Any],
    calendar_id: str,
    day: dt.date,
    tz: dt.tzinfo,
    delegated_user: str = "",
) -> List[CalendarEvent]:
    """
    Reads events from Google Calendar API for one day.
    Requires service account credentials that can access the target calendar.
    """
    if not service_account_info or not calendar_id:
        return []

    scope = ["https://www.googleapis.com/auth/calendar.readonly"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    if delegated_user:
        creds = creds.with_subject(delegated_user)

    start = dt.datetime(day.year, day.month, day.day, 0, 0, tzinfo=tz)
    end = start + dt.timedelta(days=1)
    params = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": "100",
    }

    session = AuthorizedSession(creds)
    calendar_encoded = quote(calendar_id, safe="")
    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_encoded}/events"
    resp = session.get(url, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json() if resp.content else {}
    items = payload.get("items", []) if isinstance(payload, dict) else []

    out: List[CalendarEvent] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        s = _parse_google_event_time(item.get("start", {}), tz)
        e = _parse_google_event_time(item.get("end", {}), tz)
        if s is None or e is None or e <= s:
            continue
        out.append(CalendarEvent(start=s, end=e, summary=str(item.get("summary", "(제목 없음)") or "(제목 없음)")))
    return out
