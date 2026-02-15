# storage/repo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import datetime as dt

from neuroframe.engine import UserBaseline


@dataclass
class RepoUser:
    username: str
    onboarded: bool
    baseline: UserBaseline


class NeuroRepo:
    """
    App-level interface (backend-agnostic).
    Later you can swap gsheets -> sqlite without touching Streamlit pages.
    """
    def __init__(self, gsheets_client):
        self.db = gsheets_client

    # ---- auth ----
    def create_user(self, username: str, password: str) -> bool:
        return self.db.create_user(username, password)

    def verify_login(self, username: str, password: str) -> bool:
        return self.db.verify_login(username, password)

    def touch_login(self, username: str) -> None:
        self.db.update_last_login(username)

    # ---- user baseline ----
    def get_user(self, username: str) -> Optional[RepoUser]:
        row = self.db.get_user(username)
        if not row:
            return None
        b = self._row_to_baseline(row)
        onboarded = bool(self.db.user_to_baseline_dict(row).get("onboarded", False))
        return RepoUser(username=username, onboarded=onboarded, baseline=b)

    def update_user_baseline(self, username: str, patch: Dict[str, Any]) -> bool:
        # patch keys should match sheet headers
        return self.db.upsert_user_baseline(username, patch)

    # ---- daily log ----
    def get_daily_log(self, username: str, date: dt.date) -> Optional[Dict[str, Any]]:
        return self.db.get_daily_log(username, date)

    def upsert_daily_log(self, username: str, date: dt.date, data: Dict[str, Any]) -> None:
        self.db.upsert_daily_log(username, date, data)

    def get_daily_logs_for_user(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.db.get_daily_logs_for_user(username, limit=limit)

    # ---- daily check-in ----
    def get_checkin(self, username: str, date: dt.date) -> Optional[Dict[str, Any]]:
        return self.db.get_checkin(username, date)

    def get_checkins_for_user(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.db.get_checkins_for_user(username, limit=limit)

    def upsert_checkin(self, username: str, date: dt.date, data: Dict[str, Any]) -> None:
        self.db.upsert_checkin(username, date, data)

    # ---- internal ----
    def _row_to_baseline(self, row: Dict[str, Any]) -> UserBaseline:
        d = self.db.user_to_baseline_dict(row)

        # times are stored as "HH:MM" strings in sheet
        def parse_time(s: str, fallback_hm: str) -> dt.time:
            s = (s or fallback_hm).strip()
            hh, mm = s.split(":")
            return dt.time(int(hh), int(mm))

        return UserBaseline(
            baseline_sleep_start=parse_time(d.get("baseline_sleep_start", ""), "23:30"),
            baseline_wake=parse_time(d.get("baseline_wake", ""), "07:30"),
            chronotype_shift_hours=float(d.get("chronotype_shift_hours", 0.0)),
            caffeine_half_life_hours=float(d.get("caffeine_half_life_hours", 5.0)),
            caffeine_sensitivity=float(d.get("caffeine_sensitivity", 1.0)),
            baseline_offset=float(d.get("baseline_offset", 0.0)),
            circadian_weight=float(d.get("circadian_weight", 1.0)),
            sleep_pressure_weight=float(d.get("sleep_pressure_weight", 1.2)),
            drug_weight=float(d.get("drug_weight", 0.004)),
            load_weight=float(d.get("load_weight", 0.2)),
        )
