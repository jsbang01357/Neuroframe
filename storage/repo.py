# storage/repo.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import datetime as dt
import json

from neuroframe.engine import UserBaseline


@dataclass
class RepoUser:
    username: str
    onboarded: bool
    baseline: UserBaseline
    is_shift_worker: bool = False
    uses_adhd_medication: bool = False
    medication_tags: Dict[str, bool] = field(default_factory=dict)


class NeuroRepo:
    """
    App-level interface (backend-agnostic).
    Later you can swap gsheets -> sqlite without touching Streamlit pages.
    """
    def __init__(self, gsheets_client):
        self.db = gsheets_client

    def _audit_error(self, action: str, username: str, err: Exception) -> None:
        try:
            self.db.append_admin_log("error", action, username, str(err))
        except Exception:
            return

    # ---- auth ----
    def create_user(self, username: str, password: str) -> bool:
        try:
            return self.db.create_user(username, password)
        except Exception as e:
            self._audit_error("create_user", username, e)
            raise

    def verify_login(self, username: str, password: str) -> bool:
        try:
            return self.db.verify_login(username, password)
        except Exception as e:
            self._audit_error("verify_login", username, e)
            return False

    def touch_login(self, username: str) -> None:
        try:
            self.db.update_last_login(username)
        except Exception as e:
            self._audit_error("touch_login", username, e)

    # ---- user baseline ----
    def get_user(self, username: str) -> Optional[RepoUser]:
        try:
            row = self.db.get_user(username)
        except Exception as e:
            self._audit_error("get_user", username, e)
            return None
        if not row:
            return None
        b = self._row_to_baseline(row)
        onboarded = bool(self.db.user_to_baseline_dict(row).get("onboarded", False))
        is_shift_worker = str(row.get("is_shift_worker", "false")).strip().lower() in ("1", "true", "t", "yes", "y")
        uses_adhd_medication = str(row.get("uses_adhd_medication", "false")).strip().lower() in ("1", "true", "t", "yes", "y")
        tags = self._row_to_medication_tags(row)
        return RepoUser(
            username=username,
            onboarded=onboarded,
            baseline=b,
            is_shift_worker=is_shift_worker,
            uses_adhd_medication=uses_adhd_medication,
            medication_tags=tags,
        )

    def update_user_baseline(self, username: str, patch: Dict[str, Any]) -> bool:
        # patch keys should match sheet headers
        try:
            return self.db.upsert_user_baseline(username, patch)
        except Exception as e:
            self._audit_error("update_user_baseline", username, e)
            return False

    def count_password_rows(self) -> Dict[str, int]:
        try:
            return self.db.count_password_rows()
        except Exception as e:
            self._audit_error("count_password_rows", "", e)
            return {"total": 0, "hashed": 0, "plaintext": 0, "empty": 0}

    def migrate_plaintext_passwords(self) -> int:
        try:
            return self.db.migrate_plaintext_passwords()
        except Exception as e:
            self._audit_error("migrate_plaintext_passwords", "", e)
            return 0

    # ---- daily log ----
    def get_daily_log(self, username: str, date: dt.date) -> Optional[Dict[str, Any]]:
        try:
            return self.db.get_daily_log(username, date)
        except Exception as e:
            self._audit_error("get_daily_log", username, e)
            return None

    def upsert_daily_log(self, username: str, date: dt.date, data: Dict[str, Any]) -> None:
        try:
            self.db.upsert_daily_log(username, date, data)
        except Exception as e:
            self._audit_error("upsert_daily_log", username, e)
            raise RuntimeError(f"daily log 저장 실패: {e}") from e

    def get_daily_logs_for_user(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            return self.db.get_daily_logs_for_user(username, limit=limit)
        except Exception as e:
            self._audit_error("get_daily_logs_for_user", username, e)
            return []

    # ---- daily check-in ----
    def get_checkin(self, username: str, date: dt.date) -> Optional[Dict[str, Any]]:
        try:
            return self.db.get_checkin(username, date)
        except Exception as e:
            self._audit_error("get_checkin", username, e)
            return None

    def get_checkins_for_user(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            return self.db.get_checkins_for_user(username, limit=limit)
        except Exception as e:
            self._audit_error("get_checkins_for_user", username, e)
            return []

    def upsert_checkin(self, username: str, date: dt.date, data: Dict[str, Any]) -> None:
        try:
            self.db.upsert_checkin(username, date, data)
        except Exception as e:
            self._audit_error("upsert_checkin", username, e)
            raise RuntimeError(f"checkin 저장 실패: {e}") from e

    def get_recent_admin_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return self.db.get_recent_admin_logs(limit=limit)
        except Exception:
            return []

    # ---- privacy / portability ----
    def export_user_data(self, username: str) -> Dict[str, Any]:
        try:
            return self.db.export_user_data(username)
        except Exception as e:
            self._audit_error("export_user_data", username, e)
            return {"exported_at": "", "username": username, "user": {}, "daily_logs": [], "checkins": []}

    def delete_user_data(self, username: str, anonymize: bool = False) -> Dict[str, int]:
        try:
            return self.db.delete_user_data(username, anonymize=anonymize)
        except Exception as e:
            self._audit_error("delete_user_data", username, e)
            raise RuntimeError(f"사용자 데이터 삭제 실패: {e}") from e

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

    def _row_to_medication_tags(self, row: Dict[str, Any]) -> Dict[str, bool]:
        default_tags = {
            "atomoxetine": False,
            "ssri": False,
            "aripiprazole": False,
            "beta_blocker": False,
        }
        raw = row.get("profile_json", "")
        if raw:
            try:
                obj = json.loads(str(raw))
            except Exception:
                obj = {}
            if isinstance(obj, dict):
                out = dict(default_tags)
                for k in default_tags:
                    out[k] = bool(obj.get(k, False))
                return out

        # Fallback for installations that may store these as separate columns.
        out = dict(default_tags)
        for k in default_tags:
            out[k] = str(row.get(k, "false")).strip().lower() in ("1", "true", "t", "yes", "y")
        return out
