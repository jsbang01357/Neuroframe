# storage/gsheets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import datetime as dt

import gspread


# ----------------------------
# Config
# ----------------------------

@dataclass
class GSheetsConfig:
    spreadsheet_name: str = "NeuroFrame_DB"
    users_ws: str = "users"
    logs_ws: str = "daily_logs"

    # Required columns (MVP)
    # users:
    #   username, password, created_at, last_login,
    #   baseline_sleep_start, baseline_wake, chronotype_shift_hours,
    #   caffeine_half_life_hours, caffeine_sensitivity,
    #   baseline_offset, circadian_weight, sleep_pressure_weight, drug_weight, load_weight,
    #   onboarded
    #
    # daily_logs:
    #   date, username, sleep_override_on, sleep_cross_midnight, sleep_start, wake_time,
    #   doses_json, workload_level, subjective_clarity


# ----------------------------
# Utilities
# ----------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _date_iso(d: dt.date) -> str:
    return d.isoformat()

def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def _coerce_bool(x: Any, default: bool = False) -> bool:
    if x is None or x == "":
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return default

def _ensure_headers(ws, headers: List[str]):
    """If worksheet is empty, set header row."""
    existing = ws.row_values(1)
    if not existing:
        ws.append_row(headers)
        return
    # If existing headers differ, we won't auto-migrate here (avoid breaking data).
    # You can handle migration manually if needed.

def _get_all_records(ws) -> List[Dict[str, Any]]:
    # gspread get_all_records() treats first row as header
    return ws.get_all_records()

def _find_row_index_by_key(ws, key_col: str, key_value: str) -> Optional[int]:
    """
    Returns 1-based row index in sheet where key_col == key_value.
    Requires header row.
    """
    headers = ws.row_values(1)
    if not headers:
        return None
    try:
        key_idx = headers.index(key_col) + 1
    except ValueError:
        return None

    col_vals = ws.col_values(key_idx)  # includes header at index 0
    for i in range(2, len(col_vals) + 1):  # start from row 2
        if str(col_vals[i - 1]).strip() == str(key_value).strip():
            return i
    return None

def _update_row_dict(ws, row_idx: int, patch: Dict[str, Any]):
    headers = ws.row_values(1)
    if not headers:
        raise RuntimeError("Worksheet has no header row.")
    row = ws.row_values(row_idx)
    # Pad row to header length
    if len(row) < len(headers):
        row += [""] * (len(headers) - len(row))

    header_to_pos = {h: i for i, h in enumerate(headers)}
    for k, v in patch.items():
        if k not in header_to_pos:
            continue
        row[header_to_pos[k]] = "" if v is None else str(v)

    # Update entire row in one shot
    ws.update(f"A{row_idx}:{_col_letter(len(headers))}{row_idx}", [row])

def _append_row_dict(ws, data: Dict[str, Any]):
    headers = ws.row_values(1)
    if not headers:
        raise RuntimeError("Worksheet has no header row.")
    row = ["" for _ in headers]
    header_to_pos = {h: i for i, h in enumerate(headers)}
    for k, v in data.items():
        if k not in header_to_pos:
            continue
        row[header_to_pos[k]] = "" if v is None else str(v)
    ws.append_row(row)

def _col_letter(n: int) -> str:
    """1 -> A, 2 -> B ... 27 -> AA"""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ----------------------------
# Main client
# ----------------------------

class NeuroGSheets:
    def __init__(self, gc: gspread.Client, cfg: Optional[GSheetsConfig] = None):
        self.gc = gc
        self.cfg = cfg or GSheetsConfig()

        self.sh = self.gc.open(self.cfg.spreadsheet_name)
        self.users = self.sh.worksheet(self.cfg.users_ws)
        self.logs = self.sh.worksheet(self.cfg.logs_ws)

        self._init_schema()

    def _init_schema(self):
        users_headers = [
            "username", "password", "created_at", "last_login",
            "baseline_sleep_start", "baseline_wake",
            "chronotype_shift_hours",
            "caffeine_half_life_hours", "caffeine_sensitivity",
            "baseline_offset",
            "circadian_weight", "sleep_pressure_weight", "drug_weight", "load_weight",
            "onboarded",
        ]
        logs_headers = [
            "date", "username",
            "sleep_override_on", "sleep_cross_midnight",
            "sleep_start", "wake_time",
            "doses_json",
            "workload_level",
            "subjective_clarity",
            "updated_at",
        ]
        _ensure_headers(self.users, users_headers)
        _ensure_headers(self.logs, logs_headers)

    # -------- Users --------

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        row_idx = _find_row_index_by_key(self.users, "username", username)
        if row_idx is None:
            return None
        headers = self.users.row_values(1)
        row = self.users.row_values(row_idx)
        row += [""] * (len(headers) - len(row))
        return {headers[i]: row[i] for i in range(len(headers))}

    def create_user(self, username: str, password: str) -> bool:
        if self.get_user(username) is not None:
            return False
        data = {
            "username": username,
            "password": password,
            "created_at": _now_iso(),
            "last_login": "",
            "baseline_sleep_start": "",
            "baseline_wake": "",
            "chronotype_shift_hours": "0",
            "caffeine_half_life_hours": "5",
            "caffeine_sensitivity": "1",
            "baseline_offset": "0",
            "circadian_weight": "1",
            "sleep_pressure_weight": "1.2",
            "drug_weight": "0.004",
            "load_weight": "0.2",
            "onboarded": "false",
        }
        _append_row_dict(self.users, data)
        return True

    def verify_login(self, username: str, password: str) -> bool:
        u = self.get_user(username)
        if not u:
            return False
        return str(u.get("password", "")) == str(password)

    def update_last_login(self, username: str):
        row_idx = _find_row_index_by_key(self.users, "username", username)
        if row_idx is None:
            return
        _update_row_dict(self.users, row_idx, {"last_login": _now_iso()})

    def upsert_user_baseline(self, username: str, patch: Dict[str, Any]) -> bool:
        """
        patch keys should match users columns.
        Example:
          {"baseline_sleep_start":"23:30","baseline_wake":"07:30","caffeine_half_life_hours":5,...,"onboarded":"true"}
        """
        row_idx = _find_row_index_by_key(self.users, "username", username)
        if row_idx is None:
            return False
        _update_row_dict(self.users, row_idx, patch)
        return True

    def user_to_baseline_dict(self, user_row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sheet row -> baseline kwargs-friendly dict."""
        return {
            "baseline_sleep_start": user_row.get("baseline_sleep_start", ""),
            "baseline_wake": user_row.get("baseline_wake", ""),
            "chronotype_shift_hours": _coerce_float(user_row.get("chronotype_shift_hours", 0.0), 0.0),
            "caffeine_half_life_hours": _coerce_float(user_row.get("caffeine_half_life_hours", 5.0), 5.0),
            "caffeine_sensitivity": _coerce_float(user_row.get("caffeine_sensitivity", 1.0), 1.0),
            "baseline_offset": _coerce_float(user_row.get("baseline_offset", 0.0), 0.0),
            "circadian_weight": _coerce_float(user_row.get("circadian_weight", 1.0), 1.0),
            "sleep_pressure_weight": _coerce_float(user_row.get("sleep_pressure_weight", 1.2), 1.2),
            "drug_weight": _coerce_float(user_row.get("drug_weight", 0.004), 0.004),
            "load_weight": _coerce_float(user_row.get("load_weight", 0.2), 0.2),
            "onboarded": _coerce_bool(user_row.get("onboarded", "false"), False),
        }

    # -------- Daily logs --------

    def get_daily_log(self, username: str, date: dt.date) -> Optional[Dict[str, Any]]:
        date_s = _date_iso(date)
        records = _get_all_records(self.logs)
        for r in records:
            if str(r.get("username", "")).strip() == username and str(r.get("date", "")).strip() == date_s:
                return r
        return None

    def upsert_daily_log(self, username: str, date: dt.date, data: Dict[str, Any]) -> None:
        """
        Upsert by (date, username).
        Required keys recommended:
          sleep_override_on, sleep_cross_midnight, sleep_start, wake_time, doses_json, workload_level, subjective_clarity
        """
        date_s = _date_iso(date)
        # Find row index by scanning (small MVP data -> OK)
        headers = self.logs.row_values(1)
        if not headers:
            raise RuntimeError("daily_logs sheet has no header row.")

        # Pull columns to match quickly
        # We'll search by reading all records and then compute row index.
        # Note: get_all_records() starts from row2, so index math:
        records = _get_all_records(self.logs)
        found_row_idx = None
        for i, r in enumerate(records, start=2):
            if str(r.get("username", "")).strip() == username and str(r.get("date", "")).strip() == date_s:
                found_row_idx = i
                break

        payload = {"date": date_s, "username": username, "updated_at": _now_iso()}
        payload.update(data)

        if found_row_idx is None:
            _append_row_dict(self.logs, payload)
        else:
            _update_row_dict(self.logs, found_row_idx, payload)
