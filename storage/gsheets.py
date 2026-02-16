# storage/gsheets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import datetime as dt
import random
import time

import gspread
from gspread.exceptions import APIError, WorksheetNotFound

from .security import hash_password, is_password_hashed, verify_password


# ----------------------------
# Config
# ----------------------------

@dataclass
class GSheetsConfig:
    spreadsheet_name: str = "NeuroFrame_DB"
    users_ws: str = "users"
    logs_ws: str = "daily_logs"
    checkins_ws: str = "checkins"
    admin_logs_ws: str = "admin_logs"

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
    #   doses_json, shift_blocks_json, day_shift_hours, workload_level, subjective_clarity


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


def _is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, APIError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in (429, 500, 502, 503, 504):
            return True
    text = str(exc).lower()
    return any(k in text for k in ("timeout", "temporarily", "rate limit", "connection reset", "503"))


def _with_retry(op: str, fn, attempts: int = 4, base_delay: float = 0.25):
    last_err: Optional[Exception] = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i >= attempts - 1 or not _is_retryable_api_error(e):
                break
            delay = base_delay * (2 ** i) + random.uniform(0.0, 0.15)
            time.sleep(delay)
    raise RuntimeError(f"GSheets operation failed: {op}: {last_err}") from last_err

def _ensure_headers(ws, headers: List[str]):
    """Ensure worksheet has required headers (append missing columns)."""
    existing = _with_retry("row_values(header)", lambda: ws.row_values(1))
    if not existing:
        _with_retry("append_row(header)", lambda: ws.append_row(headers))
        return
    updated = list(existing)
    changed = False
    for h in headers:
        if h not in updated:
            updated.append(h)
            changed = True
    if changed:
        _with_retry("update(header)", lambda: ws.update(f"A1:{_col_letter(len(updated))}1", [updated]))

def _get_all_records(ws) -> List[Dict[str, Any]]:
    # gspread get_all_records() treats first row as header
    return _with_retry("get_all_records", lambda: ws.get_all_records())

def _find_row_index_by_key(ws, key_col: str, key_value: str) -> Optional[int]:
    """
    Returns 1-based row index in sheet where key_col == key_value.
    Requires header row.
    """
    headers = _with_retry("row_values(header)", lambda: ws.row_values(1))
    if not headers:
        return None
    try:
        key_idx = headers.index(key_col) + 1
    except ValueError:
        return None

    col_vals = _with_retry("col_values", lambda: ws.col_values(key_idx))  # includes header at index 0
    for i in range(2, len(col_vals) + 1):  # start from row 2
        if str(col_vals[i - 1]).strip() == str(key_value).strip():
            return i
    return None

def _update_row_dict(ws, row_idx: int, patch: Dict[str, Any]):
    headers = _with_retry("row_values(header)", lambda: ws.row_values(1))
    if not headers:
        raise RuntimeError("Worksheet has no header row.")
    row = _with_retry("row_values(row)", lambda: ws.row_values(row_idx))
    # Pad row to header length
    if len(row) < len(headers):
        row += [""] * (len(headers) - len(row))

    header_to_pos = {h: i for i, h in enumerate(headers)}
    for k, v in patch.items():
        if k not in header_to_pos:
            continue
        row[header_to_pos[k]] = "" if v is None else str(v)

    # Update entire row in one shot
    _with_retry("update(row)", lambda: ws.update(f"A{row_idx}:{_col_letter(len(headers))}{row_idx}", [row]))

def _append_row_dict(ws, data: Dict[str, Any]):
    headers = _with_retry("row_values(header)", lambda: ws.row_values(1))
    if not headers:
        raise RuntimeError("Worksheet has no header row.")
    row = ["" for _ in headers]
    header_to_pos = {h: i for i, h in enumerate(headers)}
    for k, v in data.items():
        if k not in header_to_pos:
            continue
        row[header_to_pos[k]] = "" if v is None else str(v)
    _with_retry("append_row", lambda: ws.append_row(row))

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
        self._read_cache: Dict[str, Tuple[float, Any]] = {}
        self._default_ttl_sec = 20.0

        self.sh = _with_retry("open(spreadsheet)", lambda: self.gc.open(self.cfg.spreadsheet_name))
        self.users = self._get_or_create_ws(self.cfg.users_ws)
        self.logs = self._get_or_create_ws(self.cfg.logs_ws)
        self.checkins = self._get_or_create_ws(self.cfg.checkins_ws)
        self.admin_logs = self._get_or_create_ws(self.cfg.admin_logs_ws)

        self._init_schema()

    def _get_or_create_ws(self, title: str):
        try:
            return _with_retry(f"worksheet({title})", lambda: self.sh.worksheet(title))
        except WorksheetNotFound:
            return _with_retry(f"add_worksheet({title})", lambda: self.sh.add_worksheet(title=title, rows=1000, cols=30))

    def _cache_get(self, key: str) -> Optional[Any]:
        item = self._read_cache.get(key)
        if not item:
            return None
        exp, value = item
        if time.time() >= exp:
            self._read_cache.pop(key, None)
            return None
        return value

    def _cache_set(self, key: str, value: Any, ttl_sec: Optional[float] = None) -> Any:
        ttl = float(ttl_sec if ttl_sec is not None else self._default_ttl_sec)
        self._read_cache[key] = (time.time() + max(1.0, ttl), value)
        return value

    def _cache_invalidate(self, prefix: str) -> None:
        to_del = [k for k in self._read_cache if k.startswith(prefix)]
        for k in to_del:
            self._read_cache.pop(k, None)

    def _records_cached(self, key: str, ws, ttl_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        return self._cache_set(key, _get_all_records(ws), ttl_sec=ttl_sec)

    def _init_schema(self):
        users_headers = [
            "username", "password", "created_at", "last_login",
            "baseline_sleep_start", "baseline_wake",
            "chronotype_shift_hours",
            "caffeine_half_life_hours", "caffeine_sensitivity",
            "baseline_offset",
            "circadian_weight", "sleep_pressure_weight", "drug_weight", "load_weight",
            "is_shift_worker", "uses_adhd_medication",
            "onboarded",
        ]
        logs_headers = [
            "date", "username",
            "sleep_override_on", "sleep_cross_midnight",
            "sleep_start", "wake_time",
            "doses_json",
            "shift_blocks_json",
            "day_shift_hours",
            "workload_level",
            "subjective_clarity",
            "updated_at",
        ]
        checkins_headers = [
            "date", "username",
            "subjective_clarity",
            "focus_success",
            "actual_focus_minutes",
            "energy_satisfaction",
            "notes",
            "updated_at",
        ]
        admin_logs_headers = [
            "timestamp",
            "level",
            "action",
            "username",
            "detail",
        ]
        _ensure_headers(self.users, users_headers)
        _ensure_headers(self.logs, logs_headers)
        _ensure_headers(self.checkins, checkins_headers)
        _ensure_headers(self.admin_logs, admin_logs_headers)

    # -------- Users --------

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        rows = self._records_cached("users:all", self.users)
        key = str(username).strip()
        for r in rows:
            if str(r.get("username", "")).strip() == key:
                return r
        return None

    def create_user(self, username: str, password: str) -> bool:
        if self.get_user(username) is not None:
            return False
        data = {
            "username": username,
            "password": hash_password(password),
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
            "is_shift_worker": "false",
            "uses_adhd_medication": "false",
            "onboarded": "false",
        }
        _append_row_dict(self.users, data)
        self._cache_invalidate("users:")
        return True

    def verify_login(self, username: str, password: str) -> bool:
        row_idx = _find_row_index_by_key(self.users, "username", username)
        if row_idx is None:
            return False
        u = self.get_user(username)
        if not u:
            return False

        stored = str(u.get("password", ""))
        provided = str(password)

        if is_password_hashed(stored):
            return verify_password(provided, stored)

        # Legacy plaintext fallback: allow login once, then migrate to hash.
        if stored == provided:
            _update_row_dict(self.users, row_idx, {"password": hash_password(provided)})
            return True

        return False

    def update_last_login(self, username: str):
        row_idx = _find_row_index_by_key(self.users, "username", username)
        if row_idx is None:
            return
        _update_row_dict(self.users, row_idx, {"last_login": _now_iso()})
        self._cache_invalidate("users:")

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
        self._cache_invalidate("users:")
        return True

    def count_password_rows(self) -> Dict[str, int]:
        rows = self._records_cached("users:all", self.users)
        total = 0
        hashed = 0
        plaintext = 0
        empty = 0
        for r in rows:
            total += 1
            pw = str(r.get("password", "") or "").strip()
            if not pw:
                empty += 1
                continue
            if is_password_hashed(pw):
                hashed += 1
            else:
                plaintext += 1
        return {"total": total, "hashed": hashed, "plaintext": plaintext, "empty": empty}

    def migrate_plaintext_passwords(self) -> int:
        """
        One-time migration helper.
        Hashes any non-empty plaintext password row in users sheet.
        Returns number of migrated rows.
        """
        headers = _with_retry("row_values(users_header)", lambda: self.users.row_values(1))
        if not headers:
            raise RuntimeError("users sheet has no header row.")

        records = self._records_cached("users:all", self.users)
        migrated = 0
        for i, r in enumerate(records, start=2):
            pw = str(r.get("password", "") or "").strip()
            if not pw:
                continue
            if is_password_hashed(pw):
                continue
            _update_row_dict(self.users, i, {"password": hash_password(pw)})
            migrated += 1
        if migrated > 0:
            self._cache_invalidate("users:")
        return migrated

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
        records = self._records_cached("logs:all", self.logs)
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
        headers = _with_retry("row_values(logs_header)", lambda: self.logs.row_values(1))
        if not headers:
            raise RuntimeError("daily_logs sheet has no header row.")

        # Pull columns to match quickly
        # We'll search by reading all records and then compute row index.
        # Note: get_all_records() starts from row2, so index math:
        records = self._records_cached("logs:all", self.logs)
        found_rows: List[int] = []
        for i, r in enumerate(records, start=2):
            if str(r.get("username", "")).strip() == username and str(r.get("date", "")).strip() == date_s:
                found_rows.append(i)

        payload = {"date": date_s, "username": username, "updated_at": _now_iso()}
        payload.update(data)

        if not found_rows:
            _append_row_dict(self.logs, payload)
        else:
            _update_row_dict(self.logs, found_rows[0], payload)
            for idx in sorted(found_rows[1:], reverse=True):
                _with_retry("delete_rows(duplicate_daily_log)", lambda i=idx: self.logs.delete_rows(i))
        self._cache_invalidate("logs:")

    def get_daily_logs_for_user(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        logs = self._records_cached("logs:all", self.logs)
        out = [r for r in logs if str(r.get("username", "")).strip() == str(username).strip()]
        out.sort(key=lambda r: str(r.get("date", "")), reverse=True)
        if limit is not None:
            return out[: max(0, int(limit))]
        return out

    # -------- Daily check-ins --------

    def get_checkin(self, username: str, date: dt.date) -> Optional[Dict[str, Any]]:
        date_s = _date_iso(date)
        records = self._records_cached("checkins:all", self.checkins)
        for r in records:
            if str(r.get("username", "")).strip() == username and str(r.get("date", "")).strip() == date_s:
                return r
        return None

    def get_checkins_for_user(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = self._records_cached("checkins:all", self.checkins)
        out = [r for r in rows if str(r.get("username", "")).strip() == str(username).strip()]
        out.sort(key=lambda r: str(r.get("date", "")), reverse=True)
        if limit is not None:
            return out[: max(0, int(limit))]
        return out

    def upsert_checkin(self, username: str, date: dt.date, data: Dict[str, Any]) -> None:
        date_s = _date_iso(date)
        headers = _with_retry("row_values(checkins_header)", lambda: self.checkins.row_values(1))
        if not headers:
            raise RuntimeError("checkins sheet has no header row.")

        records = self._records_cached("checkins:all", self.checkins)
        found_rows: List[int] = []
        for i, r in enumerate(records, start=2):
            if str(r.get("username", "")).strip() == username and str(r.get("date", "")).strip() == date_s:
                found_rows.append(i)

        payload = {"date": date_s, "username": username, "updated_at": _now_iso()}
        payload.update(data)
        if not found_rows:
            _append_row_dict(self.checkins, payload)
        else:
            _update_row_dict(self.checkins, found_rows[0], payload)
            for idx in sorted(found_rows[1:], reverse=True):
                _with_retry("delete_rows(duplicate_checkin)", lambda i=idx: self.checkins.delete_rows(i))
        self._cache_invalidate("checkins:")

    def append_admin_log(self, level: str, action: str, username: str, detail: str) -> None:
        payload = {
            "timestamp": _now_iso(),
            "level": str(level or "info"),
            "action": str(action or ""),
            "username": str(username or ""),
            "detail": str(detail or ""),
        }
        try:
            _append_row_dict(self.admin_logs, payload)
            self._cache_invalidate("admin_logs:")
        except Exception:
            # Never fail main flow because audit log append failed.
            return

    def get_recent_admin_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._records_cached("admin_logs:all", self.admin_logs)
        out = list(rows)
        out.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        return out[: max(0, int(limit))]
