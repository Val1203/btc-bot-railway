import json
import time
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound, APIError

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

TRADES_SHEET = "trades"
DAILY_SHEET = "daily_summary"
WEEKLY_SHEET = "weekly_summary"

RETRY_STATUS = ("429", "500", "503")  # rate limit / internal error / service unavailable


def get_gsheet_client(service_json_str: str):
    """Retourne un client gspread authentifié à partir du JSON du compte de service."""
    info = json.loads(service_json_str)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def _retryable(callable_fn, *args, retries=5, base_delay=1.0, **kwargs):
    """Exécute callable_fn avec retries exponentiels sur erreurs API Google."""
    delay = base_delay
    for _ in range(retries):
        try:
            return callable_fn(*args, **kwargs)
        except APIError as e:
            msg = str(e)
            if any(code in msg for code in RETRY_STATUS):
                time.sleep(delay)
                delay = min(delay * 2, 10)
                continue
            raise
    return callable_fn(*args, **kwargs)


def _get_or_create_worksheet(sh, title, headers=None, rows=1000, cols=12):
    try:
        return sh.worksheet(title)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
        if headers:
            _retryable(ws.append_row, headers, value_input_option="USER_ENTERED")
        return ws


def ensure_worksheets(gc, gsheet_id):
    """Garantit l'existence des 3 feuilles attendues et les retourne."""
    sh = gc.open_by_key(gsheet_id)

    ws_trades = _get_or_create_worksheet(
        sh, TRADES_SHEET,
        headers=["timestamp","symbol","entry_price","exit_price","qty","pnl_usdc","result","day","week_iso"],
        rows=2000, cols=12
    )
    ws_daily = _get_or_create_worksheet(
        sh, DAILY_SHEET,
        headers=["day","symbol","trades","wins","losses","pnl_usdc"],
        rows=400, cols=10
    )
    ws_weekly = _get_or_create_worksheet(
        sh, WEEKLY_SHEET,
        headers=["week_iso","symbol","trades","wins","losses","pnl_usdc"],
        rows=400, cols=10
    )
    return ws_trades, ws_daily, ws_weekly


def _safe_get_all_records(ws):
    return _retryable(ws.get_all_records)


def _safe_append_row(ws, values):
    return _retryable(ws.append_row, values, value_input_option="USER_ENTERED")


def _safe_update(ws, a1_range, values_row):
    return _retryable(ws.update, a1_range, [values_row])


def append_trade_row(ws_trades, when_dt, symbol, entry_price, exit_price, qty, pnl_usdc, result):
    d = when_dt.date().isoformat()
    y, w, _ = when_dt.isocalendar()
    week_iso = f"{y}-W{str(w).zfill(2)}"
    _safe_append_row(ws_trades, [
        when_dt.isoformat(), symbol, float(entry_price), float(exit_price),
        float(qty), float(pnl_usdc), result, d, week_iso
    ])


def _aggregate_trades(sh, key_field, key_value, symbol):
    ws_trades = sh.worksheet(TRADES_SHEET)
    data = _safe_get_all_records(ws_trades)
    rows = [
        r for r in data
        if r.get("result") in ("WIN","LOSS")
        and r.get("symbol") == symbol
        and r.get(key_field) == key_value
    ]
    pnl = sum(float(r.get("pnl_usdc", 0)) for r in rows)
    wins = sum(1 for r in rows if r.get("result") == "WIN")
    losses = sum(1 for r in rows if r.get("result") == "LOSS")
    return dict(trades=len(rows), wins=wins, losses=losses, pnl_usdc=round(pnl, 6))


def _upsert(ws, key_col_name, key_value, symbol, agg):
    lignes = _safe_get_all_records(ws)
    idx = None
    for i, r in enumerate(lignes, start=2):  # 1 = header
        if r.get(key_col_name) == key_value and r.get("symbol") == symbol:
            idx = i
            break
    values = [key_value, symbol, agg["trades"], agg["wins"], agg["losses"], agg["pnl_usdc"]]
    if idx:
        _safe_update(ws, f"A{idx}:F{idx}", values)
    else:
        _safe_append_row(ws, values)


def upsert_daily_summary(ws_daily, day_iso, symbol):
    sh = ws_daily.spreadsheet
    agg = _aggregate_trades(sh, "day", day_iso, symbol)
    _upsert(ws_daily, "day", day_iso, symbol, agg)


def upsert_weekly_summary(ws_weekly, day_iso, symbol):
    dt = datetime.fromisoformat(day_iso)
    y, w, _ = dt.isocalendar()
    week_iso = f"{y}-W{str(w).zfill(2)}"
    sh = ws_weekly.spreadsheet
    agg = _aggregate_trades(sh, "week_iso", week_iso, symbol)
    _upsert(ws_weekly, "week_iso", week_iso, symbol, agg)
