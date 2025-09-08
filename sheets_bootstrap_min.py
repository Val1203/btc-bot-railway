# sheets_bootstrap_min.py
import os, json, time
import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _client():
    sa_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return gspread.authorize(creds)

def _open():
    return _client().open_by_key(os.environ["GSHEET_ID"])

def _retry(fn, *args, **kwargs):
    delay = 1.0
    for _ in range(5):
        try:
            return fn(*args, **kwargs)
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 10)
    # dernière tentative
    return fn(*args, **kwargs)

def setup():
    """Crée/ouvre les feuilles 'trades', 'daily', 'weekly' et pose les en-têtes."""
    try:
        sh = _open()

        try:
            trades = sh.worksheet("trades")
        except WorksheetNotFound:
            trades = sh.add_worksheet("trades", rows=1000, cols=20)
            trades.append_row(
                ["ts","date","side","qty_base","price","quote_value",
                 "fee_quote","position_id","pnl_quote"]
            )

        try:
            daily = sh.worksheet("daily")
        except WorksheetNotFound:
            daily = sh.add_worksheet("daily", rows=1000, cols=2)
            daily.append_row(["date","pnl_quote"])

        try:
            weekly = sh.worksheet("weekly")
        except WorksheetNotFound:
            weekly = sh.add_worksheet("weekly", rows=1000, cols=3)
            weekly.append_row(["week_start_date","week_end_date","pnl_quote"])

        print("[GSheets] Setup OK ✅")
        return {"sh": sh, "trades": trades, "daily": daily, "weekly": weekly}
    except Exception as e:
        print("[GSheets] Setup error:", e)
        return None

def append_trade(row):
    """Ajoute une ligne de trade dans l’onglet 'trades'."""
    sh = _open()
    ws = _retry(sh.worksheet, "trades")
    _retry(ws.append_row, row, value_input_option="USER_ENTERED")

def upsert_daily(date_str, pnl_quote):
    """Insère ou met à jour le PnL du jour dans 'daily' (col B)."""
    sh = _open()
    ws = _retry(sh.worksheet, "daily")
    values = ws.get_all_values()
    idx = None
    for i in range(2, len(values) + 1):  # saute les en-têtes
        if values[i - 1][0] == date_str:
            idx = i
            break
    if idx:
        _retry(ws.update_cell, idx, 2, pnl_quote)
    else:
        _retry(ws.append_row, [date_str, pnl_quote], value_input_option="USER_ENTERED")
