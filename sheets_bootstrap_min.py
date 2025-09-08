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
        except APIError:
            time.sleep(delay)
            delay = min(delay * 2, 10)
    # dernière tentative (laisse remonter l’erreur si ça casse)
    return fn(*args, **kwargs)

def _get_or_create(sh, title, headers):
    try:
        ws = sh.worksheet(title)
    except WorksheetNotFound:
        ws = _retry(sh.add_worksheet, title=title, rows=1000, cols=max(10, len(headers) + 2))
        _retry(ws.append_row, headers, value_input_option="USER_ENTERED")
        _retry(ws.freeze, rows=1)
    return ws

def setup():
    sh = _open()

    trades_headers = ["ts","date","side","qty_base","price","quote_value","fee_quote","position_id","pnl_quote"]
    daily_headers  = ["date","pnl_quote"]
    weekly_headers = ["week_start_date","week_end_date","pnl_quote"]

    ws_trades = _get_or_create(sh, "trades",     trades_headers)
    ws_daily  = _get_or_create(sh, "daily_pnl",  daily_headers)
    ws_weekly = _get_or_create(sh, "weekly_pnl", weekly_headers)
    return ws_trades, ws_daily, ws_weekly
