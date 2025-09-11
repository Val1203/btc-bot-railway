import json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_gsheet_client(service_json_str: str):
    info = json.loads(service_json_str)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def ensure_worksheets(gc, gsheet_id):
    sh = gc.open_by_key(gsheet_id)
    # trades
    try:
        ws_trades = sh.worksheet("trades")
    except gspread.WorksheetNotFound:
        ws_trades = sh.add_worksheet(title="trades", rows=1000, cols=12)
        ws_trades.append_row([
            "timestamp","symbol","entry_price","exit_price","qty",
            "pnl_usdc","result","day","week_iso"
        ])
    # daily_summary
    try:
        ws_daily = sh.worksheet("daily_summary")
    except gspread.WorksheetNotFound:
        ws_daily = sh.add_worksheet(title="daily_summary", rows=365, cols=10)
        ws_daily.append_row(["day","symbol","trades","wins","losses","pnl_usdc"])
    # weekly_summary
    try:
        ws_weekly = sh.worksheet("weekly_summary")
    except gspread.WorksheetNotFound:
        ws_weekly = sh.add_worksheet(title="weekly_summary", rows=200, cols=10)
        ws_weekly.append_row(["week_iso","symbol","trades","wins","losses","pnl_usdc"])
    return ws_trades, ws_daily, ws_weekly

def append_trade_row(ws_trades, when_dt, symbol, entry_price, exit_price, qty, pnl_usdc, result):
    d = when_dt.date().isoformat()
    y, w, _ = when_dt.isocalendar()
    week_iso = f"{y}-W{str(w).zfill(2)}"
    ws_trades.append_row([
        when_dt.isoformat(), symbol, float(entry_price), float(exit_price),
        float(qty), float(pnl_usdc), result, d, week_iso
    ], value_input_option="USER_ENTERED")

def _aggregate_trades(ws_trades, key_field, key_value, symbol):
    data = ws_trades.get_all_records()
    rows = [r for r in data if r.get("result") in ("WIN","LOSS")
            and r.get("symbol") == symbol and r.get(key_field) == key_value]
    pnl = sum(float(r.get("pnl_usdc",0)) for r in rows)
    wins = sum(1 for r in rows if r.get("result") == "WIN")
    losses = sum(1 for r in rows if r.get("result") == "LOSS")
    return dict(trades=len(rows), wins=wins, losses=losses, pnl_usdc=round(pnl, 6))

def _upsert(ws, key_col_name, key_value, symbol, agg):
    rows = ws.get_all_records()
    idx = None
    for i, r in enumerate(rows, start=2):  # 1 = header
        if r.get(key_col_name) == key_value and r.get("symbol") == symbol:
            idx = i; break
    values = [key_value, symbol, agg["trades"], agg["wins"], agg["losses"], agg["pnl_usdc"]]
    if idx:
        ws.update(f"A{idx}:F{idx}", [values])
    else:
        ws.append_row(values, value_input_option="USER_ENTERED")

def upsert_daily_summary(ws_daily, day_iso, symbol):
    ws_trades = ws_daily.spreadsheet.worksheet("trades")
    agg = _aggregate_trades(ws_trades, "day", day_iso, symbol)
    _upsert(ws_daily, "day", day_iso, symbol, agg)

def upsert_weekly_summary(ws_weekly, day_iso, symbol):
    dt = datetime.fromisoformat(day_iso)
    y, w, _ = dt.isocalendar()
    week_iso = f"{y}-W{str(w).zfill(2)}"
    ws_trades = ws_weekly.spreadsheet.worksheet("trades")
    agg = _aggregate_trades(ws_trades, "week_iso", week_iso, symbol)
    _upsert(ws_weekly, "week_iso", week_iso, symbol, agg)
