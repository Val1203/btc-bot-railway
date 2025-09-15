# --- AJOUTS/REMPLACEMENTS DANS sheets_bootstrap_min.py ---

import time
from gspread.exceptions import APIError

RETRY_STATUS = ("429", "500", "503")  # rate limit / internal error / service unavailable

def _retryable(callable_fn, *args, retries=5, base_delay=1.0, **kwargs):
    """
    Exécute callable_fn avec retries exponentiels sur erreurs API Google.
    """
    delay = base_delay
    for attempt in range(retries):
        try:
            return callable_fn(*args, **kwargs)
        except APIError as e:
            msg = str(e)
            if any(code in msg for code in RETRY_STATUS):
                time.sleep(delay)
                delay = min(delay * 2, 10)  # max 10s entre essais
                continue
            raise  # autres erreurs : remonter
    # dernier essai
    return callable_fn(*args, **kwargs)

def _safe_get_all_records(ws):
    return _retryable(ws.get_all_records)

def _safe_append_row(ws, values):
    return _retryable(ws.append_row, values, value_input_option="USER_ENTERED")

def _safe_update(ws, a1_range, values):
    return _retryable(ws.update, a1_range, [values])

# Utiliser _safe_* partout :

def append_trade_row(ws_trades, when_dt, symbol, entry_price, exit_price, qty, pnl_usdc, result):
    d = when_dt.date().isoformat()
    y, w, _ = when_dt.isocalendar()
    week_iso = f"{y}-W{str(w).zfill(2)}"
    _safe_append_row(ws_trades, [
        when_dt.isoformat(), symbol, float(entry_price), float(exit_price),
        float(qty), float(pnl_usdc), result, d, week_iso
    ])

def _upsert(ws, key_col_name, key_value, symbol, agg):
    rows = _safe_get_all_records(ws)
    idx = None
    for i, r in enumerate(rows, start=2):  # 1 = header
        if r.get(key_col_name) == key_value and r.get("symbol") == symbol:
            idx = i
            break
    values = [key_value, symbol, agg["trades"], agg["wins"], agg["losses"], agg["pnl_usdc"]]
    if idx:
        _safe_update(ws, f"A{idx}:F{idx}", values)
    else:
        _safe_append_row(ws, values)
