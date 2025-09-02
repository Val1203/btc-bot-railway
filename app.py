# -*- coding: utf-8 -*-
"""
app.py — Bot BTC/USDC pour Railway
- Variables faciles à modifier depuis Railway (ENV prioritaire)
- Option: fichier config.json (secondaire)
- Écrit les trades & bilans dans Google Sheets
- Stratégie simple: RSI + écart EMA, TP/SL, gardes-fous

⚠️ Aucune garantie de profit. Utilisation à vos risques.
"""

import os, json, time, math, uuid, traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Binance (Spot)
from binance.spot import Spot as BinanceClient
# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

UTC = timezone.utc
def utcnow() -> datetime: return datetime.now(tz=UTC)

# =========================
# ======= CONFIG I/O ======
# =========================

def as_bool(x: Any, default: bool = False) -> bool:
    """Convertit de façon robuste divers formats (true/false, 1/0, yes/no, on/off, vrai/faux)."""
    if x is None: return default
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in ("true","1","yes","y","on","vrai")

def parse_env_value(raw: Optional[str], default: Any = None) -> Any:
    """Parsage simple pour ENV: bool, int/float, sinon texte."""
    if raw is None: return default
    s = raw.strip()
    sl = s.lower()
    # bool?
    if sl in ("true","false","vrai","faux"):
        return sl in ("true","vrai")
    # number?
    try:
        if "." in s:
            return float(s)
        return int(s)
    except:
        return s

def load_config_file() -> Dict[str, Any]:
    path = os.getenv("CONFIG_PATH", "config.json")
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[CONFIG] Chargé depuis {path}")
            return data
    except Exception as e:
        print("[CONFIG] Erreur lecture config.json:", e)
        return {}

CFG = load_config_file()

def cfg(key: str, default: Any = None) -> Any:
    """ENV prioritaire, puis config.json, sinon défaut."""
    if key in os.environ:
        return parse_env_value(os.getenv(key), default)
    if key in CFG:
        return CFG[key]
    return default

# =========================
# ========= PARAMS ========
# =========================

# Secrets: 2 façons — préférer SECRETS_JSON (clé Google + Binance)
#   SECRETS_JSON = {
#     "BINANCE_API_KEY": "...",
#     "BINANCE_API_SECRET": "...",
#     "GOOGLE_SERVICE_ACCOUNT_JSON": {...}
#   }
def load_secrets():
    raw = os.getenv("SECRETS_JSON")
    if raw:
        try:
            data = json.loads(raw)
            return (
                data.get("BINANCE_API_KEY", ""),
                data.get("BINANCE_API_SECRET", ""),
                data.get("GOOGLE_SERVICE_ACCOUNT_JSON", {}),
            )
        except Exception as e:
            print("[SECRETS] Impossible de parser SECRETS_JSON:", e)
    # fallback: variables séparées
    api = os.getenv("BINANCE_API_KEY", "")
    sec = os.getenv("BINANCE_API_SECRET", "")
    gjson_raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    gjson = {}
    if gjson_raw:
        try:
            gjson = json.loads(gjson_raw) if isinstance(gjson_raw, str) else gjson_raw
        except Exception as e:
            print("[SECRETS] GOOGLE_SERVICE_ACCOUNT_JSON invalide:", e)
    return api, sec, gjson

BINANCE_API_KEY, BINANCE_API_SECRET, GOOGLE_SA_DICT = load_secrets()

SYMBOL               = str(cfg("SYMBOL", "BTCUSDC"))
QUOTE                = "USDC"
BASE                 = "BTC"

# Flags & sécurité
DRY_RUN              = as_bool(cfg("DRY_RUN", True), True)
TRADING_ENABLED      = as_bool(cfg("TRADING_ENABLED", True), True)
MAX_CONCURRENT_POS   = int(cfg("MAX_CONCURRENT_POSITIONS", 1))
DAILY_MAX_LOSS_USDC  = float(cfg("DAILY_MAX_LOSS_USDC", 15))
CONSECUTIVE_LOSS_LIMIT = int(cfg("CONSECUTIVE_LOSS_LIMIT", 3))
COOLDOWN_MINUTES     = int(cfg("COOLDOWN_MINUTES", 60))
MIN_USDC_RESERVE     = float(cfg("MIN_USDC_RESERVE", 0))  # garder un fond de USDC

# Stratégie
INTERVAL             = str(cfg("INTERVAL", "1m"))
EMA_PERIOD           = int(cfg("EMA_PERIOD", 50))
RSI_PERIOD           = int(cfg("RSI_PERIOD", 14))
RSI_BUY              = float(cfg("RSI_BUY", 33))
EMA_DEV_BUY_PCT      = float(cfg("EMA_DEV_BUY_PCT", 0.0015))

# Ordres
ORDER_USDC           = float(cfg("ORDER_USDC", 500))
TAKE_PROFIT_PCT      = float(cfg("TAKE_PROFIT_PCT", 0.006))
STOP_LOSS_PCT        = float(cfg("STOP_LOSS_PCT", 0.004))

# Boucle
LOOP_SLEEP_SECONDS   = int(cfg("LOOP_SLEEP_SECONDS", 15))

# Google Sheets
GSHEET_ID            = str(cfg("GSHEET_ID", ""))

# Debug d’entrée
print("ENV DEBUG:", {
  "DRY_RUN": os.getenv("DRY_RUN"),
  "TRADING_ENABLED": os.getenv("TRADING_ENABLED"),
  "ORDER_USDC": os.getenv("ORDER_USDC"),
  "TAKE_PROFIT_PCT": os.getenv("TAKE_PROFIT_PCT"),
  "STOP_LOSS_PCT": os.getenv("STOP_LOSS_PCT"),
  "DAILY_MAX_LOSS_USDC": os.getenv("DAILY_MAX_LOSS_USDC"),
  "CONSECUTIVE_LOSS_LIMIT": os.getenv("CONSECUTIVE_LOSS_LIMIT"),
  "SYMBOL": os.getenv("SYMBOL"),
})
print("CFG DEBUG:", {
  "DRY_RUN": CFG.get("DRY_RUN"),
  "TRADING_ENABLED": CFG.get("TRADING_ENABLED"),
  "ORDER_USDC": CFG.get("ORDER_USDC"),
  "TAKE_PROFIT_PCT": CFG.get("TAKE_PROFIT_PCT"),
  "STOP_LOSS_PCT": CFG.get("STOP_LOSS_PCT"),
  "DAILY_MAX_LOSS_USDC": CFG.get("DAILY_MAX_LOSS_USDC"),
  "CONSECUTIVE_LOSS_LIMIT": CFG.get("CONSECUTIVE_LOSS_LIMIT"),
  "SYMBOL": CFG.get("SYMBOL"),
})

# =========================
# ===== GOOGLE SHEETS =====
# =========================

def init_gsheets():
    if not GSHEET_ID or not GOOGLE_SA_DICT:
        print("[GSheets] GSHEET_ID ou Service Account JSON manquant → logs console uniquement.")
        return None, None, None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(GOOGLE_SA_DICT, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEET_ID)
        ws_names = [w.title for w in sh.worksheets()]
        if "trades" not in ws_names:
            sh.add_worksheet("trades", rows=1000, cols=20)
            sh.worksheet("trades").append_row(
                ["ts","date","side","qty_base","price","quote_value","fee_quote","position_id","pnl_quote"]
            )
        if "daily_pnl" not in ws_names:
            sh.add_worksheet("daily_pnl", rows=500, cols=10)
            sh.worksheet("daily_pnl").append_row(["date","pnl_quote"])
        if "weekly_pnl" not in ws_names:
            sh.add_worksheet("weekly_pnl", rows=200, cols=10)
            sh.worksheet("weekly_pnl").append_row(["week_start_date","week_end_date","pnl_quote"])
        return sh, sh.worksheet("trades"), sh.worksheet("daily_pnl")
    except Exception as e:
        print("[GSheets] Erreur init:", e)
        return None, None, None

SHEET, WS_TRADES, WS_DAILY = init_gsheets()

def gs_append_trades(row):
    try:
        if WS_TRADES: WS_TRADES.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        print("[GSheets] append trades:", e)

def gs_append_daily(date_str, pnl):
    try:
        if SHEET: SHEET.worksheet("daily_pnl").append_row([date_str, pnl], value_input_option="USER_ENTERED")
    except Exception as e:
        print("[GSheets] append daily:", e)

def gs_append_weekly(week_start, week_end, pnl):
    try:
        if SHEET: SHEET.worksheet("weekly_pnl").append_row([week_start, week_end, pnl], value_input_option="USER_ENTERED")
    except Exception as e:
        print("[GSheets] append weekly:", e)

def gs_read_df(sheet_name):
    if not SHEET: return pd.DataFrame()
    try:
        ws = SHEET.worksheet(sheet_name)
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        print("[GSheets] read df:", e)
        return pd.DataFrame()

# =========================
# ====== BINANCE I/O ======
# =========================

def new_binance_client():
    if DRY_RUN or not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return BinanceClient()
    return BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

client = new_binance_client()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
def get_klines(symbol, interval="1m", limit=200):
    return client.klines(symbol, interval, limit=limit)

def get_price(symbol):
    data = client.ticker_price(symbol)
    return float(data["price"])

def get_filters(symbol):
    ex = client.exchange_info(symbol=symbol)
    info = ex["symbols"][0]
    fs = {f["filterType"]: f for f in info["filters"]}
    return fs

FILTERS = get_filters(SYMBOL)

def round_step(value, step):
    step = float(step)
    return math.floor(float(value) / step) * step

def format_qty_price(qty, price):
    lot = FILTERS["LOT_SIZE"]
    tick = FILTERS["PRICE_FILTER"]
    q_step = float(lot["stepSize"])
    p_tick = float(tick["tickSize"])
    return round_step(qty, q_step), round_step(price, p_tick)

def account_balances():
    if DRY_RUN or not BINANCE_API_KEY:
        return {"USDC": float("inf"), "BTC": 0.0}
    acc = client.account()
    b = {x["asset"]: float(x["free"]) for x in acc["balances"]}
    return {"USDC": b.get("USDC", 0.0), "BTC": b.get("BTC", 0.0)}

def place_market_buy_usdc(usdc_amount):
    if DRY_RUN:
        px = get_price(SYMBOL)
        qty = usdc_amount / px
        qty, _ = format_qty_price(qty, px)
        return {
            "symbol": SYMBOL, "side": "BUY", "type": "MARKET",
            "fills": [{"price": str(px), "qty": str(qty)}],
            "executedQty": str(qty), "cummulativeQuoteQty": str(qty * px),
        }
    return client.new_order(symbol=SYMBOL, side="BUY", type="MARKET", quoteOrderQty=str(usdc_amount))

def place_limit_sell(qty, price):
    qty, price = format_qty_price(qty, price)
    if DRY_RUN:
        return {"orderId": 1, "status": "NEW", "origQty": str(qty), "price": str(price)}
    return client.new_order(symbol=SYMBOL, side="SELL", type="LIMIT", timeInForce="GTC",
                            quantity=str(qty), price=str(price))

def cancel_order(order_id):
    if DRY_RUN: return
    try:
        client.cancel_order(symbol=SYMBOL, orderId=order_id)
    except Exception:
        pass

def get_order(order_id):
    if DRY_RUN: return {"status": "NEW"}
    return client.get_order(symbol=SYMBOL, orderId=order_id)

def market_sell(qty):
    if DRY_RUN:
        px = get_price(SYMBOL)
        qty, _ = format_qty_price(qty, px)
        return {"symbol": SYMBOL, "side": "SELL", "type": "MARKET",
                "executedQty": str(qty), "cummulativeQuoteQty": str(qty * px)}
    return client.new_order(symbol=SYMBOL, side="SELL", type="MARKET", quantity=str(qty))

# =========================
# ===== INDICATEURS =======
# =========================

def compute_indicators():
    ks = get_klines(SYMBOL, INTERVAL, limit=max(EMA_PERIOD, RSI_PERIOD) + 100)
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(ks, columns=cols)
    df["close"] = df["close"].astype(float)
    close = df["close"].astype(float)

    # EMA
    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / (loss.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))

    return float(close.iloc[-1]), float(ema.iloc[-1]), float(rsi.iloc[-1])

# =========================
# ====== ÉTAT / PNL =======
# =========================

positions = []  # [{id, qty, entry, tp, sl, tp_order_id}]
daily_realized_pnl = 0.0
current_day = utcnow().date()
consecutive_losses = 0
cooldown_until: Optional[datetime] = None
last_week_summary: Optional[datetime] = None

def record_trade(side, qty, price, position_id, pnl_quote=None, fee_quote=0.0):
    global daily_realized_pnl, consecutive_losses, cooldown_until
    quote_value = qty * price
    ts = utcnow().isoformat()
    date_str = str(utcnow().date())
    row = [ts, date_str, side, qty, price, quote_value, fee_quote, position_id, pnl_quote if pnl_quote is not None else ""]
    print("[TRADE]", row)
    gs_append_trades(row)
    if pnl_quote is not None:
        daily_realized_pnl += pnl_quote
        if pnl_quote < 0:
            consecutive_losses += 1
            if CONSECUTIVE_LOSS_LIMIT > 0 and consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                cooldown_until = utcnow() + timedelta(minutes=COOLDOWN_MINUTES)
                print(f"[GUARD] {consecutive_losses} pertes consécutives → pause jusqu'à {cooldown_until.isoformat()}")
        else:
            consecutive_losses = 0

def do_daily_and_weekly_summaries_if_needed():
    global current_day, daily_realized_pnl, last_week_summary
    today = utcnow().date()
    # Daily
    if today != current_day:
        gs_append_daily(str(current_day), daily_realized_pnl)
        print(f"[DAILY] {current_day} PNL = {daily_realized_pnl:.2f} {QUOTE}")
        daily_realized_pnl = 0.0
        current_day = today
    # Weekly (lundi → récap semaine précédente)
    week_start = today - timedelta(days=today.weekday())
    if last_week_summary is None:
        last_week_summary = week_start
    if week_start > last_week_summary:
        df = gs_read_df("trades")
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            prev_week_start = last_week_summary
            prev_week_end = week_start - timedelta(days=1)
            mask = (df["date"] >= pd.Timestamp(prev_week_start)) & (df["date"] <= pd.Timestamp(prev_week_end))
            pnl = df.loc[mask, "pnl_quote"].apply(lambda x: float(x) if str(x) not in ("", "None") else 0.0).sum()
            gs_append_weekly(str(prev_week_start), str(prev_week_end), float(pnl))
            print(f"[WEEKLY] {prev_week_start} → {prev_week_end} PNL = {pnl:.2f} {QUOTE}")
        last_week_summary = week_start

# =========================
# ====== LOGIQUE BOT ======
# =========================

def can_open_new_position():
    if not TRADING_ENABLED:
        print("[GUARD] TRADING_ENABLED=false → pas d'achats.")
        return False
    if len(positions) >= MAX_CONCURRENT_POS:
        return False
    # Perte journalière max
    df = gs_read_df("trades")
    if not df.empty:
        today_str = str(utcnow().date())
        loss_today = df[df["date"] == today_str]["pnl_quote"].apply(lambda x: float(x) if str(x) not in ("", "None") else 0.0).sum()
        if loss_today <= -DAILY_MAX_LOSS_USDC:
            print("[GUARD] Perte journalière max atteinte → on stoppe les entrées aujourd'hui.")
            return False
    # Cooldown
    global cooldown_until
    if cooldown_until and utcnow() < cooldown_until:
        return False
    # Réserve USDC
    bal = account_balances()["USDC"]
    if bal != float("inf") and (bal - ORDER_USDC) < MIN_USDC_RESERVE:
        print("[GUARD] Réserve USDC atteinte, pas d'achat.")
        return False
    return True

def maybe_open_position():
    if not can_open_new_position():
        return
    try:
        price, ema, rsi = compute_indicators()
        if (price <= ema * (1 - EMA_DEV_BUY_PCT)) and (rsi < RSI_BUY):
            usdc_bal = account_balances()["USDC"]
            notional = ORDER_USDC if usdc_bal == float("inf") else min(ORDER_USDC, max(0.0, usdc_bal - MIN_USDC_RESERVE))
            if notional < 5:
                return
            order = place_market_buy_usdc(notional)
            if "cummulativeQuoteQty" in order and "executedQty" in order:
                qty = float(order["executedQty"])
                avg_price = float(order["cummulativeQuoteQty"]) / qty
            else:
                fills = order.get("fills", [])
                if not fills: return
                qty = float(fills[0]["qty"]); avg_price = float(fills[0]["price"])
            qty, _ = format_qty_price(qty, avg_price)
            tp_price = avg_price * (1 + TAKE_PROFIT_PCT)
            sl_price = avg_price * (1 - STOP_LOSS_PCT)
            tp_order = place_limit_sell(qty, tp_price)
            tp_order_id = tp_order.get("orderId", None)
            pos = {"id": str(uuid.uuid4())[:12], "qty": qty, "entry": avg_price,
                   "tp": tp_price, "sl": sl_price, "tp_order_id": tp_order_id}
            positions.append(pos)
            record_trade("BUY", qty, avg_price, pos["id"])
    except Exception as e:
        print("[OPEN ERROR]", e)
        traceback.print_exc()

def manage_positions():
    for pos in list(positions):
        try:
            px = get_price(SYMBOL)
            # TP atteint ?
            tp_hit = False
            if pos["tp_order_id"] is not None:
                od = get_order(pos["tp_order_id"])
                tp_hit = (od.get("status") == "FILLED")
            if DRY_RUN and px >= pos["tp"]:
                tp_hit = True
            if tp_hit:
                sell_price = pos["tp"]
                pnl = (sell_price - pos["entry"]) * pos["qty"]
                record_trade("SELL", pos["qty"], sell_price, pos["id"], pnl_quote=pnl)
                positions.remove(pos)
                continue
            # SL atteint ?
            if px <= pos["sl"]:
                if pos["tp_order_id"]:
                    cancel_order(pos["tp_order_id"])
                res = market_sell(pos["qty"])
                if "cummulativeQuoteQty" in res and "executedQty" in res:
                    sell_price = float(res["cummulativeQuoteQty"]) / float(res["executedQty"])
                else:
                    sell_price = px
                pnl = (sell_price - pos["entry"]) * pos["qty"]
                record_trade("SELL", pos["qty"], sell_price, pos["id"], pnl_quote=pnl)
                positions.remove(pos)
        except Exception as e:
            print("[MANAGE ERROR]", e)

def main_loop():
    print(f"--- BOT démarré | DRY_RUN={DRY_RUN} | TRADING_ENABLED={TRADING_ENABLED} | Symbol={SYMBOL} ---")
    print(f"[PARAMS] ORDER_USDC={ORDER_USDC} TP={TAKE_PROFIT_PCT} SL={STOP_LOSS_PCT} RSI_BUY={RSI_BUY} EMA_DEV={EMA_DEV_BUY_PCT}")
    while True:
        try:
            do_daily_and_weekly_summaries_if_needed()
            maybe_open_position()
            manage_positions()
            time.sleep(LOOP_SLEEP_SECONDS)
        except KeyboardInterrupt:
            print("Arrêt demandé.")
            break
        except Exception as e:
            print("[LOOP ERROR]", e)
            time.sleep(5)

if __name__ == "__main__":
    main_loop()
