import os
import time
import json
import math
import signal
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

import pytz
import pandas as pd
import numpy as np
import ccxt
from tenacity import retry, wait_exponential, stop_after_attempt

from sheets_bootstrap_min import (
    get_gsheet_client,
    ensure_worksheets,
    append_trade_row,
    upsert_daily_summary,
    upsert_weekly_summary,
)

# -----------------------------
# Logging propre 24/7
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("bot")

# -----------------------------
# Lecture ENV (Railway)
# -----------------------------
def env(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return cast(v) if cast != bool else (str(v).lower() in ("1","true","yes","on"))

BINANCE_API_KEY       = env("BINANCE_API_KEY")
BINANCE_API_SECRET    = env("BINANCE_API_SECRET")
DRY_RUN               = env("DRY_RUN", "true", bool)
TRADING_ENABLED       = env("TRADING_ENABLED", "false", bool)
SYMBOL                = env("SYMBOL")                    # ex: "BTC/USDT"
ORDER_USDC            = Decimal(env("ORDER_USDC", "10")) # montant par trade
TAKE_PROFIT_PCT       = Decimal(env("TAKE_PROFIT_PCT", "0.6"))/Decimal(100)
STOP_LOSS_PCT         = Decimal(env("STOP_LOSS_PCT", "0.4"))/Decimal(100)
RSI_BUY               = int(env("RSI_BUY", "33"))
EMA_DEV_BUY_PCT       = Decimal(env("EMA_DEV_BUY_PCT", "0.5"))/Decimal(100)
MAX_CAP_USDC          = Decimal(env("MAX_CAP_USDC", "200"))
DAILY_TARGET_USDC     = Decimal(env("DAILY_TARGET_USDC", "5"))
MAX_TRADES_PER_DAY    = int(env("MAX_TRADES_PER_DAY", "20"))
MIN_USDC_RESERVE      = Decimal(env("MIN_USDC_RESERVE", "0"))
MAX_CONCURRENT_POSITIONS = int(env("MAX_CONCURRENT_POSITIONS", "1"))
DAILY_MAX_LOSS_USDC   = Decimal(env("DAILY_MAX_LOSS_USDC", "20"))
CONSECUTIVE_LOSS_LIMIT= int(env("CONSECUTIVE_LOSS_LIMIT", "3"))
COOLDOWN_MINUTES      = int(env("COOLDOWN_MINUTES", "30"))
GSHEET_ID             = env("GSHEET_ID")
GOOGLE_SERVICE_JSON   = env("GOOGLE_SERVICE_ACCOUNT_JSON")
DAILY_LOSSES_LIMIT    = int(env("DAILY_LOSSES_LIMIT", "10"))
DAY_RESET_TZ          = env("DAY_RESET_TZ", "Europe/Paris")

# -----------------------------
# Bourse/Exchange (ccxt)
# -----------------------------
def build_exchange():
    params = {
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
    return ccxt.binance(params)

exchange = build_exchange()

# -----------------------------
# RSI & EMA
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

# -----------------------------
# Google Sheets client
# -----------------------------
gc = get_gsheet_client(GOOGLE_SERVICE_JSON)
ws_trades, ws_daily, ws_weekly = ensure_worksheets(gc, GSHEET_ID)

# -----------------------------
# État runtime
# -----------------------------
open_positions = []  # simple: une liste de dicts {"qty","entry","time"}
consecutive_losses = 0
trades_done_today = 0
invested_capital = Decimal("0")
daily_pnl = Decimal("0")
daily_losses_count = 0
cooldown_until = None

tz = pytz.timezone(DAY_RESET_TZ)
def now_tz():
    return datetime.now(tz)

current_day_key = now_tz().strftime("%Y-%m-%d")

def new_day_reset():
    global trades_done_today, invested_capital, daily_pnl, daily_losses_count, consecutive_losses
    trades_done_today = 0
    invested_capital = Decimal("0")
    daily_pnl = Decimal("0")
    daily_losses_count = 0
    consecutive_losses = 0

# -----------------------------
# Utils
# -----------------------------
def quantize_price(p):   # arrondis simples
    return Decimal(p).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

def quantize_qty(qty):
    return Decimal(qty).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def fetch_ohlcv(symbol, timeframe="1m", limit=200):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def fetch_free_usdt():
    if DRY_RUN:
        # Dans DRY_RUN, on suppose un capital large sauf MIN_USDC_RESERVE
        return Decimal("100000")
    balance = exchange.fetch_balance()
    for k in ("USDT","USDC","BUSD"):
        if k in balance['free']:
            return Decimal(str(balance['free'][k]))
    return Decimal("0")

def place_market_buy(symbol, usdc_amount):
    if DRY_RUN:
        return {"filled": float(usdc_amount), "price": float(get_last_price(symbol))}
    price = Decimal(str(get_last_price(symbol)))
    qty = (Decimal(usdc_amount) / price)
    order = exchange.create_market_buy_order(symbol, float(qty))
    trade_price = Decimal(str(order['price'] or price))
    filled = Decimal(str(order['cost'] or (qty*trade_price)))
    return {"filled": float(filled), "price": float(trade_price)}

def place_market_sell(symbol, qty):
    if DRY_RUN:
        return {"filled": float(qty), "price": float(get_last_price(symbol))}
    order = exchange.create_market_sell_order(symbol, float(qty))
    trade_price = Decimal(str(order['price'] or get_last_price(symbol)))
    return {"filled": float(qty), "price": float(trade_price)}

def get_last_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker['last'])

# -----------------------------
# Règles d'entrée/sortie
# -----------------------------
def should_enter(prices: pd.Series) -> bool:
    r = rsi(prices).iloc[-1]
    em = ema(prices).iloc[-1]
    last = prices.iloc[-1]
    under_ema = (Decimal(str(em - last)).copy_abs() / Decimal(str(em))) >= EMA_DEV_BUY_PCT and last < em
    return (r <= RSI_BUY) and under_ema

def exit_levels(entry_price: Decimal):
    tp = entry_price * (Decimal("1.0") + TAKE_PROFIT_PCT)
    sl = entry_price * (Decimal("1.0") - STOP_LOSS_PCT)
    return (tp, sl)

# -----------------------------
# Gestion des signaux système
# -----------------------------
RUNNING = True
def handle_sigterm(signum, frame):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# -----------------------------
# Boucle 24/7
# -----------------------------
log.info("Bot démarré. DRY_RUN=%s TRADING_ENABLED=%s SYMBOL=%s", DRY_RUN, TRADING_ENABLED, SYMBOL)

while RUNNING:
    try:
        # Reset journalier à la TZ choisie
        day_key = now_tz().strftime("%Y-%m-%d")
        if day_key != current_day_key:
            # cloture des bilans jour/semaine à minuit TZ
            upsert_daily_summary(ws_daily, day_key, SYMBOL)  # calcule si données existent
            upsert_weekly_summary(ws_weekly, day_key, SYMBOL)
            new_day_reset()
            current_day_key = day_key

        if cooldown_until and now_tz() < cooldown_until:
            time.sleep(5)
            continue
        else:
            cooldown_until = None

        # Sécurité : limites journalières
        if daily_pnl <= -DAILY_MAX_LOSS_USDC or daily_losses_count >= DAILY_LOSSES_LIMIT:
            log.warning("Limite quotidienne atteinte. Pause jusqu’à minuit.")
            # mise en cooldown très long (jusqu’à minuit)
            tomorrow = (now_tz() + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            cooldown_until = tomorrow
            time.sleep(5)
            continue

        # Télécharger marché
        ohlcv = fetch_ohlcv(SYMBOL, timeframe="1m", limit=200)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
        prices = df["close"].astype(float)

        # Gestion des positions existantes (take profit / stop loss)
        last_price = Decimal(str(prices.iloc[-1]))
        still_open = []
        for pos in open_positions:
            tp, sl = exit_levels(pos["entry"])
            if last_price >= tp:
                # sortie gagnante
                sell_res = place_market_sell(SYMBOL, pos["qty"])
                pnl = (Decimal(str(sell_res["price"])) - pos["entry"]) * pos["qty"]
                daily_pnl += pnl
                consecutive_losses = 0
                status = "WIN"
                append_trade_row(ws_trades, now_tz(), SYMBOL, pos["entry"], Decimal(str(sell_res["price"])),
                                 pos["qty"], pnl, status)
                log.info("TP atteint: +%.4f USDC", float(pnl))
            elif last_price <= sl:
                # sortie perdante
                sell_res = place_market_sell(SYMBOL, pos["qty"])
                pnl = (Decimal(str(sell_res["price"])) - pos["entry"]) * pos["qty"]
                daily_pnl += pnl
                consecutive_losses += 1
                daily_losses_count += 1
                status = "LOSS"
                append_trade_row(ws_trades, now_tz(), SYMBOL, pos["entry"], Decimal(str(sell_res["price"])),
                                 pos["qty"], pnl, status)
                log.info("SL touché: %.4f USDC", float(pnl))
                if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                    log.warning("Perte consécutive limite atteinte. Cooldown %d min.", COOLDOWN_MINUTES)
                    cooldown_until = now_tz() + timedelta(minutes=COOLDOWN_MINUTES)
            else:
                # conserver
                still_open.append(pos)
        open_positions = still_open

        # Entrées
        if TRADING_ENABLED and len(open_positions) < MAX_CON_
