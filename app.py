import os, sys, time, signal, logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

import pytz
import pandas as pd
import numpy as np
import ccxt
from tenacity import retry, wait_exponential, stop_after_attempt

# Import en module (plus robuste que "from … import …")
import sheets_bootstrap_min as sbm

# =========================
# Logging -> stdout
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("bot")

# =========================
# ENV helpers (FR/EN)
# =========================
TRUE_SET  = {"1","true","yes","on","vrai","oui","y","t"}
FALSE_SET = {"0","false","no","off","faux","non","n","f"}

def env_raw(name, default=None):
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def env_bool(name, default="false"):
    v = str(env_raw(name, default)).strip().lower()
    if v in TRUE_SET:  return True
    if v in FALSE_SET: return False
    return v not in ("",)

def env_decimal(name, default):
    return Decimal(str(env_raw(name, default)))

def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        return s
    QUOTES = ("USDT","USDC","BUSD","BTC","ETH","EUR","USD")
    for q in QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)] + "/" + q
    return s

# =========================
# Lecture ENV (Railway)
# =========================
BINANCE_API_KEY    = env_raw("BINANCE_API_KEY")
BINANCE_API_SECRET = env_raw("BINANCE_API_SECRET")

# compat SYMBOLE
_symbol_raw = os.getenv("SYMBOL") or os.getenv("SYMBOLE")
if not _symbol_raw:
    raise RuntimeError("Veuillez définir la variable 'SYMBOL' (ou 'SYMBOLE').")
SYMBOL = normalize_symbol(_symbol_raw)

DRY_RUN            = env_bool("DRY_RUN", "true")
TRADING_ENABLED    = env_bool("TRADING_ENABLED", "false")

ORDER_USDC             = env_decimal("ORDER_USDC", "50")
TAKE_PROFIT_PCT        = env_decimal("TAKE_PROFIT_PCT", "0.6")/Decimal(100)
STOP_LOSS_PCT          = env_decimal("STOP_LOSS_PCT", "0.4")/Decimal(100)
RSI_BUY                = int(env_raw("RSI_BUY", "40"))
EMA_DEV_BUY_PCT        = env_decimal("EMA_DEV_BUY_PCT", "0.3")/Decimal(100)

MAX_TRADES_PER_DAY     = int(env_raw("MAX_TRADES_PER_DAY", "18"))
MAX_CONCURRENT_POSITIONS = int(env_raw("MAX_CONCURRENT_POSITIONS", "1"))

MAX_CAP_USDC           = env_decimal("MAX_CAP_USDC", "600")
MIN_USDC_RESERVE       = env_decimal("MIN_USDC_RESERVE", "0")
DAILY_MAX_LOSS_USDC    = env_decimal("DAILY_MAX_LOSS_USDC", "30")
DAILY_LOSSES_LIMIT     = int(env_raw("DAILY_LOSSES_LIMIT", "6"))
CONSECUTIVE_LOSS_LIMIT = int(env_raw("CONSECUTIVE_LOSS_LIMIT", "3"))
COOLDOWN_MINUTES       = int(env_raw("COOLDOWN_MINUTES", "45"))
DAY_RESET_TZ           = env_raw("DAY_RESET_TZ", "Europe/Paris")

GSHEET_ID              = env_raw("GSHEET_ID")
GOOGLE_SERVICE_JSON    = env_raw("GOOGLE_SERVICE_ACCOUNT_JSON")

log.info(
    "CONFIGURATION | SYMBOL=%s | ORDER_USDC=%s | TP=%.4f %% | SL=%.4f %% | RSI_BUY=%s | "
    "EMA_DEV_BUY_PCT=%.4f %% | MAX_CAP_USDC=%s | MAX_TRADES_PER_DAY=%d | "
    "MAX_CONCURRENT_POSITIONS=%d | DAILY_MAX_LOSS_USDC=%s | DAILY_LOSSES_LIMIT=%d | "
    "CONSECUTIVE_LOSS_LIMIT=%d | COOLDOWN_MINUTES=%d | TZ=%s | DRY_RUN=%s | TRADING_ENABLED=%s",
    SYMBOL, ORDER_USDC, float(TAKE_PROFIT_PCT*100), float(STOP_LOSS_PCT*100), RSI_BUY,
    float(EMA_DEV_BUY_PCT*100), MAX_CAP_USDC, MAX_TRADES_PER_DAY,
    MAX_CONCURRENT_POSITIONS, DAILY_MAX_LOSS_USDC, DAILY_LOSSES_LIMIT,
    CONSECUTIVE_LOSS_LIMIT, COOLDOWN_MINUTES, DAY_RESET_TZ, DRY_RUN, TRADING_ENABLED
)

# =========================
# Exchange (ccxt)
# =========================
def build_exchange():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
exchange = build_exchange()

# =========================
# Indicateurs
# =========================
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

# =========================
# Google Sheets
# =========================
gc = sbm.get_gsheet_client(GOOGLE_SERVICE_JSON)
ws_trades, ws_daily, ws_weekly = sbm.ensure_worksheets(gc, GSHEET_ID)
try:
    sh = gc.open_by_key(GSHEET_ID)
    log.info("Google Sheets connecté | Tableur : '%s' | Onglets : %s",
             sh.title, ", ".join([w.title for w in sh.worksheets()]))
except Exception as e:
    log.warning("Google Sheets initialisé, mais info feuille non lue: %s", e)

# =========================
# État runtime
# =========================
open_positions = []  # dicts {"qty","entry","time"}
consecutive_losses = 0
trades_done_today = 0
invested_capital = Decimal("0")
daily_pnl = Decimal("0")
daily_losses_count = 0
cooldown_until = None

tz = pytz.timezone(DAY_RESET_TZ)
now_tz = lambda: datetime.now(tz)
current_day_key = now_tz().strftime("%Y-%m-%d")

def new_day_reset():
    global trades_done_today, invested_capital, daily_pnl, daily_losses_count, consecutive_losses
    trades_done_today = 0
    invested_capital = Decimal("0")
    daily_pnl = Decimal("0")
    daily_losses_count = 0
    consecutive_losses = 0

# =========================
# Utils
# =========================
def quantize_qty(qty):
    return Decimal(qty).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def fetch_ohlcv(symbol, timeframe="1m", limit=200):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def fetch_free_stables():
    if DRY_RUN:
        return Decimal("100000")
    bal = exchange.fetch_balance()
    for k in ("USDT","USDC","BUSD"):
        if k in bal.get("free", {}):
            return Decimal(str(bal["free"][k]))
    return Decimal("0")

def get_last_price(symbol):
    return float(exchange.fetch_ticker(symbol)["last"])

def place_market_buy(symbol, usdc_amount):
    if DRY_RUN:
        return {"filled": float(usdc_amount), "price": float(get_last_price(symbol))}
    price = Decimal(str(get_last_price(symbol)))
    qty = Decimal(usdc_amount) / price
    order = exchange.create_market_buy_order(symbol, float(qty))
    trade_price = Decimal(str(order.get("price") or price))
    filled_cost = Decimal(str(order.get("cost") or (qty*trade_price)))
    return {"filled": float(filled_cost), "price": float(trade_price)}

def place_market_sell(symbol, qty):
    if DRY_RUN:
        return {"filled": float(qty), "price": float(get_last_price(symbol))}
    order = exchange.create_market_sell_order(symbol, float(qty))
    trade_price = Decimal(str(order.get("price") or get_last_price(symbol)))
    return {"filled": float(qty), "price": float(trade_price)}

# =========================
# Règles
# =========================
def should_enter(prices: pd.Series) -> bool:
    r = rsi(prices).iloc[-1]
    em = ema(prices).iloc[-1]
    last = prices.iloc[-1]
    under_ema = (Decimal(str(em - last)).copy_abs() / Decimal(str(em))) >= EMA_DEV_BUY_PCT and last < em
    return (r <= RSI_BUY) and under_ema

def exit_levels(entry_price: Decimal):
    tp = entry_price * (Decimal("1")+TAKE_PROFIT_PCT)
    sl = entry_price * (Decimal("1")-STOP_LOSS_PCT)
    return tp, sl

# =========================
# Signals
# =========================
RUNNING = True
def handle_sigterm(*_):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# =========================
# Boucle 24/7
# =========================
log.info("Bot démarré.")

while RUNNING:
    try:
        # Reset journalier (TZ)
        day_key = now_tz().strftime("%Y-%m-%d")
        if day_key != current_day_key:
            sbm.upsert_daily_summary(ws_daily, current_day_key, SYMBOL)
            sbm.upsert_weekly_summary(ws_weekly, current_day_key, SYMBOL)
            new_day_reset()
            current_day_key = day_key

        if cooldown_until and now_tz() < cooldown_until:
            time.sleep(5); continue
        else:
            cooldown_until = None

        # Limites journalières
        if daily_pnl <= -DAILY_MAX_LOSS_USDC or daily_losses_count >= DAILY_LOSSES_LIMIT:
            log.warning("Limites journalières atteintes. Pause jusqu'à minuit.")
            tomorrow = (now_tz() + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            cooldown_until = tomorrow
            time.sleep(5); continue

        # Marché
        ohlcv = fetch_ohlcv(SYMBOL, timeframe="1m", limit=200)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
        prices = df["close"].astype(float)
        last_price = Decimal(str(prices.iloc[-1]))

        # DEBUG raison d'entrée
        r = rsi(prices).iloc[-1]
        em = ema(prices).iloc[-1]
        dev_pct = abs((em - prices.iloc[-1]) / em) * 100
        log.info("DEBUG | last=%.2f | EMA=%.2f | dev=%.3f%% | RSI=%.1f",
                 float(prices.iloc[-1]), float(em), float(dev_pct), float(r))

        # Sorties TP/SL
        still_open = []
        for pos in open_positions:
            tp, sl = exit_levels(pos["entry"])
            if last_price >= tp:
                sell_res = place_market_sell(SYMBOL, pos["qty"])
                pnl = (Decimal(str(sell_res["price"])) - pos["entry"]) * pos["qty"]
                daily_pnl += pnl
                consecutive_losses = 0
                sbm.append_trade_row(ws_trades, now_tz(), SYMBOL, pos["entry"], Decimal(str(sell_res["price"])),
                                     pos["qty"], pnl, "WIN")
                log.info("TP atteint: +%.4f USDC", float(pnl))
            elif last_price <= sl:
                sell_res = place_market_sell(SYMBOL, pos["qty"])
                pnl = (Decimal(str(sell_res["price"])) - pos["entry"]) * pos["qty"]
                daily_pnl += pnl
                consecutive_losses += 1
                daily_losses_count += 1
                sbm.append_trade_row(ws_trades, now_tz(), SYMBOL, pos["entry"], Decimal(str(sell_res["price"])),
                                     pos["qty"], pnl, "LOSS")
                log.info("SL touché: %.4f USDC", float(pnl))
                if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                    log.warning("Perte consécutive limite atteinte. Cooldown %d min.", COOLDOWN_MINUTES)
                    cooldown_until = now_tz() + timedelta(minutes=COOLDOWN_MINUTES)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Entrées
        if TRADING_ENABLED and len(open_positions) < MAX_CONCURRENT_POSITIONS:
            if trades_done_today < MAX_TRADES_PER_DAY and invested_capital + ORDER_USDC <= MAX_CAP_USDC:
                free_stable = fetch_free_stables()
                if free_stable - ORDER_USDC >= MIN_USDC_RESERVE:
                    if should_enter(prices):
                        buy_res = place_market_buy(SYMBOL, ORDER_USDC)
                        entry_price = Decimal(str(buy_res["price"]))
                        qty = quantize_qty(ORDER_USDC / entry_price)
                        open_positions.append({"entry": entry_price, "qty": qty, "time": now_tz()})
                        invested_capital += ORDER_USDC
                        trades_done_today += 1
                        log.info("Entrée @ %s qty=%s (trades jour: %d)", entry_price, qty, trades_done_today)

        # Bilans toutes les 5 minutes (réduit charge API)
        if int(time.time()) % 300 < 2:
            sbm.upsert_daily_summary(ws_daily, now_tz().strftime("%Y-%m-%d"), SYMBOL)
            sbm.upsert_weekly_summary(ws_weekly, now_tz().strftime("%Y-%m-%d"), SYMBOL)

        time.sleep(4)

    except ccxt.NetworkError as e:
        log.warning("NetworkError: %s", e); time.sleep(5)
    except Exception as e:
        log.exception("Erreur boucle principale: %s", e); time.sleep(5)

log.info("Bot arrêté proprement.")
