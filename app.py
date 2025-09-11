#!/usr/bin/env python3
# app.py

from __future__ import annotations

import os
import json
import time
import uuid
import math
import traceback
from datetime import datetime, timezone, date

# ========= Imports optionnels (on s'adapte à ce qui est installé) =========
EXCHANGE = None
CLIENT = None
USE_CCXT = False
USE_BINANCE_SDK = False

try:
    import ccxt  # type: ignore
    USE_CCXT = True
except Exception:
    USE_CCXT = False

if not USE_CCXT:
    try:
        from binance.client import Client  # type: ignore
        from binance.enums import HistoricalKlinesType  # type: ignore
        USE_BINANCE_SDK = True
    except Exception:
        USE_BINANCE_SDK = False

# ========= Google Sheets (optionnel) =========
try:
    import sheets_bootstrap_min as gsh  # ton module
except Exception:
    gsh = None  # on passera en "print only"

# ========= Helpers diag & sécurité =========
def coalesce(v, default):
    return default if v in (None, "", "Aucun", "None", "null") else v

def log_signal(prefix, price, rsi, ema, rsi_buy, ema_dev,
               trading_enabled, risk_lock, free_quote, need_quote):
    try:
        dev = abs(price - ema) / ema if ema else 0.0
    except Exception:
        dev = 0.0
    print(
        f"[SIGNAL] {prefix} price={price:.2f} rsi={rsi:.2f} ema={ema:.2f} "
        f"dev={dev:.4f} rsi_ok={(rsi <= rsi_buy)} dev_ok={(dev >= ema_dev)} "
        f"ENABLED={trading_enabled} RISK_LOCK={risk_lock} "
        f"free_quote={free_quote:.2f} need={need_quote:.2f}",
        flush=True,
    )

# ========= Paramètres (ENV + garde-fous) =========
SYMBOL         = os.getenv("SYMBOL", "BTCUSDC").upper()
BASE_ASSET     = SYMBOL[:-4] if SYMBOL.endswith(("USDC","USDT","BUSD")) else SYMBOL.split("/")[0]
QUOTE_ASSET    = os.getenv("QUOTE_ASSET", "USDC" if SYMBOL.endswith("USDC") else "USDT").upper()
TIMEFRAME      = os.getenv("TIMEFRAME", "1m")
RSI_PERIOD     = int(coalesce(os.getenv("RSI_PERIOD"), 14))
EMA_PERIOD     = int(coalesce(os.getenv("EMA_PERIOD"), 200))
RSI_BUY        = float(coalesce(os.getenv("RSI_BUY"), 33))
EMA_DEV        = float(coalesce(os.getenv("EMA_DEV"), 0.0035))   # 0.35%
TAKE_PROFIT_PCT= float(coalesce(os.getenv("TAKE_PROFIT_PCT"), 0.007))
STOP_LOSS_PCT  = float(coalesce(os.getenv("STOP_LOSS_PCT"), 0.004))
ORDER_USDC     = float(coalesce(os.getenv("ORDER_USDC"), 300))
TRADING_ENABLED= str(coalesce(os.getenv("TRADING_ENABLED"), "true")).lower() == "true"
DRY_RUN        = str(coalesce(os.getenv("DRY_RUN"), "false")).lower() == "true"
DAILY_CAP      = float(coalesce(os.getenv("DAILY_LOSSES_LIMIT"), 999))
if DAILY_CAP <= 0: DAILY_CAP = 999
CONSECUTIVE_LOSS_LIMIT = int(coalesce(os.getenv("CONSECUTIVE_LOSS_LIMIT"), 3))
LOOP_SLEEP     = float(coalesce(os.getenv("LOOP_SLEEP"), 1))   # seconde(s)
STATE_FILE     = os.getenv("STATE_FILE", "/mnt/data/state.json")

print("[ENV DEBUG]", {
    'DRY_RUN': DRY_RUN, 'TRADING_ENABLED': TRADING_ENABLED, 'ORDER_USDC': str(ORDER_USDC),
    'TAKE_PROFIT_PCT': f"{TAKE_PROFIT_PCT:.3f}", 'STOP_LOSS_PCT': f"{STOP_LOSS_PCT:.3f}",
    'DAILY_MAX_LOSS_USDC': os.getenv("DAILY_MAX_LOSS_USDC","8"),
    'CONSECUTIVE_LOSS_LIMIT': str(CONSECUTIVE_LOSS_LIMIT),
    'DAILY_LOSSES_LIMIT': str(DAILY_CAP if DAILY_CAP != 999 else "999 (off)"),
    'SYMBOL': SYMBOL
}, flush=True)

# ========= Initialisation Exchange =========
def init_exchange():
    global EXCHANGE, CLIENT, USE_CCXT, USE_BINANCE_SDK
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if USE_CCXT:
        try:
            EXCHANGE = ccxt.binance({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            EXCHANGE.load_markets()
            print("[INIT] ccxt binance prêt", flush=True)
            return
        except Exception as e:
            print("[WARN] ccxt indisponible:", e, flush=True)
            EXCHANGE = None
            USE_CCXT = False

    if USE_BINANCE_SDK:
        try:
            CLIENT = Client(api_key, api_secret)
            print("[INIT] python-binance prêt", flush=True)
            return
        except Exception as e:
            print("[WARN] python-binance indisponible:", e, flush=True)
            CLIENT = None
            USE_BINANCE_SDK = False

    print("[INIT] Aucune lib d'échange disponible → mode DRY_RUN forcé", flush=True)

init_exchange()

# ========= Utilitaires Exchange =========
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500):
    """Retourne une liste [[ts, open, high, low, close, vol], ...] en ms."""
    if USE_CCXT and EXCHANGE:
        return EXCHANGE.fetch_ohlcv(symbol.replace("", "/") if "/" in symbol else f"{BASE_ASSET}/{QUOTE_ASSET}", timeframe, limit=limit)
    if USE_BINANCE_SDK and CLIENT:
        # map timeframe
        tf_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "3m": Client.KLINE_INTERVAL_3MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
        }
        interval = tf_map.get(timeframe, Client.KLINE_INTERVAL_1MINUTE)
        kl = CLIENT.get_klines(symbol=symbol, interval=interval, limit=limit, klines_type=HistoricalKlinesType.SPOT)
        # ts, open, high, low, close, volume, ...
        return [[k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in kl]
    # fallback (pas d'exchange)
    raise RuntimeError("Aucune source OHLCV disponible")

def get_filters(symbol: str):
    """Retourne (step_qty, tick_price, min_notional). Fallbacks raisonnables si inconnu."""
    step_qty, tick_price, min_notional = 1e-6, 0.01, 5.0
    try:
        if USE_CCXT and EXCHANGE:
            m = EXCHANGE.markets[symbol] if symbol in EXCHANGE.markets else EXCHANGE.markets.get(f"{BASE_ASSET}/{QUOTE_ASSET}")
            if m:
                step_qty = m["limits"]["amount"]["min"] or 1e-6
                tick_price = m["precision"]["price"]
                if isinstance(tick_price, int):  # ccxt sometimes stores precision as digits
                    tick_price = 10 ** (-tick_price)
                min_notional = m["limits"]["cost"]["min"] or 5.0
        elif USE_BINANCE_SDK and CLIENT:
            info = CLIENT.get_symbol_info(symbol)
            for f in info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step_qty = float(f["stepSize"])
                elif f["filterType"] == "PRICE_FILTER":
                    tick_price = float(f["tickSize"])
                elif f["filterType"] in ("MIN_NOTIONAL","NOTIONAL"):
                    min_notional = float(f.get("minNotional") or f.get("notional"))
    except Exception as e:
        print("[WARN] get_filters fallback:", e, flush=True)
    return step_qty, tick_price, min_notional

def quantize_step(value, step):
    if step <= 0: return value
    return math.floor(value / step) * step

def round_to_tick(price, tick):
    if tick <= 0: return price
    return round(math.floor(price / tick) * tick, int(max(-math.log10(tick), 0)))

def get_free_quote_balance():
    try:
        if USE_CCXT and EXCHANGE:
            bal = EXCHANGE.fetch_balance()
            return float(bal[QUOTE_ASSET]["free"])
        if USE_BINANCE_SDK and CLIENT:
            b = CLIENT.get_asset_balance(asset=QUOTE_ASSET)
            return float(b["free"])
    except Exception as e:
        print("[WARN] get_free_quote_balance:", e, flush=True)
    return 0.0

def place_market_order(side: str, qty_base: float):
    if DRY_RUN or (not (USE_CCXT or USE_BINANCE_SDK)):
        print(f"[DRY] {side} qty={qty_base}", flush=True)
        return {"id": f"dry-{uuid.uuid4().hex[:8]}", "filled": qty_base, "price": None}
    try:
        if USE_CCXT and EXCHANGE:
            if side == "BUY":
                return EXCHANGE.create_market_buy_order(f"{BASE_ASSET}/{QUOTE_ASSET}", qty_base)
            else:
                return EXCHANGE.create_market_sell_order(f"{BASE_ASSET}/{QUOTE_ASSET}", qty_base)
        if USE_BINANCE_SDK and CLIENT:
            if side == "BUY":
                return CLIENT.order_market_buy(symbol=SYMBOL, quantity=qty_base)
            else:
                return CLIENT.order_market_sell(symbol=SYMBOL, quantity=qty_base)
    except Exception as e:
        print("[ERR] place_market_order:", e, flush=True)
        return None

# ========= Indicateurs =========
def ema(values, period):
    k = 2 / (period + 1.0)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val

def rsi(values, period=14):
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        if change >= 0: gains += change
        else: losses -= change
    avg_gain = gains / period
    avg_loss = losses / period if losses else 0.0
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float('inf')
    rsi_val = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float('inf')
        rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

# ========= Persistance état =========
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"position": None, "day": str(date.today()), "day_loss": 0.0, "consec_losses": 0}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

STATE = load_state()

# ========= Google Sheets wrappers =========
def gs_append_trade(ts_iso, side, qty_base, price, fee_quote, position_id, pnl_quote=None):
    """Ajoute une ligne trades : ts, date, side, qty_base, price, quote_value, fee_quote, position_id, pnl_quote"""
    quote_value = qty_base * price if (qty_base and price) else 0.0
    date_str = ts_iso.split("T")[0]
    row = [ts_iso, date_str, side, qty_base, price, quote_value, fee_quote or 0.0, position_id or "", "" if pnl_quote is None else pnl_quote]
    try:
        if gsh and hasattr(gsh, "append_trade_row"):
            gsh.append_trade_row(row)
        else:
            print("[GSheets] (mock) row:", row, flush=True)
    except Exception as e:
        print("[GSheets] erreur append:", e, flush=True)

def gs_daily_ready_log():
    try:
        print("[GSheets] Bilan journalier prêt ✅", flush=True)
    except Exception:
        pass

gs_daily_ready_log()

# ========= Log CFG (si tu récupères une config Sheet, garde-fou 'Aucun') =========
CFG = {
    'DRY_RUN': 'Aucun', 'TRADING_ENABLED': 'Aucun', 'ORDER_USDC': 'Aucun',
    'TAKE_PROFIT_PCT': 'Aucun', 'STOP_LOSS_PCT': 'Aucun',
    'DAILY_MAX_LOSS_USDC': 'Aucun', 'CONSECUTIVE_LOSS_LIMIT': 'Aucun',
    'DAILY_LOSSES_LIMIT': 'Aucun', 'SYMBOL': 'Aucun'
}
print("[CFG DEBUG]", CFG, flush=True)

# ========= Log param synthèse =========
print(f"--- BOT démarré | DRY_RUN={'Vrai' if DRY_RUN else 'Faux'} | TRADING_ENABLED={'Vrai' if TRADING_ENABLED else 'Faux'} | Symbole={SYMBOL} ---", flush=True)
print(f"[PARAMS] ORDER_USDC={int(ORDER_USDC)}, TP={TAKE_PROFIT_PCT}, SL={STOP_LOSS_PCT} RSI_BUY={RSI_BUY}, EMA_DEV={EMA_DEV}", flush=True)

# ========= Moteur d'une itération =========
def main_once():
    global STATE

    # Reset journalier si changement de jour UTC
    today = str(date.today())
    if STATE.get("day") != today:
        STATE["day"] = today
        STATE["day_loss"] = 0.0
        STATE["consec_losses"] = 0
        save_state(STATE)
        print("[INFO] Nouveau jour → compteurs remis à zéro", flush=True)

    # Récup OHLCV
    limit = max(RSI_PERIOD, EMA_PERIOD) + 5
    ohlcv = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    closes = [c[4] for c in ohlcv]
    if len(closes) < max(RSI_PERIOD, EMA_PERIOD) + 1:
        print("[WAIT] Pas assez d'historique", flush=True)
        return

    last_price = float(closes[-1])
    ema_val = float(ema(closes[-EMA_PERIOD:], EMA_PERIOD))
    rsi_val = float(rsi(closes[-(RSI_PERIOD+1):], RSI_PERIOD))

    # Position courante
    pos = STATE.get("position")

    # Gestion SELL (TP/SL) si position ouverte
    if pos:
        entry = float(pos["entry_price"])
        qty   = float(pos["qty"])
        tp_px = entry * (1 + TAKE_PROFIT_PCT)
        sl_px = entry * (1 - STOP_LOSS_PCT)

        log_signal("SELL-CHECK", last_price, rsi_val, ema_val, RSI_BUY, EMA_DEV,
                   TRADING_ENABLED, (STATE["day_loss"] >= DAILY_CAP), 0.0, 0.0)

        should_sell = last_price >= tp_px or last_price <= sl_px
        if should_sell:
            # arrondi qty/price selon filtres
            step_qty, tick_price, _ = get_filters(SYMBOL)
            sell_qty = max(quantize_step(qty, step_qty), step_qty)
            sell_px  = last_price  # au marché; le prix de remplissage sera proche
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            res = place_market_order("SELL", sell_qty)
            # estimation PnL (approx. sans frais)
            pnl_quote = (sell_px - entry) * sell_qty
            # Maj état
            if pnl_quote < 0:
                STATE["day_loss"] += abs(pnl_quote)
                STATE["consec_losses"] += 1
            else:
                STATE["consec_losses"] = 0
            STATE["position"] = None
            save_state(STATE)
            # Sheets
            gs_append_trade(ts, "SELL", sell_qty, sell_px, 0.0, pos["position_id"], pnl_quote)
            print(f"[TRADE] SELL qty={sell_qty} px≈{sell_px:.2f} pnl={pnl_quote:.2f} USDC | day_loss={STATE['day_loss']:.2f} consec={STATE['consec_losses']}", flush=True)
        return  # si position ouverte on ne cherche pas d'entrée

    # Vérrous de risque / habilitation
    risk_lock = (STATE["day_loss"] >= DAILY_CAP) or (STATE["consec_losses"] >= CONSECUTIVE_LOSS_LIMIT)
    free_q = get_free_quote_balance()
    log_signal("BUY-CHECK", last_price, rsi_val, ema_val, RSI_BUY, EMA_DEV,
               TRADING_ENABLED, risk_lock, free_q, ORDER_USDC)

    if not TRADING_ENABLED or risk_lock:
        return

    # Conditions d'entrée (RSI + Ecart EMA)
    ema_gap_ok = (abs(last_price - ema_val) / ema_val) >= EMA_DEV
    rsi_ok = rsi_val <= RSI_BUY
    if not (ema_gap_ok and rsi_ok):
        return

    # Filtres d'échange
    step_qty, tick_price, min_notional = get_filters(SYMBOL)
    if ORDER_USDC < min_notional:
        print(f"[BLOCK] ORDER_USDC({ORDER_USDC}) < MIN_NOTIONAL({min_notional})", flush=True)
        return
    if free_q + 1e-6 < ORDER_USDC:
        print(f"[BLOCK] Solde {QUOTE_ASSET} insuffisant: free={free_q:.2f}, need={ORDER_USDC:.2f}", flush=True)
        return

    # Calcul quantité & arrondi
    qty = ORDER_USDC / last_price
    qty = max(quantize_step(qty, step_qty), step_qty)

    # Envoi BUY
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res = place_market_order("BUY", qty)
    position_id = uuid.uuid4().hex[:12]
    STATE["position"] = {"position_id": position_id, "entry_price": last_price, "qty": qty, "ts": ts}
    save_state(STATE)
    gs_append_trade(ts, "BUY", qty, last_price, 0.0, position_id, pnl_quote=None)
    print(f"[TRADE] BUY qty={qty} px≈{last_price:.2f} (TP={TAKE_PROFIT_PCT*100:.2f}%, SL={STOP_LOSS_PCT*100:.2f}%)", flush=True)

# ========= Boucle infinie 24/7 =========
def run_forever():
    print("--- BOT loop up ---", flush=True)
    while True:
        try:
            main_once()
        except KeyboardInterrupt:
            print("Arrêt demandé (CTRL+C).", flush=True)
            break
        except Exception as e:
            print("[FATAL] Exception:", repr(e), flush=True)
            traceback.print_exc()
            time.sleep(5)
        time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    run_forever()
