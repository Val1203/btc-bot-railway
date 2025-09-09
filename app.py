#!/usr/bin/env python3
# app.py
from __future__ import annotations

import csv
import os
import time
import uuid
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# =========================
# Config & utilitaires
# =========================

PARIS_TZ = ZoneInfo("Europe/Paris")
TRADES_CSV_PATH = os.getenv("TRADES_CSV_PATH", "trades.csv")  # fermé par défaut

def _D(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))

def _env_decimal(name: str, default: str = "0") -> Decimal:
    return _D(os.getenv(name, default))

def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw) if raw != "" else default
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if raw in ("1", "true", "vrai", "yes", "y"):
        return True
    if raw in ("0", "false", "faux", "no", "n"):
        return False
    return default

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("bot")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger

logger = setup_logger()

@dataclass
class Config:
    DRY_RUN: bool
    TRADING_ENABLED: bool
    SYMBOL: str
    ORDER_USDC: Decimal
    TAKE_PROFIT_PCT: Decimal
    STOP_LOSS_PCT: Decimal
    RSI_BUY: Decimal
    EMA_DEV: Decimal
    DAILY_MAX_LOSS_USDC: Decimal
    CONSECUTIVE_LOSS_LIMIT: int
    DAILY_LOSSES_LIMIT: int
    LOOP_SLEEP_SECONDS: int

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            DRY_RUN=_env_bool("DRY_RUN", False),
            TRADING_ENABLED=_env_bool("TRADING_ENABLED", True),
            SYMBOL=os.getenv("SYMBOL", "BTCUSDC"),
            ORDER_USDC=_env_decimal("ORDER_USDC", "300"),
            TAKE_PROFIT_PCT=_env_decimal("TAKE_PROFIT_PCT", "0.007"),
            STOP_LOSS_PCT=_env_decimal("STOP_LOSS_PCT", "0.004"),
            RSI_BUY=_env_decimal("RSI_BUY", "33"),
            EMA_DEV=_env_decimal("EMA_DEV", "0.0035"),
            DAILY_MAX_LOSS_USDC=_env_decimal("DAILY_MAX_LOSS_USDC", "8"),
            CONSECUTIVE_LOSS_LIMIT=_env_int("CONSECUTIVE_LOSS_LIMIT", 3),
            DAILY_LOSSES_LIMIT=_env_int("DAILY_LOSSES_LIMIT", 0),
            LOOP_SLEEP_SECONDS=_env_int("LOOP_SLEEP_SECONDS", 5),
        )
        # Logs style "ENV DEBUG" / "CFG DEBUG" (comme tes captures)
        env_debug = {
            "DRY_RUN": str(cfg.DRY_RUN).lower(),
            "TRADING_ENABLED": str(cfg.TRADING_ENABLED).lower(),
            "ORDER_USDC": str(cfg.ORDER_USDC),
            "TAKE_PROFIT_PCT": str(cfg.TAKE_PROFIT_PCT),
            "STOP_LOSS_PCT": str(cfg.STOP_LOSS_PCT),
            "DAILY_MAX_LOSS_USDC": str(cfg.DAILY_MAX_LOSS_USDC),
            "CONSECUTIVE_LOSS_LIMIT": str(cfg.CONSECUTIVE_LOSS_LIMIT),
            "DAILY_LOSSES_LIMIT": str(cfg.DAILY_LOSSES_LIMIT),
            "SYMBOL": cfg.SYMBOL,
        }
        cfg_debug = {k: "Aucun" for k in env_debug.keys()}
        logger.info(f"ENV DEBUG : {env_debug}")
        logger.info(f"CFG DEBUG : {cfg_debug}")
        return cfg

# =========================
# Parsers & sources trades
# =========================

def parse_iso_dt(s: str) -> datetime:
    """
    Parse timestamps type:
      - '2025-09-09T02:22:57.28924+00:00'
      - '2025-09-09T02:22:57Z'
      - '2025-09-09 02:22:57'
    Retourne un datetime timezone-aware (UTC si non précisé).
    """
    s = s.strip()
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        # fallback tolérant
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            raise

def paris_today_bounds(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    if now is None:
        now = datetime.now(PARIS_TZ)
    start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=PARIS_TZ)
    end = start + timedelta(days=1)
    return start, end

def load_trades_today() -> List[Dict[str, Any]]:
    """
    ⚠️ Par défaut lit un CSV local `trades.csv`.
    Colonnes minimales:
      closed_at,pnl_usdc
    Exemple:
      2025-09-09T02:22:57Z,-1.2311384
    Adapte cette fonction à ta source réelle (GSheets / DB / API).
    """
    trades: List[Dict[str, Any]] = []
    if not os.path.exists(TRADES_CSV_PATH):
        return trades

    start, end = paris_today_bounds()
    with open(TRADES_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = parse_iso_dt(row["closed_at"])
                # convertir en Europe/Paris pour le filtrage
                ts_paris = ts.astimezone(PARIS_TZ)
                if not (start <= ts_paris < end):
                    continue
                pnl = _D(row["pnl_usdc"])
                trades.append(
                    {
                        "pnl_usdc": pnl,
                        "closed_at": ts.isoformat(),
                        "id": row.get("order_id") or str(uuid.uuid4()),
                    }
                )
            except Exception:
                # ligne incomplète -> on ignore
                continue
    return trades

# =========================
# Garde-fou journalier
# =========================

def summarize_day(trades_today: List[Dict[str, Any]]) -> Dict[str, Any]:
    daily_pnl = sum((_D(t.get("pnl_usdc", 0)) for t in trades_today), _D(0))
    consecutive_losses = 0
    for t in reversed(trades_today):
        if _D(t.get("pnl_usdc", 0)) < 0:
            consecutive_losses += 1
        else:
            break
    daily_loss_count = sum(1 for t in trades_today if _D(t.get("pnl_usdc", 0)) < 0)
    return {
        "daily_pnl": daily_pnl,
        "consecutive_losses": consecutive_losses,
        "daily_loss_count": daily_loss_count,
        "trades_count": len(trades_today),
    }

def should_stop_today(
    trades_today: List[Dict[str, Any]],
    *,
    daily_max_loss_usdc: Decimal,
    consecutive_loss_limit: int = 0,
    daily_losses_limit: int = 0,
) -> Tuple[bool, Optional[str]]:
    stats = summarize_day(trades_today)
    pnl = stats["daily_pnl"]
    cons = stats["consecutive_losses"]
    loss_cnt = stats["daily_loss_count"]

    # 1) Limite USDC absolue (pas de division par ORDER_USDC)
    if daily_max_loss_usdc > 0 and pnl <= -daily_max_loss_usdc:
        reason = (
            "[GARDE][STOP] Limite de perte journalière atteinte : "
            f"PNL_journalier={pnl:.4f} USDC ≤ -{daily_max_loss_usdc:.4f} USDC. "
            f"Trades_clos_aujourd'hui={stats['trades_count']}."
        )
        return True, reason

    # 2) Pertes consécutives
    if consecutive_loss_limit > 0 and cons >= consecutive_loss_limit:
        reason = (
            "[GARDE][STOP] Limite de pertes consécutives atteinte : "
            f"pertes_consécutives={cons} ≥ {consecutive_loss_limit}. "
            f"PNL_journalier={pnl:.4f} USDC."
        )
        return True, reason

    # 3) Nombre total de trades perdants du jour
    if daily_losses_limit > 0 and loss_cnt >= daily_losses_limit:
        reason = (
            "[GARDE][STOP] Limite de trades perdants (jour) atteinte : "
            f"perdants_jour={loss_cnt} ≥ {daily_losses_limit}. "
            f"PNL_journalier={pnl:.4f} USDC."
        )
        return True, reason

    return False, None

class StopNotifier:
    """
    Évite le spam: on logge la raison d'arrêt au plus UNE fois par jour.
    """
    def __init__(self):
        self._last_date: Optional[str] = None
        self._already_sent: bool = False

    def reset_if_new_day(self, now_paris: datetime):
        key = now_paris.strftime("%Y-%m-%d")
        if self._last_date != key:
            self._last_date = key
            self._already_sent = False

    def notify_once(self, message: str):
        if not self._already_sent:
            logger.warning(message)
            self._already_sent = True

# =========================
# Stratégie / boucle
# =========================

def strategy_tick(cfg: Config) -> None:
    """
    💡 ICI branche ta stratégie réelle:
      - récupération du prix / signaux
      - décision BUY/SELL
      - envoi d'ordres si ALLOW_NEW_ENTRIES
    Dans cet exemple, on ne place pas d'ordres (squelette propre).
    """
    # Exemple de log "heartbeat" stratégique :
    # logger.info("[TICK] Vérification des signaux…")
    pass

def main() -> None:
    logger.info("[GSheets] Bilan journalier prêt ✅")  # pour rester fidèle à tes logs
    cfg = Config.from_env()

    logger.info(f"--- BOT démarré | DRY_RUN={'Faux' if not cfg.DRY_RUN else 'Vrai'} "
                f"| TRADING_ENABLED={'Vrai' if cfg.TRADING_ENABLED else 'Faux'} "
                f"| Symbole={cfg.SYMBOL} ---")

    logger.info(f"[PARAMS] ORDER_USDC={cfg.ORDER_USDC}, TP={cfg.TAKE_PROFIT_PCT}, "
                f"SL={cfg.STOP_LOSS_PCT} RSI_BUY={cfg.RSI_BUY}, EMA_DEV={cfg.EMA_DEV}")

    notifier = StopNotifier()

    while True:
        now_paris = datetime.now(PARIS_TZ)
        notifier.reset_if_new_day(now_paris)

        trades_today = load_trades_today()
        # Log récap quotidien (facultatif, décommenter si utile)
        # stats = summarize_day(trades_today)
        # logger.info(f"[QUOTIDIEN] {now_paris.strftime('%Y-%m-%d')} PNL = {stats['daily_pnl']:.2f} USDC")

        stop, reason = should_stop_today(
            trades_today,
            daily_max_loss_usdc=cfg.DAILY_MAX_LOSS_USDC,
            consecutive_loss_limit=cfg.CONSECUTIVE_LOSS_LIMIT,
            daily_losses_limit=cfg.DAILY_LOSSES_LIMIT,
        )

        if stop:
            # On bloque seulement les NOUVELLES ENTRÉES pour aujourd'hui
            notifier.notify_once(reason)
        else:
            # Sécurité: on ne trade que si TRADING_ENABLED
            if cfg.TRADING_ENABLED:
                strategy_tick(cfg)

        # Toujours tourner (24/7), même quand on stoppe les entrées
        time.sleep(cfg.LOOP_SLEEP_SECONDS)

# =========================
# Entrée programme
# =========================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Arrêt manuel (CTRL+C).")
