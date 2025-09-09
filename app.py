#!/usr/bin/env python3
# app.py
from __future__ import annotations

import csv
import os
import sys
import time
import uuid
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict

# =========================
# Config & utilitaires
# =========================

PARIS_TZ = ZoneInfo("Europe/Paris")
TRADES_CSV_PATH = os.getenv("TRADES_CSV_PATH", "trades.csv")  # export de ta feuille

def _D(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))

def _parse_decimal_maybe_comma(s: Any, default: str = "0") -> Decimal:
    """
    Parse un décimal robuste aux virgules françaises ('1,23') et aux champs vides.
    """
    if s is None:
        return Decimal(default)
    st = str(s).strip()
    if st == "":
        return Decimal(default)
    st = st.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        return Decimal(st)
    except (InvalidOperation, ValueError):
        return Decimal(default)

def _env_decimal(name: str, default: str = "0") -> Decimal:
    return _parse_decimal_maybe_comma(os.getenv(name, default), default)

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

# ---------- Logger corrigé (stdout pour INFO/DEBUG, stderr pour WARNING+) ----------
class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("bot")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(fmt)

    # stdout: DEBUG & INFO
    h_out = logging.StreamHandler(sys.stdout)
    h_out.setLevel(logging.DEBUG)
    h_out.addFilter(MaxLevelFilter(logging.WARNING))
    h_out.setFormatter(formatter)

    # stderr: WARNING, ERROR, CRITICAL
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)
    h_err.setFormatter(formatter)

    logger.addHandler(h_out)
    logger.addHandler(h_err)
    logger.propagate = False
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
        logger.info("[GSheets] Bilan journalier prêt ✅")
        logger.info(f"ENV DEBUG : {env_debug}")
        logger.info(f"CFG DEBUG : {cfg_debug}")
        return cfg

# =========================
# Parsers & sources trades
# =========================

def parse_iso_dt(s: str) -> datetime:
    """
    Supporte:
      '2025-09-09T02:22:57.28924+00:00'
      '2025-09-09T02:22:57Z'
      '2025-09-09 02:22:57'
    Retourne un datetime aware (UTC si absent).
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

# ---------- Calcul PNL net (format "nouvelle feuille") ----------

def compute_pnl_net_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reçoit des lignes hétérogènes (BUY & SELL) avec au moins :
      - 'ts' (timestamp ISO)
      - 'side' ('BUY'/'SELL')
      - 'position_id'
      - 'fee_quote' (string/float/Decimal)
      - 'pnl_quote' (présent sur SELL, parfois vide)
      - 'date' (YYYY-MM-DD) optionnelle
    Retourne seulement les SELL, chacun enrichi de 'pnl_net' et 'closed_at'.
    """
    fees_by_pos: Dict[str, Decimal] = defaultdict(Decimal)
    # Additionner les frais de TOUTE la position (peu importe le jour d'ouverture)
    for r in rows:
        pid = str(r.get("position_id", "")).strip()
        if not pid:
            # si pas d'identifiant, on impute à une position unique
            pid = "__nopos__"
            r["position_id"] = pid
        fees_by_pos[pid] += _parse_decimal_maybe_comma(r.get("fee_quote", "0"))

    out: List[Dict[str, Any]] = []
    for r in rows:
        side = str(r.get("side", "")).upper()
        if side != "SELL":
            continue
        pid = str(r.get("position_id"))
        gross = _parse_decimal_maybe_comma(r.get("pnl_quote", "0"))
        fees = fees_by_pos.get(pid, Decimal("0"))
        ts_raw = r.get("ts") or r.get("closed_at") or ""
        try:
            closed = parse_iso_dt(str(ts_raw))
        except Exception:
            # si timestamp manquant, on considère maintenant (rare)
            closed = datetime.now(timezone.utc)
        out.append(
            {
                "position_id": pid,
                "pnl_usdc": gross - fees,     # <- NET en quote (USDC)
                "pnl_net": gross - fees,      # alias explicite
                "pnl_quote": gross,
                "fees_total": fees,
                "closed_at": closed.isoformat(),
                "date": r.get("date"),
            }
        )
    # Tri par timestamp de clôture
    out.sort(key=lambda x: x["closed_at"])
    return out

def load_trades_today() -> List[Dict[str, Any]]:
    """
    Charge les trades **clôturés aujourd'hui (Europe/Paris)**.
    Auto-détection du format CSV.
    - Ancien format : closed_at,pnl_usdc
    - Nouveau format : ts,date,side,fee_quote,position_id,pnl_quote,...
    Retourne une liste de dicts SELL:
      {"pnl_usdc": Decimal, "closed_at": iso, "position_id": "...", "date": "YYYY-MM-DD"}
    """
    trades: List[Dict[str, Any]] = []
    if not os.path.exists(TRADES_CSV_PATH):
        return trades

    with open(TRADES_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip() for h in (reader.fieldnames or [])]

        is_old = ("closed_at" in fieldnames and "pnl_usdc" in fieldnames)
        is_new = ("ts" in fieldnames and "side" in fieldnames and "position_id" in fieldnames)

        all_rows = list(reader)

        if is_old and not is_new:
            # Ancien format : déjà au format "trade clos" (une ligne = un trade clos)
            for row in all_rows:
                try:
                    ts = parse_iso_dt(row["closed_at"])
                    pnl = _parse_decimal_maybe_comma(row["pnl_usdc"], "0")
                    trades.append(
                        {
                            "pnl_usdc": pnl,
                            "closed_at": ts.isoformat(),
                            "position_id": row.get("position_id") or str(uuid.uuid4()),
                            "date": None,
                        }
                    )
                except Exception:
                    continue
        else:
            # Nouveau format : on calcule le PNL net par position et on ne garde que les SELL
            sell_rows = compute_pnl_net_rows(all_rows)
            trades.extend(sell_rows)

    # Filtrage "aujourd'hui" (Europe/Paris) sur le timestamp de clôture
    start, end = paris_today_bounds()
    today_trades = []
    for t in trades:
        try:
            ts = parse_iso_dt(t["closed_at"]).astimezone(PARIS_TZ)
            if start <= ts < end:
                today_trades.append(t)
        except Exception:
            continue

    return today_trades

# =========================
# Garde-fou journalier (sur PNL NET)
# =========================

def summarize_day(trades_today: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    trades_today doit contenir uniquement des trades CLOS (SELL),
    avec 'pnl_usdc' = PNL NET par trade.
    """
    daily_pnl = sum((_D(t.get("pnl_usdc", 0)) for t in trades_today), Decimal("0"))

    # Pertes consécutives -> on regarde les trades dans l'ordre chronologique
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

    # 1) Limite USDC absolue (PNL NET du jour)
    if daily_max_loss_usdc > 0 and pnl <= -daily_max_loss_usdc:
        reason = (
            "[GARDE][STOP] Limite de perte journalière (NET) atteinte : "
            f"PNL_net_journalier={pnl:.4f} USDC ≤ -{daily_max_loss_usdc:.4f} USDC. "
            f"Trades_clos_aujourd'hui={stats['trades_count']}."
        )
        return True, reason

    # 2) Pertes consécutives
    if consecutive_loss_limit > 0 and cons >= consecutive_loss_limit:
        reason = (
            "[GARDE][STOP] Limite de pertes consécutives atteinte : "
            f"pertes_consécutives={cons} ≥ {consecutive_loss_limit}. "
            f"PNL_net_journalier={pnl:.4f} USDC."
        )
        return True, reason

    # 3) Nombre total de trades perdants du jour
    if daily_losses_limit > 0 and loss_cnt >= daily_losses_limit:
        reason = (
            "[GARDE][STOP] Limite de trades perdants (jour) atteinte : "
            f"perdants_jour={loss_cnt} ≥ {daily_losses_limit}. "
            f"PNL_net_journalier={pnl:.4f} USDC."
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
            logger.warning(message)  # WARNING -> stderr (rouge)
            self._already_sent = True

# =========================
# Stratégie / boucle
# =========================

def strategy_tick(cfg: Config) -> None:
    """
    Branche ici ta stratégie réelle :
      - récupération du prix / signaux
      - décision BUY/SELL
      - envoi d'ordres si autorisé
    """
    # logger.info("[TICK] Vérification des signaux…")
    pass

def main() -> None:
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

        # (Optionnel) log du PnL NET du jour :
        # stats = summarize_day(trades_today)
        # logger.info(f"[QUOTIDIEN] {now_paris.strftime('%Y-%m-%d')} PNL_NET = {stats['daily_pnl']:.2f} USDC "
        #             f"(trades={stats['trades_count']}, pertes_cons={stats['consecutive_losses']})")

        stop, reason = should_stop_today(
            trades_today,
            daily_max_loss_usdc=cfg.DAILY_MAX_LOSS_USDC,
            consecutive_loss_limit=cfg.CONSECUTIVE_LOSS_LIMIT,
            daily_losses_limit=cfg.DAILY_LOSSES_LIMIT,
        )

        if stop:
            # On bloque seulement les nouvelles entrées pour aujourd'hui
            notifier.notify_once(reason)
        else:
            if cfg.TRADING_ENABLED:
                strategy_tick(cfg)

        # Tourne en continu (24/7), même quand on stoppe les entrées
        time.sleep(cfg.LOOP_SLEEP_SECONDS)

# =========================
# Entrée programme
# =========================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Arrêt manuel (CTRL+C).")
